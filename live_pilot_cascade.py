"""
live_pilot_cascade.py
Two-arm live downstream execution benchmark using held-out tasks.

Holdout note
------------
94 tasks are reserved (~27.5% of the full 342-task pool, slightly above the
standard 20% but acceptable given dataset size). This pilot uses 30 of those
94 tasks (31.9% of holdout used; 68 tasks remain in reserve), never seen
during training.

Arms
----
  UNGATED : Agent A -> Agent B -> grade          (baseline Pass@1)
  GATED   : Agent A -> [gate] -> Agent B -> grade

Gate = two-stage cascade, Operating point PRE-REGISTERED from OOF:
  Stage 1 : Random Forest on 11 surface features (NLI columns dropped so the
            live inference features exactly match the training features).
            PASS_THRESHOLD = 0.45 (Youden-derived on 11-feature OOF; at this
            point only 3/248 clean training handoffs fall below, while most
            structural corruptions — truncation, over-summarization — are
            blocked cheaply).
  Stage 2 : REAL LLM judge (SUFFICIENT/INSUFFICIENT) on every row Stage 1
            would PASS. This is the band where semantic corruptions
            (invert_objective, fake_completion, ...) hide, indistinguishable
            from clean handoffs on surface features (AUDIT: ~0.50 AUROC).

  BLOCK (Stage 1 or Stage 2) -> RECOVERY = regenerate code from the ORIGINAL
  problem statement directly with Agent B. This is the proven recovery route
  (see live_downstream_pilot.py): unlike an Agent-A retry fed the possibly
  corrupted handoff, Agent B on the pristine problem is always at least as
  informative as any handoff, so a false block costs ~nothing. Judge errors
  fail OPEN (PASS) so a broken judge call never damages a good handoff.

Task mix (30 total)
-------------------
  15 clean handoffs    : test false-positive cost (gate wrongly blocking good handoffs)
  15 corrupted handoffs: test true-positive benefit (gate catching bad handoffs)

  Corruption uses 4 validated reliable types (no linguistic-pattern dependency),
  cycling evenly across all 15 corrupted tasks (each type appears 3-4 times):
    truncation, over_summarization, invert_objective, fake_completion

Output
------
  live_pilot_results.csv   -- per-task detail
  Console summary          -- Pass@1 both arms (+ McNemar, bootstrap CI),
                              recovery/regression counts, judge stats
"""

import concurrent.futures
import os
import sys
import threading
import warnings

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

warnings.filterwarnings("ignore")

import joblib  # noqa: F401  (kept for API parity)
import numpy as np
import pandas as pd
from scipy.stats import binom
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from dotenv import load_dotenv
load_dotenv()

from agents import agent_a_plan, agent_b_code, N_KEYS
from rollout_runner import grade
from corruption import truncate, over_summarize, invert_objective, fake_completion
from feature_extraction import (
    compute_cosine_similarity, compute_entity_overlap, compute_length_ratio,
    compute_sentence_count_ratio, compute_section_coverage, compute_verbatim_copy_rate,
    compute_trailing_specificity, compute_constraint_count, compute_role_pronoun_rate,
    compute_novel_api_rate, compute_function_name_preserved,
)
from two_stage_cascade_optionB_llmjudge import _call_judge_pooled

# ─── Config ───────────────────────────────────────────────────────────────────

TRAINING_CSV   = "final_dataset_full_features.csv"
HOLDOUT_FILE   = "holdout_task_ids.txt"
OUTPUT_CSV     = "live_pilot_results.csv"
PASS_THRESHOLD = 0.45          # pre-registered 11-feature OOF Youden operating point
N_PILOT_TASKS  = 94            # full holdout: 47 clean + 47 corrupted
USE_LLM_JUDGE  = True          # Stage-2 judge on the Stage-1 PASS band (fail-open)
WORKERS        = 3             # keep concurrency low: keys can share a per-model quota
MAX_RESUME_ROUNDS = 5          # rerun error/rate-limited rows until the 30-task set is clean

# NLI columns are NOT used here: they were a constant-proxy at inference and
# silently shifted scores. Identical 11 features train and serve the gate.
FEATURE_COLS = [
    "cosine_similarity", "entity_overlap", "length_ratio", "sentence_count_ratio",
    "section_coverage", "verbatim_copy_rate", "trailing_specificity", "constraint_count",
    "role_pronoun_rate", "novel_api_rate", "function_name_preserved",
]

# 4 validated corruption types — these are STRESS corruptions that reliably
# realize (the audit's rollout_runner discards corruptions that don't break the
# code — a "weak negative" is not a valid damaged handoff). compose() applies
# `invert_objective` first; if the handoff had no matchable keyword (it is by
# design a no-op when nothing can be inverted), the truncation fallback
# guarantees the corrupted sample is genuinely damaged.
def _corrupt_invert(handoff: str, entry_point: str) -> str:
    out = invert_objective(handoff)
    return out if out != handoff else truncate(handoff, keep_fraction=0.12)


def _corrupt_fake(handoff: str, entry_point: str) -> str:
    return fake_completion(truncate(handoff, keep_fraction=0.60))


CORRUPTION_CYCLE = [
    ("truncation",         lambda h, ep: truncate(h, keep_fraction=0.10)),
    ("over_summarization", lambda h, ep: over_summarize(h, sentence_count=1)),
    ("invert_objective",   _corrupt_invert),
    ("fake_completion",    _corrupt_fake),
]


# ─── Load holdout tasks ───────────────────────────────────────────────────────

def load_holdout_tasks(n: int = N_PILOT_TASKS) -> list[dict]:
    """
    Returns the first `n` reachable holdout tasks.
    All 94 holdout IDs are confirmed reachable in the task maps.
    """
    from humaneval_tasks import HUMANEVAL_TASKS
    from mbpp_tasks import MBPP_TASKS
    from tasks.coding_tasks import TASKS as hand_tasks
    from tasks.hard_tasks_3 import HARD_TASKS_3

    all_tasks = {t["id"]: t for t in HUMANEVAL_TASKS + MBPP_TASKS + hand_tasks + HARD_TASKS_3}

    with open(HOLDOUT_FILE) as f:
        holdout_ids = [l.strip() for l in f if l.strip()]

    tasks = []
    for tid in holdout_ids:
        if tid in all_tasks:
            tasks.append(all_tasks[tid])
        if len(tasks) == n:
            break
    return tasks


# ─── Stage-1 model (train at startup from features CSV) ──────────────────────

def train_stage1_model():
    """
    Trains the Stage-1 RF gate on final_dataset_full_features.csv using the
    SAME 11 feature columns used at inference time (no NLI features).
    """
    df = pd.read_csv(TRAINING_CSV)
    X = df[FEATURE_COLS].values
    y = df["label"].values
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=6, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )),
    ])
    model.fit(X, y)
    return model


# ─── Feature extraction (live, for a single handoff) ─────────────────────────

def extract_live_features(handoff: str, problem: str, entry_point: str) -> dict:
    return {
        "cosine_similarity":       compute_cosine_similarity(handoff, problem),
        "entity_overlap":          compute_entity_overlap(handoff, problem),
        "length_ratio":            compute_length_ratio(handoff, problem),
        "sentence_count_ratio":    compute_sentence_count_ratio(handoff, problem),
        "section_coverage":        compute_section_coverage(handoff, problem, entry_point),
        "verbatim_copy_rate":      compute_verbatim_copy_rate(handoff, problem),
        "trailing_specificity":    compute_trailing_specificity(handoff, problem),
        "constraint_count":        compute_constraint_count(handoff, problem),
        "role_pronoun_rate":       compute_role_pronoun_rate(handoff, problem),
        "novel_api_rate":          compute_novel_api_rate(handoff, problem),
        "function_name_preserved": compute_function_name_preserved(handoff, problem, entry_point),
    }


# ─── Gate: Stage-1 RF (cheap block) + Stage-2 LLM judge (escalated PASS) ─────

model_lock = threading.Lock()

def cascade_gate(handoff: str, problem: str, entry_point: str,
                 stage1_model) -> dict:
    """
    Two-stage gate. Returns:
      decision       : "PASS" or "BLOCK"
      stage          : 1 (RF block) or 2 (judge verdict) or None on a Stage-1 pass
                       that the judge approved
      score          : RF predicted probability of a good handoff
      judge_verdict  : 1 = SUFFICIENT, 0 = INSUFFICIENT, None = not escalated
    """
    feats = extract_live_features(handoff, problem, entry_point)
    X     = pd.DataFrame([feats])[FEATURE_COLS].values

    with model_lock:
        score = float(stage1_model.predict_proba(X)[0, 1])

    # Stage 1: cheap structural block
    if score < PASS_THRESHOLD:
        return {"decision": "BLOCK", "stage": 1, "score": score,
                "judge_verdict": None}

    # Stage 2: escalate; semantic corruptions hide at high Stage-1 scores.
    if USE_LLM_JUDGE:
        try:
            verdict = _call_judge_pooled(problem, handoff)   # 1=SUFFICIENT
        except Exception:
            verdict = 1                                       # fail OPEN
        if verdict == 0:                                      # INSUFFICIENT
            return {"decision": "BLOCK", "stage": 2, "score": score,
                    "judge_verdict": 0}
        return {"decision": "PASS", "stage": 2, "score": score,
                "judge_verdict": 1}

    return {"decision": "PASS", "stage": 1, "score": score,
            "judge_verdict": None}


print_lock = threading.Lock()

def process_task(i: int, task: dict, stage1_model) -> dict:
    tid        = task["id"]
    problem    = task["problem"]
    ep         = task["entry_point"]
    test_code  = task["test_code"]
    is_corrupt = (i % 2 == 1)   # odd-indexed tasks get corruption
    ctype      = "clean"

    logs = []
    def log(msg): logs.append(msg)

    log(f"[{i+1:02d}/{N_PILOT_TASKS}] {tid}  ({ep})")

    # ── Agent A: generate initial handoff ────────────────────────────────
    try:
        raw_handoff = agent_a_plan(problem)
    except Exception as exc:
        log(f"  Agent A failed: {exc} — skipping")
        with print_lock:
            print("\n".join(logs) + "\n")
        return None

    # ── Inject corruption if this is a corrupted task ───────────────────
    if is_corrupt:
        ctype_name, corrupt_fn = CORRUPTION_CYCLE[(i // 2) % len(CORRUPTION_CYCLE)]
        handoff = corrupt_fn(raw_handoff, ep)
        ctype = ctype_name
        log(f"  Corruption injected: {ctype}")
    else:
        handoff = raw_handoff
        log(f"  Clean handoff")

    # ── Arm 1: Ungated ───────────────────────────────────────────────────
    ungated_error = False
    try:
        code_ungated = agent_b_code(handoff, ep)
        pass_ungated = grade(code_ungated, ep, test_code)
    except Exception as exc:
        ungated_error = True
        log(f"  Ungated Agent B error: {exc}")
        pass_ungated = False
    log(f"  Ungated  : {'PASS' if pass_ungated else 'FAIL'}"
        f"{'  [API error — will be re-run]' if ungated_error else ''}")

    # ── Arm 2: Gated (two-stage cascade) ─────────────────────────────────
    gate_result = cascade_gate(handoff, problem, ep, stage1_model)
    decision   = gate_result["decision"]
    gate_stage = gate_result["stage"]
    gate_score = gate_result["score"]
    judge_verdict = gate_result["judge_verdict"]

    log(f"  Gate     : {decision}  (Stage {gate_stage}, score={gate_score:.3f}"
        f"{f', judge=INSUFFICIENT' if judge_verdict == 0 else ''})")

    if decision == "PASS":
        # Guard approved the handoff — identical code to the ungated arm.
        pass_gated = pass_ungated
        action     = "handoff_code"
        gated_error = ungated_error   # same code path; a U error taints this row too
    else:
        # Blocked — recover from the ORIGINAL problem (proven route).
        gated_error = False
        try:
            code_gated = agent_b_code(problem, ep)
            pass_gated = grade(code_gated, ep, test_code)
            action     = f"fallback_raw_stage{gate_stage}"
        except Exception as exc:
            gated_error = True
            log(f"  Fallback error: {exc}")
            pass_gated = False
            action     = f"fallback_raw_stage{gate_stage}_error"

    log(f"  Gated    : {'PASS' if pass_gated else 'FAIL'}  [{action}] ({gate_score:.3f})"
        f"{'  [API error — will be re-run]' if gated_error else ''}")

    with print_lock:
        print("\n".join(logs) + "\n")

    return {
        "task_id":           tid,
        "source":            task.get("source", ""),
        "entry_point":       ep,
        "corruption_type":   ctype,
        "is_corrupted":      is_corrupt,
        "stage1_score":      round(gate_score, 4),
        "stage1_blocked":    gate_stage == 1 and decision == "BLOCK",
        "judge_escalated":   judge_verdict is not None,
        "judge_verdict":     judge_verdict,
        "gate_stage":        gate_stage,
        "gate_decision":     decision,
        "action":            action,
        "ungated_passed":    pass_ungated,
        "gated_passed":      pass_gated,
        "ungated_error":     ungated_error,
        "gated_error":       gated_error,
    }


# ─── Statistics ──────────────────────────────────────────────────────────────

def mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar on discordant pairs (b=regression, c=recovery)."""
    total = b + c
    if total == 0:
        return 1.0
    p = 0.0
    for k in range(min(b, c) + 1):
        p += binom.pmf(k, total, 0.5)
    return 2.0 * min(p, 1.0)


def bootstrap_ci(df, n_boot=5000, seed=42) -> tuple:
    """Percentile CI on mean(gated) - mean(ungated) per paired task row."""
    delta = (df["gated_passed"].astype(int) - df["ungated_passed"].astype(int)).to_numpy()
    rng = np.random.default_rng(seed)
    draws = rng.choice(delta, size=(n_boot, len(delta)), replace=True).mean(axis=1)
    return np.percentile(draws, 2.5), np.percentile(draws, 97.5)


# ─── Main ─────────────────────────────────────────────────────────────────────

RESULT_COLUMNS = [
    "task_id", "source", "entry_point", "corruption_type", "is_corrupted",
    "stage1_score", "stage1_blocked", "judge_escalated", "judge_verdict",
    "gate_stage", "gate_decision", "action",
    "ungated_passed", "gated_passed", "ungated_error", "gated_error",
]


def load_existing_results(path: str) -> pd.DataFrame:
    """Loads prior results IF they use the current schema (has error flags).
    Older runs (different gate design) are deliberately ignored."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=RESULT_COLUMNS)
    df = pd.read_csv(path)
    if "ungated_error" not in df.columns:
        print(f"  NOTE: {path} uses the old schema — ignoring stale rows "
              f"(different gate design, not comparable).")
        return pd.DataFrame(columns=RESULT_COLUMNS)
    missing = [c for c in RESULT_COLUMNS if c not in df.columns]
    for c in missing:
        df[c] = False if c.endswith("error") else ""
    return df[RESULT_COLUMNS]


def is_row_clean(row) -> bool:
    try:
        return not bool(row["ungated_error"]) and not bool(row["gated_error"])
    except Exception:
        return False


def run_round(to_run: list, stage1_model) -> dict:
    """Runs a subset of (index, task) pairs; returns {task_id: result_dict}."""
    out = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(process_task, i, task, stage1_model): task["id"]
            for i, task in to_run
        }
        for future in concurrent.futures.as_completed(futures):
            tid = futures[future]
            try:
                res = future.result()
            except Exception as exc:
                print(f"  worker crashed on {tid}: {exc}")
                continue
            if res is not None:
                out[tid] = res
    return out


def main():
    print("=" * 72)
    print("  Live Pilot — Two-Stage Cascade (RF + LLM Judge), Two-Arm Evaluation")
    print(f"  Stage-1 threshold : {PASS_THRESHOLD}  (pre-registered 11-feature OOF Youden)")
    print(f"  Stage-2           : LLM judge on escalated PASS band, fail-open "
          f"{'ON' if USE_LLM_JUDGE else 'OFF'}")
    print(f"  Recovery          : Agent B on ORIGINAL problem when gate blocks")
    print(f"  Corruption        : 4 validated types, cycling across 15 corrupted tasks")
    print(f"  Pilot tasks       : {N_PILOT_TASKS}  (15 clean + 15 corrupted, {N_PILOT_TASKS}/94 holdout)")
    print("=" * 72)

    print("\nTraining Stage-1 gate from features CSV (11 features, no NLI)...")
    stage1_model = train_stage1_model()
    print("Stage-1 gate ready.")

    tasks = load_holdout_tasks(N_PILOT_TASKS)
    print(f"Loaded {len(tasks)} holdout tasks.\n")

    # ── Resume-aware round loop: (re)run only incomplete / API-errored rows ──
    rounds = 0
    while rounds < MAX_RESUME_ROUNDS:
        rounds += 1
        existing = load_existing_results(OUTPUT_CSV)
        done = {r["task_id"]: r for _, r in existing.iterrows() if is_row_clean(r)}

        to_run = [(i, t) for i, t in enumerate(tasks)
                  if t["id"] not in done]
        if not to_run:
            print(f"[round {rounds}] all {len(done)} tasks complete and clean — done.")
            break

        n_err = sum(1 for _, r in existing.iterrows() if not is_row_clean(r))
        print(f"[round {rounds}] {len(to_run)} task(s) to (re)run "
              f"({len(done)} clean done, {n_err} prior API-error rows rerun) "
              f"using {WORKERS} workers...")

        results = run_round(to_run, stage1_model)
        if not results:
            print("  No rows returned this round — stopping to avoid spinning.")
            break

        combined = [r.to_dict() for r in done.values()] + [results[tid] for tid in results]
        pd.DataFrame(combined, columns=RESULT_COLUMNS).to_csv(OUTPUT_CSV, index=False)
        n_bad = sum(not is_row_clean(r) for r in combined)
        print(f"  Saved {len(combined)} rows; {n_bad} with API errors will be rerun.\n")

    # ── Final dataset & statistics ──────────────────────────────────────────
    df = load_existing_results(OUTPUT_CSV)
    df = df[df["task_id"].isin({t["id"] for t in tasks})].copy()

    n = len(df)
    print(f"Final clean rows: {n}/{N_PILOT_TASKS}")
    if df["ungated_error"].any() or df["gated_error"].any():
        bad = df[df["ungated_error"] | df["gated_error"]]["task_id"].tolist()
        print(f"  WARNING — rows still tainted by API errors: {bad}")
    if n == 0:
        print("No results to report.")
        return

    clean_df   = df[~df["is_corrupted"]]
    corrupt_df = df[df["is_corrupted"]]

    ung_all  = df["ungated_passed"].mean() * 100
    gat_all  = df["gated_passed"].mean()   * 100
    ung_cln  = clean_df["ungated_passed"].mean() * 100 if len(clean_df)   else float("nan")
    gat_cln  = clean_df["gated_passed"].mean()   * 100 if len(clean_df)   else float("nan")
    ung_crp  = corrupt_df["ungated_passed"].mean() * 100 if len(corrupt_df) else float("nan")
    gat_crp  = corrupt_df["gated_passed"].mean()   * 100 if len(corrupt_df) else float("nan")

    # Per-task paired outcomes
    n_both_pass = ((df["ungated_passed"]) & (df["gated_passed"])).sum()
    n_both_fail = ((~df["ungated_passed"]) & (~df["gated_passed"])).sum()
    b = ((df["ungated_passed"]) & (~df["gated_passed"])).sum()   # regression
    c = ((~df["ungated_passed"]) & (df["gated_passed"])).sum()   # recovery
    mcnemar = mcnemar_p(int(b), int(c))
    ci_lo, ci_hi = bootstrap_ci(df)

    n_intercepted = (corrupt_df["gate_decision"] == "BLOCK").sum()
    n_fp          = (clean_df["gate_decision"] == "BLOCK").sum()

    # Confusion: did a BLOCK target a handoff that actually failed ungated?
    tp_block = ((df["gate_decision"] == "BLOCK") & (~df["ungated_passed"])).sum()
    fp_block = ((df["gate_decision"] == "BLOCK") & (df["ungated_passed"])).sum()

    print("=" * 72)
    print("  LIVE CASCADE PILOT RESULTS (two-stage, fallback recovery)")
    print("=" * 72)
    print(f"\n  {'Metric':<45} {'Ungated':>8} {'Gated':>8} {'Delta':>8}")
    print(f"  {'-'*69}")
    print(f"  {'Pass@1 — All tasks (n=' + str(n) + ')':<45} {ung_all:>7.1f}% {gat_all:>7.1f}% {gat_all-ung_all:>+7.1f}%")
    print(f"  {'Pass@1 — Clean tasks (n=' + str(len(clean_df)) + ')':<45} {ung_cln:>7.1f}% {gat_cln:>7.1f}% {gat_cln-ung_cln:>+7.1f}%")
    print(f"  {'Pass@1 — Corrupted tasks (n=' + str(len(corrupt_df)) + ')':<45} {ung_crp:>7.1f}% {gat_crp:>7.1f}% {gat_crp-ung_crp:>+7.1f}%")
    print(f"\n  Per-pair outcomes: both-pass {n_both_pass} | both-fail {n_both_fail} | "
          f"gate hurt {b} | gate recovered {c}")
    print(f"  McNemar (exact, 2-sided)  : p={'%.4f' % mcnemar}   "
          f"{'significant' if mcnemar < 0.05 else 'n.s.'} at alpha=0.05")
    print(f"  Bootstrap 95% CI for delta: [{ci_lo:+.3f}, {ci_hi:+.3f}]")

    print(f"\n  Corrupted runs intercepted before Agent B : {n_intercepted}/{len(corrupt_df)}")
    print(f"  False blocks on clean handoffs            : {n_fp}/{len(clean_df)}  (recovered via fallback)")
    print(f"  BLOCK -> recovery success                 : "
          f"{c + ((df['gate_decision']=='BLOCK') & df['gated_passed'] & df['ungated_passed']).sum()} blocked that passed")
    print(f"  Blocked handoffs that would have passed   : {fp_block} (false-positive cost, zero after fallback)")

    if USE_LLM_JUDGE and df["judge_escalated"].any():
        esc = df[df["judge_escalated"]]
        flipped = df[(df["judge_escalated"]) & (df["judge_verdict"] == 0)]
        print(f"\n  Stage-2 LLM judge: escalated {esc['judge_escalated'].sum()}/{len(df)} "
              f"Stage-1 passes, flipped {len(flipped)} to INSUFFICIENT (recovered)")

    if len(corrupt_df):
        print("\n  Per-corruption-type Pass@1 (U vs G):")
        for ctype in ["truncation", "over_summarization", "invert_objective", "fake_completion"]:
            sub = corrupt_df[corrupt_df["corruption_type"] == ctype]
            if len(sub) == 0:
                continue
            u = 100 * sub["ungated_passed"].mean()
            g = 100 * sub["gated_passed"].mean()
            print(f"    {ctype:<20} n={len(sub):<2}  {u:5.1f}%  ->  {g:5.1f}%" )

    print("\n  Results saved to : " + OUTPUT_CSV)
    print("=" * 72)


if __name__ == "__main__":
    main()