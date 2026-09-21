"""
Handoff PRM — Two-Stage Cascade, Option B: Real LLM-Judge Stage 2
=====================================================================

Corrected design after diagnostic: semantic corruptions (invert_objective,
negate_edge_case, signature_rename, wrong_algorithm_name) score HIGH on
Stage 1 (0.79-0.83), indistinguishable from clean handoffs (0.78) —
they do NOT sit in an uncertain middle band. Stage 1 is confidently
wrong on these, not unsure.

New escalation rule: escalate every row Stage 1 would PASS (score >= tau),
since that's where a wrongly-passed semantic corruption hides, camouflaged
among genuinely good handoffs. Rows Stage 1 confidently BLOCKS are trusted
as-is (blocking is the cheap, safe default; passing is the costly-if-wrong
decision that deserves a real check).

Stage 2 = an actual LLM-judge call (not the weak nli_entailment_mean
feature), applied ONLY to the escalated subset to keep cost bounded.
Uses multiple API keys in parallel (same pattern as your Step 5/6 pipeline).

Requires: pip install google-generativeai pandas numpy scikit-learn
"""

import ast
import os
import queue
import re
import threading
import time
import warnings
import concurrent.futures
from contextlib import contextmanager
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors as genai_errors

load_dotenv()

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

# ----------------------------------------------------------------------
# 0. Config
# ----------------------------------------------------------------------

INPUT_PATH = "final_dataset_full_features.csv"
OUTPUT_PATH = "cascade_results_optionB.csv"

# Pass threshold: rows scoring >= this are escalated for verification
# before being trusted as "pass". Rows below are trusted as "block"
# directly from Stage 1 (cheap corruptions like over_summarization,
# tool_result_drop, truncation are reliably caught here already).
PASS_THRESHOLD = 0.50

# Keys loaded from .env (GEMINI_KEYS=['key1','key2',...] or GEMINI_API_KEY=key)
def _load_keys() -> list[str]:
    raw = os.getenv("GEMINI_KEYS", "").strip()
    if raw:
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, (list, tuple)) and parsed:
                return [str(k).strip() for k in parsed if str(k).strip()]
        except (ValueError, SyntaxError):
            pass
    single = os.getenv("GEMINI_API_KEY", "").strip()
    if single:
        return [single]
    raise RuntimeError(
        "No API keys found. Set GEMINI_KEYS=['key1','key2',...] "
        "or GEMINI_API_KEY=key in your .env"
    )

GEMINI_KEYS = _load_keys()
N_KEYS       = len(GEMINI_KEYS)

# Judge model — gemini-3.5-flash-lite, 10 RPM/key hard cap
MODEL_NAME  = "gemini-3.5-flash-lite"

# Per-key RPM cap: hard limit is 15 RPM, we enforce 10 to leave a buffer.
JUDGE_RPM_PER_KEY = 10
JUDGE_MAX_RETRIES = 5

FEATURE_COLS = [
    "cosine_similarity", "entity_overlap", "length_ratio", "sentence_count_ratio",
    "section_coverage", "verbatim_copy_rate", "trailing_specificity", "constraint_count",
    "role_pronoun_rate", "novel_api_rate", "function_name_preserved",
    "nli_entailment_mean",
]

JUDGE_PROMPT_TEMPLATE = """You are verifying whether a handoff message correctly and \
completely represents the original problem statement below, with no factual, \
algorithmic, or schema errors introduced.

ORIGINAL PROBLEM:
{problem}

HANDOFF MESSAGE TO VERIFY:
{handoff}

Check specifically for:
- Inverted objectives (e.g. ascending swapped for descending, max swapped for min)
- Inverted edge-case/boundary logic (e.g. inclusive swapped for exclusive)
- Wrong algorithm or data-structure names
- Renamed or altered function signatures
- False or conflicting constraints not in the original
- Hollow/generic filler text replacing real reasoning

Respond with EXACTLY ONE WORD: SUFFICIENT or INSUFFICIENT.
"""


# ----------------------------------------------------------------------
# 1. Stage 1 — reuse existing model to get OOF scores
# ----------------------------------------------------------------------

def get_stage1_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=6, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )),
    ])


def compute_stage1_oof(X, y, groups, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    oof_proba = np.zeros(len(y))
    for train_idx, test_idx in gkf.split(X, y, groups):
        assert len(set(groups[train_idx]) & set(groups[test_idx])) == 0, "LEAKAGE detected"
        model = get_stage1_model()
        model.fit(X[train_idx], y[train_idx])
        oof_proba[test_idx] = model.predict_proba(X[test_idx])[:, 1]
    return oof_proba


# ----------------------------------------------------------------------
# 2. Stage 2 — real LLM-judge call, parallelized with key pool
# ----------------------------------------------------------------------

# ── Per-key rate limiter ──────────────────────────────────────────────
class _RateLimiter:
    """Enforces a minimum gap between API calls on one key."""
    def __init__(self, rpm: int):
        self._min_interval = 60.0 / rpm
        self._last_call    = 0.0
        self._lock         = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            elapsed   = time.monotonic() - self._last_call
            sleep_for = self._min_interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._last_call = time.monotonic()


# ── Thread-safe key pool with daily-quota retirement ──────────────────
class _KeyPool:
    """Queue of (genai.Client, _RateLimiter) pairs.

    acquire() yields (entry, retire_callback).  Call retire_callback() to
    permanently remove a key whose daily quota is gone; otherwise the key
    is returned to the pool automatically on exit.
    """
    def __init__(self, keys: list[str], rpm: int):
        self._lock    = threading.Lock()
        self._q: queue.Queue = queue.Queue()
        self._active: set    = set()
        self._entries        = []
        for key in keys:
            c   = genai.Client(api_key=key)
            lim = _RateLimiter(rpm)
            entry = (c, lim)
            self._entries.append(entry)
            self._active.add(id(entry))
            self._q.put(entry)

    @contextmanager
    def acquire(self):
        item     = self._q.get()
        retired  = [False]

        def retire():
            retired[0] = True
            with self._lock:
                self._active.discard(id(item))
                try:
                    self._entries.remove(item)
                except ValueError:
                    pass
            print(f"    [judge pool] key retired (daily quota) — "
                  f"{self.size()} key(s) still active")

        try:
            yield item, retire
        finally:
            if not retired[0]:
                self._q.put(item)

    def size(self) -> int:
        with self._lock:
            return len(self._active)


# Module-level pool — created once, shared across all threads.
_JUDGE_POOL = _KeyPool(GEMINI_KEYS, JUDGE_RPM_PER_KEY)

JUDGE_GEN_CONFIG = types.GenerateContentConfig(max_output_tokens=16)


def _call_judge_pooled(problem: str, handoff: str) -> int:
    """
    Calls the LLM judge once using the shared key pool.
    - Rate limiting: enforced per key via _RateLimiter.wait().
    - Quota retirement: if a key's daily quota is gone it is retired and
      the call is retried automatically on the next available key.
    - Fail-open on parse errors: returns 1 (SUFFICIENT) so a bad judge
      response doesn't wrongly flip a Stage-1 pass to a block.
    Raises RuntimeError only when all keys have been retired.
    """
    prompt = JUDGE_PROMPT_TEMPLATE.format(problem=problem, handoff=handoff)

    while True:
        if _JUDGE_POOL.size() == 0:
            raise RuntimeError("All judge API keys have exhausted their daily quota")

        with _JUDGE_POOL.acquire() as ((client, limiter), retire):
            for attempt in range(JUDGE_MAX_RETRIES):
                limiter.wait()
                try:
                    response = client.models.generate_content(
                        model=MODEL_NAME,
                        contents=prompt,
                        config=JUDGE_GEN_CONFIG,
                    )
                    text = (response.text or "").strip().upper()
                    if "INSUFFICIENT" in text:
                        return 0
                    if "SUFFICIENT" in text:
                        return 1
                    print(f"    [judge] unparseable response: {text[:80]!r} — failing open")
                    return 1

                except genai_errors.ClientError as err:
                    if getattr(err, "code", None) != 429:
                        print(f"    [judge] non-quota API error: {err} — failing open")
                        return 1
                    msg = str(getattr(err, "message", "")) + str(err)
                    if "PerDay" in msg:
                        retire()
                        break   # outer while retries with next key
                    delay = (2.0 ** attempt)
                    print(f"    [judge pool] rate limited, sleeping {delay:.0f}s "
                          f"(attempt {attempt+1}/{JUDGE_MAX_RETRIES})")
                    time.sleep(delay + 1.0)

                except Exception as exc:
                    print(f"    [judge] unexpected error: {exc} — failing open")
                    return 1
            else:
                # All retries exhausted on this key without a quota hit
                print("    [judge] max retries exceeded — failing open")
                return 1
            # Broke out of for-loop because key was retired; outer while retries.


def run_stage2_parallel(escalated_df: pd.DataFrame) -> dict:
    """
    Runs the LLM judge on every escalated row using the shared _JUDGE_POOL.
    Workers acquire keys via the pool (rate-limited, retire-on-quota).
    Returns {row_index: verdict (0 or 1)}.
    """
    rows    = list(escalated_df.iterrows())
    results = {}
    n_workers = N_KEYS   # one worker per live key

    print(f"Running Stage-2 LLM judge on {len(rows)} escalated rows "
          f"({N_KEYS} keys, {n_workers} workers, {JUDGE_RPM_PER_KEY} RPM/key)...")

    def worker(idx_row):
        idx, row = idx_row
        verdict = _call_judge_pooled(row["problem"], row["handoff"])
        return idx, verdict

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(worker, item): item[0] for item in rows}
        for n, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                idx, verdict = future.result()
            except concurrent.futures.CancelledError:
                continue
            except Exception as exc:
                idx = futures[future]
                print(f"    [judge] worker error for row {idx}: {exc} — failing open")
                results[idx] = 1
                continue
            results[idx] = verdict
            if (n + 1) % 20 == 0 or (n + 1) == len(rows):
                print(f"  {n + 1}/{len(rows)} judged  "
                      f"({_JUDGE_POOL.size()} key(s) still active)")

    return results


# ----------------------------------------------------------------------
# 3. Main
# ----------------------------------------------------------------------

def main():
    print(f"Loading {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows, {df['task_id'].nunique()} unique tasks")

    X = df[FEATURE_COLS].values
    y = df["label"].values
    groups = df["task_id"].values

    print("\nDeriving Stage-1 OOF scores...")
    oof_proba = compute_stage1_oof(X, y, groups)
    df["stage1_score"] = oof_proba
    print(f"Stage-1 overall OOF AUROC: {roc_auc_score(y, oof_proba):.4f}")

    stage1_pred = (df["stage1_score"] >= PASS_THRESHOLD).astype(int)
    escalated_mask = stage1_pred == 1  # escalate every row Stage 1 would PASS
    n_escalated = escalated_mask.sum()
    print(f"\nEscalation rule: score >= {PASS_THRESHOLD} (Stage-1 'pass' verdict)")
    print(f"Rows escalated to Stage 2: {n_escalated} / {len(df)} ({100*n_escalated/len(df):.1f}%)")
    print("\nEscalated rows by corruption type:")
    print(df.loc[escalated_mask, "corruption_type"].value_counts())

    escalated_df = df.loc[escalated_mask]

    # ---- Real LLM-judge call on escalated subset ----
    stage2_verdicts = run_stage2_parallel(escalated_df)

    cascade_pred = stage1_pred.copy()
    for idx, verdict in stage2_verdicts.items():
        cascade_pred.loc[idx] = verdict

    # ---- Evaluation ----
    y_escalated = y[escalated_mask.values]
    stage1_only_escalated = stage1_pred.loc[escalated_mask].values  # all 1s by construction
    cascade_escalated = cascade_pred.loc[escalated_mask].values

    print("\n" + "=" * 70)
    print("PERFORMANCE ON THE ESCALATED (STAGE-1 'PASS') SUBSET")
    print("=" * 70)
    print(f"n = {n_escalated}")
    print(f"Stage-1-only accuracy (all predicted PASS by construction): "
          f"{accuracy_score(y_escalated, stage1_only_escalated):.4f}")
    print(f"Cascade (real LLM-judge Stage 2) accuracy:                  "
          f"{accuracy_score(y_escalated, cascade_escalated):.4f}")
    print(f"Cascade (real LLM-judge Stage 2) F1:                        "
          f"{f1_score(y_escalated, cascade_escalated):.4f}")

    print("\n" + "=" * 70)
    print("OVERALL: STAGE-1-ONLY vs. FULL CASCADE (all 483 rows)")
    print("=" * 70)
    print(f"Stage-1-only accuracy: {accuracy_score(y, stage1_pred):.4f}")
    print(f"Full-cascade accuracy: {accuracy_score(y, cascade_pred):.4f}")
    print(f"Stage-1-only F1:       {f1_score(y, stage1_pred):.4f}")
    print(f"Full-cascade F1:       {f1_score(y, cascade_pred):.4f}")

    # ---- Per-corruption-type recovery: did Stage 2 catch the 4 weak types? ----
    print("\n" + "=" * 70)
    print("PER-CORRUPTION-TYPE ACCURACY, ESCALATED SUBSET ONLY")
    print("=" * 70)
    for ctype in escalated_df["corruption_type"].unique():
        mask = (df["corruption_type"] == ctype) & escalated_mask
        if mask.sum() == 0:
            continue
        y_c = y[mask.values]
        s1_c = stage1_pred.loc[mask].values
        casc_c = cascade_pred.loc[mask].values
        print(f"{ctype:<25} n={mask.sum():<4} "
              f"Stage1-only acc={accuracy_score(y_c, s1_c):.3f}  "
              f"Cascade acc={accuracy_score(y_c, casc_c):.3f}")

    df["cascade_prediction"] = cascade_pred
    df["escalated_to_stage2"] = escalated_mask
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
