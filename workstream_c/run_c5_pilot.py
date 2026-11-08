"""C5 Pilot Evaluation Runner.

Evaluates 10 tasks on seed 42 across:
1. no_gate: Executes initial clean handoff directly.
2. blind_retry: Retries up to 2 times with a generic prompt if Stage 1 blocks.
3. shap_hinted_retry: Retries up to 2 times with the worst SHAP feature hint if Stage 1 blocks.

Logs:
- Hint text, triggering feature, and stage-1 scores before/after retry.
- Block rate and retry distribution.
- Tokens, latency, and function-name mismatch tracking.
- Corrected call estimate for 100 tasks x 3 seeds.
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["MPLCONFIGDIR"] = os.path.join(os.environ.get("TEMP", "C:/tmp"), "matplotlib")

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workstream_c.harness.agents_interface import (
    AGENT_B_SYSTEM_PROMPT,
    LLMAgentClient,
)
from workstream_c.harness.config import (
    DEFAULT_MODEL_A,
    DEFAULT_MODEL_B,
    HEADLINE_THRESHOLD_RAW,
    RESULTS_DIR,
)
from workstream_c.harness.execution_sandbox import grade_code
from workstream_c.harness.feature_pipeline import Stage1Classifier
from workstream_c.harness.key_manager import KeyPool, get_key_slice
from workstream_c.harness.task_loader import load_live_test_tasks
from workstream_c.run_hidden_problem import (
    BLIND_RETRY_TEMPLATE,
    SHAP_HINTED_RETRY_TEMPLATE,
)

logger = logging.getLogger("C5Pilot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

MANIFEST_PATH = REPO_ROOT / "workstream_c" / "manifests" / "manifest_hidden_problem_100.json"
CLEAN_HANDOFFS_PATH = REPO_ROOT / "workstream_c" / "results" / "clean_agent_a_handoffs.csv"
PILOT_CSV_PATH = RESULTS_DIR / "c5_pilot_results.csv"
PILOT_REPORT_PATH = RESULTS_DIR / "c5_pilot_report.md"

PILOT_FIELDNAMES = [
    "task_id",
    "source",
    "condition",          # "no_gate" | "blind_retry" | "shap_hinted_retry"
    "seed",
    "initial_score",
    "initial_decision",   # "PASS" | "BLOCK"
    "retries_used",       # 0, 1, 2
    "final_stage1_score",
    "triggering_feature",
    "hint_text",
    "retry_history",      # JSON string list of {retry: int, score_before, score_after, feat, hint}
    "outcome",            # 1 (passed unit tests) | 0 (failed)
    "tokens_prompt",
    "tokens_completion",
    "latency_ms",
    "error_message",
    "fn_mismatch_initial",
    "fn_mismatch_resolved",
]


def run_c5_pilot(
    limit: int = 10,
    seed: int = 42,
    key_slice: str = "15:18",  # Keys 16 to 18
    threshold: float = HEADLINE_THRESHOLD_RAW,
    max_retries: int = 2,
    model_a: str = DEFAULT_MODEL_A,
    model_b: str = DEFAULT_MODEL_B,
):
    print("=" * 80, flush=True)
    print(f"C5 PILOT EVALUATION (10 tasks, Seed {seed}, Key Slice: {key_slice})", flush=True)
    print("=" * 80, flush=True)

    # 1. Load manifest tasks
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)
    task_ids = manifest_data["tasks"][:limit]
    all_tasks = {t["id"]: t for t in load_live_test_tasks()}
    tasks = [all_tasks[tid] for tid in task_ids]
    print(f"Loaded {len(tasks)} tasks: {task_ids}", flush=True)

    # 2. Load cached initial clean handoffs for seed
    with open(CLEAN_HANDOFFS_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        clean_lookup = {r["task_id"]: r["handoff_text"] for r in reader if int(r["seed"]) == seed}

    keys = get_key_slice(key_slice)
    key_pool = KeyPool(keys=keys)
    print(f"KeyPool initialized with slice '{key_slice}': {len(key_pool.keys)} active keys.", flush=True)
    
    # Dedicated cache for C5 pilot
    pilot_cache_path = REPO_ROOT / "cache" / f"llm_cache_c5_pilot_s{seed}.json"
    client = LLMAgentClient(key_pool=key_pool, cache_path=pilot_cache_path)
    
    # Pre-populate pilot cache with main run cached Agent B responses for seed 42 to guarantee zero redundant calls
    main_cache_path = REPO_ROOT / "cache" / f"llm_cache_s{seed}.json"
    if main_cache_path.exists():
        with open(main_cache_path, "r", encoding="utf-8") as f:
            main_cache = json.load(f)
            client.cache.update(main_cache)
        print(f"Pre-loaded {len(main_cache)} entries from main cache s{seed}.", flush=True)

    classifier = Stage1Classifier()
    conditions = ["no_gate", "blind_retry", "shap_hinted_retry"]
    results = []

    # Pre-score all initial handoffs
    initial_diagnostics = {}
    for task in tasks:
        tid = task["id"]
        ep = task["entry_point"]
        prob = task["problem"]
        ht = clean_lookup[tid]
        score_res = classifier.score(ht, prob, entry_point=ep, threshold=threshold)
        initial_diagnostics[tid] = score_res

    print("\nInitial Handoff Quality Scores (Seed 42):", flush=True)
    for tid in task_ids:
        diag = initial_diagnostics[tid]
        dec = "BLOCK" if diag["score"] < threshold else "PASS"
        print(f"  {tid:<10}: Score = {diag['score']:.4f} -> {dec:<5} | Worst Feature: {diag.get('worst_feature')}", flush=True)

    for t_idx, task in enumerate(tasks, 1):
        tid = task["id"]
        source = task.get("source", "unknown")
        ep = task["entry_point"]
        prob = task["problem"]
        test_c = task["test_code"]
        init_ht = clean_lookup[tid]
        init_diag = initial_diagnostics[tid]
        init_score = init_diag["score"]
        init_dec = "BLOCK" if init_score < threshold else "PASS"

        # Check if initial handoff causes a function name mismatch in generated code
        init_b_res = client.code_agent_b(
            handoff_or_problem=init_ht,
            entry_point=ep,
            model=model_b,
            temperature=0.0,
            seed=seed,
            is_raw_problem=False,
        )
        init_pass, init_err, _ = grade_code(init_b_res.get("text", ""), ep, test_c)
        init_fn_mismatch = bool("not found in generated code" in init_err)

        for cond in conditions:
            total_p_tokens = 0
            total_c_tokens = 0
            total_latency_ms = 0.0
            retries_used = 0
            retry_history = []
            current_ht = init_ht
            current_score = init_score
            current_diag = init_diag

            if cond == "no_gate":
                final_ht = init_ht
                final_score = init_score
                b_res = init_b_res
                passed = init_pass
                err_msg = init_err
                fn_mismatch_resolved = False
                total_p_tokens += b_res.get("prompt_tokens", 0)
                total_c_tokens += b_res.get("completion_tokens", 0)
                total_latency_ms += b_res.get("latency_ms", 0.0)
            else:
                # Retry logic
                while retries_used < max_retries and current_score < threshold:
                    retries_used += 1
                    score_before = current_score
                    trig_feat = current_diag.get("worst_feature", "unknown")
                    hint_txt = current_diag.get("hint", "")

                    if cond == "blind_retry":
                        retry_prompt = BLIND_RETRY_TEMPLATE.format(problem=prob, previous_handoff=current_ht)
                    else:  # shap_hinted_retry
                        retry_prompt = SHAP_HINTED_RETRY_TEMPLATE.format(problem=prob, previous_handoff=current_ht, hint=hint_txt)

                    # Agent A sampled at temp=0.7 with seed variation
                    retry_res = client.generate(
                        prompt=retry_prompt,
                        model=model_a,
                        temperature=0.7,
                        seed=seed + retries_used * 100,
                    )
                    total_p_tokens += retry_res.get("prompt_tokens", 0)
                    total_c_tokens += retry_res.get("completion_tokens", 0)
                    total_latency_ms += retry_res.get("latency_ms", 0.0)

                    current_ht = retry_res.get("text", current_ht)
                    current_diag = classifier.score(current_ht, prob, entry_point=ep, threshold=threshold)
                    score_after = current_diag["score"]
                    current_score = score_after

                    retry_history.append({
                        "retry": retries_used,
                        "score_before": round(score_before, 4),
                        "score_after": round(score_after, 4),
                        "delta": round(score_after - score_before, 4),
                        "decision": "PASS" if score_after >= threshold else "BLOCK",
                        "triggering_feature": trig_feat,
                        "hint": hint_txt,
                    })
                    logger.info(f"[{tid}] {cond} Retry {retries_used}: {score_before:.3f} -> {score_after:.3f} ({trig_feat})")

                final_ht = current_ht
                final_score = current_score

                # Run Agent B on final handoff
                if retries_used == 0:
                    b_res = init_b_res
                    passed = init_pass
                    err_msg = init_err
                    total_p_tokens += b_res.get("prompt_tokens", 0)
                    total_c_tokens += b_res.get("completion_tokens", 0)
                    total_latency_ms += b_res.get("latency_ms", 0.0)
                else:
                    b_res = client.code_agent_b(
                        handoff_or_problem=final_ht,
                        entry_point=ep,
                        model=model_b,
                        temperature=0.0,
                        seed=seed,
                        is_raw_problem=False,
                    )
                    total_p_tokens += b_res.get("prompt_tokens", 0)
                    total_c_tokens += b_res.get("completion_tokens", 0)
                    total_latency_ms += b_res.get("latency_ms", 0.0)

                    passed, err_msg, grade_ms = grade_code(b_res.get("text", ""), ep, test_c)
                    total_latency_ms += grade_ms

                # Check if retry resolved function name mismatch
                final_fn_mismatch = bool("not found in generated code" in err_msg)
                fn_mismatch_resolved = bool(init_fn_mismatch and not final_fn_mismatch)

            # Record result row
            row = {
                "task_id": tid,
                "source": source,
                "condition": cond,
                "seed": seed,
                "initial_score": round(init_score, 4),
                "initial_decision": init_dec,
                "retries_used": retries_used,
                "final_stage1_score": round(final_score, 4),
                "triggering_feature": init_diag.get("worst_feature", ""),
                "hint_text": init_diag.get("hint", ""),
                "retry_history": json.dumps(retry_history),
                "outcome": 1 if passed else 0,
                "tokens_prompt": total_p_tokens,
                "tokens_completion": total_c_tokens,
                "latency_ms": round(total_latency_ms, 2),
                "error_message": err_msg,
                "fn_mismatch_initial": init_fn_mismatch,
                "fn_mismatch_resolved": fn_mismatch_resolved,
            }
            results.append(row)
            print(f"  [{cond:<18}] {tid}: Retries={retries_used}, Init={init_score:.3f} -> Final={final_score:.3f}, Passed={passed}", flush=True)

    # 3. Write CSV
    with open(PILOT_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PILOT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSaved pilot results to {PILOT_CSV_PATH}", flush=True)

    # 4. Generate Pilot Report
    import pandas as pd
    res_df = pd.DataFrame(results)

    n_tasks = len(tasks)
    initial_blocked = (res_df[res_df["condition"] == "no_gate"]["initial_decision"] == "BLOCK").sum()
    block_rate = (initial_blocked / n_tasks) * 100

    print("\n" + "=" * 80, flush=True)
    print("C5 PILOT SUMMARY METRICS", flush=True)
    print("=" * 80, flush=True)
    print(f"Tasks Evaluated:              {n_tasks}")
    print(f"Initial Clean Block Rate:     {initial_blocked}/{n_tasks} ({block_rate:.1f}%)")

    for cond in conditions:
        sub = res_df[res_df["condition"] == cond]
        p1 = sub["outcome"].mean() * 100
        mean_r = sub["retries_used"].mean()
        r_dist = sub["retries_used"].value_counts().to_dict()
        mean_p_tok = sub["tokens_prompt"].mean()
        mean_c_tok = sub["tokens_completion"].mean()
        mean_lat = sub["latency_ms"].mean()
        print(f"\nCondition: {cond}")
        print(f"  Pass@1:                     {p1:.2f}% ({sub['outcome'].sum()}/{len(sub)})")
        print(f"  Mean Retries:               {mean_r:.2f} (Distribution: {r_dist})")
        print(f"  Mean Prompt Tokens:         {mean_p_tok:.1f}")
        print(f"  Mean Completion Tokens:     {mean_c_tok:.1f}")
        print(f"  Mean Latency:               {mean_lat:.1f} ms")

    blocked_no_gate = res_df[(res_df["condition"] == "no_gate") & (res_df["initial_decision"] == "BLOCK")]
    fn_mismatches_initial = blocked_no_gate["fn_mismatch_initial"].sum()
    print(f"\nFunction-Name Mismatch Tracking on Blocked Handoffs:")
    print(f"  Blocked handoffs with initial fn mismatch: {fn_mismatches_initial} / {len(blocked_no_gate)}")

    blind_resolved = res_df[res_df["condition"] == "blind_retry"]["fn_mismatch_resolved"].sum()
    shap_resolved = res_df[res_df["condition"] == "shap_hinted_retry"]["fn_mismatch_resolved"].sum()
    print(f"  Blind Retry resolved mismatches:           {blind_resolved}")
    print(f"  SHAP-Hinted Retry resolved mismatches:     {shap_resolved}")

    # Count API calls from pilot cache vs pre-loaded cache
    pilot_calls = len(client.cache) - (len(main_cache) if main_cache_path.exists() else 0)
    print(f"\nAPI Calls Used in this Pilot Run: {pilot_calls}")
    print(f"Key Slice Used: {key_slice} (Keys 16 to 18 only; Keys 1 to 15 untouched)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--key-slice", type=str, default="15:18")
    args = parser.parse_args()
    run_c5_pilot(limit=args.limit, seed=args.seed, key_slice=args.key_slice)
