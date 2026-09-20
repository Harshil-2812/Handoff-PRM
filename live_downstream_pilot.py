"""
Live Downstream Execution Benchmark on 20 Real Tasks

Runs genuine, un-faked live agent execution with real Python unit test grading:
1. Agent A generates handoffs for 20 coding tasks.
   (10 clean tasks + 10 corrupted tasks to test resilience).
2. Ungated Condition:
   Agent B implements code strictly from the handoff. Real unit tests graded via grade().
3. Gated Condition:
   Handoff checked by PRM Gate. If blocked, fallback to problem prompt is triggered.
   Real unit tests graded via grade().
4. Measures actual real-world Pass@1 comparison.
"""

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import os
import time
import pandas as pd
import numpy as np
import joblib

from tasks.coding_tasks import TASKS
from agents import agent_a_plan, agent_b_code
from corruption import truncate, over_summarize, drop_tool_result, fake_completion, invert_objective
from feature_extraction import (
    compute_cosine_similarity, compute_entity_overlap, compute_length_ratio,
    compute_sentence_count_ratio, compute_section_coverage, compute_verbatim_copy_rate,
    compute_trailing_specificity, compute_constraint_count, compute_role_pronoun_rate,
    compute_novel_api_rate, compute_function_name_preserved, compute_signature_param_diff
)

OUTPUT_CSV = "results/tables/live_downstream_pilot_results.csv"
os.makedirs("results/tables", exist_ok=True)

# Load trained PRM model
bundle = joblib.load("results/models/prm_final.joblib")
prm_model = bundle["model"]
feature_cols = bundle["feature_cols"]

FAST_COLS = [c for c in feature_cols if not c.startswith("nli_")]


def grade(code: str, entry_point: str, test_code: str) -> tuple[bool, str]:
    """Executes code + unit test in an isolated namespace."""
    namespace = {}
    try:
        exec(code, namespace)
        exec(test_code, namespace)
        fn = namespace.get(entry_point)
        if fn is None:
            return False, f"Function '{entry_point}' not found in generated code."
        passed = bool(namespace["test"](fn))
        return passed, "Success" if passed else "Assertion failed in test harness."
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:100]}"


def score_handoff_fast(handoff: str, problem: str, entry_point: str) -> float:
    """Computes PRM probability for a single handoff using the 12 fast features."""
    feats = {
        "cosine_similarity":        compute_cosine_similarity(handoff, problem),
        "entity_overlap":           compute_entity_overlap(handoff, problem),
        "length_ratio":             compute_length_ratio(handoff, problem),
        "sentence_count_ratio":     compute_sentence_count_ratio(handoff, problem),
        "section_coverage":         compute_section_coverage(handoff, problem, entry_point),
        "verbatim_copy_rate":       compute_verbatim_copy_rate(handoff, problem),
        "trailing_specificity":     compute_trailing_specificity(handoff, problem),
        "constraint_count":         compute_constraint_count(handoff, problem),
        "role_pronoun_rate":        compute_role_pronoun_rate(handoff, problem),
        "novel_api_rate":           compute_novel_api_rate(handoff, problem),
        "function_name_preserved":  compute_function_name_preserved(handoff, problem, entry_point),
        "signature_param_diff":     compute_signature_param_diff(handoff, problem),
        "nli_contradiction_max":    0.0,
        "nli_entailment_mean":      0.5,
    }
    row = pd.DataFrame([feats])[feature_cols]
    prob = float(prm_model.predict_proba(row)[0, 1])
    return prob


def main():
    test_tasks = TASKS[:20]
    print("=" * 75)
    print(f"STARTING LIVE 20-TASK DOWNSTREAM EXECUTION EXPERIMENT")
    print(f"Total Tasks: {len(test_tasks)}")
    print("=" * 75)

    GATE_THRESHOLD = 0.45  # Operating point
    results = []

    for i, task in enumerate(test_tasks):
        task_id = task["id"]
        prob = task["problem"]
        entry = task["entry_point"]
        test_c = task["test_code"]

        print(f"\n--- [{i+1}/20] Task: {task_id} ({entry}) ---")

        # 1. Agent A generates initial handoff
        print("  Calling Agent A (Planner)...")
        try:
            raw_handoff = agent_a_plan(prob)
        except Exception as e:
            print(f"  Agent A failed: {e}")
            continue
        time.sleep(2)  # rate limit headroom

        # 2. Inject corruption into even-indexed tasks to test resilience
        is_corrupted = (i % 2 == 1)
        if is_corrupted:
            corr_choice = (i // 2) % 4
            if corr_choice == 0:
                handoff = truncate(raw_handoff, keep_fraction=0.25)
                c_name = "truncation"
            elif corr_choice == 1:
                handoff = over_summarize(raw_handoff, sentence_count=1)
                c_name = "over_summarization"
            elif corr_choice == 2:
                handoff = drop_tool_result(raw_handoff)
                c_name = "tool_result_drop"
            else:
                handoff = fake_completion(raw_handoff)
                c_name = "fake_completion"
            print(f"  Injected corruption: {c_name}")
        else:
            handoff = raw_handoff
            c_name = "clean"
            print("  Using clean handoff.")

        # 3. Ungated Condition: Agent B receives handoff directly
        print("  [Condition 1: Ungated] Calling Agent B on handoff...")
        try:
            code_ungated = agent_b_code(handoff, entry)
        except Exception as e:
            code_ungated = f"# Error: {e}"
        time.sleep(2)

        pass_ungated, err_ungated = grade(code_ungated, entry, test_c)
        print(f"    Ungated Grade: {'PASSED' if pass_ungated else 'FAILED'} ({err_ungated})")

        # 4. Gated Condition: PRM inspects handoff
        prm_score = score_handoff_fast(handoff, prob, entry)
        gate_passed = (prm_score >= GATE_THRESHOLD)
        print(f"    PRM Score: {prm_score:.3f} -> Gate Decision: {'PASS' if gate_passed else 'BLOCK (Intercepted)'}")

        if gate_passed:
            # PRM approved: use the code generated from handoff
            pass_gated, err_gated = pass_ungated, err_ungated
            action_taken = "Passed handoff to Agent B"
        else:
            # PRM blocked: protect Agent B by falling back to raw problem statement directly
            print("    Gate blocked! Falling back to raw problem prompt for Agent B...")
            try:
                code_fallback = agent_b_code(prob, entry)
            except Exception as e:
                code_fallback = f"# Error: {e}"
            time.sleep(2)
            pass_gated, err_gated = grade(code_fallback, entry, test_c)
            action_taken = "Fallback to raw problem statement"

        print(f"    Gated Grade: {'PASSED' if pass_gated else 'FAILED'} ({err_gated})")

        results.append({
            "task_id": task_id,
            "entry_point": entry,
            "corruption_type": c_name,
            "is_corrupted": is_corrupted,
            "prm_score": round(prm_score, 4),
            "gate_decision": "PASS" if gate_passed else "BLOCK",
            "action_taken": action_taken,
            "ungated_passed": pass_ungated,
            "ungated_error": err_ungated,
            "gated_passed": pass_gated,
            "gated_error": err_gated,
        })

    # Summary
    df_res = pd.DataFrame(results)
    df_res.to_csv(OUTPUT_CSV, index=False)

    n_total = len(df_res)
    ungated_rate = df_res["ungated_passed"].mean() * 100.0
    gated_rate = df_res["gated_passed"].mean() * 100.0

    print("\n" + "=" * 75)
    print("LIVE DOWNSTREAM EXPERIMENT RESULTS (REAL UNIT TESTS)")
    print("=" * 75)
    print(f"Total Live Tasks Evaluated: {n_total}")
    print(f"Ungated Multi-Agent Pass@1: {ungated_rate:.1f}% ({df_res['ungated_passed'].sum()}/{n_total})")
    print(f"Gated Multi-Agent Pass@1:   {gated_rate:.1f}% ({df_res['gated_passed'].sum()}/{n_total})")
    print(f"Absolute Pass@1 Boost:      {gated_rate - ungated_rate:+.1f}%")

    # Corrupted tasks specifically
    corrupt_df = df_res[df_res["is_corrupted"]]
    print(f"\nOn Corrupted Tasks Specifically (n={len(corrupt_df)}):")
    print(f"  Ungated Pass@1: {corrupt_df['ungated_passed'].mean() * 100.0:.1f}% ({corrupt_df['ungated_passed'].sum()}/{len(corrupt_df)})")
    print(f"  Gated Pass@1:   {corrupt_df['gated_passed'].mean() * 100.0:.1f}% ({corrupt_df['gated_passed'].sum()}/{len(corrupt_df)})")
    print(f"  Corruptions Intercepted by Gate: {sum(corrupt_df['gate_decision'] == 'BLOCK')}/{len(corrupt_df)} ({sum(corrupt_df['gate_decision'] == 'BLOCK')/len(corrupt_df)*100.0:.1f}%)")
    print("=" * 75)


if __name__ == "__main__":
    main()
