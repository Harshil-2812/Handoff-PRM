"""
Phase 7 & 8 -- Gated vs Ungated and SHAP-Guided vs Blind Retry Experiment Script

Evaluates multi-agent coding performance across 3 pipeline configurations:
  1. Ungated (Baseline): Agent A -> Agent B -> Code Execution
  2. Gated (Blind Retry): Agent A -> PRM Gate (Generic feedback on rejection) -> Agent B
  3. Gated (SHAP Retry): Agent A -> PRM Gate (SHAP specific reason on rejection) -> Agent B

Saves full event log to results/gating/gated_experiment.csv
and summary table to results/tables/gating_comparison.csv
"""

import csv
import json
import os
import time
import pandas as pd

from tasks.all_tasks import ALL_TASKS
from agents import agent_a_plan, agent_a_retry_plan, agent_b_code, QuotaExhausted
from gate import HandoffGate
from rollout_runner import grade

OUTPUT_CSV = "results/gating/gated_experiment.csv"
SUMMARY_CSV = "results/tables/gating_comparison.csv"
FIELDNAMES = [
    "task_id", "source", "condition",
    "initial_score", "initial_pass",
    "final_score", "final_pass",
    "retries_used", "final_outcome",
    "initial_handoff_length", "final_handoff_length"
]

def load_completed_experiment_keys(path: str = OUTPUT_CSV) -> set:
    if not os.path.exists(path):
        return set()
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return {f"{row['task_id']}::{row['condition']}" for row in csv.DictReader(f) if row.get("task_id")}
    except Exception:
        return set()

def select_eval_tasks(n_per_source: int = 5):
    """Selects a balanced sample of tasks from each source."""
    by_source = {}
    for t in ALL_TASKS:
        src = t.get("source", "hand_crafted")
        if src not in by_source:
            by_source[src] = []
        by_source[src].append(t)
    
    selected = []
    for src, task_list in by_source.items():
        # Pick n_per_source tasks deterministically
        selected.extend(task_list[:n_per_source])
    return selected

def run_experiment(max_retries: int = 2, n_per_source: int = 4):
    os.makedirs("results/gating", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    gate = HandoffGate()
    eval_tasks = select_eval_tasks(n_per_source=n_per_source)
    completed_keys = load_completed_experiment_keys()

    print(f"Selected {len(eval_tasks)} evaluation tasks across sources.")
    print(f"Gate threshold: {gate.threshold}")

    is_new = not os.path.exists(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0
    f_out = open(OUTPUT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f_out, fieldnames=FIELDNAMES)
    if is_new:
        writer.writeheader()

    conditions = ["ungated", "gated_blind", "gated_shap"]

    try:
        for task in eval_tasks:
            task_id = task["id"]
            source = task.get("source", "hand_crafted")
            problem = task["problem"]
            entry_point = task["entry_point"]
            test_code = task["test_code"]

            print(f"\n==================== Task: {task_id} ({source}) ====================")

            # 1. Generate initial handoff once for fair comparison across conditions
            try:
                h0_key = f"{task_id}::initial_h0"
                initial_handoff = agent_a_plan(problem)
            except QuotaExhausted as e:
                print(f"Quota exhausted: {e}. Stopping experiment.")
                break
            except Exception as e:
                print(f"Failed to generate initial handoff for {task_id}: {e}")
                continue

            # Initial gate check
            res_initial = gate.check(initial_handoff, problem)
            init_score = res_initial["score"]
            init_pass = res_initial["pass"]

            print(f"Initial Handoff Score: {init_score:.4f} | PRM Pass: {init_pass}")

            for cond in conditions:
                exp_key = f"{task_id}::{cond}"
                if exp_key in completed_keys:
                    print(f"  [Skipping] {cond} already recorded for {task_id}")
                    continue

                print(f"\n  --- Running Condition: {cond} ---")

                current_handoff = initial_handoff
                current_res = res_initial
                retries_used = 0

                if cond == "ungated":
                    # Direct generation without gating/retries
                    final_handoff = current_handoff
                    final_res = current_res
                else:
                    # Gated loop with retries if rejected
                    while retries_used < max_retries and not current_res["pass"]:
                        retries_used += 1
                        if cond == "gated_blind":
                            feedback = (
                                "Your handoff message was flagged as incomplete or low quality by the PRM gate. "
                                "Please rewrite it with explicit function parameters, logic steps, and edge cases."
                            )
                        else:  # gated_shap
                            feedback = f"PRM Gate Feedback: {current_res['reason']}"

                        print(f"    Retry {retries_used}/{max_retries} prompt feedback: {current_res.get('worst_feature', 'general')}")
                        try:
                            current_handoff = agent_a_retry_plan(problem, current_handoff, feedback)
                            current_res = gate.check(current_handoff, problem)
                            print(f"    New score: {current_res['score']:.4f} | Pass: {current_res['pass']}")
                        except QuotaExhausted as e:
                            print(f"Quota exhausted during retry: {e}")
                            raise
                        except Exception as e:
                            print(f"Error during retry {retries_used}: {e}")
                            break
                    
                    final_handoff = current_handoff
                    final_res = current_res

                # Agent B code generation & grading
                try:
                    code = agent_b_code(final_handoff, entry_point)
                    outcome = 1 if grade(code, entry_point, test_code) else 0
                except QuotaExhausted as e:
                    print(f"Quota exhausted during Agent B generation: {e}")
                    raise
                except Exception as e:
                    print(f"Error running Agent B code: {e}")
                    outcome = 0

                print(f"  Result [{cond}]: Final PRM Score: {final_res['score']:.4f} | Retries: {retries_used} | Final Code Outcome: {'PASS' if outcome==1 else 'FAIL'}")

                row = {
                    "task_id": task_id,
                    "source": source,
                    "condition": cond,
                    "initial_score": init_score,
                    "initial_pass": 1 if init_pass else 0,
                    "final_score": final_res["score"],
                    "final_pass": 1 if final_res["pass"] else 0,
                    "retries_used": retries_used,
                    "final_outcome": outcome,
                    "initial_handoff_length": len(initial_handoff.split()),
                    "final_handoff_length": len(final_handoff.split())
                }

                writer.writerow(row)
                f_out.flush()
                completed_keys.add(exp_key)

    except QuotaExhausted:
        print("\nExperiment paused due to quota limit. Safe to resume anytime.")
    finally:
        f_out.close()

    # Aggregate & output summary comparison
    if os.path.exists(OUTPUT_CSV) and os.path.getsize(OUTPUT_CSV) > 0:
        res_df = pd.read_csv(OUTPUT_CSV)
        summary = res_df.groupby("condition").agg(
            task_count=("task_id", "count"),
            pass_rate=("final_outcome", "mean"),
            avg_retries=("retries_used", "mean"),
            avg_prm_score=("final_score", "mean"),
            avg_word_count=("final_handoff_length", "mean")
        ).reset_index()

        summary.to_csv(SUMMARY_CSV, index=False)
        print("\n=== Experiment Summary ===")
        print(summary.to_string(index=False))

if __name__ == "__main__":
    run_experiment()
