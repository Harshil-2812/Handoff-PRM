"""
Runs the full data-generation pipeline:
  1. For each task, run Agent A -> Agent B, check pass/fail (original rollout).
  2. If it PASSED, corrupt the handoff (all 4 types) and re-run Agent B on each.
  3. Label: original = 1. Corrupted version = 0 if it now fails, discarded if
     it still passes (weak negative -- doesn't prove the corruption mattered).
  4. Save everything to a CSV for the feature extraction / training step.

Run this AFTER agents.py and corruption.py both work standalone.
Usage: python rollout_runner.py

Resumable: rows are appended to the CSV as each task finishes, and tasks
already present in the CSV are skipped on the next run. Free-tier quota is
small enough that a full dataset takes several days, so a run that loses its
completed work on the way out is not affordable.
"""

import csv
import os
import traceback

from tasks.all_tasks import ALL_TASKS
from agents import agent_a_plan, agent_b_code, QuotaExhausted
from corruption import CORRUPTION_FUNCTIONS

OUTPUT_CSV = "rollouts.csv"
FIELDNAMES = ["task_id", "problem", "handoff", "corruption_type", "label", "source"]


def load_completed_task_ids(path: str = OUTPUT_CSV) -> set:
    """Task IDs already in the CSV, so a resumed run doesn't re-spend quota."""
    if not os.path.exists(path):
        return set()
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return {row["task_id"] for row in csv.DictReader(f) if row.get("task_id")}
    except (KeyError, csv.Error):
        print(f"WARNING: {path} is unreadable; starting fresh (existing file untouched)")
        return set()


def _open_writer(path: str = OUTPUT_CSV):
    """Opens the CSV for append, writing a header only if the file is new."""
    is_new = not os.path.exists(path) or os.path.getsize(path) == 0
    f = open(path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    if is_new:
        writer.writeheader()
    return f, writer


def grade(code: str, entry_point: str, test_code: str) -> bool:
    """
    Executes Agent B's generated code plus the task's test function in an
    isolated namespace and returns True/False. Wrapped in try/except because
    generated code will sometimes be flat-out broken -- that's an expected
    outcome, not a bug in this harness.
    """
    namespace = {}
    try:
        exec(code, namespace)
        exec(test_code, namespace)
        fn = namespace.get(entry_point)
        if fn is None:
            return False
        return bool(namespace["test"](fn))
    except Exception:
        return False


def run_all(tasks=ALL_TASKS, verbose=True):
    all_rows = []
    completed = load_completed_task_ids()
    if completed and verbose:
        print(f"Resuming: {len(completed)} task(s) already in {OUTPUT_CSV}, skipping those.")

    f, writer = _open_writer()
    try:
        for task in tasks:
            if task["id"] in completed:
                if verbose:
                    print(f"\n=== {task['id']} === (already done, skipping)")
                continue

            if verbose:
                print(f"\n=== {task['id']} ===")

            task_rows = []

            # --- Step 1: original rollout ---
            try:
                handoff = agent_a_plan(task["problem"])
                code = agent_b_code(handoff, task["entry_point"])
                passed = grade(code, task["entry_point"], task["test_code"])
            except QuotaExhausted as err:
                # Not a task outcome -- we never observed one. Stop the run so
                # this task is retried intact tomorrow rather than discarded.
                print(f"\nSTOPPING: {err}")
                print("Completed tasks are saved. Re-run to resume where this left off.")
                break
            except Exception:
                if verbose:
                    print(f"  original rollout errored, skipping task")
                    traceback.print_exc()
                continue

            if verbose:
                print(f"  original rollout: {'PASS' if passed else 'FAIL'}")

            if not passed:
                # Discard: can't attribute this failure to the handoff since
                # there's nothing to compare it against (no corruption applied).
                continue

            task_rows.append({
                "task_id": task["id"],
                "problem": task["problem"],
                "handoff": handoff,
                "corruption_type": "none",
                "label": 1,
                "source": task.get("source", "unknown"),
            })

            # --- Step 2: corrupt and re-run for each corruption type ---
            quota_hit = False
            for corruption_type, corrupt_fn in CORRUPTION_FUNCTIONS.items():
                try:
                    corrupted_handoff = corrupt_fn(handoff)
                    corrupted_code = agent_b_code(corrupted_handoff, task["entry_point"])
                    corrupted_passed = grade(corrupted_code, task["entry_point"], task["test_code"])
                except QuotaExhausted as err:
                    print(f"\nSTOPPING: {err}")
                    quota_hit = True
                    break
                except Exception:
                    if verbose:
                        print(f"  corruption '{corruption_type}' errored, skipping")
                    continue

                if verbose:
                    print(f"  corruption '{corruption_type}': {'still PASS (discard)' if corrupted_passed else 'now FAILS (label 0)'}")

                if corrupted_passed:
                    # weak negative -- corruption didn't actually break anything, discard
                    continue

                task_rows.append({
                    "task_id": task["id"],
                    "problem": task["problem"],
                    "handoff": corrupted_handoff,
                    "corruption_type": corruption_type,
                    "label": 0,
                    "source": task.get("source", "unknown"),
                })

            if quota_hit:
                # Partial task: drop it rather than writing a task whose
                # corruption set is incomplete for reasons unrelated to the data.
                print(f"  discarding partial {task['id']} (will be redone on resume)")
                print("Completed tasks are saved. Re-run to resume where this left off.")
                break

            # --- Checkpoint: flush this task's rows before starting the next ---
            writer.writerows(task_rows)
            f.flush()
            all_rows.extend(task_rows)
            if verbose:
                print(f"  saved {len(task_rows)} row(s)")
    finally:
        f.close()

    if verbose:
        n_pos = sum(1 for r in all_rows if r["label"] == 1)
        n_neg = sum(1 for r in all_rows if r["label"] == 0)
        total = len(load_completed_task_ids())
        print(f"\nThis run added {len(all_rows)} labeled examples "
              f"({n_pos} positive, {n_neg} negative)")
        print(f"{OUTPUT_CSV} now covers {total} task(s) total")

    return all_rows


if __name__ == "__main__":
    run_all()
