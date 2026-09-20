"""
Selects a final, "never touched until the last reported number" holdout set,
and lets you run the pipeline on everything EXCEPT that holdout -- so quota
is spent only on the training/tuning pool, and the holdout tasks are never
even attempted until you're ready for the final confirmatory evaluation.

Why holdout candidates are restricted to NOT-YET-COMPLETED tasks:
every task already in rollouts.csv has already been used in this project's
reported model comparison, ANOVA, feature-ablation study, and threshold
locking. Those tasks have already influenced every modeling decision made
so far, so none of them can serve as a genuine, unbiased final holdout --
only tasks that haven't been attempted at all yet are eligible.

Usage:
    python select_holdout.py select    # do this ONCE, writes holdout_task_ids.txt
    python select_holdout.py run       # runs rollout_runner on training pool only
                                        # (everything NOT in holdout_task_ids.txt)
    python select_holdout.py run-holdout   # LATER, once ready for the final
                                        # confirmatory number: runs rollout_runner
                                        # on ONLY the holdout tasks
"""

import csv
import os
import random
import sys
from collections import defaultdict

from tasks.all_tasks import ALL_TASKS

HOLDOUT_FRACTION = 0.20
SEED = 42
HOLDOUT_FILE = "holdout_task_ids.txt"
ROLLOUTS_CSV = "rollouts.csv"


def already_completed_task_ids(path=ROLLOUTS_CSV):
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {row["task_id"] for row in csv.DictReader(f) if row.get("task_id")}


def select_holdout():
    if os.path.exists(HOLDOUT_FILE):
        print(f"{HOLDOUT_FILE} already exists -- refusing to overwrite an "
              f"existing holdout selection (that would break its whole point). "
              f"Delete it manually first if you really mean to re-select.")
        return

    completed = already_completed_task_ids()
    candidates = [t for t in ALL_TASKS if t["id"] not in completed]

    by_source = defaultdict(list)
    for t in candidates:
        by_source[t.get("source", "unknown")].append(t["id"])

    rng = random.Random(SEED)
    holdout_ids = []
    for source, ids in by_source.items():
        ids_sorted = sorted(ids)  # deterministic order before shuffling
        rng.shuffle(ids_sorted)
        n_holdout = round(len(ids_sorted) * HOLDOUT_FRACTION)
        holdout_ids.extend(ids_sorted[:n_holdout])
        print(f"  {source}: {len(ids_sorted)} eligible -> {n_holdout} held out")

    holdout_ids = sorted(set(holdout_ids))
    with open(HOLDOUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(holdout_ids) + "\n")

    print(f"\nWrote {len(holdout_ids)} holdout task IDs to {HOLDOUT_FILE}")
    print(f"({len(candidates)} tasks were eligible; {len(completed)} already-"
          f"completed tasks were excluded from eligibility, since they're "
          f"already 'spent' on prior analysis)")
    print("\nThis file is now the fixed, permanent holdout selection. "
          "Commit it to your repo and never regenerate it.")


def load_holdout_ids():
    if not os.path.exists(HOLDOUT_FILE):
        print(f"{HOLDOUT_FILE} not found -- run `python select_holdout.py select` first.")
        sys.exit(1)
    with open(HOLDOUT_FILE, encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def run_training_pool():
    from rollout_runner import run_all
    holdout_ids = load_holdout_ids()
    training_tasks = [t for t in ALL_TASKS if t["id"] not in holdout_ids]
    print(f"Running on {len(training_tasks)} training-pool tasks "
          f"({len(holdout_ids)} holdout tasks excluded, will not be attempted).")
    run_all(tasks=training_tasks)


def run_holdout_only():
    from rollout_runner import run_all
    holdout_ids = load_holdout_ids()
    holdout_tasks = [t for t in ALL_TASKS if t["id"] in holdout_ids]
    print(f"Running on {len(holdout_tasks)} HOLDOUT tasks. "
          f"This should only be done once, for the final confirmatory number, "
          f"after all model selection and tuning is fully finished.")
    confirm = input("Type YES to proceed: ")
    if confirm.strip() != "YES":
        print("Aborted.")
        return
    run_all(tasks=holdout_tasks)


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("select", "run", "run-holdout"):
        print(__doc__)
        sys.exit(1)

    if sys.argv[1] == "select":
        select_holdout()
    elif sys.argv[1] == "run":
        run_training_pool()
    elif sys.argv[1] == "run-holdout":
        run_holdout_only()
