"""
generate_corruptions.py
-----------------------
Phase 1 of the corruption pipeline.

Reads step4_verified.csv (label=1 rows = verified clean handoffs), applies all
10 corruption functions from corruption.py locally (NO API calls), and writes
every (task, corruption_type) pair to corrupted_handoffs.csv.

signature_rename needs the entry_point, which is resolved via the curated task
list.  Any corruption that produces the *same* text as the original is silently
dropped (the corruption found nothing to change -- no point grading it).

Output columns
--------------
  task_id          original task id
  problem          original problem statement
  original_handoff clean handoff that passed Agent B
  handoff          the corrupted version of that handoff
  corruption_type  name of the corruption applied
  entry_point      function name (needed by run_corruptions.py for grading)
  source           hand_easy / hand_hard / humaneval / mbpp

Usage
-----
    python generate_corruptions.py
"""

import csv
import os
import sys

sys.path.insert(0, ".")

from corruption import CORRUPTION_FUNCTIONS
from tasks.curated_250 import CURATED_250

INPUT_CSV  = "step4_verified.csv"
OUTPUT_CSV = "corrupted_handoffs.csv"

FIELDNAMES = [
    "task_id", "problem", "original_handoff", "handoff",
    "corruption_type", "entry_point", "source",
]

# ── task lookup: id -> entry_point ────────────────────────────────────────────
_task_lookup = {t["id"]: t for t in CURATED_250}


def load_pass_rows(path: str) -> list[dict]:
    """Return only the verified-pass (label=1) rows from step4_verified.csv."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("label") == "1":
                rows.append(row)
    return rows


def load_already_done(path: str) -> set:
    """(task_id, corruption_type) pairs already written — supports resume."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {
            (r["task_id"], r["corruption_type"])
            for r in csv.DictReader(f)
            if r.get("task_id") and r.get("corruption_type")
        }


def main():
    pass_rows = load_pass_rows(INPUT_CSV)
    already_done = load_already_done(OUTPUT_CSV)

    # Deduplicate: one clean handoff per task_id (take the first)
    seen_tasks: dict[str, dict] = {}
    for row in pass_rows:
        tid = row["task_id"]
        if tid not in seen_tasks:
            seen_tasks[tid] = row
    clean_rows = list(seen_tasks.values())

    print("=" * 64)
    print("  Corruption generator (Phase 1 — no API calls)")
    print(f"  Input CSV       : {INPUT_CSV}")
    print(f"  Output CSV      : {OUTPUT_CSV}")
    print(f"  Clean handoffs  : {len(clean_rows)}")
    print(f"  Corruption types: {len(CORRUPTION_FUNCTIONS)}")
    print(f"  Max output rows : {len(clean_rows) * len(CORRUPTION_FUNCTIONS)}")
    print(f"  Already done    : {len(already_done)} (task,type) pairs")
    print("=" * 64)

    is_new = not os.path.exists(OUTPUT_CSV) or os.path.getsize(OUTPUT_CSV) == 0
    f = open(OUTPUT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
    if is_new:
        writer.writeheader()

    n_written = 0
    n_skipped_done = 0
    n_skipped_noop = 0

    try:
        for row in clean_rows:
            tid         = row["task_id"]
            problem     = row.get("problem", "")
            handoff     = row["handoff"]
            source      = row.get("source", "")
            task_info   = _task_lookup.get(tid, {})
            entry_point = task_info.get("entry_point", "")

            for ctype, corrupt_fn in CORRUPTION_FUNCTIONS.items():
                if (tid, ctype) in already_done:
                    n_skipped_done += 1
                    continue

                # Apply corruption (signature_rename needs entry_point)
                try:
                    if ctype == "signature_rename":
                        corrupted = corrupt_fn(handoff, entry_point=entry_point)
                    else:
                        corrupted = corrupt_fn(handoff)
                except Exception as exc:
                    print(f"  WARN  {tid}/{ctype} errored: {exc}")
                    continue

                # Drop no-ops (corruption changed nothing)
                if corrupted.strip() == handoff.strip():
                    n_skipped_noop += 1
                    continue

                writer.writerow({
                    "task_id":          tid,
                    "problem":          problem,
                    "original_handoff": handoff,
                    "handoff":          corrupted,
                    "corruption_type":  ctype,
                    "entry_point":      entry_point,
                    "source":           source,
                })
                n_written += 1

        f.flush()
    finally:
        f.close()

    print()
    print(f"  Written         : {n_written} corrupted rows")
    print(f"  Skipped (done)  : {n_skipped_done}")
    print(f"  Skipped (no-op) : {n_skipped_noop}")
    print(f"  Output          : {OUTPUT_CSV}")
    print("=" * 64)


if __name__ == "__main__":
    main()
