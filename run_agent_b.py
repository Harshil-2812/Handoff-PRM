"""
Step 4: Run Agent B in parallel on all saved clean handoffs (label=1,
corruption_type=none) from rollouts.csv, grade with real unit tests,
and write results to step4_verified.csv.

Only tasks that PASS are kept as verified positives.
Tasks that fail are written with label=0 and label_source="measured"
so we know Agent B actually ran (not a corruption artifact).

Usage:
    python run_agent_b.py              # dry-run print plan, no API calls
    python run_agent_b.py --execute    # actually run (NOT DRY RUN)

DO NOT pass --execute until you're ready.
"""

import argparse
import csv
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from tasks.curated_250 import CURATED_250
from agents import agent_b_code, N_KEYS, MODEL_B, QuotaExhausted
from rollout_runner import grade

# ─── Config ───────────────────────────────────────────────────────────────────

INPUT_CSV  = "rollouts.csv"
OUTPUT_CSV = "step4_verified.csv"
FIELDNAMES = ["task_id", "problem", "handoff", "corruption_type",
              "label", "label_source", "source"]

# Use all available keys as workers — true parallelism
N_WORKERS = N_KEYS


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_clean_handoffs(path: str) -> list[dict]:
    """Load label=1 / corruption_type=none rows — the original Agent A handoffs."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("label") == "1" and row.get("corruption_type", "").strip() == "none":
                rows.append(row)
    return rows


def build_task_lookup() -> dict:
    """task_id -> {entry_point, test_code, problem} — built from the 250-task curated set."""
    return {
        t["id"]: {
            "entry_point": t.get("entry_point", ""),
            "test_code":   t.get("test_code", ""),
            "problem":     t.get("problem", ""),
        }
        for t in CURATED_250
    }


def load_already_done(path: str) -> set:
    """task_ids already written to output CSV, so a resumed run skips them."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {row["task_id"] for row in csv.DictReader(f) if row.get("task_id")}


def _open_writer(path: str):
    is_new = not os.path.exists(path) or os.path.getsize(path) == 0
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=FIELDNAMES)
    if is_new:
        w.writeheader()
    return f, w


# ─── Per-task worker (runs in a thread) ───────────────────────────────────────

def process_task(row: dict, task_info: dict) -> dict:
    """
    Run Agent B on one handoff, grade it, return a result dict.
    Called from multiple threads simultaneously.
    """
    task_id     = row["task_id"]
    handoff     = row["handoff"]
    source      = row.get("source", "unknown")
    problem     = row.get("problem", "")
    entry_point = task_info["entry_point"]
    test_code   = task_info["test_code"]

    t_start = time.monotonic()
    try:
        code   = agent_b_code(handoff, entry_point)
        passed = grade(code, entry_point, test_code)
    except QuotaExhausted:
        raise   # bubble up to the main loop to stop the run
    except Exception as exc:
        return {
            "task_id":        task_id,
            "ok":             False,
            "error":          str(exc),
            "elapsed":        time.monotonic() - t_start,
            "csv_row":        None,
        }

    label = 1 if passed else 0
    return {
        "task_id":  task_id,
        "ok":       True,
        "passed":   passed,
        "elapsed":  time.monotonic() - t_start,
        "csv_row": {
            "task_id":          task_id,
            "problem":          problem,
            "handoff":          handoff,
            "corruption_type":  "none",
            "label":            label,
            "label_source":     "measured",   # ← provenance: real Agent B + grade
            "source":           source,
        },
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(dry_run: bool = True):
    # Load inputs
    clean_rows  = load_clean_handoffs(INPUT_CSV)
    task_lookup = build_task_lookup()
    already_done = load_already_done(OUTPUT_CSV)

    # Filter to tasks we can actually process
    pending = []
    skipped_no_task = 0
    skipped_done    = 0
    for row in clean_rows:
        tid = row["task_id"]
        if tid in already_done:
            skipped_done += 1
            continue
        if tid not in task_lookup:
            skipped_no_task += 1
            continue
        pending.append(row)

    print("=" * 64)
    print(f"  Step 4 — Agent B parallel runner")
    print(f"  Model       : {MODEL_B}")
    print(f"  API keys    : {N_KEYS}  (workers = {N_WORKERS})")
    print(f"  Clean rows  : {len(clean_rows)}")
    print(f"  Already done: {skipped_done}")
    print(f"  No task info: {skipped_no_task}")
    print(f"  To process  : {len(pending)}")
    print(f"  Output CSV  : {OUTPUT_CSV}")
    print("=" * 64)

    if dry_run:
        print("\n  DRY RUN — no API calls made.")
        print("  Pass --execute to actually run.\n")
        if pending:
            print(f"  First 5 tasks queued:")
            for r in pending[:5]:
                ti = task_lookup[r["task_id"]]
                print(f"    {r['task_id']}  ep={ti['entry_point']}")
        return

    if not pending:
        print("\n  Nothing to do — all tasks already in output CSV.\n")
        return

    # ── Execute ──
    f, writer = _open_writer(OUTPUT_CSV)
    n_pass = 0
    n_fail = 0
    n_err  = 0
    quota_hit = False
    wall_start = time.monotonic()

    try:
        with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {
                executor.submit(
                    process_task,
                    row,
                    task_lookup[row["task_id"]]
                ): row["task_id"]
                for row in pending
            }

            for future in as_completed(futures):
                tid = futures[future]
                try:
                    result = future.result()
                except QuotaExhausted as exc:
                    if not quota_hit:
                        print(f"\n  QUOTA HIT — daily quota exhausted: {exc}")
                        print(f"  Cancelling pending tasks; draining already-finished "
                              f"results so no progress is lost...")
                        quota_hit = True
                        # Cancel futures that haven't started yet.
                        # Futures that already finished in another thread are
                        # still in the as_completed queue — we keep draining
                        # them so every completed result gets written to disk.
                        for fut in futures:
                            fut.cancel()
                    # Skip this failed task and continue draining completed ones.
                    n_err += 1
                    continue
                except Exception:
                    print(f"  ERROR  {tid}")
                    traceback.print_exc()
                    n_err += 1
                    continue

                if not result["ok"]:
                    print(f"  ERROR  {tid}  ({result['elapsed']:.1f}s)  {result['error']}")
                    n_err += 1
                    continue

                status = "PASS" if result["passed"] else "FAIL"
                print(f"  [{status}]  {tid}  ({result['elapsed']:.1f}s)")

                if result["passed"]:
                    n_pass += 1
                else:
                    n_fail += 1

                # Write immediately — every flush is a checkpoint.
                # On the next run load_already_done() skips these task_ids.
                writer.writerow(result["csv_row"])
                f.flush()

    finally:
        f.close()

    wall = time.monotonic() - wall_start
    print()
    print("=" * 64)
    print(f"  Done in {wall:.0f}s")
    print(f"  PASS (label=1): {n_pass}")
    print(f"  FAIL (label=0): {n_fail}")
    print(f"  Errors (skip) : {n_err}")
    if quota_hit:
        print(f"  !! Quota hit — re-run to resume (already-done rows are skipped)")
    print(f"  Results in    : {OUTPUT_CSV}")
    print("=" * 64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually run Agent B. Without this flag it's a dry-run only.",
    )
    args = parser.parse_args()
    main(dry_run=not args.execute)
