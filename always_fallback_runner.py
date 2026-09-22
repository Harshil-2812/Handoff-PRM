"""
always_fallback_runner.py
=========================
Runs the "always-fallback" baseline: Agent B on the ORIGINAL RAW PROBLEM
for every PASS row in live_pilot_results.csv.

- BLOCKED rows already have this data (gated_passed = Agent B on raw problem).
- This script fills in the 47 PASS rows that are missing it.

Uses the same 16-key parallel pool from agents.py (WORKERS=3 to be safe
on RPM limits, resume-safe, re-runs error rows until clean).

Output: always_fallback_results.csv  -- per-task {task_id, afb_passed, afb_error}
        Merged report printed at end showing exact always-fallback vs gated vs ungated.
"""
import concurrent.futures
import os
import sys
import threading

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from agents import agent_b_code, N_KEYS
from rollout_runner import grade

# ── Load task pool ──────────────────────────────────────────────────────────
from humaneval_tasks import HUMANEVAL_TASKS
from mbpp_tasks import MBPP_TASKS
from tasks.coding_tasks import TASKS as hand_tasks
from tasks.hard_tasks_3 import HARD_TASKS_3

ALL_TASKS = {t["id"]: t for t in HUMANEVAL_TASKS + MBPP_TASKS + hand_tasks + HARD_TASKS_3}

PILOT_CSV  = "live_pilot_results.csv"
OUTPUT_CSV = "always_fallback_results.csv"
WORKERS    = 3
MAX_ROUNDS = 5

print_lock = threading.Lock()


def run_afb_task(row: dict) -> dict:
    """Run Agent B on the raw problem (never the handoff)."""
    tid = row["task_id"]
    task = ALL_TASKS.get(tid)
    if task is None:
        with print_lock:
            print(f"  [SKIP] {tid} not found in task pool")
        return {"task_id": tid, "afb_passed": False, "afb_error": "task_not_found"}

    problem    = task["problem"]
    ep         = task["entry_point"]
    test_code  = task["test_code"]

    with print_lock:
        print(f"  Running AFB on {tid} ({ep})")

    try:
        code = agent_b_code(problem, ep)   # Agent B sees RAW PROBLEM
        passed = grade(code, ep, test_code)
        with print_lock:
            print(f"    {tid}: {'PASS' if passed else 'FAIL'}")
        return {"task_id": tid, "afb_passed": passed, "afb_error": ""}
    except Exception as exc:
        with print_lock:
            print(f"    {tid}: ERROR — {exc}")
        return {"task_id": tid, "afb_passed": False, "afb_error": str(exc)[:120]}


def is_clean(row) -> bool:
    return not bool(row.get("afb_error", ""))


def main():
    print("=" * 72)
    print("  Always-Fallback Runner — Agent B on raw problem for PASS rows")
    print(f"  {WORKERS} workers, {N_KEYS} API keys, resume-safe")
    print("=" * 72)

    # Identify rows to run: PASS rows from live_pilot_results.csv
    pilot = pd.read_csv(PILOT_CSV)
    pass_rows = pilot[pilot["gate_decision"] == "PASS"]
    pass_ids  = set(pass_rows["task_id"])
    print(f"\nPASS rows to fill: {len(pass_ids)}")

    rounds = 0
    while rounds < MAX_ROUNDS:
        rounds += 1

        # Load existing results
        if os.path.exists(OUTPUT_CSV):
            done_df = pd.read_csv(OUTPUT_CSV)
            done = {r["task_id"]: r for _, r in done_df.iterrows()
                    if is_clean(r.to_dict())}
        else:
            done = {}

        to_run = [r.to_dict() for _, r in pass_rows.iterrows()
                  if r["task_id"] not in done]

        if not to_run:
            print(f"[round {rounds}] All {len(done)} rows complete.")
            break

        n_err = len(pass_ids) - len(done)
        print(f"\n[round {rounds}] Running {len(to_run)} rows "
              f"({len(done)} clean, {n_err} pending) with {WORKERS} workers...")

        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futs = {ex.submit(run_afb_task, row): row["task_id"] for row in to_run}
            for fut in concurrent.futures.as_completed(futs):
                tid = futs[fut]
                try:
                    res = fut.result()
                    results[res["task_id"]] = res
                except Exception as exc:
                    print(f"  Worker crashed on {tid}: {exc}")

        # Merge and save
        all_res = list(done.values()) + [results[t] for t in results]
        pd.DataFrame(all_res).to_csv(OUTPUT_CSV, index=False)
        n_bad = sum(1 for r in all_res if not is_clean(r))
        print(f"  Saved {len(all_res)} rows; {n_bad} with errors will be re-run.")

    # ── Final merge and report ──────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  ALWAYS-FALLBACK vs GATED vs UNGATED — FINAL COMPARISON")
    print("=" * 72)

    afb_df = pd.read_csv(OUTPUT_CSV)
    afb_map = dict(zip(afb_df["task_id"], afb_df["afb_passed"].astype(bool)))

    # Combine with pilot: PASS rows use afb_map, BLOCK rows use gated_passed
    pilot = pd.read_csv(PILOT_CSV)
    pilot["is_corrupted"] = pilot["is_corrupted"].astype(str).str.lower().map(
        lambda x: True if x in ("true","1","yes") else False)
    for c in ["ungated_passed","gated_passed"]:
        pilot[c] = pilot[c].astype(str).str.lower().map(
            lambda x: True if x in ("true","1","yes") else False)

    def get_afb(row):
        if row["gate_decision"] == "BLOCK":
            return bool(row["gated_passed"])   # same code path
        else:
            return afb_map.get(row["task_id"], False)

    pilot["afb_passed"] = pilot.apply(get_afb, axis=1)

    cln = pilot[~pilot["is_corrupted"]]
    crp = pilot[pilot["is_corrupted"]]

    def pp(series): return f"{series.mean()*100:.1f}% ({series.sum()}/{len(series)})"

    W = 32
    print(f"\n  {'Condition':<{W}} {'Ungated':>16} {'Always-FB':>16} {'Gated':>16}")
    print(f"  {'-'*(W+50)}")
    for label, sub in [("All (n=94)", pilot), ("Clean (n=47)", cln), ("Corrupted (n=47)", crp)]:
        u = pp(sub["ungated_passed"])
        a = pp(sub["afb_passed"])
        g = pp(sub["gated_passed"])
        print(f"  {label:<{W}} {u:>16} {a:>16} {g:>16}")

    # McNemar: gated vs always-fallback
    from scipy.stats import binom
    import numpy as np

    b_ga = ((pilot["gated_passed"]) & (~pilot["afb_passed"])).sum()
    c_ga = ((~pilot["gated_passed"]) & (pilot["afb_passed"])).sum()
    total_ga = int(b_ga) + int(c_ga)
    if total_ga > 0:
        p_ga = 2*min(sum(binom.pmf(k, total_ga, 0.5) for k in range(min(int(b_ga),int(c_ga))+1)), 1.0)
    else:
        p_ga = 1.0

    rng = np.random.default_rng(42)
    delta = (pilot["gated_passed"].astype(int) - pilot["afb_passed"].astype(int)).to_numpy()
    draws = rng.choice(delta, size=(10000, len(delta)), replace=True).mean(axis=1)
    ci_lo, ci_hi = float(np.percentile(draws, 2.5))*100, float(np.percentile(draws, 97.5))*100

    print(f"\n  McNemar (gated vs always-fallback): b={b_ga} c={c_ga} p={p_ga:.4f}")
    print(f"  Bootstrap 95% CI (gated-afb):       [{ci_lo:+.1f}, {ci_hi:+.1f}] pp")
    print(f"\n  PASS rows breakdown:")
    print(f"    Clean PASS rows (n=40):  AFB = {pp(cln[cln['gate_decision']=='PASS']['afb_passed'])}")
    print(f"    Corrupt PASS rows (n=7): AFB = {pp(crp[crp['gate_decision']=='PASS']['afb_passed'])}")

    # Save merged CSV for paper_metrics.py to pick up
    pilot.to_csv("live_pilot_results_with_afb.csv", index=False)
    print(f"\n  Merged results -> live_pilot_results_with_afb.csv")
    print("=" * 72)


if __name__ == "__main__":
    main()
