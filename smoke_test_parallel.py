"""
Smoke test: parallel Agent B calls using the key pool.

Fires 3 Agent B tasks simultaneously in a ThreadPoolExecutor.
If all 3 complete and return non-empty code, the key pool + threading is working.

Run with:
    python smoke_test_parallel.py
"""

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from agents import agent_b_code, N_KEYS, GEMINI_KEYS, MODEL_B

# ─── 3 sample tasks (real handoffs from rollouts.csv) ─────────────────────────

TASKS = [
    {
        "id": "task_002",
        "entry_point": "is_palindrome",
        "handoff": """\
Handoff Message for Agent B:

1. **Function Purpose**: Write a Python function that checks whether a given
   string is a palindrome. The check must be case-insensitive and ignore all spaces.

2. **Function Signature**: `def is_palindrome(s: str) -> bool`

3. **Key Edge Cases**:
   - Empty string: return True
   - Single character: return True
   - Mixed case: "Racecar" -> True
   - Spaces: "race car" -> True

4. **Approach**: Strip spaces, lowercase, compare to its reverse.
""",
    },
    {
        "id": "he_000",
        "entry_point": "has_close_elements",
        "handoff": """\
Handoff Message for Agent B:

1. **Function Purpose**: Check if any two numbers in a list are closer than a
   given threshold.

2. **Function Signature**: `def has_close_elements(numbers: list, threshold: float) -> bool`

3. **Key Edge Cases**:
   - Empty list: return False
   - Single element: return False
   - Threshold of 0: duplicates count

4. **Approach**: Sort the list, check consecutive differences.
""",
    },
    {
        "id": "he_001",
        "entry_point": "separate_paren_groups",
        "handoff": """\
Handoff Message for Agent B:

1. **Function Purpose**: Separate groups of balanced, non-nested parentheses
   from a string into a list of strings. Ignore spaces.

2. **Function Signature**: `def separate_paren_groups(paren_string: str) -> list`

3. **Key Edge Cases**:
   - Empty string: return []
   - Nested groups must be treated as one unit

4. **Approach**: Scan left to right, track depth. Flush when depth hits 0.
""",
    },
]


def run_one(task: dict, worker_id: int) -> dict:
    """Run Agent B on one task. Returns result dict."""
    tid = task["id"]
    t_start = time.monotonic()
    thread_name = threading.current_thread().name
    print(f"  [worker-{worker_id} / {thread_name}] START  {tid}")
    try:
        code = agent_b_code(task["handoff"], task["entry_point"])
        elapsed = time.monotonic() - t_start
        print(f"  [worker-{worker_id}] DONE   {tid}  ({elapsed:.1f}s)  "
              f"code length={len(code)} chars")
        return {"id": tid, "ok": True, "code": code, "elapsed": elapsed}
    except Exception as exc:
        elapsed = time.monotonic() - t_start
        print(f"  [worker-{worker_id}] ERROR  {tid}  ({elapsed:.1f}s)  {exc}")
        return {"id": tid, "ok": False, "error": str(exc), "elapsed": elapsed}


def main():
    print("=" * 60)
    print(f"  Smoke test: parallel Agent B calls")
    print(f"  Keys loaded : {N_KEYS}")
    print(f"  Model       : {MODEL_B}")
    print(f"  Tasks       : {len(TASKS)}")
    print(f"  Workers     : {min(N_KEYS, len(TASKS))} (min of keys vs tasks)")
    print("=" * 60)
    print()

    n_workers = min(N_KEYS, len(TASKS))
    wall_start = time.monotonic()

    results = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(run_one, task, i): task
            for i, task in enumerate(TASKS)
        }
        for future in as_completed(futures):
            results.append(future.result())

    wall_elapsed = time.monotonic() - wall_start

    print()
    print("=" * 60)
    print("  Results")
    print("=" * 60)
    n_ok = 0
    for r in results:
        status = "PASS" if r["ok"] else "FAIL"
        print(f"  [{status}] {r['id']}  ({r['elapsed']:.1f}s)")
        if r["ok"]:
            n_ok += 1
            # Show first 3 lines of generated code
            first_lines = r["code"].splitlines()[:3]
            for line in first_lines:
                print(f"         {line}")
        else:
            print(f"         ERROR: {r['error']}")
    print()
    print(f"  {n_ok}/{len(TASKS)} tasks succeeded")
    print(f"  Wall time: {wall_elapsed:.1f}s  "
          f"(serial equivalent would be ~{sum(r['elapsed'] for r in results):.1f}s)")
    print("=" * 60)

    if n_ok == len(TASKS):
        print("\n  Key pool + parallel threading is WORKING.\n")
    else:
        print("\n  Some tasks failed — check errors above.\n")


if __name__ == "__main__":
    main()
