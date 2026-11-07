"""Unit test verifying that pick_corruption is 100% deterministic across processes and independent of PYTHONHASHSEED."""

import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKER_SCRIPT = Path(__file__).resolve().parent / "worker_pick_corruption.py"


def run_worker_with_hashseed(seed_val: str) -> dict:
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = seed_val
    res = subprocess.run(
        [sys.executable, str(WORKER_SCRIPT)],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return json.loads(res.stdout.strip())


def test_pick_corruption_deterministic_across_processes():
    res_seed_0 = run_worker_with_hashseed("0")
    res_seed_12345 = run_worker_with_hashseed("12345")

    assert res_seed_0 == res_seed_12345, "Mismatch between PYTHONHASHSEED=0 and 12345!"
    print(f"PASSED: Verified {len(res_seed_0)} (task, seed) pairs across 2 distinct process hash seeds.")


if __name__ == "__main__":
    test_pick_corruption_deterministic_across_processes()
