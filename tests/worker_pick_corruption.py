"""Worker script for testing process-level determinism of pick_corruption."""

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from workstream_c.run_live import pick_corruption

task_ids = [f"he_{i:03d}" for i in range(1, 15)] + [f"mbpp_{j}" for j in range(10, 25)]
seeds = [42, 43, 44]

results = {}
for tid in task_ids:
    for s in seeds:
        name, _ = pick_corruption(tid, s)
        results[f"{tid}_{s}"] = name

print(json.dumps(results))

