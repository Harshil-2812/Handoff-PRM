"""Task Loader and Manifest Validation for Workstream C.

Enforces:
1. Manifest-driven loading of held-out live test tasks.
2. Runtime assertion guaranteeing ZERO leakage / zero overlap with the train/dev pool.
3. Filtering by task limit, source, and specific IDs.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set

from workstream_c.harness.config import MANIFEST_DIR, REPO_ROOT
from tasks.all_tasks import ALL_TASKS


def load_manifest_task_ids(manifest_name: str = "manifest_live_test.json") -> List[str]:
    """Loads ordered task IDs from the specified manifest file."""
    manifest_path = MANIFEST_DIR / manifest_name
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest file {manifest_path} not found. Run `workstream_c/manifests/build_manifests.py` first."
        )
    with open(manifest_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["tasks"]


def load_live_test_tasks(
    limit: Optional[int] = None,
    source_filter: Optional[str] = None,
    task_id_filter: Optional[List[str]] = None,
) -> List[Dict]:
    """Loads verified tasks for the live evaluation from the frozen live test manifest.
    
    Includes a strict runtime assertion ensuring zero task ID overlap with the train/dev pool.
    """
    # 1. Load train/dev manifest IDs
    train_dev_ids = set(load_manifest_task_ids("manifest_train_dev.json"))
    
    # 2. Load live test manifest IDs
    live_test_ids = load_manifest_task_ids("manifest_live_test.json")
    
    # 3. CRITICAL ASSERTION: Assert zero overlap
    overlap = set(live_test_ids) & train_dev_ids
    assert len(overlap) == 0, f"PROTOCOL VIOLATION: Leakage detected! {len(overlap)} tasks overlap with train/dev pool: {overlap}"
    
    # 4. Map task objects from ALL_TASKS
    task_map = {t["id"]: t for t in ALL_TASKS}
    tasks = []
    for tid in live_test_ids:
        if tid not in task_map:
            raise KeyError(f"Task ID '{tid}' from live manifest not found in ALL_TASKS!")
        task = task_map[tid]
        
        # Check source filter
        if source_filter and task.get("source") != source_filter:
            continue
            
        # Check task id filter
        if task_id_filter and tid not in task_id_filter:
            continue
            
        tasks.append(task)
        if limit is not None and len(tasks) >= limit:
            break
            
    return tasks

