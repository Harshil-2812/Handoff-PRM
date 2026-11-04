"""
Deterministic Task Manifest Partitioning for Handoff-PRM.

Partitions the 675 tasks in tasks/all_tasks.py into three strictly disjoint pools:
1. manifest_train_dev.txt (300 tasks):
   - Allocated to Workstream B (Aravind G) for offline training, 5-fold CV, and threshold tuning.
   - Drawn strictly from completed rollouts in rollouts.csv.
2. manifest_live_test.txt (200 tasks):
   - Allocated to Workstream C (Harshil) for live end-to-end evaluation.
   - Disjoint held-out tasks, never touched during training or tuning.
3. manifest_organic_pool.txt (175 tasks):
   - Allocated to Workstream A (Avaneesh N) for organic temperature-sampled handoffs.
   - Disjoint from both train/dev and live test sets.

Fixed Random Seed: 42
"""

import os
import random
import pandas as pd
from tasks.all_tasks import ALL_TASKS

SEED = 42
MANIFEST_DIR = "manifests"
ROLLOUTS_CSV = "rollouts.csv"

def partition_tasks():
    os.makedirs(MANIFEST_DIR, exist_ok=True)
    rng = random.Random(SEED)

    # 1. Load rollouts tasks
    df_rollouts = pd.read_csv(ROLLOUTS_CSV)
    rollout_tasks = set(df_rollouts["task_id"].unique())
    all_tasks_map = {t["id"]: t for t in ALL_TASKS}
    all_ids = set(all_tasks_map.keys())

    # 2. Group into in_rollouts and unseen by source
    in_r = {}
    unseen = {}
    for t in ALL_TASKS:
        s = t["source"]
        tid = t["id"]
        if tid in rollout_tasks:
            in_r.setdefault(s, []).append(tid)
        else:
            unseen.setdefault(s, []).append(tid)

    # Shuffle deterministically within each source group
    for s in in_r:
        rng.shuffle(in_r[s])
    for s in unseen:
        rng.shuffle(unseen[s])

    # 3. Train/dev selection: exactly 300 tasks from in_rollouts
    target_train = {"mbpp": 150, "humaneval": 95, "hand_hard": 30, "hand_easy": 25}
    train_dev = []
    pool_remaining = {}

    for s, count in target_train.items():
        train_dev.extend(in_r[s][:count])
        pool_remaining[s] = in_r[s][count:] + unseen.get(s, [])

    # 4. Live test and organic pool selection from pool_remaining (375 tasks)
    target_live = {"mbpp": 148, "humaneval": 37, "hand_hard": 9, "hand_easy": 6}
    live_test = []
    organic_pool = []

    for s, ids in pool_remaining.items():
        rng.shuffle(ids)
        n_live = target_live.get(s, len(ids) // 2)
        live_test.extend(ids[:n_live])
        organic_pool.extend(ids[n_live:])

    # 5. Sanity verifications
    s_train = set(train_dev)
    s_live = set(live_test)
    s_org = set(organic_pool)

    assert len(s_train) == 300, f"Expected 300 train_dev, got {len(s_train)}"
    assert len(s_live) == 200, f"Expected 200 live_test, got {len(s_live)}"
    assert len(s_org) == 175, f"Expected 175 organic_pool, got {len(s_org)}"
    assert len(s_train.intersection(s_live)) == 0, "Train and Live overlap detected!"
    assert len(s_train.intersection(s_org)) == 0, "Train and Organic overlap detected!"
    assert len(s_live.intersection(s_org)) == 0, "Live and Organic overlap detected!"
    assert len(s_train) + len(s_live) + len(s_org) == len(ALL_TASKS), "Missing tasks in union!"

    # Sort deterministically for reproducible ordering
    train_dev.sort()
    live_test.sort()
    organic_pool.sort()

    # 6. Save text manifests
    path_train = os.path.join(MANIFEST_DIR, "manifest_train_dev.txt")
    path_live = os.path.join(MANIFEST_DIR, "manifest_live_test.txt")
    path_org = os.path.join(MANIFEST_DIR, "manifest_organic_pool.txt")

    with open(path_train, "w", encoding="utf-8") as f:
        f.write("\n".join(train_dev) + "\n")

    with open(path_live, "w", encoding="utf-8") as f:
        f.write("\n".join(live_test) + "\n")

    with open(path_org, "w", encoding="utf-8") as f:
        f.write("\n".join(organic_pool) + "\n")

    # 7. Save metadata CSV
    meta_rows = []
    for tid, pool_name in [(t, "train_dev") for t in train_dev] + \
                          [(t, "live_test") for t in live_test] + \
                          [(t, "organic_pool") for t in organic_pool]:
        task_obj = all_tasks_map[tid]
        meta_rows.append({
            "task_id": tid,
            "source": task_obj["source"],
            "entry_point": task_obj["entry_point"],
            "manifest_pool": pool_name,
            "was_in_rollouts_csv": tid in rollout_tasks,
        })

    df_meta = pd.DataFrame(meta_rows)
    path_meta = os.path.join(MANIFEST_DIR, "manifest_metadata.csv")
    df_meta.to_csv(path_meta, index=False)

    print("=" * 60)
    print("TASK MANIFEST PARTITION COMPLETE (Seed: 42)")
    print("=" * 60)
    print(f"1. Train/Dev Pool:   {len(train_dev):3d} tasks -> {path_train}")
    print(f"2. Live Test Pool:   {len(live_test):3d} tasks -> {path_live}")
    print(f"3. Organic Pool:     {len(organic_pool):3d} tasks -> {path_org}")
    print(f"Total Partitioned:   {len(ALL_TASKS):3d} tasks (100% disjoint, zero leakage)")
    print(f"Metadata Manifest:   {path_meta}")
    print("=" * 60)
    print("\nBreakdown by Source across Pools:")
    print(pd.crosstab(df_meta["source"], df_meta["manifest_pool"], margins=True))

if __name__ == "__main__":
    partition_tasks()
