"""Build and verify formal task manifests for Workstream C.

Enforces strict disjointness between:
1. Train/Dev Pool (248 tasks from features_v2.csv)
2. Live Test Set (300 tasks: 84 HumanEval, 7 hand-crafted, 209 MBPP)
3. Organic Pool (127 tasks: remaining MBPP)

Assertions verify:
- Train/Dev & Live Test are disjoint.
- Train/Dev & Organic Pool are disjoint.
- Live Test & Organic Pool are disjoint.
"""

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from tasks.all_tasks import ALL_TASKS

SEED = 42
MANIFEST_DIR = Path(__file__).resolve().parent

def build_manifests():
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load train/dev task IDs
    v2_df = pd.read_csv(Path(__file__).resolve().parent.parent.parent / "features_v2.csv")
    train_dev_ids = sorted(list(set(v2_df["task_id"].unique())))
    assert len(train_dev_ids) == 248, f"Expected 248 train/dev tasks, got {len(train_dev_ids)}"
    
    # 2. Get untouched tasks
    all_dict = {t["id"]: t for t in ALL_TASKS}
    all_ids = set(all_dict.keys())
    untouched_ids = all_ids - set(train_dev_ids)
    assert len(untouched_ids) == 427, f"Expected 427 untouched tasks, got {len(untouched_ids)}"
    
    # Separate by source
    he_ids = sorted([tid for tid in untouched_ids if all_dict[tid]["source"] == "humaneval"])
    easy_ids = sorted([tid for tid in untouched_ids if all_dict[tid]["source"] == "hand_easy"])
    hard_ids = sorted([tid for tid in untouched_ids if all_dict[tid]["source"] == "hand_hard"])
    mbpp_ids = sorted([tid for tid in untouched_ids if all_dict[tid]["source"] == "mbpp"])
    
    # Shuffle MBPP with fixed seed 42 to select 209 for live test and 127 for organic
    rng = random.Random(SEED)
    mbpp_shuffled = list(mbpp_ids)
    rng.shuffle(mbpp_shuffled)
    
    mbpp_live = sorted(mbpp_shuffled[:209])
    mbpp_organic = sorted(mbpp_shuffled[209:])
    
    live_test_ids = sorted(he_ids + easy_ids + hard_ids + mbpp_live)
    organic_ids = sorted(mbpp_organic)
    
    assert len(live_test_ids) == 300, f"Expected 300 live test tasks, got {len(live_test_ids)}"
    assert len(organic_ids) == 127, f"Expected 127 organic pool tasks, got {len(organic_ids)}"
    
    # Assert zero overlap
    s_train = set(train_dev_ids)
    s_live = set(live_test_ids)
    s_org = set(organic_ids)
    
    assert len(s_train & s_live) == 0, f"Leakage: {len(s_train & s_live)} tasks overlap train/dev and live test!"
    assert len(s_train & s_org) == 0, f"Leakage: {len(s_train & s_org)} tasks overlap train/dev and organic!"
    assert len(s_live & s_org) == 0, f"Leakage: {len(s_live & s_org)} tasks overlap live test and organic!"
    
    # Write JSON and TXT manifests
    with open(MANIFEST_DIR / "manifest_train_dev.json", "w", encoding="utf-8") as f:
        json.dump({"tasks": train_dev_ids, "count": len(train_dev_ids), "seed": SEED}, f, indent=2)
    with open(MANIFEST_DIR / "manifest_train_dev.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(train_dev_ids) + "\n")
        
    with open(MANIFEST_DIR / "manifest_live_test.json", "w", encoding="utf-8") as f:
        json.dump({
            "tasks": live_test_ids,
            "count": len(live_test_ids),
            "sources": {
                "humaneval": len(he_ids),
                "hand_easy": len(easy_ids),
                "hand_hard": len(hard_ids),
                "mbpp": len(mbpp_live),
            },
            "seed": SEED
        }, f, indent=2)
    with open(MANIFEST_DIR / "manifest_live_test.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(live_test_ids) + "\n")
        
    with open(MANIFEST_DIR / "manifest_organic.json", "w", encoding="utf-8") as f:
        json.dump({"tasks": organic_ids, "count": len(organic_ids), "seed": SEED}, f, indent=2)
    with open(MANIFEST_DIR / "manifest_organic.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(organic_ids) + "\n")
        
    print(f"Manifests successfully created in {MANIFEST_DIR}:")
    print(f"  Train/Dev pool: {len(train_dev_ids)} tasks")
    print(f"  Live test pool: {len(live_test_ids)} tasks ({len(he_ids)} HE, {len(easy_ids)+len(hard_ids)} hand-crafted, {len(mbpp_live)} MBPP)")
    print(f"  Organic pool:   {len(organic_ids)} tasks")
    print("Zero overlap confirmed across all pools.")

if __name__ == "__main__":
    build_manifests()
