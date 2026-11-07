"""
Deterministic deduplication script for Workstream C main evaluation results.

For each (task_id, seed):
- Keeps all clean rows (dropping identical duplicate keys).
- For corrupted handoffs, retains only the corruption type that appears FIRST in the file.
- Drops all rows belonging to subsequent/different corruption types caused by process restart hash drift.
- Logs per seed: initial rows, kept rows, removed rows, and affected task IDs.
"""

import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DedupMainResults")

RESULTS_DIR = Path(__file__).resolve().parent / "results"
SEEDS = [42, 43, 44]


def dedup_seed_csv(seed: int, dry_run: bool = False) -> Dict:
    csv_path = RESULTS_DIR / f"live_results_main_s{seed}.csv"
    if not csv_path.exists():
        logger.warning(f"File {csv_path} does not exist. Skipping.")
        return {}

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        fieldnames = reader[0].keys() if reader else []

    first_corr: Dict[Tuple[str, int], str] = {}
    seen_keys: Set[Tuple[str, str, int, str]] = set()
    kept_rows: List[Dict[str, str]] = []
    dropped_corr_rows: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    dropped_duplicates: int = 0

    for r in reader:
        tid = r["task_id"]
        arm = r["arm"]
        s = int(r["seed"])
        htype = r["handoff_type"]

        if htype == "clean":
            key = (tid, arm, s, htype)
            if key not in seen_keys:
                seen_keys.add(key)
                kept_rows.append(r)
            else:
                dropped_duplicates += 1
        else:
            if (tid, s) not in first_corr:
                first_corr[(tid, s)] = htype

            chosen = first_corr[(tid, s)]
            if htype == chosen:
                key = (tid, arm, s, htype)
                if key not in seen_keys:
                    seen_keys.add(key)
                    kept_rows.append(r)
                else:
                    dropped_duplicates += 1
            else:
                dropped_corr_rows[tid][htype] += 1

    total_removed = len(reader) - len(kept_rows)

    # Check for partial arm distributions on kept rows
    task_htype_arms = defaultdict(lambda: defaultdict(set))
    for r in kept_rows:
        task_htype_arms[r["task_id"]][r["handoff_type"]].add(r["arm"])

    partial_tasks = {}
    for tid, htypes in task_htype_arms.items():
        for h, arms in htypes.items():
            if len(arms) != 7:
                partial_tasks[(tid, h)] = sorted(list(arms))

    report = {
        "seed": seed,
        "original_rows": len(reader),
        "kept_rows": len(kept_rows),
        "total_removed": total_removed,
        "dropped_secondary_corruption_rows": sum(sum(v.values()) for v in dropped_corr_rows.values()),
        "dropped_duplicates": dropped_duplicates,
        "affected_tasks_count": len(dropped_corr_rows),
        "affected_tasks": {tid: dict(corrs) for tid, corrs in dropped_corr_rows.items()},
        "partial_tasks": partial_tasks,
    }

    if not dry_run and total_removed > 0:
        tmp_path = csv_path.with_name(f"{csv_path.name}.tmp")
        with open(tmp_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept_rows)
        tmp_path.replace(csv_path)
        logger.info(f"Overwrote {csv_path} with {len(kept_rows)} deduplicated rows.")

    return report


def main():
    logger.info("Starting deterministic deduplication across main seed results...")
    for s in SEEDS:
        rep = dedup_seed_csv(s, dry_run=False)
        print(f"\n==================== SEED {s} DEDUP REPORT ====================")
        print(f"Original rows: {rep['original_rows']}")
        print(f"Kept rows:     {rep['kept_rows']}")
        print(f"Removed rows:  {rep['total_removed']}")
        print(f"  - Dropped secondary corruption rows: {rep['dropped_secondary_corruption_rows']}")
        print(f"  - Dropped duplicate exact keys:      {rep['dropped_duplicates']}")
        print(f"Affected tasks with secondary corruptions: {rep['affected_tasks_count']}")
        print(f"Partial task handoffs (boundary tasks):    {rep['partial_tasks']}")
        print("===============================================================\n")


if __name__ == "__main__":
    main()

