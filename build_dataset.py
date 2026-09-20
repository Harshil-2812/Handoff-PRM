"""
Reads rollouts.csv, computes all 13 features for every row, and saves
features.csv for training.

Also resolves entry_point from task metadata for the function_name_preserved
feature.

Usage: python build_dataset.py
"""

import pandas as pd
from tqdm import tqdm

from feature_extraction import extract_features_batch
from tasks.all_tasks import ALL_TASKS

INPUT_CSV  = "rollouts.csv"
OUTPUT_CSV = "features.csv"


def build_entry_point_lookup():
    """Returns {task_id: entry_point} for all tasks."""
    return {task["id"]: task.get("entry_point", "") for task in ALL_TASKS}


def main():
    df = pd.read_csv(INPUT_CSV)
    entry_point_lookup = build_entry_point_lookup()
    
    out_df = extract_features_batch(df, entry_point_lookup=entry_point_lookup)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(out_df)} feature rows to {OUTPUT_CSV}")
    print(out_df["label"].value_counts())
    print(f"\nFeature columns: {[c for c in out_df.columns if c not in ['task_id','source','corruption_type','label']]}")


if __name__ == "__main__":
    main()
