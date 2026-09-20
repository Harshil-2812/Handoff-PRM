"""
Reads rollouts.csv (produced by rollout_runner.py), computes the 3 structural
features for every row, and saves the final labeled feature dataset that
train_classifier.py trains on.

Usage: python build_dataset.py
"""

import pandas as pd
from tqdm import tqdm
from feature_extraction import extract_features

INPUT_CSV = "rollouts.csv"
OUTPUT_CSV = "features.csv"


def main():
    df = pd.read_csv(INPUT_CSV)
    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="extracting features"):
        feats = extract_features(row["handoff"], row["problem"])
        feats["task_id"] = row["task_id"]
        feats["source"] = row.get("source", "unknown")
        feats["corruption_type"] = row["corruption_type"]
        feats["label"] = row["label"]
        rows.append(feats)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved {len(out_df)} feature rows to {OUTPUT_CSV}")
    print(out_df["label"].value_counts())


if __name__ == "__main__":
    main()
