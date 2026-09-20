"""
Robustness Analysis using OUT-OF-FOLD predictions.

Uses the OOF predictions from model_comparison.py (results/calibration/oof_predictions.csv)
rather than re-running the trained model, giving honest held-out performance numbers.

Outputs:
  results/robustness/robustness_by_corruption.csv
  results/robustness/robustness_by_source.csv

Usage: python analyze_robustness.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

OOF_PATH     = "results/calibration/oof_predictions.csv"
FEAT_PATH    = "features.csv"
OUTPUT_DIR   = "results/robustness"


def load_oof_with_metadata():
    """Merge OOF predictions with corruption_type and source from features.csv."""
    oof  = pd.read_csv(OOF_PATH)
    feat = pd.read_csv(FEAT_PATH)

    # oof has: index (matches features.csv row order), y_true, oof_prob
    # Merge on index position
    merged = feat.copy()
    model_col = "XGBoost" if "XGBoost" in oof.columns else "LogisticRegression"
    merged["oof_prob"] = oof[model_col].values
    merged["y_true"]   = oof["label"].values

    return merged


def robustness_by_corruption(df):
    positives = df[df["label"] == 1]
    results = []

    for corr in sorted(df["corruption_type"].unique()):
        if corr == "none":
            continue
        sub = pd.concat([positives, df[df["corruption_type"] == corr]]).drop_duplicates()
        y    = sub["label"].values
        prob = sub["oof_prob"].values

        if len(np.unique(y)) < 2:
            continue

        auroc = roc_auc_score(y, prob)
        auprc = average_precision_score(y, prob)
        pred  = (prob >= 0.5).astype(int)
        acc   = accuracy_score(y, pred)
        n_neg = len(df[df["corruption_type"] == corr])

        results.append({
            "corruption_type":  corr,
            "n_negatives":      n_neg,
            "oof_auroc":        round(auroc, 4),
            "oof_auprc":        round(auprc, 4),
            "oof_accuracy":     round(acc, 4),
        })

    return pd.DataFrame(results)


def robustness_by_source(df):
    results = []
    for src in sorted(df["source"].unique()):
        sub = df[df["source"] == src]
        y    = sub["label"].values
        prob = sub["oof_prob"].values
        if len(np.unique(y)) < 2:
            continue
        auroc = roc_auc_score(y, prob)
        auprc = average_precision_score(y, prob)
        results.append({
            "source":       src,
            "n_total":      len(sub),
            "n_positives":  int((y == 1).sum()),
            "n_negatives":  int((y == 0).sum()),
            "oof_auroc":    round(auroc, 4),
            "oof_auprc":    round(auprc, 4),
        })
    return pd.DataFrame(results)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(OOF_PATH):
        print(f"ERROR: {OOF_PATH} not found. Run model_comparison.py first.")
        exit(1)

    df = load_oof_with_metadata()
    print(f"Loaded {len(df)} OOF predictions")

    corr_df = robustness_by_corruption(df)
    corr_df.to_csv(f"{OUTPUT_DIR}/robustness_by_corruption.csv", index=False)
    print("\n=== OOF Robustness by Corruption Type ===")
    print(corr_df.to_string(index=False))

    src_df = robustness_by_source(df)
    src_df.to_csv(f"{OUTPUT_DIR}/robustness_by_source.csv", index=False)
    print("\n=== OOF Robustness by Source ===")
    print(src_df.to_string(index=False))
