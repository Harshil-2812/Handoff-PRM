"""
Held-Out Corruption Type Evaluation.

For each corruption type C in the dataset:
  - Train on all rows EXCEPT those where corruption_type == C
  - Test on: all label=1 rows + all rows where corruption_type == C
  - Report AUROC for each held-out corruption type

This is the key generalization test: can the model detect a corruption
type it has never seen during training?

Usage: python held_out_corruption_eval.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import stats

FEATURES_CSV = "features.csv"
OUTPUT_DIR   = "results/robustness"


def load_data():
    df = pd.read_csv(FEATURES_CSV)
    feature_cols = [
        c for c in df.columns
        if c not in ["task_id", "source", "corruption_type", "label"]
    ]
    return df, feature_cols


def build_model():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("lr",     LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
    ])


def held_out_eval(df, feature_cols):
    """Leave-one-corruption-type-out evaluation."""
    corruption_types = [c for c in df["corruption_type"].unique() if c != "none"]
    positives = df[df["label"] == 1]
    results = []

    print("=" * 70)
    print("Held-Out Corruption Type Evaluation")
    print("=" * 70)

    for held_out in sorted(corruption_types):
        # Train: all data EXCEPT held-out corruption type
        train_df = df[df["corruption_type"] != held_out]
        # Test: positives + held-out corruption type examples
        test_neg  = df[df["corruption_type"] == held_out]
        test_df   = pd.concat([positives, test_neg]).drop_duplicates()

        if len(test_df["label"].unique()) < 2:
            print(f"  {held_out:<30} SKIP (only one class in test set)")
            continue

        X_train = train_df[feature_cols].values
        y_train = train_df["label"].values
        X_test  = test_df[feature_cols].values
        y_test  = test_df["label"].values

        model = build_model()
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]

        auroc = roc_auc_score(y_test, probs)
        auprc = average_precision_score(y_test, probs)
        n_neg = len(test_neg)
        n_pos = len(positives)

        print(f"  {held_out:<30} n_neg={n_neg:3d}  AUROC={auroc:.4f}  AUPRC={auprc:.4f}")
        results.append({
            "held_out_corruption": held_out,
            "n_neg_test":          n_neg,
            "n_pos_test":          n_pos,
            "train_size":          len(train_df),
            "auroc":               round(auroc, 4),
            "auprc":               round(auprc, 4),
        })

    return pd.DataFrame(results)


def cross_source_eval(df, feature_cols):
    """Leave-one-source-out evaluation."""
    sources = df["source"].unique()
    results = []

    print("\n" + "=" * 70)
    print("Cross-Source Generalization Evaluation")
    print("=" * 70)

    for held_out_src in sorted(sources):
        train_df = df[df["source"] != held_out_src]
        test_df  = df[df["source"] == held_out_src]

        if len(test_df["label"].unique()) < 2:
            print(f"  {held_out_src:<20} SKIP (only one class)")
            continue
        if len(train_df["label"].unique()) < 2:
            print(f"  {held_out_src:<20} SKIP (only one class in train)")
            continue

        X_train = train_df[feature_cols].values
        y_train = train_df["label"].values
        X_test  = test_df[feature_cols].values
        y_test  = test_df["label"].values

        model = build_model()
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]

        auroc = roc_auc_score(y_test, probs)
        auprc = average_precision_score(y_test, probs)

        print(f"  Test source: {held_out_src:<15}  n={len(test_df):4d}  AUROC={auroc:.4f}  AUPRC={auprc:.4f}")
        results.append({
            "test_source":  held_out_src,
            "n_test":       len(test_df),
            "n_train":      len(train_df),
            "auroc":        round(auroc, 4),
            "auprc":        round(auprc, 4),
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df, feature_cols = load_data()
    print(f"Loaded {len(df)} rows, {len(feature_cols)} features")
    print(f"Features: {feature_cols}")
    print(f"Corruption types: {sorted(df['corruption_type'].unique())}")

    heldout_df = held_out_eval(df, feature_cols)
    heldout_df.to_csv(f"{OUTPUT_DIR}/held_out_corruption_eval.csv", index=False)
    print(f"\nSaved -> {OUTPUT_DIR}/held_out_corruption_eval.csv")

    cross_df = cross_source_eval(df, feature_cols)
    cross_df.to_csv(f"{OUTPUT_DIR}/cross_source_eval.csv", index=False)
    print(f"Saved -> {OUTPUT_DIR}/cross_source_eval.csv")
