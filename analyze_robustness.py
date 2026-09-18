"""
Phase 9 -- Robustness Analysis Script

Evaluates performance breakdowns of the trained PRM model:
  1. Across corruption types (truncation, entity_omission, tool_result_drop, over_summarization)
  2. Across task sources (hand_easy, hand_hard, humaneval, mbpp)

Outputs CSVs to results/robustness/
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, confusion_matrix

def run_robustness_analysis():
    os.makedirs("results/robustness", exist_ok=True)

    df = pd.read_csv("features.csv")
    bundle = joblib.load("results/models/prm_final.joblib")
    model = bundle.get("model") or bundle.get("lr_model")
    feature_cols = bundle["feature_cols"]

    X = df[feature_cols].values
    y = df["label"].values

    # Get predictions
    probs = model.predict_proba(X)[:, 1]
    df["pred_prob"] = probs
    
    threshold = 0.33
    df["pred_label"] = (probs >= threshold).astype(int)

    positives = df[df["label"] == 1]

    # 1. Performance by Corruption Type
    corruption_stats = []
    for corr in df["corruption_type"].unique():
        if corr == "none":
            continue
        sub_df = pd.concat([positives, df[df["corruption_type"] == corr]])
        
        y_sub = sub_df["label"].values
        p_sub = sub_df["pred_prob"].values
        l_sub = sub_df["pred_label"].values

        auroc = roc_auc_score(y_sub, p_sub)
        auprc = average_precision_score(y_sub, p_sub)
        acc = accuracy_score(y_sub, l_sub)

        # Confusion matrix on sub_df
        tn, fp, fn, tp = confusion_matrix(y_sub, l_sub).ravel()
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        corruption_stats.append({
            "corruption_type": corr,
            "count_negatives": len(df[df["corruption_type"] == corr]),
            "auroc": auroc,
            "auprc": auprc,
            "accuracy": acc,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr
        })

    corr_df = pd.DataFrame(corruption_stats)
    corr_df.to_csv("results/robustness/robustness_by_corruption.csv", index=False)

    print("=== Robustness by Corruption Type ===")
    print(corr_df.to_string(index=False))

    # 2. Performance by Task Source
    source_stats = []
    for src in df["source"].unique():
        sub_df = df[df["source"] == src]
        if len(sub_df["label"].unique()) < 2:
            continue

        y_sub = sub_df["label"].values
        p_sub = sub_df["pred_prob"].values
        l_sub = sub_df["pred_label"].values

        auroc = roc_auc_score(y_sub, p_sub)
        auprc = average_precision_score(y_sub, p_sub)
        acc = accuracy_score(y_sub, l_sub)

        source_stats.append({
            "source": src,
            "total_samples": len(sub_df),
            "positives": int((y_sub == 1).sum()),
            "negatives": int((y_sub == 0).sum()),
            "auroc": auroc,
            "auprc": auprc,
            "accuracy": acc
        })

    src_df = pd.DataFrame(source_stats)
    src_df.to_csv("results/robustness/robustness_by_source.csv", index=False)

    print("\n=== Robustness by Source ===")
    print(src_df.to_string(index=False))

if __name__ == "__main__":
    run_robustness_analysis()
