"""
Phase 3 -- Length-ratio diagnostic & feature ablation script.

Analyzes the statistical relationship between length_ratio and corruption_type,
evaluates per-corruption AUROC for length_ratio, and compares GroupKFold CV
performance of full 3-feature model vs ablation (no length_ratio, length_ratio alone).
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

def run_diagnostic():
    os.makedirs("results/robustness", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)

    df = pd.read_csv("features.csv")
    print(f"Loaded features.csv with {len(df)} rows.")
    print("Columns:", list(df.columns))
    
    # 1. Summary stats of length_ratio per corruption_type
    stats_by_corr = df.groupby("corruption_type")["length_ratio"].agg(["count", "mean", "std", "min", "median", "max"]).reset_index()
    print("\n=== Length Ratio by Corruption Type ===")
    print(stats_by_corr.to_string(index=False))

    # 2. One-way ANOVA for length_ratio across corruption_type
    groups = [group["length_ratio"].values for name, group in df.groupby("corruption_type")]
    f_stat, p_val = stats.f_oneway(*groups)
    
    # Calculate eta-squared (SS_between / SS_total)
    grand_mean = df["length_ratio"].mean()
    ss_total = np.sum((df["length_ratio"] - grand_mean)**2)
    ss_between = np.sum([len(g) * (np.mean(g) - grand_mean)**2 for g in groups])
    eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

    print(f"\nANOVA F-statistic: {f_stat:.4f}, p-value: {p_val:.4e}, Eta-squared: {eta_sq:.4f}")

    # 3. Per-corruption AUROC for length_ratio alone
    positives = df[df["label"] == 1]
    per_corr_auroc = []
    
    print("\n=== Per-Corruption Type Discrimination (length_ratio alone) ===")
    for corr in df["corruption_type"].unique():
        if corr == "none":
            continue
        sub_df = pd.concat([positives, df[df["corruption_type"] == corr]])
        auroc = roc_auc_score(sub_df["label"], sub_df["length_ratio"])
        auprc = average_precision_score(sub_df["label"], sub_df["length_ratio"])
        count = len(df[df["corruption_type"] == corr])
        per_corr_auroc.append({
            "corruption_type": corr,
            "sample_count": count,
            "length_ratio_auroc": auroc,
            "length_ratio_auprc": auprc
        })
        print(f"Corruption: {corr:<20} | Count: {count:<3} | AUROC: {auroc:.4f} | AUPRC: {auprc:.4f}")

    # 4. GroupKFold Ablation Study (5-fold, grouped by task_id)
    gkf = GroupKFold(n_splits=5)

    features_full = ["cosine_similarity", "entity_overlap", "length_ratio"]
    features_no_lr = ["cosine_similarity", "entity_overlap"]
    features_lr_only = ["length_ratio"]

    models = {
        "XGBoost": lambda: XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42, eval_metric="logloss"),
        "LogisticRegression": lambda: LogisticRegression(random_state=42)
    }

    feature_sets = {
        "Full_3_Features": features_full,
        "No_Length_Ratio": features_no_lr,
        "Length_Ratio_Only": features_lr_only
    }

    cv_results = []

    for model_name, model_fn in models.items():
        for fset_name, fcols in feature_sets.items():
            oof_preds = np.zeros(len(df))
            
            for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=df["task_id"])):
                X_train, y_train = df.iloc[train_idx][fcols], df.iloc[train_idx]["label"]
                X_val, y_val = df.iloc[val_idx][fcols], df.iloc[val_idx]["label"]
                
                clf = model_fn()
                clf.fit(X_train, y_train)
                
                if hasattr(clf, "predict_proba"):
                    probs = clf.predict_proba(X_val)[:, 1]
                else:
                    probs = clf.predict(X_val)
                
                oof_preds[val_idx] = probs

            overall_auroc = roc_auc_score(df["label"], oof_preds)
            overall_auprc = average_precision_score(df["label"], oof_preds)
            overall_brier = brier_score_loss(df["label"], oof_preds)
            
            cv_results.append({
                "model": model_name,
                "feature_set": fset_name,
                "features_used": ", ".join(fcols),
                "oof_auroc": overall_auroc,
                "oof_auprc": overall_auprc,
                "oof_brier": overall_brier
            })
            print(f"Model: {model_name:<18} | FeatureSet: {fset_name:<18} | AUROC: {overall_auroc:.4f} | AUPRC: {overall_auprc:.4f} | Brier: {overall_brier:.4f}")

    # Save output CSVs
    diag_df = pd.DataFrame(per_corr_auroc)
    diag_df.to_csv("results/robustness/length_ratio_diagnostic.csv", index=False)
    
    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv("results/robustness/length_ratio_ablation.csv", index=False)
    
    print("\nSaved diagnostics to results/robustness/length_ratio_diagnostic.csv and length_ratio_ablation.csv")

if __name__ == "__main__":
    run_diagnostic()
