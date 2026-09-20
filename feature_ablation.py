"""
A5 -- Full Feature Ablation Script

Extends the original length-ratio diagnostic to cover ALL three features
individually, completing the ablation study required for the paper.

Feature sets evaluated:
  Full_3_Features          : cosine_similarity, entity_overlap, length_ratio
  No_Length_Ratio          : cosine_similarity, entity_overlap
  No_Cosine_Similarity     : entity_overlap, length_ratio
  No_Entity_Overlap        : cosine_similarity, length_ratio
  Cosine_Similarity_Only   : cosine_similarity          [NEW]
  Entity_Overlap_Only      : entity_overlap              [NEW]
  Length_Ratio_Only        : length_ratio

CV strategy: 5-fold GroupKFold grouped by task_id (no task leaks across folds)
Models     : XGBoost, LogisticRegression
Metrics    : OOF AUROC (+ per-fold std), AUPRC, Brier score

Outputs:
  results/robustness/feature_ablation.csv         -- 14 rows (7 sets x 2 models)
  results/robustness/per_feature_discrimination.csv -- per-feature x corruption AUROC
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression


def get_all_features(df):
    return [c for c in df.columns if c not in ["task_id", "source", "corruption_type", "label"]]


def get_feature_sets(all_features):
    fsets = {
        "Full_All_Features":         all_features,
        "No_Length_Ratio":           [f for f in all_features if f != "length_ratio"],
        "No_Cosine_Similarity":      [f for f in all_features if f != "cosine_similarity"],
        "No_Entity_Overlap":         [f for f in all_features if f != "entity_overlap"],
        "Core_3_Features":           ["cosine_similarity", "entity_overlap", "length_ratio"],
        "Cosine_Similarity_Only":    ["cosine_similarity"],
        "Entity_Overlap_Only":       ["entity_overlap"],
        "Length_Ratio_Only":         ["length_ratio"],
    }
    if "function_name_preserved" in all_features:
        fsets["Function_Name_Preserved_Only"] = ["function_name_preserved"]
    if "section_coverage" in all_features:
        fsets["Section_Coverage_Only"] = ["section_coverage"]
    return fsets


def _make_model(name: str):
    if name == "XGBoost":
        return XGBClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            random_state=42, eval_metric="logloss",
        )
    return LogisticRegression(random_state=42)


def run_ablation():
    os.makedirs("results/robustness", exist_ok=True)

    df = pd.read_csv("features.csv")
    all_features = get_all_features(df)
    FEATURE_SETS = get_feature_sets(all_features)
    FEATURE_COLS = all_features

    print(f"Loaded features.csv  ->  {len(df)} rows, {df['task_id'].nunique()} unique tasks")
    print(f"Features ({len(all_features)}):", all_features)
    print("Label distribution:\n", df["label"].value_counts().to_string(), "\n")

    # -----------------------------------------------------------------------
    # 1.  GroupKFold ablation -- all feature sets x both models
    # -----------------------------------------------------------------------
    gkf = GroupKFold(n_splits=5)
    cv_results = []

    for model_name in ("XGBoost", "LogisticRegression"):
        for fset_name, fcols in FEATURE_SETS.items():
            oof_preds = np.zeros(len(df))
            fold_aurocs = []

            for fold, (train_idx, val_idx) in enumerate(
                gkf.split(df, groups=df["task_id"])
            ):
                X_tr = df.iloc[train_idx][fcols]
                y_tr = df.iloc[train_idx]["label"]
                X_va = df.iloc[val_idx][fcols]
                y_va = df.iloc[val_idx]["label"]

                clf = _make_model(model_name)
                clf.fit(X_tr, y_tr)

                probs = (
                    clf.predict_proba(X_va)[:, 1]
                    if hasattr(clf, "predict_proba")
                    else clf.predict(X_va)
                )
                oof_preds[val_idx] = probs
                fold_aurocs.append(roc_auc_score(y_va, probs))

            oof_auroc = roc_auc_score(df["label"], oof_preds)
            oof_auprc = average_precision_score(df["label"], oof_preds)
            oof_brier = brier_score_loss(df["label"], oof_preds)
            std_auroc  = float(np.std(fold_aurocs))
            mean_fold_auroc = float(np.mean(fold_aurocs))

            cv_results.append({
                "model":           model_name,
                "feature_set":     fset_name,
                "features_used":   ", ".join(fcols),
                "oof_auroc":       round(oof_auroc, 6),
                "mean_fold_auroc": round(mean_fold_auroc, 6),
                "std_fold_auroc":  round(std_auroc, 6),
                "oof_auprc":       round(oof_auprc, 6),
                "oof_brier":       round(oof_brier, 6),
            })
            print(
                f"  {model_name:<20} | {fset_name:<25} | "
                f"AUROC {oof_auroc:.4f} +/-{std_auroc:.4f} | "
                f"AUPRC {oof_auprc:.4f} | Brier {oof_brier:.4f}"
            )

    ablation_df = pd.DataFrame(cv_results)
    out_path = "results/robustness/feature_ablation.csv"
    ablation_df.to_csv(out_path, index=False)
    print(f"\nSaved ablation table  ->  {out_path}  ({len(ablation_df)} rows)\n")

    # -----------------------------------------------------------------------
    # 2.  Per-corruption AUROC for each single feature
    #     (extends length_ratio_diagnostic.csv pattern to all 3 features)
    # -----------------------------------------------------------------------
    positives = df[df["label"] == 1]
    disc_results = []

    print("=== Per-Corruption Discrimination (single feature) ===")
    for feat in FEATURE_COLS:
        for corr in df["corruption_type"].unique():
            if corr == "none":
                continue
            sub_df = pd.concat([positives, df[df["corruption_type"] == corr]])
            count  = len(df[df["corruption_type"] == corr])
            auroc  = roc_auc_score(sub_df["label"], sub_df[feat])
            auprc  = average_precision_score(sub_df["label"], sub_df[feat])
            disc_results.append({
                "feature":         feat,
                "corruption_type": corr,
                "sample_count":    count,
                "auroc":           round(auroc, 6),
                "auprc":           round(auprc, 6),
            })
            print(
                f"  {feat:<22} | {corr:<25} | "
                f"n={count:<4} | AUROC {auroc:.4f} | AUPRC {auprc:.4f}"
            )

    disc_df = pd.DataFrame(disc_results)
    disc_path = "results/robustness/per_feature_discrimination.csv"
    disc_df.to_csv(disc_path, index=False)
    print(f"\nSaved discrimination table  ->  {disc_path}  ({len(disc_df)} rows)")

    # -----------------------------------------------------------------------
    # 3.  One-way ANOVA for each feature across corruption types
    # -----------------------------------------------------------------------
    print("\n=== ANOVA: feature separability across corruption types ===")
    for feat in FEATURE_COLS:
        groups    = [g[feat].values for _, g in df.groupby("corruption_type")]
        f_stat, p = stats.f_oneway(*groups)
        grand_m   = df[feat].mean()
        ss_tot    = np.sum((df[feat] - grand_m) ** 2)
        ss_bet    = np.sum([len(g) * (np.mean(g) - grand_m) ** 2 for g in groups])
        eta_sq    = ss_bet / ss_tot if ss_tot > 0 else 0.0
        print(
            f"  {feat:<22} | F={f_stat:.4f}  p={p:.2e}  eta_sq={eta_sq:.4f}"
        )


if __name__ == "__main__":
    run_ablation()
