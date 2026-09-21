"""
Handoff PRM — Final Model Sweep on Cleaned 12-Feature Set
============================================================

Loads final_dataset_full_features.csv (the Step 7 output), drops the
two problematic features (nli_contradiction_max — max-saturation
artifact correlated with length; signature_param_diff — 65% missing),
and runs the same 4-model, 5-fold task-grouped OOF comparison used
for the 3-feature baseline, so the two results are directly comparable.

Also reports per-corruption-type AUROC (like the original paper's
Table IX) using the best model's OOF predictions.

Requirements: pandas numpy scikit-learn xgboost
"""

import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    accuracy_score,
    f1_score,
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed — skipping XGBoost.")

INPUT_PATH = "final_dataset_full_features.csv"
RESULTS_PATH = "model_comparison_results_12feat.csv"
PER_CORRUPTION_PATH = "per_corruption_type_auroc_12feat.csv"

# Cleaned feature set: dropped nli_contradiction_max (saturation artifact,
# correlated with length rather than true contradiction) and
# signature_param_diff (65% missing / uncomputable).
FEATURE_COLS = [
    "cosine_similarity", "entity_overlap", "length_ratio", "sentence_count_ratio",
    "section_coverage", "verbatim_copy_rate", "trailing_specificity", "constraint_count",
    "role_pronoun_rate", "novel_api_rate", "function_name_preserved",
    "nli_entailment_mean",
]


def get_models():
    """Fresh, unfit model pipelines. Imputer added since a couple of
    features (e.g. entity_overlap on empty-entity rows) can occasionally
    be NaN at the edges."""
    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=6, class_weight="balanced",
                random_state=42, n_jobs=-1,
            )),
        ]),
        "Small MLP": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(16,), max_iter=2000, random_state=42, early_stopping=True,
            )),
        ]),
    }
    if HAS_XGB:
        # XGBoost handles NaN natively — no imputer needed
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            eval_metric="logloss", random_state=42, n_jobs=-1,
        )
    return models


def run_oof_comparison(X, y, groups, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    results = []
    oof_predictions = {}

    for name in get_models().keys():
        print(f"\nRunning {name} under {n_splits}-fold GroupKFold...")
        oof_proba = np.zeros(len(y))
        fold_aurocs = []
        latencies = []

        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
            train_tasks = set(groups[train_idx])
            test_tasks = set(groups[test_idx])
            assert len(train_tasks & test_tasks) == 0, (
                f"LEAKAGE in fold {fold_idx} for {name}: "
                f"{len(train_tasks & test_tasks)} overlapping task_ids"
            )

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = get_models()[name]
            model.fit(X_train, y_train)

            start = time.perf_counter()
            proba = model.predict_proba(X_test)[:, 1]
            elapsed = time.perf_counter() - start
            latencies.append((elapsed / len(X_test)) * 1000)

            oof_proba[test_idx] = proba
            if len(np.unique(y_test)) > 1:
                fold_aurocs.append(roc_auc_score(y_test, proba))

        oof_pred_label = (oof_proba >= 0.5).astype(int)
        results.append({
            "Model": name,
            "OOF_AUROC_mean": np.mean(fold_aurocs),
            "OOF_AUROC_std": np.std(fold_aurocs),
            "OOF_AUROC_overall": roc_auc_score(y, oof_proba),
            "OOF_AUPRC": average_precision_score(y, oof_proba),
            "Brier": brier_score_loss(y, oof_proba),
            "LogLoss": log_loss(y, oof_proba),
            "Accuracy": accuracy_score(y, oof_pred_label),
            "F1": f1_score(y, oof_pred_label),
            "Latency_ms_per_example": np.mean(latencies),
        })
        oof_predictions[name] = oof_proba
        print(f"  {name}: OOF AUROC = {np.mean(fold_aurocs):.4f} ± {np.std(fold_aurocs):.4f}")

    results_df = pd.DataFrame(results).sort_values("OOF_AUROC_mean", ascending=False)
    return results_df, oof_predictions


def per_corruption_auroc(df, y, best_model_oof_proba):
    """AUROC of clean-vs-corrupted, split out per corruption type.
    Positives (label=1, corruption_type='none') are compared against
    each corruption type's negatives separately, matching the original
    paper's Table IX format."""
    rows = []
    positive_mask = df["corruption_type"] == "none"
    pos_proba = best_model_oof_proba[positive_mask.values]
    pos_y = y[positive_mask.values]

    for ctype in df.loc[df["corruption_type"] != "none", "corruption_type"].unique():
        neg_mask = df["corruption_type"] == ctype
        neg_proba = best_model_oof_proba[neg_mask.values]
        neg_y = y[neg_mask.values]

        combined_proba = np.concatenate([pos_proba, neg_proba])
        combined_y = np.concatenate([pos_y, neg_y])

        n = neg_mask.sum()
        if len(np.unique(combined_y)) > 1:
            auroc = roc_auc_score(combined_y, combined_proba)
        else:
            auroc = np.nan

        rows.append({"corruption_type": ctype, "n": n, "auroc": auroc})

    return pd.DataFrame(rows).sort_values("auroc", ascending=False)


def main():
    print(f"Loading {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows, {df['task_id'].nunique()} unique tasks")
    print(f"Using {len(FEATURE_COLS)} features: {FEATURE_COLS}")

    missing_cols = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing expected feature columns: {missing_cols}")

    X = df[FEATURE_COLS].values
    y = df["label"].values
    groups = df["task_id"].values

    print(f"\nClass balance: {np.bincount(y)} (0=insufficient, 1=sufficient)")
    print(f"Feature matrix shape: {X.shape}")
    print(f"NaN count per feature:\n{df[FEATURE_COLS].isna().sum()}")

    results_df, oof_predictions = run_oof_comparison(X, y, groups, n_splits=5)

    print("\n" + "=" * 70)
    print("FINALIZED 12-FEATURE OUT-OF-FOLD MODEL COMPARISON")
    print("=" * 70)
    print(results_df.to_string(index=False))
    results_df.to_csv(RESULTS_PATH, index=False)
    print(f"\nSaved results to {RESULTS_PATH}")

    best_model_name = results_df.iloc[0]["Model"]
    print(f"\nBest model: {best_model_name} — computing per-corruption-type AUROC breakdown...")
    per_corr_df = per_corruption_auroc(df, y, oof_predictions[best_model_name])
    print("\n" + "=" * 70)
    print(f"PER-CORRUPTION-TYPE AUROC ({best_model_name}, clean vs. each corruption type)")
    print("=" * 70)
    print(per_corr_df.to_string(index=False))
    per_corr_df.to_csv(PER_CORRUPTION_PATH, index=False)
    print(f"\nSaved to {PER_CORRUPTION_PATH}")

    print("\n" + "=" * 70)
    print("COMPARISON vs. 3-FEATURE BASELINE (fill in your earlier numbers)")
    print("=" * 70)
    print("3-feature baseline: LR=0.8215  RF=0.8452  MLP=0.7925  XGB=0.8507")
    print(f"12-feature result:  see table above")


if __name__ == "__main__":
    main()
