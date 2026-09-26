"""
train_models.py — Handoff-PRM Classifier Training (Paper Reference Copy)
=========================================================================
Trains all 5 classifier variants (LR, XGBoost, RF, MLP, SVM) used in Table VI
of the paper. Uses 5-fold GroupKFold to prevent data leakage across tasks.

This is the teacher-reference version. The actual training runs were executed via:
  model_sweep_12features_final.py  (primary 483-row dataset)
  model_comparison_sweep.py        (earlier 177-row dataset sweep)

Paper sections: §VI-A, §VI-B, Table VI

Key design decisions:
  1. GroupKFold by task_id: all rollouts from the same task land in the same fold,
     preventing the model from memorising a task's "style" across train/test splits.
  2. OOF (out-of-fold) predictions: AUROC/AUPRC/Brier/logloss are computed on
     OOF predictions, not held-out test split, for maximum data efficiency given n=483.
  3. Class imbalance: dataset has ~2:1 pos:neg ratio; models use class_weight='balanced'
     or scale_pos_weight where applicable.
  4. No NLI features in Stage-1: nli_entailment_mean is excluded from all 5 models
     because the CrossEncoder adds ~80ms per sample (unacceptable for real-time gating).
     NLI is used only in Stage-2 of the cascade.

Reproduce Table VI:
  python train_models.py --dataset ../../final_dataset_full_features.csv
"""

import argparse
import warnings
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    log_loss, f1_score
)
import xgboost as xgb
import joblib
import shap

warnings.filterwarnings("ignore")

# ─── Feature set (Table V of paper) ──────────────────────────────────────────
FEATURE_COLS = [
    "cosine_similarity",     # semantic similarity (sentence-transformers)
    "entity_overlap",        # NE / noun-chunk recall (spaCy)
    "length_ratio",          # word count ratio handoff / problem
    "sentence_count_ratio",  # handoff sentences / problem sentences
    "section_coverage",      # fraction of expected sections present
    "verbatim_copy_rate",    # 4-gram verbatim overlap rate
    "trailing_specificity",  # info density of last 30% of handoff
    "constraint_count",      # explicit constraint statement count
    "role_pronoun_rate",     # first-person pronoun rate
    "novel_api_rate",        # fraction of backtick IDs not in problem
    "function_name_preserved",  # 1.0 if entry_point name in handoff
]

# NLI excluded from Stage-1 feature set (see docstring above)
NLI_EXCLUDED = ["nli_entailment_mean", "nli_contradiction_max"]


def build_models():
    """Returns dict of {name: sklearn-compatible estimator}."""
    rf = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("clf", RandomForestClassifier(
            n_estimators=100, max_depth=3, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        )),
    ])
    xgb_clf = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("clf", xgb.XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            reg_alpha=1.0, reg_lambda=1.0,
            scale_pos_weight=2.0,  # ~2:1 class ratio
            eval_metric="logloss", random_state=42, verbosity=0
        )),
    ])
    lr = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=0.1, penalty="l2", solver="lbfgs",
                                    class_weight="balanced", max_iter=1000,
                                    random_state=42)),
    ])
    mlp = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(32, 16), activation="relu", solver="adam",
            max_iter=500, early_stopping=True, random_state=42
        )),
    ])
    svm = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=1.0, probability=True,
                    class_weight="balanced", random_state=42)),
    ])
    return {
        "RandomForest": rf,
        "XGBoost": xgb_clf,
        "LogisticRegression": lr,
        "SmallMLP": mlp,
        "SVM": svm,
    }


def evaluate_oof(y_true, y_prob, threshold=0.5):
    """Compute all metrics used in Table VI from OOF predictions."""
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "oof_auroc": round(roc_auc_score(y_true, y_prob), 4),
        "oof_auprc": round(average_precision_score(y_true, y_prob), 4),
        "oof_brier": round(brier_score_loss(y_true, y_prob), 4),
        "oof_logloss": round(log_loss(y_true, y_prob), 4),
        "oof_f1": round(f1_score(y_true, y_pred), 4),
    }


def main(dataset_path: str, output_dir: str = ".", seed: int = 42):
    df = pd.read_csv(dataset_path)
    X = df[FEATURE_COLS].values
    y = df["label"].values
    groups = df["task_id"].values

    gkf = GroupKFold(n_splits=5)
    models = build_models()
    results = {}

    print(f"Dataset: {len(df)} rows, {len(FEATURE_COLS)} features, "
          f"{y.sum()} positives ({y.mean():.1%})")
    print(f"Tasks: {len(np.unique(groups))} unique")
    print()

    best_auroc = 0.0
    best_model_name = None
    best_model_fitted = None

    for name, model in models.items():
        print(f"Training {name}...")
        oof_probs = cross_val_predict(
            model, X, y, groups=groups, cv=gkf, method="predict_proba"
        )[:, 1]

        metrics = evaluate_oof(y, oof_probs)
        results[name] = metrics
        print(f"  OOF AUROC={metrics['oof_auroc']:.4f}  AUPRC={metrics['oof_auprc']:.4f}"
              f"  Brier={metrics['oof_brier']:.4f}  LogLoss={metrics['oof_logloss']:.4f}")

        if metrics["oof_auroc"] > best_auroc:
            best_auroc = metrics["oof_auroc"]
            best_model_name = name

    # Refit best model on full data for deployment
    print(f"\nBest model: {best_model_name} (AUROC={best_auroc:.4f})")
    best_model_fitted = models[best_model_name]
    best_model_fitted.fit(X, y)

    # SHAP explainer (for gate.py)
    clf = best_model_fitted.named_steps["clf"]
    X_imp = best_model_fitted.named_steps["imputer"].transform(X)
    explainer = shap.TreeExplainer(clf)

    # Save model bundle
    model_path = f"{output_dir}/handoff_prm.joblib"
    joblib.dump({
        "model": best_model_fitted,
        "explainer": explainer,
        "feature_cols": FEATURE_COLS,
        "model_name": best_model_name,
        "oof_auroc": best_auroc,
        "seed": seed,
    }, model_path)
    print(f"Model saved: {model_path}")

    # Save results JSON
    results_path = f"{output_dir}/cv_results_primary.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Handoff-PRM classifiers")
    parser.add_argument("--dataset", default="../../final_dataset_full_features.csv",
                        help="Path to feature CSV")
    parser.add_argument("--output_dir", default=".", help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.dataset, args.output_dir, args.seed)
