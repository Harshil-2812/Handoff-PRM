"""
TABLE VIII -- Full GroupKFold Model Comparison (re-run on new dataset)

Evaluates Logistic Regression, XGBoost, Random Forest, and SVM (RBF)
using 5-fold GroupKFold cross-validation on the new, larger features.csv.
Reports per-fold AUROC std so variance numbers match the paper table.

LLM-as-Judge baseline row is left as n/a for Brier/Log Loss (no probabilities)
and is BLOCKED pending confirmation of the baseline split -- per task notes.

Outputs:
  results/tables/cv_results.csv          -- full metric table
  results/calibration/oof_predictions.csv -- raw OOF probabilities
  results/models/prm_final.joblib        -- trained XGBoost + LR + SHAP explainer
  handoff_prm.joblib                     -- compatibility copy for gate.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, brier_score_loss, log_loss
)
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
import time


def get_feature_cols(df):
    return [c for c in df.columns if c not in ["task_id", "source", "corruption_type", "label"]]


def _clone(name: str):
    """Return a fresh unfitted model instance."""
    if name == "LogisticRegression":
        return LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    if name == "XGBoost":
        return XGBClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            random_state=42, eval_metric="logloss",
        )
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=50, max_depth=3, class_weight="balanced", random_state=42)
    if name == "SVM_RBF":
        return make_pipeline(
            StandardScaler(),
            SVC(probability=True, class_weight="balanced", random_state=42)
        )
    raise ValueError(f"Unknown model: {name}")


def run_model_comparison():
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/calibration", exist_ok=True)

    df = pd.read_csv("features.csv")
    FEATURE_COLS = get_feature_cols(df)
    print(f"Loaded features.csv  ->  {len(df)} rows, {df['task_id'].nunique()} tasks")
    print(f"Using {len(FEATURE_COLS)} features: {FEATURE_COLS}")
    X = df[FEATURE_COLS].values
    y = df["label"].values
    groups = df["task_id"].values

    gkf = GroupKFold(n_splits=5)
    model_names = ["LogisticRegression", "XGBoost", "RandomForest", "SVM_RBF"]

    cv_summary = []
    oof_store  = {}          # {model_name: oof_prob_array}

    for name in model_names:
        oof_probs   = np.zeros(len(df))
        fold_aurocs = []
        fold_auprcs = []

        # Measure inference latency on a single example
        dummy_clf = _clone(name)
        dummy_clf.fit(X[:10], y[:10])
        t0 = time.perf_counter()
        for _ in range(100):
            dummy_clf.predict_proba(X[:1])
        latency_ms = (time.perf_counter() - t0) / 100 * 1000

        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            clf = _clone(name)
            clf.fit(X[train_idx], y[train_idx])
            probs = clf.predict_proba(X[val_idx])[:, 1]
            oof_probs[val_idx] = probs
            fold_aurocs.append(roc_auc_score(y[val_idx], probs))
            fold_auprcs.append(average_precision_score(y[val_idx], probs))

        # OOF aggregate metrics
        auroc  = roc_auc_score(y, oof_probs)
        auprc  = average_precision_score(y, oof_probs)
        brier  = brier_score_loss(y, oof_probs)
        logloss = log_loss(y, oof_probs)

        preds_50 = (oof_probs >= 0.5).astype(int)
        acc  = accuracy_score(y, preds_50)
        prec = precision_score(y, preds_50, zero_division=0)
        rec  = recall_score(y, preds_50, zero_division=0)
        f1   = f1_score(y, preds_50, zero_division=0)

        mean_fold_auroc = float(np.mean(fold_aurocs))
        std_fold_auroc  = float(np.std(fold_aurocs))

        cv_summary.append({
            "model":            name,
            "oof_auroc":        round(auroc, 4),
            "mean_fold_auroc":  round(mean_fold_auroc, 4),
            "std_fold_auroc":   round(std_fold_auroc, 4),
            "oof_auprc":        round(auprc, 4),
            "oof_brier":        round(brier, 4),
            "oof_logloss":      round(logloss, 4),
            "accuracy":         round(acc, 4),
            "precision":        round(prec, 4),
            "recall":           round(rec, 4),
            "f1_score":         round(f1, 4),
            "latency_ms":       round(latency_ms, 2),
        })

        oof_store[name] = oof_probs

        print(
            f"{name:<20} | AUROC {auroc:.4f} +/-{std_fold_auroc:.4f} | "
            f"AUPRC {auprc:.4f} | Brier {brier:.4f} | "
            f"F1 {f1:.4f} | {latency_ms:.2f} ms/ex"
        )

    # LLM-as-Judge baseline placeholder row (BLOCKED -- see task notes)
    cv_summary.append({
        "model":           "LLM-as-Judge (baseline)",
        "oof_auroc":       "BLOCKED",
        "mean_fold_auroc": "BLOCKED",
        "std_fold_auroc":  "BLOCKED",
        "oof_auprc":       "BLOCKED",
        "oof_brier":       "n/a",
        "oof_logloss":     "n/a",
        "accuracy":        "BLOCKED",
        "precision":       "BLOCKED",
        "recall":          "BLOCKED",
        "f1_score":        "BLOCKED",
        "latency_ms":      "~3933",
    })

    summary_df = pd.DataFrame(cv_summary)
    summary_df.to_csv("results/tables/cv_results.csv", index=False)
    print(f"\nSaved model comparison  ->  results/tables/cv_results.csv")

    # Save OOF predictions for downstream calibration
    oof_df = pd.DataFrame(oof_store)
    oof_df["label"]   = y
    oof_df["task_id"] = groups
    oof_df.to_csv("results/calibration/oof_predictions.csv", index=False)
    print(f"Saved OOF predictions  ->  results/calibration/oof_predictions.csv")

    # Train final model on full dataset (XGBoost primary + LR secondary)
    best_model_name = (
        summary_df[summary_df["model"] != "LLM-as-Judge (baseline)"]
        .sort_values("oof_auroc", ascending=False)
        .iloc[0]["model"]
    )
    print(f"\nBest OOF AUROC: {best_model_name}")

    final_xgb = XGBClassifier(
        n_estimators=50, max_depth=3, learning_rate=0.1,
        random_state=42, eval_metric="logloss",
    )
    final_xgb.fit(X, y)
    explainer = shap.TreeExplainer(final_xgb)

    final_lr = LogisticRegression(random_state=42, max_iter=1000)
    final_lr.fit(X, y)

    bundle = {
        "model":           final_xgb,
        "lr_model":        final_lr,
        "explainer":       explainer,
        "feature_cols":    FEATURE_COLS,
        "best_model_name": best_model_name,
    }
    joblib.dump(bundle, "results/models/prm_final.joblib")

    # gate.py compatibility copy
    joblib.dump(
        {"model": final_xgb, "explainer": explainer, "feature_cols": FEATURE_COLS},
        "handoff_prm.joblib",
    )

    print("Saved  ->  results/models/prm_final.joblib")
    print("Saved  ->  handoff_prm.joblib  (gate.py compatibility)")


if __name__ == "__main__":
    run_model_comparison()
