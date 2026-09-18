"""
Phase 4 -- Full GroupKFold Model Comparison & Training Script

Evaluates multiple classifiers (Logistic Regression, XGBoost, Random Forest, SVM)
using 5-fold GroupKFold CV on the 3-feature representation.
Saves comprehensive out-of-fold metrics and the final trained model artifact.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, brier_score_loss, log_loss
)
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
import shap

def run_model_comparison():
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/calibration", exist_ok=True)

    df = pd.read_csv("features.csv")
    feature_cols = ["cosine_similarity", "entity_overlap", "length_ratio"]
    X = df[feature_cols].values
    y = df["label"].values
    groups = df["task_id"].values

    models = {
        "LogisticRegression": LogisticRegression(random_state=42),
        "XGBoost": XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42, eval_metric="logloss"),
        "RandomForest": RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42),
        "SVM_RBF": SVC(probability=True, random_state=42)
    }

    gkf = GroupKFold(n_splits=5)
    cv_summary = []
    oof_predictions = {}

    for name, model in models.items():
        oof_probs = np.zeros(len(df))
        fold_aurocs = []
        fold_auprcs = []
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_va = X[val_idx], y[val_idx]

            # Fit clone of model
            if name == "XGBoost":
                clf = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42, eval_metric="logloss")
            elif name == "LogisticRegression":
                clf = LogisticRegression(random_state=42)
            elif name == "RandomForest":
                clf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
            elif name == "SVM_RBF":
                clf = SVC(probability=True, random_state=42)
            
            clf.fit(X_tr, y_tr)
            probs = clf.predict_proba(X_va)[:, 1]
            oof_probs[val_idx] = probs

            fold_aurocs.append(roc_auc_score(y_va, probs))
            fold_auprcs.append(average_precision_score(y_va, probs))

        # Overall OOF metrics
        auroc = roc_auc_score(y, oof_probs)
        auprc = average_precision_score(y, oof_probs)
        brier = brier_score_loss(y, oof_probs)
        logloss = log_loss(y, oof_probs)
        
        # Binary metrics at 0.5 threshold
        preds_50 = (oof_probs >= 0.5).astype(int)
        acc = accuracy_score(y, preds_50)
        prec = precision_score(y, preds_50)
        rec = recall_score(y, preds_50)
        f1 = f1_score(y, preds_50)

        mean_fold_auroc = float(np.mean(fold_aurocs))
        std_fold_auroc = float(np.std(fold_aurocs))

        cv_summary.append({
            "model": name,
            "oof_auroc": auroc,
            "mean_fold_auroc": mean_fold_auroc,
            "std_fold_auroc": std_fold_auroc,
            "oof_auprc": auprc,
            "oof_brier": brier,
            "oof_logloss": logloss,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1
        })

        oof_predictions[name] = oof_probs
        print(f"Model: {name:<20} | OOF AUROC: {auroc:.4f} (+/- {std_fold_auroc:.4f}) | AUPRC: {auprc:.4f} | Brier: {brier:.4f} | F1: {f1:.4f}")

    summary_df = pd.DataFrame(cv_summary)
    summary_df.to_csv("results/tables/cv_results.csv", index=False)
    
    # Save OOF predictions for calibration step
    oof_df = pd.DataFrame(oof_predictions)
    oof_df["label"] = y
    oof_df["task_id"] = groups
    oof_df.to_csv("results/calibration/oof_predictions.csv", index=False)

    # Select best model (LogisticRegression or XGBoost)
    # Fit final model on full dataset
    best_model_name = summary_df.sort_values("oof_auroc", ascending=False).iloc[0]["model"]
    print(f"\nBest performing model based on OOF AUROC: {best_model_name}")

    # Train both XGBoost and LogisticRegression on full dataset and save
    final_xgb = XGBClassifier(n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42, eval_metric="logloss")
    final_xgb.fit(X, y)
    
    final_lr = LogisticRegression(random_state=42)
    final_lr.fit(X, y)

    # We save XGBoost as primary prm_final.joblib since SHAP explainer is natively integrated with tree models
    explainer = shap.TreeExplainer(final_xgb)
    
    joblib.dump({
        "model": final_xgb,
        "lr_model": final_lr,
        "explainer": explainer,
        "feature_cols": feature_cols,
        "best_model_name": best_model_name
    }, "results/models/prm_final.joblib")
    
    # Also save to root handoff_prm.joblib for compatibility
    joblib.dump({
        "model": final_xgb,
        "explainer": explainer,
        "feature_cols": feature_cols
    }, "handoff_prm.joblib")

    print("Saved final model to results/models/prm_final.joblib and handoff_prm.joblib")

if __name__ == "__main__":
    run_model_comparison()
