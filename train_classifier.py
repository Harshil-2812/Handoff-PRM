"""
Trains the Handoff PRM (XGBoost) on features.csv, reports Precision / Recall
/ AUROC, and saves the trained model + a SHAP explainer for later use in the
real-time gating step.

Usage: python train_classifier.py
"""

import pandas as pd
import xgboost as xgb
import shap
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score, classification_report

FEATURES_CSV = "features.csv"
MODEL_OUT = "handoff_prm.joblib"


def main():
    df = pd.read_csv(FEATURES_CSV)
    FEATURE_COLS = [c for c in df.columns if c not in ["task_id", "source", "corruption_type", "label"]]
    print(f"Using {len(FEATURE_COLS)} features: {FEATURE_COLS}")
    X = df[FEATURE_COLS]
    y = df["label"]

    # NOTE: with a small dataset (first run), a plain train/test split is fine.
    # Once you have a few thousand rows, switch this to k-fold cross-validation
    # for a more reliable estimate.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    print("=== Evaluation ===")
    print(f"Precision: {precision_score(y_test, preds):.3f}")
    print(f"Recall:    {recall_score(y_test, preds):.3f}")
    print(f"AUROC:     {roc_auc_score(y_test, probs):.3f}")
    print()
    print(classification_report(y_test, preds))

    # Feature importance via SHAP -- this is what powers the "tell Agent A
    # why it failed" step later.
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    mean_abs_shap = pd.Series(
        abs(shap_values).mean(axis=0), index=FEATURE_COLS
    ).sort_values(ascending=False)
    print("\n=== Mean |SHAP value| per feature (higher = more influential) ===")
    print(mean_abs_shap)

    joblib.dump({"model": model, "explainer": explainer, "feature_cols": FEATURE_COLS}, MODEL_OUT)
    print(f"\nSaved trained model + explainer to {MODEL_OUT}")


if __name__ == "__main__":
    main()
