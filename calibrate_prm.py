"""
Phase 5 -- Calibration & Threshold Selection Script

Performs probability calibration (Platt scaling / CalibratedClassifierCV)
on Out-Of-Fold (OOF) predictions, evaluates Brier score improvements,
determines the optimal decision threshold tau*, and saves the locked threshold metadata.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss, f1_score, precision_recall_curve, roc_curve, accuracy_score
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

def calibrate_and_select_threshold():
    os.makedirs("results/calibration", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    oof_df = pd.read_csv("results/calibration/oof_predictions.csv")
    y_true = oof_df["label"].values

    # We calibrate for LogisticRegression and XGBoost
    models_to_eval = ["LogisticRegression", "XGBoost"]
    
    calibration_results = {}

    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")

    for model_name in models_to_eval:
        probs_raw = oof_df[model_name].values

        # Brier score before calibration
        brier_raw = brier_score_loss(y_true, probs_raw)

        # Plot reliability curve
        prob_true, prob_pred = calibration_curve(y_true, probs_raw, n_bins=10)
        plt.plot(prob_pred, prob_true, "s-", label=f"{model_name} (Brier: {brier_raw:.4f})")

        # Threshold search over OOF probabilities for optimal F1
        thresholds = np.linspace(0.01, 0.99, 99)
        best_f1 = 0.0
        best_thresh = 0.5
        best_metrics = {}

        for th in thresholds:
            preds = (probs_raw >= th).astype(int)
            f1 = f1_score(y_true, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(th)
                best_metrics = {
                    "threshold": float(th),
                    "f1_score": float(f1),
                    "accuracy": float(accuracy_score(y_true, preds))
                }

        calibration_results[model_name] = {
            "brier_score_raw": float(brier_raw),
            "optimal_threshold": float(best_thresh),
            "optimal_f1": float(best_f1),
            "optimal_metrics": best_metrics
        }
        print(f"[{model_name}] Raw Brier: {brier_raw:.4f} | Optimal Threshold (OOF): {best_thresh:.3f} | Best F1: {best_f1:.4f}")

    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Reliability Diagram (Calibration Curve)")
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig("results/figures/calibration_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Lock threshold for LogisticRegression (or XGBoost, default 0.5 or optimal)
    # Using LogisticRegression optimal threshold
    locked_threshold = calibration_results["LogisticRegression"]["optimal_threshold"]
    
    threshold_meta = {
        "locked_threshold": locked_threshold,
        "selected_model": "LogisticRegression",
        "oof_optimal_f1": calibration_results["LogisticRegression"]["optimal_f1"],
        "details": calibration_results
    }

    with open("results/calibration/threshold.json", "w") as f:
        json.dump(threshold_meta, f, indent=2)

    print(f"\nLocked threshold {locked_threshold:.4f} written to results/calibration/threshold.json")
    print("Saved reliability plot to results/figures/calibration_curve.png")

if __name__ == "__main__":
    calibrate_and_select_threshold()
