"""
Phase 10 -- Figure Generation Script

Creates publication-ready figures for the research paper/report and saves them to results/figures/
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score, roc_auc_score
import shap

# Set styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({"font.size": 12, "axes.labelsize": 14, "axes.titlesize": 14, "figure.titlesize": 16})

def generate_all_figures():
    os.makedirs("results/figures", exist_ok=True)
    df = pd.read_csv("features.csv")
    feature_cols = ["cosine_similarity", "entity_overlap", "length_ratio"]
    X = df[feature_cols].values
    y = df["label"].values

    # 1. Model Comparison AUROC & AUPRC
    if os.path.exists("results/tables/cv_results.csv"):
        cv_df = pd.read_csv("results/tables/cv_results.csv")
        fig, ax = plt.subplots(figsize=(8, 5))
        x_indices = np.arange(len(cv_df))
        width = 0.35
        
        ax.bar(x_indices - width/2, cv_df["oof_auroc"], width, label="AUROC", color="#2b5c8f")
        ax.bar(x_indices + width/2, cv_df["oof_auprc"], width, label="AUPRC", color="#d95f02")
        
        ax.set_xticks(x_indices)
        ax.set_xticklabels(cv_df["model"], rotation=15)
        ax.set_ylim(0.85, 1.0)
        ax.set_ylabel("Metric Score")
        ax.set_title("5-Fold GroupKFold Cross-Validation Performance")
        ax.legend(loc="lower right")
        
        for i, row in cv_df.iterrows():
            ax.text(i - width/2, row["oof_auroc"] + 0.005, f"{row['oof_auroc']:.3f}", ha="center", fontsize=10)
            ax.text(i + width/2, row["oof_auprc"] + 0.005, f"{row['oof_auprc']:.3f}", ha="center", fontsize=10)

        plt.savefig("results/figures/model_comparison_auroc.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("Generated results/figures/model_comparison_auroc.png")

    # 2. ROC & PR Curves
    if os.path.exists("results/calibration/oof_predictions.csv"):
        oof_df = pd.read_csv("results/calibration/oof_predictions.csv")
        
        # ROC Curves
        plt.figure(figsize=(7, 6))
        for col in ["LogisticRegression", "XGBoost", "RandomForest", "SVM_RBF"]:
            if col in oof_df.columns:
                fpr, tpr, _ = roc_curve(oof_df["label"], oof_df[col])
                score = roc_auc_score(oof_df["label"], oof_df[col])
                plt.plot(fpr, tpr, label=f"{col} (AUC = {score:.3f})", lw=2)
        
        plt.plot([0, 1], [0, 1], "k--", label="Random Chance")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Out-of-Fold Receiver Operating Characteristic (ROC)")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig("results/figures/roc_curves.png", dpi=300, bbox_inches="tight")
        plt.close()

        # PR Curves
        plt.figure(figsize=(7, 6))
        for col in ["LogisticRegression", "XGBoost", "RandomForest", "SVM_RBF"]:
            if col in oof_df.columns:
                precision, recall, _ = precision_recall_curve(oof_df["label"], oof_df[col])
                score = average_precision_score(oof_df["label"], oof_df[col])
                plt.plot(recall, precision, label=f"{col} (AUPRC = {score:.3f})", lw=2)
        
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Out-of-Fold Precision-Recall (PR) Curves")
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.savefig("results/figures/pr_curves.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("Generated results/figures/roc_curves.png and pr_curves.png")

    # 3. Ablation Study Chart
    if os.path.exists("results/robustness/length_ratio_ablation.csv"):
        abl_df = pd.read_csv("results/robustness/length_ratio_ablation.csv")
        plt.figure(figsize=(8, 5))
        sns.barplot(data=abl_df, x="feature_set", y="oof_auroc", hue="model", palette="Blues_d")
        plt.ylim(0.80, 1.0)
        plt.ylabel("Out-of-Fold AUROC")
        plt.xlabel("Feature Set")
        plt.title("Feature Ablation Study: Impact of Length Ratio")
        plt.savefig("results/figures/ablation_study.png", dpi=300, bbox_inches="tight")
        plt.close()
        print("Generated results/figures/ablation_study.png")

    # 4. SHAP Feature Importance Plot
    if os.path.exists("results/models/prm_final.joblib"):
        bundle = joblib.load("results/models/prm_final.joblib")
        explainer = bundle.get("explainer")
        xgb_model = bundle.get("model")
        if explainer and xgb_model:
            shap_values = explainer.shap_values(X)
            plt.figure(figsize=(8, 4))
            shap.summary_plot(shap_values, X, feature_names=feature_cols, show=False)
            plt.title("SHAP Feature Importance (TreeExplainer)")
            plt.savefig("results/figures/shap_summary.png", dpi=300, bbox_inches="tight")
            plt.close()
            print("Generated results/figures/shap_summary.png")

if __name__ == "__main__":
    generate_all_figures()
