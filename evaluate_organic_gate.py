"""
Offline PRM Gate Evaluation on Organic Communication Failures (Workstream A & Gate 1)
======================================================================================
Author: Avaneesh N (Software and Experiments)
Repository: https://github.com/Harshil-2812/Handoff-PRM
Protocol: Week 0 Protocol Freeze (Seed: 42, Operating Threshold: tau* = 0.330)

Evaluates the deployed Handoff-PRM classifier on genuine organic multi-agent handoffs:
1. Featurizes handoffs with the 12 fast surface features.
2. Scores handoffs with the trained PRM classifier (results/models/prm_final.joblib).
3. Evaluates detection performance across all 3 labeling policies:
   - Majority-vote policy (v0_majority)
   - Discard policy (v1_discard)
   - Robust-positive policy (v2_robust_positive)
4. Assesses Gate 1 criterion (AUROC >= 0.75 target).
5. Computes metrics per Agent A tier (Weaker T=0.95, Mid T=0.70, Stronger T=0.20).
6. Outputs publication-ready LaTeX tables and Markdown audit report.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
)

from tasks.all_tasks import ALL_TASKS
from feature_extraction import (
    compute_cosine_similarity, compute_entity_overlap, compute_length_ratio,
    compute_sentence_count_ratio, compute_section_coverage, compute_verbatim_copy_rate,
    compute_trailing_specificity, compute_constraint_count, compute_role_pronoun_rate,
    compute_novel_api_rate, compute_function_name_preserved, compute_signature_param_diff
)

ORGANIC_RAW_CSV = "results/tables/organic_dataset_raw.csv"
ORGANIC_FEATS_CSV = "results/tables/organic_features_12.csv"
MODEL_PATH = "results/models/prm_final.joblib"
OUTPUT_REPORT = "results/audit/organic_gate_evaluation_report.md"
OUTPUT_LATEX = "results/latex/table_organic_gate_eval.tex"

os.makedirs("results/tables", exist_ok=True)
os.makedirs("results/audit", exist_ok=True)
os.makedirs("results/latex", exist_ok=True)

FAST_FEATURES = [
    "cosine_similarity", "entity_overlap", "length_ratio", "sentence_count_ratio",
    "section_coverage", "verbatim_copy_rate", "trailing_specificity", "constraint_count",
    "role_pronoun_rate", "novel_api_rate", "function_name_preserved", "signature_param_diff"
]

TAU_STAR = 0.330  # Headline operating threshold established in paper


def extract_organic_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Extracts 12 fast features for organic handoffs, caching results to CSV."""
    if os.path.exists(ORGANIC_FEATS_CSV):
        df_cached = pd.read_csv(ORGANIC_FEATS_CSV)
        if len(df_cached) == len(df_raw):
            print(f"[Cache Hit] Loaded {len(df_cached)} featurized rows from {ORGANIC_FEATS_CSV}")
            return df_cached

    print(f"Extracting 12 fast features for {len(df_raw)} organic samples...")
    task_map = {t["id"]: t for t in ALL_TASKS}
    rows = []

    for idx, r in df_raw.iterrows():
        tid = r["task_id"]
        task = task_map.get(tid)
        if not task:
            continue
        problem = task["problem"]
        handoff = str(r["handoff_text"])
        ep = str(r["entry_point"])

        f_vec = {
            "task_id": tid,
            "tier_name": r["tier_name"],
            "agent_a_model": r["agent_a_model"],
            "agent_a_temp": r["agent_a_temp"],
            "label_majority": int(r["label_majority"]),
            "label_discard_policy": str(r["label_discard_policy"]),
            "label_robust_positive": int(r["label_robust_positive"]),
            "pass_count_of_3": int(r["pass_count_of_3"]),
            "cosine_similarity": compute_cosine_similarity(handoff, problem),
            "entity_overlap": compute_entity_overlap(handoff, problem),
            "length_ratio": compute_length_ratio(handoff, problem),
            "sentence_count_ratio": compute_sentence_count_ratio(handoff, problem),
            "section_coverage": compute_section_coverage(handoff, problem, ep),
            "verbatim_copy_rate": compute_verbatim_copy_rate(handoff, problem),
            "trailing_specificity": compute_trailing_specificity(handoff, problem),
            "constraint_count": compute_constraint_count(handoff, problem),
            "role_pronoun_rate": compute_role_pronoun_rate(handoff, problem),
            "novel_api_rate": compute_novel_api_rate(handoff, problem),
            "function_name_preserved": compute_function_name_preserved(handoff, problem, ep),
            "signature_param_diff": compute_signature_param_diff(handoff, problem),
        }
        rows.append(f_vec)
        if (idx + 1) % 25 == 0 or (idx + 1) == len(df_raw):
            print(f"  Processed [{idx+1}/{len(df_raw)}] handoffs...")

    df_feats = pd.DataFrame(rows)
    df_feats.to_csv(ORGANIC_FEATS_CSV, index=False)
    print(f"[Saved] Extracted features to {ORGANIC_FEATS_CSV}")
    return df_feats


def evaluate_gate():
    if not os.path.exists(ORGANIC_RAW_CSV):
        print(f"[ERROR] Raw organic dataset not found at {ORGANIC_RAW_CSV}")
        return

    df_raw = pd.read_csv(ORGANIC_RAW_CSV)
    df_feats = extract_organic_features(df_raw)

    print("\n" + "=" * 75)
    print("EVALUATING HANDOFF-PRM ON ORGANIC DATASET (Gate 1 Assessment)")
    print("=" * 75)
    print(f"Dataset Size:         {len(df_feats)} handoffs ({df_feats['task_id'].nunique()} tasks)")
    print(f"Features Evaluated:   {len(FAST_FEATURES)} (Stage 1 Fast Surface Features)")
    print(f"Calibrated Threshold: tau* = {TAU_STAR}")
    print("=" * 75)

    # Load PRM Model Bundle
    bundle = joblib.load(MODEL_PATH)
    clf = bundle["model"]
    feature_cols = bundle["feature_cols"]

    # Handle feature alignment (12 fast features vs 14 full features)
    X = df_feats[FAST_FEATURES].copy()
    for col in feature_cols:
        if col not in X.columns:
            # Set NLI features to neutral baseline mean (0.0 for contradiction, 0.5 for entailment)
            X[col] = 0.5 if "entailment" in col else 0.0

    X_mat = X[feature_cols].values
    probs = clf.predict_proba(X_mat)[:, 1]
    df_feats["prm_score"] = probs
    df_feats["pred_gated"] = (probs >= TAU_STAR).astype(int)

    # ── 1. Policy Evaluations ────────────────────────────────────────────────
    policies = {
        "Majority-Vote Policy (v0)": {
            "mask": np.ones(len(df_feats), dtype=bool),
            "y": df_feats["label_majority"].values
        },
        "Discard Policy (v1)": {
            "mask": df_feats["label_discard_policy"].astype(str) != "discard",
            "y": df_feats.loc[df_feats["label_discard_policy"].astype(str) != "discard", "label_discard_policy"].astype(int).values
        },
        "Robust-Positive Policy (v2)": {
            "mask": np.ones(len(df_feats), dtype=bool),
            "y": df_feats["label_robust_positive"].values
        }
    }

    policy_results = []
    for name, p_data in policies.items():
        sub_probs = probs[p_data["mask"]]
        sub_y = p_data["y"]
        sub_preds = (sub_probs >= TAU_STAR).astype(int)

        auroc = roc_auc_score(sub_y, sub_probs)
        auprc = average_precision_score(sub_y, sub_probs)
        brier = brier_score_loss(sub_y, sub_probs)
        acc = accuracy_score(sub_y, sub_preds)
        prec = precision_score(sub_y, sub_preds, zero_division=0)
        rec = recall_score(sub_y, sub_preds, zero_division=0)
        f1 = f1_score(sub_y, sub_preds, zero_division=0)

        tn, fp, fn, tp = confusion_matrix(sub_y, sub_preds, labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True Negative Rate (Interception of failures)
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

        policy_results.append({
            "Policy": name,
            "N": len(sub_y),
            "Pos": int(np.sum(sub_y == 1)),
            "Neg": int(np.sum(sub_y == 0)),
            "AUROC": auroc,
            "AUPRC": auprc,
            "Brier": brier,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "Specificity_TNR": spec,
            "NPV": npv,
            "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)
        })

    df_policy_res = pd.DataFrame(policy_results)
    print("\nPolicy Performance Summary (tau* = 0.330):")
    print(df_policy_res[["Policy", "N", "Neg", "AUROC", "AUPRC", "Accuracy", "F1", "Specificity_TNR"]].to_string(index=False))

    # ── 2. Tier-Wise Evaluations ─────────────────────────────────────────────
    tier_results = []
    for tier in df_feats["tier_name"].unique():
        sub = df_feats[df_feats["tier_name"] == tier]
        sub_y = sub["label_majority"].values
        sub_probs = sub["prm_score"].values
        sub_preds = sub["pred_gated"].values

        n_neg = int(np.sum(sub_y == 0))
        n_pos = int(np.sum(sub_y == 1))
        auroc = roc_auc_score(sub_y, sub_probs) if len(np.unique(sub_y)) > 1 else np.nan
        f1 = f1_score(sub_y, sub_preds, zero_division=0)
        acc = accuracy_score(sub_y, sub_preds)
        tn, fp, fn, tp = confusion_matrix(sub_y, sub_preds, labels=[0, 1]).ravel()
        interception_rate = tn / n_neg if n_neg > 0 else 0.0

        tier_results.append({
            "Tier": tier,
            "Model": sub["agent_a_model"].iloc[0],
            "Temp": sub["agent_a_temp"].iloc[0],
            "N": len(sub),
            "Pos": n_pos,
            "Neg": n_neg,
            "AUROC": auroc,
            "Accuracy": acc,
            "F1": f1,
            "TN": int(tn),
            "FP": int(fp),
            "Interception_Rate": interception_rate
        })

    df_tier_res = pd.DataFrame(tier_results)
    print("\nTier-Wise Breakdown:")
    print(df_tier_res.to_string(index=False))

    # ── 3. Gate 1 Condition Check ─────────────────────────────────────────────
    primary_auroc = df_policy_res.loc[df_policy_res["Policy"] == "Majority-Vote Policy (v0)", "AUROC"].values[0]
    gate1_pass = primary_auroc >= 0.75
    print("\n" + "=" * 75)
    print("GATE 1 ASSESSMENT OUTCOME")
    print("=" * 75)
    print(f"Primary Organic AUROC (Majority Vote):  {primary_auroc:.4f}")
    print(f"Pre-Registered Gate 1 Target:            AUROC >= 0.7500")
    if gate1_pass:
        print("RESULT: GATE 1 PASSED (Keeps primary 'Detection Gate' framing).")
    else:
        print("RESULT: GATE 1 BELOW 0.75 (Scientific framing adjusts to Measurement & Boundary Bounds).")
    print("=" * 75)

    # ── 4. Generate LaTeX Table ───────────────────────────────────────────────
    with open(OUTPUT_LATEX, "w", encoding="utf-8") as f:
        f.write("% Table: PRM Gate Performance on Organic Communication Failures\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\caption{Offline PRM Gate Performance on Organic Multi-Agent Handoffs ($\\tau^* = 0.330$)}\n")
        f.write("\\label{tab:organic_gate_performance}\n")
        f.write("\\begin{tabular}{lcccccccc}\n")
        f.write("\\toprule\n")
        f.write("Policy & $N$ & Pos & Neg & AUROC & AUPRC & Accuracy & F1 & Specificity \\\\\n")
        f.write("\\midrule\n")
        for _, r in df_policy_res.iterrows():
            f.write(f"{r['Policy']} & {r['N']} & {r['Pos']} & {r['Neg']} & {r['AUROC']:.3f} & {r['AUPRC']:.3f} & {r['Accuracy']:.3f} & {r['F1']:.3f} & {r['Specificity_TNR']:.3f} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    # ── 5. Generate Markdown Audit Report ─────────────────────────────────────
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("# Gate 1 Evaluation Report: PRM Performance on Organic Multi-Agent Failures\n\n")
        f.write(f"**Workstream A**: Avaneesh N (Data & Organic Communication)\n")
        f.write(f"**Evaluated Model**: `{MODEL_PATH}` (XGBoost 12-Feature Surface Scanner)\n")
        f.write(f"**Total Samples**: {len(df_feats)} organic handoffs across {df_feats['task_id'].nunique()} tasks\n")
        f.write(f"**Headline Threshold**: $\\tau^* = {TAU_STAR}$\n\n")

        f.write("## 1. Executive Summary & Gate 1 Decision\n\n")
        f.write(f"- **Primary Organic AUROC (Majority-Vote)**: **`{primary_auroc:.4f}`**\n")
        f.write(f"- **Gate 1 Pre-Registered Threshold**: `0.7500`\n")
        f.write(f"- **Gate 1 Status**: **{'PASSED ✓' if gate1_pass else 'NEGATIVE/MEASUREMENT FRAMING RECOMMENDED'}**\n")
        f.write(f"- **Natural Communication Failure Rate**: `{df_policy_res.iloc[0]['Neg'] / df_policy_res.iloc[0]['N'] * 100:.1f}%` ({df_policy_res.iloc[0]['Neg']} failures / {df_policy_res.iloc[0]['N']} samples)\n\n")

        f.write("## 2. Policy-Specific Performance Breakdown\n\n")
        f.write("| Labeling Policy | N | Pos | Neg | AUROC | AUPRC | Accuracy | F1 Score | Specificity (Interception) |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for _, r in df_policy_res.iterrows():
            f.write(f"| **{r['Policy']}** | {r['N']} | {r['Pos']} | {r['Neg']} | `{r['AUROC']:.4f}` | `{r['AUPRC']:.4f}` | `{r['Accuracy']:.3f}` | `{r['F1']:.3f}` | `{r['Specificity_TNR']*100:.1f}%` |\n")

        f.write("\n## 3. Confusion Matrix at Headline Threshold ($\\tau^* = 0.330$)\n\n")
        for _, r in df_policy_res.iterrows():
            f.write(f"### {r['Policy']}\n")
            f.write(f"- **True Positives (TP)**: {r['TP']} (Correctly allowed sufficient handoffs)\n")
            f.write(f"- **True Negatives (TN)**: {r['TN']} (Correctly intercepted organic failures)\n")
            f.write(f"- **False Positives (FP)**: {r['FP']} (Unnecessarily blocked sufficient handoffs)\n")
            f.write(f"- **False Negatives (FN)**: {r['FN']} (Missed failures allowed to Agent B)\n")
            f.write(f"- **Precision**: `{r['Precision']:.3f}` | **Recall**: `{r['Recall']:.3f}` | **NPV**: `{r['NPV']:.3f}`\n\n")

        f.write("## 4. Breakdown Across Agent A Model Tiers\n\n")
        f.write("| Tier Name | Planner Model | Temp | Samples | Pos | Neg | Fail Rate | AUROC | Interception Rate |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for _, r in df_tier_res.iterrows():
            f.write(f"| `{r['Tier']}` | `{r['Model']}` | {r['Temp']} | {r['N']} | {r['Pos']} | {r['Neg']} | `{r['Neg']/r['N']*100:.1f}%` | `{r['AUROC']:.4f}` | `{r['Interception_Rate']*100:.1f}%` |\n")

    print(f"\n[Saved] LaTeX Table:  {OUTPUT_LATEX}")
    print(f"[Saved] Audit Report: {OUTPUT_REPORT}")
    print("=" * 75)


if __name__ == "__main__":
    evaluate_gate()
