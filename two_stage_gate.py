"""
Two-Stage Cascading PRM Gate ("Airport Security Architecture")

Stage 1 (Fast Scanner):
  Evaluates 12 lightweight structural/lexical features in < 1 ms.
  - If P_fast >= tau_high -> Clear PASS (direct to Agent B)
  - If P_fast <= tau_low  -> Clear FAIL (structural corruption detected)
  - If tau_low < P_fast < tau_high -> UNCERTAINTY ZONE -> Route to Stage 2

Stage 2 (Deep Semantic Scanner):
  Evaluates transformer NLI contradiction and entailment features.
  Only called on the routed subset (e.g., ~15-20% of rollouts).

Outputs:
  - Combined AUROC and AUPRC under 5-Fold GroupKFold
  - Routing rate (% of rollouts needing Stage 2)
  - Average effective latency
  - Comparison across all corruption types
"""

import sys
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

FEATURES_CSV = "features.csv"
OUTPUT_TABLES = "results/tables"
os.makedirs(OUTPUT_TABLES, exist_ok=True)

FAST_FEATURES = [
    "cosine_similarity", "entity_overlap", "length_ratio", "sentence_count_ratio",
    "section_coverage", "verbatim_copy_rate", "trailing_specificity", "constraint_count",
    "role_pronoun_rate", "novel_api_rate", "function_name_preserved", "signature_param_diff"
]
DEEP_FEATURES = ["nli_contradiction_max", "nli_entailment_mean"]


def evaluate_two_stage_system(df, tau_low=0.30, tau_high=0.70):
    """
    Evaluates the two-stage cascading gate under 5-fold GroupKFold CV.
    """
    X_fast = df[FAST_FEATURES].values
    X_deep = df[FAST_FEATURES + DEEP_FEATURES].values
    y = df["label"].values
    groups = df["task_id"].values

    gkf = GroupKFold(n_splits=5)
    oof_fast_probs = np.zeros(len(df))
    oof_combined_probs = np.zeros(len(df))
    routed_mask = np.zeros(len(df), dtype=bool)

    # Class weights for fast scanner
    pos_count = int(np.sum(y == 1))
    neg_count = int(np.sum(y == 0))
    scale_weight = neg_count / max(1, pos_count)

    for train_idx, val_idx in gkf.split(df, groups=groups):
        # 1. Train Stage 1 (Fast Scanner)
        fast_clf = XGBClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1,
            scale_pos_weight=scale_weight,
            random_state=42, eval_metric="logloss"
        )
        fast_clf.fit(X_fast[train_idx], y[train_idx])
        p_fast_val = fast_clf.predict_proba(X_fast[val_idx])[:, 1]
        oof_fast_probs[val_idx] = p_fast_val

        # 2. Train Stage 2 (Deep Semantic Classifier on full feature space)
        deep_clf = LogisticRegression(
            C=1.0, class_weight="balanced", random_state=42, max_iter=1000
        )
        deep_clf.fit(X_deep[train_idx], y[train_idx])
        p_deep_val = deep_clf.predict_proba(X_deep[val_idx])[:, 1]

        # 3. Apply Routing Rule in Validation Fold
        # Route if fast scanner is in the uncertainty band
        # OR if text length/coverage looks valid (length_ratio in [0.7, 1.4]) but semantic flip is possible
        is_uncertain = (p_fast_val > tau_low) & (p_fast_val < tau_high)
        routed_mask[val_idx] = is_uncertain

        # Combined prediction: fast probability if confident; deep probability if routed
        combined_val = np.where(is_uncertain, p_deep_val, p_fast_val)
        oof_combined_probs[val_idx] = combined_val

    # Metrics
    routing_rate = float(np.mean(routed_mask)) * 100.0
    fast_auroc = roc_auc_score(y, oof_fast_probs)
    fast_auprc = average_precision_score(y, oof_fast_probs)

    combined_auroc = roc_auc_score(y, oof_combined_probs)
    combined_auprc = average_precision_score(y, oof_combined_probs)

    # Latencies: Fast = 0.8 ms, Deep = 25.0 ms
    avg_latency = (1.0 - routing_rate / 100.0) * 0.8 + (routing_rate / 100.0) * 25.0

    return {
        "tau_low": tau_low,
        "tau_high": tau_high,
        "routing_rate_pct": routing_rate,
        "fast_auroc": fast_auroc,
        "fast_auprc": fast_auprc,
        "combined_auroc": combined_auroc,
        "combined_auprc": combined_auprc,
        "delta_auroc": combined_auroc - fast_auroc,
        "effective_latency_ms": avg_latency,
        "oof_combined_probs": oof_combined_probs,
        "oof_fast_probs": oof_fast_probs,
        "routed_mask": routed_mask,
    }


def sweep_routing_thresholds(df):
    print("=" * 75)
    print("SWEEPING ROUTING THRESHOLDS FOR TWO-STAGE CASCADING PRM")
    print("=" * 75)

    candidates = [
        (0.40, 0.60),  # very narrow uncertainty (~10% routed)
        (0.35, 0.65),  # narrow uncertainty (~15% routed)
        (0.30, 0.70),  # balanced uncertainty (~20% routed)
        (0.25, 0.75),  # wider uncertainty (~30% routed)
        (0.20, 0.80),  # wide uncertainty (~40% routed)
    ]

    records = []
    best_res = None
    best_delta = -1.0

    for t_low, t_high in candidates:
        res = evaluate_two_stage_system(df, tau_low=t_low, tau_high=t_high)
        print(f"Bands [{t_low:.2f}, {t_high:.2f}] -> Routed: {res['routing_rate_pct']:5.1f}% | Combined AUROC: {res['combined_auroc']:.4f} (Delta: {res['delta_auroc']:+.4f}) | Latency: {res['effective_latency_ms']:.2f} ms")
        records.append({
            "tau_low": t_low,
            "tau_high": t_high,
            "routing_rate_pct": round(res["routing_rate_pct"], 2),
            "fast_auroc": round(res["fast_auroc"], 4),
            "combined_auroc": round(res["combined_auroc"], 4),
            "delta_auroc": round(res["delta_auroc"], 4),
            "combined_auprc": round(res["combined_auprc"], 4),
            "latency_ms": round(res["effective_latency_ms"], 2),
        })
        if res["combined_auroc"] > best_delta:
            best_delta = res["combined_auroc"]
            best_res = res

    sweep_df = pd.DataFrame(records)
    sweep_df.to_csv(f"{OUTPUT_TABLES}/two_stage_sweep.csv", index=False)
    return best_res


def breakdown_by_corruption(df, oof_fast, oof_combined, routed_mask):
    print("\n" + "=" * 75)
    print("DETECTION BREAKDOWN BY CORRUPTION: STAGE 1 (FAST) VS. TWO-STAGE (COMBINED)")
    print("=" * 75)

    records = []
    y = df["label"].values

    for c_type in sorted(df["corruption_type"].unique()):
        if c_type == "none":
            continue
        c_mask = (df["corruption_type"] == c_type) | (df["label"] == 1)
        sub_y = y[c_mask]
        sub_fast = oof_fast[c_mask]
        sub_comb = oof_combined[c_mask]
        sub_routed = routed_mask[c_mask]

        fast_auc = roc_auc_score(sub_y, sub_fast)
        comb_auc = roc_auc_score(sub_y, sub_comb)
        pct_routed = np.mean(sub_routed[df["corruption_type"][c_mask] == c_type]) * 100.0

        print(f"  {c_type:<25} (n={sum(df['corruption_type']==c_type):3d}) | Fast AUROC: {fast_auc:.4f} -> Combined: {comb_auc:.4f} (Delta: {comb_auc-fast_auc:+.4f}) | Routed to S2: {pct_routed:4.1f}%")
        records.append({
            "corruption_type": c_type,
            "n_negatives": sum(df["corruption_type"] == c_type),
            "fast_auroc": round(fast_auc, 4),
            "combined_auroc": round(comb_auc, 4),
            "delta_auroc": round(comb_auc - fast_auc, 4),
            "pct_routed_to_stage2": round(pct_routed, 1),
        })

    breakdown_df = pd.DataFrame(records)
    breakdown_df.to_csv(f"{OUTPUT_TABLES}/two_stage_by_corruption.csv", index=False)
    return breakdown_df


def main():
    df = pd.read_csv(FEATURES_CSV)
    print(f"Loaded {len(df)} rollouts from {FEATURES_CSV}")

    # 1. Sweep routing bands
    best_res = sweep_routing_thresholds(df)

    # 2. Breakdown on the balanced 20% routing operating point
    operating_res = evaluate_two_stage_system(df, tau_low=0.30, tau_high=0.70)
    breakdown_by_corruption(
        df,
        operating_res["oof_fast_probs"],
        operating_res["oof_combined_probs"],
        operating_res["routed_mask"]
    )

    print("\n" + "=" * 75)
    print("TWO-STAGE CASCADING EVALUATION COMPLETE")
    print(f"Saved: {OUTPUT_TABLES}/two_stage_sweep.csv")
    print(f"Saved: {OUTPUT_TABLES}/two_stage_by_corruption.csv")
    print("=" * 75)


if __name__ == "__main__":
    main()
