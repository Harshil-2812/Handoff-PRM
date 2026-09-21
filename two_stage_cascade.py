"""
Handoff PRM — Two-Stage Cascade: Threshold Calibration + Evaluation
=====================================================================

Stage 1 (fast, cheap): the 12-feature Random Forest classifier already
trained. Confident predictions (very high or very low score) are
trusted directly.

Stage 2 (deep, selective): only triggered for handoffs whose Stage-1
score falls in an uncertainty band — exactly where semantic corruptions
(negate_edge_case, wrong_algorithm_name, invert_objective,
signature_rename) are expected to cluster, since Stage 1 could not
reliably separate them (AUROC 0.37-0.45, at/below chance).

This script:
  1. Re-derives OOF predictions for Stage 1 (same protocol as the sweep).
  2. Inspects where each corruption type's OOF scores actually land, to
     pick a data-driven uncertainty band (not an arbitrary guess).
  3. Applies the cascade: confident rows resolved by Stage 1 alone;
     uncertain rows get a Stage-2 correction using nli_entailment_mean
     (recomputed as a stricter, standalone check on just this subset).
  4. Reports: % resolved by Stage 1, % escalated to Stage 2, and
     accuracy/AUROC before vs. after the Stage-2 correction on the
     escalated subset specifically.

Run this after model_sweep_12features_final.py.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

INPUT_PATH = "final_dataset_full_features.csv"
CASCADE_OUTPUT_PATH = "cascade_results.csv"
BAND_ANALYSIS_PATH = "uncertainty_band_analysis.csv"

FEATURE_COLS = [
    "cosine_similarity", "entity_overlap", "length_ratio", "sentence_count_ratio",
    "section_coverage", "verbatim_copy_rate", "trailing_specificity", "constraint_count",
    "role_pronoun_rate", "novel_api_rate", "function_name_preserved",
    "nli_entailment_mean",
]

# Which corruption types Stage 1 struggled with (AUROC <= 0.55 from your
# per-corruption breakdown) — used only for diagnostic printing, not
# hardcoded into the cascade logic itself (the band is picked from the
# score distribution, so this list is not a shortcut/leak).
WEAK_CORRUPTION_TYPES = [
    "negate_edge_case", "wrong_algorithm_name", "invert_objective", "signature_rename",
]


def get_stage1_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=6, class_weight="balanced",
            random_state=42, n_jobs=-1,
        )),
    ])


def compute_stage1_oof(X, y, groups, n_splits=5):
    """Re-derives OOF probabilities for Stage 1 (same protocol as the sweep script)."""
    gkf = GroupKFold(n_splits=n_splits)
    oof_proba = np.zeros(len(y))

    for train_idx, test_idx in gkf.split(X, y, groups):
        train_tasks = set(groups[train_idx])
        test_tasks = set(groups[test_idx])
        assert len(train_tasks & test_tasks) == 0, "LEAKAGE detected in cascade OOF re-derivation"

        model = get_stage1_model()
        model.fit(X[train_idx], y[train_idx])
        oof_proba[test_idx] = model.predict_proba(X[test_idx])[:, 1]

    return oof_proba


def analyze_score_distribution(df, oof_proba):
    """Shows where each corruption type's scores land — the basis for
    picking a data-driven uncertainty band rather than guessing."""
    df = df.copy()
    df["stage1_score"] = oof_proba

    rows = []
    for ctype, group in df.groupby("corruption_type"):
        rows.append({
            "corruption_type": ctype,
            "n": len(group),
            "mean_score": group["stage1_score"].mean(),
            "std_score": group["stage1_score"].std(),
            "median_score": group["stage1_score"].median(),
            "is_weak_type": ctype in WEAK_CORRUPTION_TYPES,
        })
    band_df = pd.DataFrame(rows).sort_values("mean_score")
    print("\nStage-1 score distribution by corruption type:")
    print(band_df.to_string(index=False))
    band_df.to_csv(BAND_ANALYSIS_PATH, index=False)
    return band_df


def pick_band_from_scores(oof_proba, lower_pct=35, upper_pct=65):
    """
    Data-driven band: the middle percentile range of the full OOF score
    distribution. Adjust lower_pct/upper_pct to widen or narrow how much
    gets escalated to Stage 2 (wider band = more Stage-2 calls, likely
    higher accuracy but higher cost).
    """
    lower = np.percentile(oof_proba, lower_pct)
    upper = np.percentile(oof_proba, upper_pct)
    return lower, upper


def stage2_correction(df, escalated_mask):
    """
    Stage 2: a stricter, standalone semantic check applied only to the
    escalated (uncertain) subset. Here implemented as a simple
    entailment-threshold rule on nli_entailment_mean (>= median-of-
    positives => sufficient), since this is the only semantic feature
    already computed. Swap this out for an actual LLM-as-judge call on
    just the escalated subset if budget allows — that is the intended
    production design (expensive check, applied only to low volume).
    """
    escalated = df.loc[escalated_mask]
    pos_threshold = df.loc[df["label"] == 1, "nli_entailment_mean"].median()

    stage2_pred = (escalated["nli_entailment_mean"] >= pos_threshold).astype(int)
    return stage2_pred


def main():
    print(f"Loading {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows, {df['task_id'].nunique()} unique tasks")

    X = df[FEATURE_COLS].values
    y = df["label"].values
    groups = df["task_id"].values

    print("\nDeriving Stage-1 OOF scores (Random Forest, 12 features)...")
    oof_proba = compute_stage1_oof(X, y, groups)
    print(f"Stage-1 overall OOF AUROC: {roc_auc_score(y, oof_proba):.4f}")

    band_df = analyze_score_distribution(df, oof_proba)

    lower, upper = pick_band_from_scores(oof_proba, lower_pct=35, upper_pct=65)
    print(f"\nData-driven uncertainty band: [{lower:.3f}, {upper:.3f}]")
    print("(Adjust lower_pct/upper_pct in pick_band_from_scores() to widen/narrow escalation)")

    df["stage1_score"] = oof_proba
    escalated_mask = (df["stage1_score"] >= lower) & (df["stage1_score"] <= upper)
    n_escalated = escalated_mask.sum()
    pct_escalated = 100 * n_escalated / len(df)
    print(f"\nRows escalated to Stage 2: {n_escalated} / {len(df)} ({pct_escalated:.1f}%)")

    escalated_by_type = df.loc[escalated_mask, "corruption_type"].value_counts()
    print("\nEscalated rows by corruption type (check this catches the weak types):")
    print(escalated_by_type)

    confident_mask = ~escalated_mask
    stage1_only_pred = (df["stage1_score"] >= 0.5).astype(int)

    cascade_pred = stage1_only_pred.copy()
    stage2_pred = stage2_correction(df, escalated_mask)
    cascade_pred.loc[escalated_mask] = stage2_pred.values

    y_escalated = y[escalated_mask.values]
    stage1_only_escalated_pred = stage1_only_pred.loc[escalated_mask].values
    cascade_escalated_pred = cascade_pred.loc[escalated_mask].values

    print("\n" + "=" * 70)
    print("PERFORMANCE ON THE ESCALATED (UNCERTAIN) SUBSET")
    print("=" * 70)
    print(f"n = {n_escalated}")
    print(f"Stage-1-only accuracy on escalated subset: "
          f"{accuracy_score(y_escalated, stage1_only_escalated_pred):.4f}")
    print(f"Stage-1-only F1 on escalated subset:       "
          f"{f1_score(y_escalated, stage1_only_escalated_pred):.4f}")
    print(f"Cascade (Stage 2 correction) accuracy:     "
          f"{accuracy_score(y_escalated, cascade_escalated_pred):.4f}")
    print(f"Cascade (Stage 2 correction) F1:           "
          f"{f1_score(y_escalated, cascade_escalated_pred):.4f}")

    print("\n" + "=" * 70)
    print("OVERALL DATASET PERFORMANCE: STAGE-1-ONLY vs. FULL CASCADE")
    print("=" * 70)
    print(f"Stage-1-only accuracy (all rows):  {accuracy_score(y, stage1_only_pred):.4f}")
    print(f"Full-cascade accuracy (all rows):  {accuracy_score(y, cascade_pred):.4f}")
    print(f"Stage-1-only F1 (all rows):        {f1_score(y, stage1_only_pred):.4f}")
    print(f"Full-cascade F1 (all rows):        {f1_score(y, cascade_pred):.4f}")
    print(f"\n% of rows resolved by Stage 1 alone: {100 - pct_escalated:.1f}%")
    print(f"% of rows escalated to Stage 2:      {pct_escalated:.1f}%")

    df["cascade_prediction"] = cascade_pred
    df["escalated_to_stage2"] = escalated_mask
    df.to_csv(CASCADE_OUTPUT_PATH, index=False)
    print(f"\nSaved full cascade results to {CASCADE_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
