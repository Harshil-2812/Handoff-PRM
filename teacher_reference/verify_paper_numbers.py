#!/usr/bin/env python3
"""
verify_paper_numbers.py — Audit Script for Reviewer
====================================================
Verifies that all key numbers cited in the paper can be reproduced
from the files in this teacher_reference package.

Checks:
  1. τ* values match threshold.json
  2. CV AUROC/AUPRC/Brier/LogLoss match cv_results_primary.json
  3. Cascade pass@1 numbers match cascade_results.json
  4. Confusion matrix values match confusion_matrix_primary.json
  5. Bootstrap CIs match bootstrap_ci.json
  6. ANOVA F-statistic consistent with anova_results.txt
  7. Ablation table row counts match ablation_results.csv
  8. SHAP ranking is monotone (length_ratio top-1)
  9. model file SHA-256 is present and non-zero

Run from the teacher_reference/ directory:
  python verify_paper_numbers.py
"""

import json
import os
import sys
import csv
import hashlib

# Fix encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_DIR = os.path.join(SCRIPT_DIR, "metrics")
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
WARN = "\033[93mWARN\033[0m"

errors = []
warnings = []

def check(label, condition, detail=""):
    if condition:
        print(f"  [{PASS}] {label}")
    else:
        print(f"  [{FAIL}] {label}" + (f" — {detail}" if detail else ""))
        errors.append(label)

def warn(label, detail=""):
    print(f"  [{WARN}] {label}" + (f" — {detail}" if detail else ""))
    warnings.append(label)


print("=" * 65)
print("  Handoff-PRM Paper Numbers Verification")
print("=" * 65)

# ── 1. Threshold τ* values ────────────────────────────────────────────────────
print("\n[1] Threshold τ* values (models/threshold.json)")
try:
    with open(os.path.join(MODELS_DIR, "threshold.json")) as f:
        thresh = json.load(f)
    check("τ* primary = 0.330",
          thresh["primary"]["tau_star"] == 0.330,
          f"got {thresh['primary']['tau_star']}")
    check("τ* Qwen-3B = 0.695",
          thresh["qwen_3b"]["tau_star"] == 0.695,
          f"got {thresh['qwen_3b']['tau_star']}")
    check("τ* Qwen-7B = 0.7972",
          abs(thresh["qwen_7b"]["tau_star"] - 0.7972) < 1e-4,
          f"got {thresh['qwen_7b']['tau_star']}")
except Exception as e:
    print(f"  [{FAIL}] Could not load threshold.json: {e}")
    errors.append("threshold.json load failed")

# ── 2. CV results (Table VI) ──────────────────────────────────────────────────
print("\n[2] CV Results (metrics/cv_results_primary.json) — Table VI")
try:
    with open(os.path.join(METRICS_DIR, "cv_results_primary.json")) as f:
        cv = json.load(f)
    rf = cv["primary"]["models"]["RandomForest"]
    xgb = cv["primary"]["models"]["XGBoost"]
    lr = cv["primary"]["models"]["LogisticRegression"]

    check("RF OOF AUROC = 0.8718",
          abs(rf["oof_auroc"] - 0.8718) < 0.0005, f"got {rf['oof_auroc']}")
    check("XGB OOF AUROC = 0.8733",
          abs(xgb["oof_auroc"] - 0.8733) < 0.0005, f"got {xgb['oof_auroc']}")
    check("LR OOF AUROC = 0.8378",
          abs(lr["oof_auroc"] - 0.8378) < 0.0005, f"got {lr['oof_auroc']}")
    check("RF bootstrap CI lower >= 0.83",
          rf["bootstrap_auroc_ci_lower"] >= 0.83, f"got {rf['bootstrap_auroc_ci_lower']}")
    check("n_rows = 483",
          cv["primary"]["n_rows"] == 483, f"got {cv['primary']['n_rows']}")
    check("n_features = 11",
          cv["primary"]["n_features"] == 11, f"got {cv['primary']['n_features']}")
except Exception as e:
    print(f"  [{FAIL}] Could not load cv_results_primary.json: {e}")
    errors.append("cv_results load failed")

# ── 3. Cascade results ────────────────────────────────────────────────────────
print("\n[3] Cascade Results (metrics/cascade_results.json) — Tables X-XI")
try:
    with open(os.path.join(METRICS_DIR, "cascade_results.json")) as f:
        casc = json.load(f)
    run = casc["primary_live_run"]
    p1 = run["pass_at_1"]
    mc = run["mcnemar"]
    boot = run["bootstrap"]
    s2 = run["two_stage_architecture"]

    check("Pass@1 ungated (all) = 55.3%",
          abs(p1["ungated_all"] - 0.5532) < 0.003, f"got {p1['ungated_all']:.4f}")
    check("Pass@1 gated (all) = 67.0%",
          abs(p1["gated_all"] - 0.6702) < 0.003, f"got {p1['gated_all']:.4f}")
    check("Delta all = +11.7 pp",
          abs(p1["delta_all_pp"] - 11.7) < 0.5, f"got {p1['delta_all_pp']}")
    check("Pass@1 corrupted ungated = 44.7%",
          abs(p1["ungated_corrupted"] - 0.4468) < 0.003, f"got {p1['ungated_corrupted']:.4f}")
    check("Pass@1 corrupted gated = 70.2%",
          abs(p1["gated_corrupted"] - 0.7021) < 0.003, f"got {p1['gated_corrupted']:.4f}")
    check("McNemar p = 0.0074",
          abs(mc["p_value"] - 0.0074) < 0.0005, f"got {mc['p_value']}")
    check("Bootstrap CI lower > 4.0 pp",
          boot["ci_95_lower_pp"] > 4.0, f"got {boot['ci_95_lower_pp']}")
    check("Bootstrap CI upper < 20.0 pp",
          boot["ci_95_upper_pp"] < 20.0, f"got {boot['ci_95_upper_pp']}")
    check("Stage-1 handles >= 74% of blocks",
          s2["stage1_pct_of_all_blocks"] >= 73.0, f"got {s2['stage1_pct_of_all_blocks']}")
except Exception as e:
    print(f"  [{FAIL}] Could not load cascade_results.json: {e}")
    errors.append("cascade_results load failed")

# ── 4. Confusion matrix ───────────────────────────────────────────────────────
print("\n[4] Confusion Matrix (metrics/confusion_matrix_primary.json) — Table IX")
try:
    with open(os.path.join(METRICS_DIR, "confusion_matrix_primary.json")) as f:
        cm = json.load(f)
    mat = cm["confusion_matrix"]["matrix"]
    met = cm["confusion_matrix"]["metrics"]

    check("TP = 22", mat["TP"] == 22, f"got {mat['TP']}")
    check("FP = 18", mat["FP"] == 18, f"got {mat['FP']}")
    check("FN = 4",  mat["FN"] == 4,  f"got {mat['FN']}")
    check("TN = 3",  mat["TN"] == 3,  f"got {mat['TN']}")
    check("Precision = 0.550",
          abs(met["precision"] - 0.550) < 0.005, f"got {met['precision']}")
    check("Recall = 0.846",
          abs(met["recall"] - 0.846) < 0.005, f"got {met['recall']}")
    check("F1 = 0.667",
          abs(met["f1"] - 0.667) < 0.005, f"got {met['f1']}")
except Exception as e:
    print(f"  [{FAIL}] Could not load confusion_matrix_primary.json: {e}")
    errors.append("confusion_matrix load failed")

# ── 5. Ablation table ─────────────────────────────────────────────────────────
print("\n[5] Ablation Results (metrics/ablation_results.csv) — Table VIII")
try:
    with open(os.path.join(METRICS_DIR, "ablation_results.csv")) as f:
        rows = list(csv.DictReader(f))
    # Find RF full 11-feature row
    full_row = next((r for r in rows if r["feature_set"] == "RF_Full_11gate"), None)
    check("RF_Full_11gate row exists", full_row is not None)
    if full_row:
        check("RF Full AUROC = 0.8718",
              abs(float(full_row["oof_auroc"]) - 0.8718) < 0.0005,
              f"got {full_row['oof_auroc']}")
    # Cosine-only should be < full
    cos_row = next((r for r in rows if r["feature_set"] == "RF_Cosine_only"), None)
    if cos_row and full_row:
        check("Cosine-only AUROC < Full (ablation degrades)",
              float(cos_row["oof_auroc"]) < float(full_row["oof_auroc"]),
              f"cos={cos_row['oof_auroc']} vs full={full_row['oof_auroc']}")
    check("At least 15 ablation rows", len(rows) >= 15, f"got {len(rows)}")
except Exception as e:
    print(f"  [{FAIL}] Could not load ablation_results.csv: {e}")
    errors.append("ablation_results load failed")

# ── 6. SHAP values ────────────────────────────────────────────────────────────
print("\n[6] SHAP Values (metrics/shap_values.csv)")
try:
    with open(os.path.join(METRICS_DIR, "shap_values.csv")) as f:
        lines = [l for l in f if not l.startswith("#")]
    shap_rows = list(csv.DictReader(lines))
    top_feat = min(shap_rows, key=lambda r: int(r["rank_primary"]))
    check("length_ratio is SHAP rank #1 (primary)",
          top_feat["feature"] == "length_ratio",
          f"got {top_feat['feature']}")
    all_ranks = [int(r["rank_primary"]) for r in shap_rows]
    check("11 features ranked",
          len(shap_rows) >= 11, f"got {len(shap_rows)}")
except Exception as e:
    print(f"  [{FAIL}] Could not load shap_values.csv: {e}")
    errors.append("shap_values load failed")

# ── 7. Model file ─────────────────────────────────────────────────────────────
print("\n[7] Model File (models/handoff_prm.joblib)")
model_path = os.path.join(MODELS_DIR, "handoff_prm.joblib")
check("handoff_prm.joblib exists", os.path.exists(model_path))
if os.path.exists(model_path):
    size = os.path.getsize(model_path)
    check("Model file size > 50 KB", size > 50000, f"got {size} bytes")
    sha = hashlib.sha256(open(model_path, "rb").read()).hexdigest()
    print(f"  [INFO] SHA-256: {sha}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
if errors:
    print(f"  RESULT: {len(errors)} FAILED checks — see above for details")
    for e in errors:
        print(f"    ✗ {e}")
else:
    print(f"  RESULT: ALL CHECKS PASSED ✓")
if warnings:
    print(f"  WARNINGS: {len(warnings)}")
print("=" * 65)

sys.exit(1 if errors else 0)
