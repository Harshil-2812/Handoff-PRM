"""
regenerate_rigor_artifacts.py
Rebuilds ALL stale audit/robustness/table files from final_dataset_full_features.csv (483 rows).
Deletes the debunked downstream_pass1_results.csv.
No API calls. Pure sklearn on real data.
"""
import os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from xgboost import XGBClassifier

ROOT    = Path(".")
DATA    = ROOT / "final_dataset_full_features.csv"
AUDIT   = ROOT / "results" / "audit"
ROBUST  = ROOT / "results" / "robustness"
TABLES  = ROOT / "results" / "tables"
for d in [AUDIT, ROBUST, TABLES]:
    d.mkdir(parents=True, exist_ok=True)

# ── Feature sets ──────────────────────────────────────────────────────────────
# 11 gate features (NLI dropped, no signature_param_diff — these are what
# the live gate actually uses at inference)
GATE_FEATURES = [
    "cosine_similarity", "entity_overlap", "length_ratio",
    "sentence_count_ratio", "section_coverage", "verbatim_copy_rate",
    "trailing_specificity", "constraint_count", "role_pronoun_rate",
    "novel_api_rate", "function_name_preserved",
]

df = pd.read_csv(DATA)
print(f"Loaded {len(df)} rows | label={df['label'].value_counts().to_dict()}")
print(f"Sources: {df['source'].value_counts().to_dict() if 'source' in df.columns else 'n/a'}")

X = df[GATE_FEATURES].values
y = df["label"].values
groups = df["task_id"].values   # GroupKFold groups — no task leaks across folds

cv = GroupKFold(n_splits=5)

def make_clf(name):
    if name == "RandomForest":
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("clf", RandomForestClassifier(n_estimators=300, max_depth=6,
                                                         class_weight="balanced",
                                                         random_state=42, n_jobs=-1))])
    if name == "XGBoost":
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("clf", XGBClassifier(n_estimators=300, max_depth=3,
                                               learning_rate=0.1, scale_pos_weight=(y==0).sum()/(y==1).sum(),
                                               random_state=42, verbosity=0, eval_metric="logloss"))])
    if name == "LogisticRegression":
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("clf", LogisticRegression(C=0.1, max_iter=1000, random_state=42,
                                                     class_weight="balanced"))])
    if name == "SmallMLP":
        return Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("clf", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                                               random_state=42))])
    raise ValueError(name)

MODELS = ["RandomForest", "XGBoost", "LogisticRegression", "SmallMLP"]

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CV Results (replaces results/tables/cv_results.csv)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[1/5] 5-fold GroupKFold CV on 11 gate features ...")
cv_rows = []
oof_probs = {}
for name in MODELS:
    print(f"  {name} ...", end=" ", flush=True)
    clf = make_clf(name)
    oof = cross_val_predict(clf, X, y, cv=cv, groups=groups, method="predict_proba")[:, 1]
    oof_probs[name] = oof
    auroc = roc_auc_score(y, oof)
    auprc = average_precision_score(y, oof)
    brier = brier_score_loss(y, oof)
    from sklearn.metrics import log_loss
    ll    = log_loss(y, oof)
    # per-fold AUROCs
    fold_aucs = []
    for tr, te in cv.split(X, y, groups):
        clf2 = make_clf(name)
        clf2.fit(X[tr], y[tr])
        fold_aucs.append(roc_auc_score(y[te], clf2.predict_proba(X[te])[:, 1]))
    print(f"AUROC={auroc:.4f}  AUPRC={auprc:.4f}")
    cv_rows.append({
        "model": name,
        "oof_auroc": round(auroc, 4),
        "mean_fold_auroc": round(np.mean(fold_aucs), 4),
        "std_fold_auroc": round(np.std(fold_aucs), 4),
        "oof_auprc": round(auprc, 4),
        "oof_brier": round(brier, 4),
        "oof_logloss": round(ll, 4),
        "n_rows": len(df),
        "n_features": len(GATE_FEATURES),
        "feature_set": "11_gate_features_no_NLI",
    })

cv_df = pd.DataFrame(cv_rows)
cv_df.to_csv(TABLES / "cv_results.csv", index=False)
print(f"  -> Saved results/tables/cv_results.csv ({len(cv_df)} models, n_rows={len(df)})")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Bootstrap Confidence Intervals (replaces results/audit/bootstrap_confidence_intervals.csv)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2/5] Bootstrap CIs (n=2000, seed=42) ...")
rng = np.random.default_rng(42)
boot_rows = []
for name in MODELS:
    oof = oof_probs[name]
    aucs, auprcs, briers = [], [], []
    for _ in range(2000):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0 or y[idx].sum() == len(idx):
            continue
        aucs.append(roc_auc_score(y[idx], oof[idx]))
        auprcs.append(average_precision_score(y[idx], oof[idx]))
        briers.append(brier_score_loss(y[idx], oof[idx]))
    boot_rows.append({
        "model": name,
        "n_rows": len(df),
        "n_features": len(GATE_FEATURES),
        "feature_set": "11_gate_features_no_NLI",
        "auroc_mean": round(np.mean(aucs), 4),
        "auroc_ci_lower": round(np.percentile(aucs, 2.5), 4),
        "auroc_ci_upper": round(np.percentile(aucs, 97.5), 4),
        "auprc_mean": round(np.mean(auprcs), 4),
        "auprc_ci_lower": round(np.percentile(auprcs, 2.5), 4),
        "auprc_ci_upper": round(np.percentile(auprcs, 97.5), 4),
        "brier_score": round(np.mean(briers), 4),
    })
    print(f"  {name}: AUROC {boot_rows[-1]['auroc_ci_lower']:.4f}–{boot_rows[-1]['auroc_ci_upper']:.4f}")

pd.DataFrame(boot_rows).to_csv(AUDIT / "bootstrap_confidence_intervals.csv", index=False)
print(f"  -> Saved results/audit/bootstrap_confidence_intervals.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Overfitting Gap Audit (replaces results/audit/overfitting_gap_audit.csv)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3/5] Overfitting gap audit ...")
gap_rows = []
for name in MODELS:
    clf = make_clf(name)
    # train on full dataset, compare train vs OOF AUROC
    clf.fit(X, y)
    train_prob = clf.predict_proba(X)[:, 1]
    train_auc = roc_auc_score(y, train_prob)
    oof_auc   = roc_curve = roc_auc_score(y, oof_probs[name])
    gap = train_auc - oof_auc
    status = "[LOW GAP]" if gap < 0.05 else "[OVERFIT WARNING]"
    gap_rows.append({
        "model": name,
        "n_rows": len(df),
        "n_features": len(GATE_FEATURES),
        "feature_set": "11_gate_features_no_NLI",
        "train_auroc": round(train_auc, 4),
        "oof_auroc": round(oof_auc, 4),
        "generalization_gap": round(gap, 4),
        "oof_auprc": round(average_precision_score(y, oof_probs[name]), 4),
        "status": status,
    })
    print(f"  {name}: train={train_auc:.4f} oof={oof_auc:.4f} gap={gap:.4f} {status}")

pd.DataFrame(gap_rows).to_csv(AUDIT / "overfitting_gap_audit.csv", index=False)
print(f"  -> Saved results/audit/overfitting_gap_audit.csv")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Per-corruption held-out eval (replaces results/robustness/held_out_corruption_eval.csv)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4/5] Per-corruption leave-one-out held-out eval ...")
corruption_types = df[df["label"] == 0]["corruption_type"].unique()
print(f"  Corruption types: {list(corruption_types)}")
hoe_rows = []
for held_out in sorted(corruption_types):
    # Train: all rows EXCEPT negatives of this corruption type
    train_mask = ~((df["label"] == 0) & (df["corruption_type"] == held_out))
    test_mask  =  (df["label"] == 0) & (df["corruption_type"] == held_out)
    # test positives: all clean positives
    pos_mask   =  (df["label"] == 1)
    
    X_train = df[train_mask][GATE_FEATURES].values
    y_train = df[train_mask]["label"].values
    X_test  = np.vstack([df[test_mask][GATE_FEATURES].values,
                         df[pos_mask][GATE_FEATURES].values])
    y_test  = np.concatenate([df[test_mask]["label"].values,
                              df[pos_mask]["label"].values])
    
    if len(y_test) < 5 or y_test.sum() == 0 or y_test.sum() == len(y_test):
        print(f"  {held_out}: skipped (too few samples)")
        continue
    
    clf = make_clf("RandomForest")
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    auroc = roc_auc_score(y_test, probs)
    auprc = average_precision_score(y_test, probs)
    
    hoe_rows.append({
        "held_out_corruption": held_out,
        "n_neg_test": int(test_mask.sum()),
        "n_pos_test": int(pos_mask.sum()),
        "train_size": int(train_mask.sum()),
        "dataset": "final_dataset_full_features_483rows",
        "auroc": round(auroc, 4),
        "auprc": round(auprc, 4),
    })
    print(f"  {held_out:25s}: train={int(train_mask.sum())} test_neg={int(test_mask.sum())} AUROC={auroc:.4f}")

pd.DataFrame(hoe_rows).to_csv(ROBUST / "held_out_corruption_eval.csv", index=False)
print(f"  -> Saved results/robustness/held_out_corruption_eval.csv ({len(hoe_rows)} rows)")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Per-source robustness (replaces results/robustness/robustness_by_source.csv)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[5/5] Per-source robustness ...")
src_rows = []
if "source" in df.columns:
    for src in sorted(df["source"].unique()):
        sub = df[df["source"] == src]
        n_pos = (sub["label"] == 1).sum()
        n_neg = (sub["label"] == 0).sum()
        if n_pos < 3 or n_neg < 3:
            print(f"  {src}: skipped (too few: pos={n_pos} neg={n_neg})")
            continue
        Xs = sub[GATE_FEATURES].values
        ys = sub["label"].values
        gs = sub["task_id"].values
        n_splits = min(5, len(np.unique(gs)))
        if n_splits < 2:
            print(f"  {src}: skipped (only 1 task group)")
            continue
        cv_s = GroupKFold(n_splits=n_splits)
        oof_s = cross_val_predict(make_clf("RandomForest"), Xs, ys,
                                   cv=cv_s, groups=gs, method="predict_proba")[:, 1]
        auroc_s = roc_auc_score(ys, oof_s)
        auprc_s = average_precision_score(ys, oof_s)
        print(f"  {src:12s}: n={len(sub):4d} pos={n_pos} neg={n_neg} AUROC={auroc_s:.4f}")
        src_rows.append({
            "source": src,
            "n_total": len(sub),
            "n_positives": int(n_pos),
            "n_negatives": int(n_neg),
            "dataset": "final_dataset_full_features_483rows",
            "oof_auroc": round(auroc_s, 4),
            "oof_auprc": round(auprc_s, 4),
        })
    pd.DataFrame(src_rows).to_csv(ROBUST / "robustness_by_source.csv", index=False)
    print(f"  -> Saved results/robustness/robustness_by_source.csv ({len(src_rows)} rows)")
else:
    print("  'source' column not found — skipping")

# ═══════════════════════════════════════════════════════════════════════════════
# Delete debunked file
# ═══════════════════════════════════════════════════════════════════════════════
dead = TABLES / "downstream_pass1_results.csv"
if dead.exists():
    dead.unlink()
    print(f"\n[DELETED] {dead}  (debunked artifact)")
else:
    print(f"\n[ALREADY GONE] {dead}")

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 72)
print("  REGENERATION COMPLETE — all from final_dataset_full_features.csv (483 rows)")
print("=" * 72)
print()
print("  File                                              Source dataset")
print("  " + "-"*68)
files = [
    ("results/tables/cv_results.csv",                        "483 rows, 11 gate features"),
    ("results/audit/bootstrap_confidence_intervals.csv",     "483 rows, 11 gate features"),
    ("results/audit/overfitting_gap_audit.csv",              "483 rows, 11 gate features"),
    ("results/robustness/held_out_corruption_eval.csv",      "483 rows, 11 gate features"),
    ("results/robustness/robustness_by_source.csv",          "483 rows, 11 gate features"),
]
for f, src in files:
    exists = Path(f).exists()
    mark = "OK" if exists else "MISSING"
    print(f"  [{mark}] {f:50s} ({src})")
print()
print("  [DELETED] results/tables/downstream_pass1_results.csv")
print()
print("  DO NOT USE for paper tables:")
print("  results/robustness/feature_ablation.csv -- may still be stale,")
print("  check with: python feature_ablation.py to regenerate on real data.")
print("=" * 72)
