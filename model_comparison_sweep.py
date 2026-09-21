"""
Handoff PRM — Full Model Comparison Sweep (rebuilt, real dataset)
==================================================================

Loads final_dataset.csv (task_id, problem, handoff, corruption_type, label,
label_source, source), extracts the three core structural features
(cosine similarity, entity overlap, length ratio), then runs a 5-fold
task-grouped out-of-fold (OOF) comparison across four model families:

    - Logistic Regression
    - XGBoost
    - Random Forest
    - Small MLP

Reports OOF AUROC, AUPRC, Brier score, log loss, accuracy, F1, and
per-example inference latency for each model — matching the reporting
format used in the original paper's Table VIII.

Requirements:
    pip install pandas numpy scikit-learn xgboost sentence-transformers spacy --break-system-packages
    python -m spacy download en_core_web_sm
"""

import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    accuracy_score,
    f1_score,
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed — skipping XGBoost. "
          "pip install xgboost --break-system-packages")

# ----------------------------------------------------------------------
# 1. Feature extraction
# ----------------------------------------------------------------------

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the three core structural features for each row:
      - cosine_similarity: MiniLM embedding similarity of handoff vs. problem
      - entity_overlap: Jaccard overlap of spaCy named entities + noun chunks
      - length_ratio: word count of handoff / word count of problem

    Returns df with three new numeric columns appended.
    """
    print("Loading sentence-transformer model (all-MiniLM-L6-v2)...")
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Loading spaCy model (en_core_web_sm)...")
    import spacy
    nlp = spacy.load("en_core_web_sm")

    handoffs = df["handoff"].astype(str).tolist()
    problems = df["problem"].astype(str).tolist()

    print(f"Embedding {len(handoffs)} handoffs and {len(problems)} problems...")
    handoff_emb = embed_model.encode(handoffs, show_progress_bar=True, convert_to_numpy=True)
    problem_emb = embed_model.encode(problems, show_progress_bar=True, convert_to_numpy=True)

    # cosine similarity, row-wise
    def cosine(a, b):
        num = np.sum(a * b, axis=1)
        denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        denom = np.where(denom == 0, 1e-8, denom)
        return num / denom

    cosine_similarity = cosine(handoff_emb, problem_emb)

    print("Computing entity overlap and length ratio...")
    entity_overlap = []
    length_ratio = []

    for h, p in zip(handoffs, problems):
        h_doc = nlp(h)
        p_doc = nlp(p)

        h_ents = set(ent.text.lower().strip() for ent in h_doc.ents) | \
                 set(chunk.text.lower().strip() for chunk in h_doc.noun_chunks)
        p_ents = set(ent.text.lower().strip() for ent in p_doc.ents) | \
                 set(chunk.text.lower().strip() for chunk in p_doc.noun_chunks)

        if len(p_ents) == 0:
            overlap = 0.0
        else:
            overlap = len(h_ents & p_ents) / len(p_ents | h_ents) if len(p_ents | h_ents) > 0 else 0.0
        entity_overlap.append(overlap)

        h_words = len(h.split())
        p_words = len(p.split())
        length_ratio.append(h_words / p_words if p_words > 0 else 0.0)

    df = df.copy()
    df["cosine_similarity"] = cosine_similarity
    df["entity_overlap"] = entity_overlap
    df["length_ratio"] = length_ratio

    return df


# ----------------------------------------------------------------------
# 2. Model definitions
# ----------------------------------------------------------------------

def get_models():
    """Returns a dict of {model_name: sklearn-compatible pipeline}."""
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=6, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Small MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(16,), max_iter=2000, random_state=42, early_stopping=True
            )),
        ]),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1,
            eval_metric="logloss", random_state=42, n_jobs=-1,
        )
    return models


# ----------------------------------------------------------------------
# 3. GroupKFold out-of-fold evaluation
# ----------------------------------------------------------------------

def run_oof_comparison(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 5):
    """
    Runs 5-fold task-grouped out-of-fold evaluation for every model.
    Returns a results DataFrame (one row per model) and a dict of
    per-model OOF prediction arrays (for optional per-fold analysis).
    """
    gkf = GroupKFold(n_splits=n_splits)
    models = get_models()
    results = []
    oof_predictions = {}

    for name, model in models.items():
        print(f"\nRunning {name} under {n_splits}-fold GroupKFold...")
        oof_proba = np.zeros(len(y))
        fold_aurocs = []
        latencies = []

        for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
            # sanity: zero task-id overlap between train/test
            train_tasks = set(groups[train_idx])
            test_tasks = set(groups[test_idx])
            assert len(train_tasks & test_tasks) == 0, (
                f"LEAKAGE DETECTED in fold {fold_idx} for {name}: "
                f"{len(train_tasks & test_tasks)} overlapping task_ids"
            )

            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model_clone = get_models()[name]  # fresh unfit clone per fold
            model_clone.fit(X_train, y_train)

            start = time.perf_counter()
            proba = model_clone.predict_proba(X_test)[:, 1]
            elapsed = time.perf_counter() - start
            per_example_ms = (elapsed / len(X_test)) * 1000
            latencies.append(per_example_ms)

            oof_proba[test_idx] = proba

            if len(np.unique(y_test)) > 1:
                fold_aurocs.append(roc_auc_score(y_test, proba))

        # Overall OOF metrics
        oof_pred_label = (oof_proba >= 0.5).astype(int)
        oof_auroc = roc_auc_score(y, oof_proba)
        oof_auprc = average_precision_score(y, oof_proba)
        oof_brier = brier_score_loss(y, oof_proba)
        oof_logloss = log_loss(y, oof_proba)
        oof_acc = accuracy_score(y, oof_pred_label)
        oof_f1 = f1_score(y, oof_pred_label)

        results.append({
            "Model": name,
            "OOF_AUROC_mean": np.mean(fold_aurocs),
            "OOF_AUROC_std": np.std(fold_aurocs),
            "OOF_AUROC_overall": oof_auroc,
            "OOF_AUPRC": oof_auprc,
            "Brier": oof_brier,
            "LogLoss": oof_logloss,
            "Accuracy": oof_acc,
            "F1": oof_f1,
            "Latency_ms_per_example": np.mean(latencies),
        })
        oof_predictions[name] = oof_proba

        print(f"  {name}: OOF AUROC = {np.mean(fold_aurocs):.4f} ± {np.std(fold_aurocs):.4f}")

    results_df = pd.DataFrame(results).sort_values("OOF_AUROC_mean", ascending=False)
    return results_df, oof_predictions


# ----------------------------------------------------------------------
# 4. Main
# ----------------------------------------------------------------------

def main():
    input_path = "final_dataset.csv"
    output_path = "model_comparison_results.csv"
    features_output_path = "final_dataset_with_features.csv"

    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows, {df['task_id'].nunique()} unique tasks")

    # Extract features (skip if already present, e.g. re-running after first pass)
    feature_cols = ["cosine_similarity", "entity_overlap", "length_ratio"]
    if not all(c in df.columns for c in feature_cols):
        df = extract_features(df)
        df.to_csv(features_output_path, index=False)
        print(f"Saved features to {features_output_path}")
    else:
        print("Features already present in input — skipping extraction.")

    X = df[feature_cols].values
    y = df["label"].values
    groups = df["task_id"].values

    print(f"\nClass balance: {np.bincount(y)} (0=insufficient, 1=sufficient)")
    print(f"Feature matrix shape: {X.shape}")

    results_df, oof_predictions = run_oof_comparison(X, y, groups, n_splits=5)

    print("\n" + "=" * 70)
    print("FINALIZED OUT-OF-FOLD MODEL COMPARISON")
    print("=" * 70)
    print(results_df.to_string(index=False))

    results_df.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
