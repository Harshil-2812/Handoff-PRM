"""Stage-1 Feature Extraction, Classifier Scoring, and SHAP Explainability."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from feature_extraction import (
    compute_cosine_similarity,
    compute_entity_overlap,
    compute_length_ratio,
    compute_sentence_count_ratio,
    compute_section_coverage,
    compute_verbatim_copy_rate,
    compute_trailing_specificity,
    compute_constraint_count,
    compute_role_pronoun_rate,
    compute_novel_api_rate,
    compute_function_name_preserved,
)

logger = logging.getLogger("FeaturePipeline")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_CACHE_DIR = REPO_ROOT / "workstream_c" / "models"
MODEL_CACHE_PATH = MODEL_CACHE_DIR / "stage1_rf.joblib"
V2_CSV = REPO_ROOT / "features_v2.csv"
V2_JSON = REPO_ROOT / "features_v2.json"

FEATURE_COLS_11 = [
    "cosine_similarity",
    "entity_overlap",
    "length_ratio",
    "sentence_count_ratio",
    "section_coverage",
    "verbatim_copy_rate",
    "trailing_specificity",
    "constraint_count",
    "role_pronoun_rate",
    "novel_api_rate",
    "function_name_preserved",
]

# Plain-language hint templates for SHAP-hinted retries (C5)
SHAP_FEATURE_HINTS = {
    "cosine_similarity": "Your handoff diverges in semantic meaning from the problem statement -- align your plan directly with the requirements.",
    "entity_overlap": "Key entities, variables, or specific terminology from the problem appear missing in your handoff -- include them explicitly.",
    "length_ratio": "Your handoff is significantly compressed compared to the problem -- include the full details, step-by-step logic, and constraints.",
    "sentence_count_ratio": "Your handoff contains too few sentences to thoroughly convey the problem details -- expand the explanation.",
    "section_coverage": "Important structural sections (such as Objective, Constraints, Edge Cases, or Signature) are missing -- organize with clear headings.",
    "verbatim_copy_rate": "Crucial problem phrases and technical details appear omitted -- preserve the exact problem definitions where relevant.",
    "trailing_specificity": "The conclusion of your handoff loses technical precision or fades out -- ensure specific edge cases and output formats are detailed to the end.",
    "constraint_count": "Explicit constraints (e.g. boundary conditions, bounds, invalid inputs) are missing or ignored -- state all constraints explicitly.",
    "role_pronoun_rate": "Your handoff contains conversational or first-person chatter -- format it strictly as a technical specification.",
    "novel_api_rate": "Your handoff invents unmentioned functions or external APIs not found in the problem -- stick strictly to required standard interfaces.",
    "function_name_preserved": "The required entry-point function name is either missing or changed -- define exactly the required function signature.",
}


def extract_11_features(handoff: str, problem: str, entry_point: str = "") -> Dict[str, float]:
    """Extracts the 11 frozen features for a (handoff, problem) pair."""
    return {
        "cosine_similarity": float(compute_cosine_similarity(handoff, problem)),
        "entity_overlap": float(compute_entity_overlap(handoff, problem)),
        "length_ratio": float(compute_length_ratio(handoff, problem)),
        "sentence_count_ratio": float(compute_sentence_count_ratio(handoff, problem)),
        "section_coverage": float(compute_section_coverage(handoff, problem, entry_point)),
        "verbatim_copy_rate": float(compute_verbatim_copy_rate(handoff, problem)),
        "trailing_specificity": float(compute_trailing_specificity(handoff, problem)),
        "constraint_count": float(compute_constraint_count(handoff, problem)),
        "role_pronoun_rate": float(compute_role_pronoun_rate(handoff, problem)),
        "novel_api_rate": float(compute_novel_api_rate(handoff, problem)),
        "function_name_preserved": float(compute_function_name_preserved(handoff, problem, entry_point)),
    }


class Stage1Classifier:
    """Manages the Stage-1 Random Forest model, probability inference, and SHAP explainability."""

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path or MODEL_CACHE_PATH
        self.feature_cols = FEATURE_COLS_11
        self.model: Optional[Pipeline] = None
        self.explainer: Optional[shap.TreeExplainer] = None
        self._load_or_train()

    def _load_or_train(self):
        # 1. Attempt loading existing cached model
        if self.model_path.exists():
            try:
                bundle = joblib.load(self.model_path)
                self.model = bundle["model"]
                self.explainer = bundle.get("explainer")
                self.feature_cols = bundle.get("feature_cols", FEATURE_COLS_11)
                logger.info(f"Loaded Stage-1 classifier from {self.model_path}")
                return
            except Exception as e:
                logger.warning(f"Could not load cached model from {self.model_path} ({e}). Rebuilding...")

        # 2. Train deterministic Random Forest on features_v2.csv
        logger.info(f"Training Stage-1 Random Forest model on {V2_CSV}...")
        df = pd.read_csv(V2_CSV)
        X = df[self.feature_cols]
        y = df["label"]

        model = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )),
        ])
        model.fit(X, y)

        rf_clf = model.named_steps["clf"]
        imputer = model.named_steps["imp"]
        X_imp = imputer.transform(X)
        explainer = shap.TreeExplainer(rf_clf, data=X_imp[:100])

        self.model = model
        self.explainer = explainer

        MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "explainer": self.explainer,
            "feature_cols": self.feature_cols,
        }, self.model_path)
        logger.info(f"Trained and saved Python 3.10-compatible Stage-1 model to {self.model_path}")

    def score(
        self,
        handoff: str,
        problem: str,
        entry_point: str = "",
        threshold: float = 0.550419,
    ) -> Dict[str, Any]:
        """Extracts features, computes P(good), checks threshold, and extracts SHAP attribution."""
        feats = extract_11_features(handoff, problem, entry_point=entry_point)
        df_x = pd.DataFrame([feats])[self.feature_cols]

        prob = float(self.model.predict_proba(df_x)[0, 1])
        passed = prob >= threshold

        worst_feature = None
        hint = None

        if not passed and self.explainer is not None:
            try:
                imp_x = self.model.named_steps["imp"].transform(df_x)
                shap_vals = self.explainer.shap_values(imp_x)
                # For binary RF, shap_values can be a list [class 0, class 1] or an array
                if isinstance(shap_vals, list) and len(shap_vals) == 2:
                    vals = shap_vals[1][0]
                elif isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) == 3:
                    vals = shap_vals[0, :, 1]
                else:
                    vals = shap_vals[0]
                worst_idx = int(np.argmin(vals))
                worst_feature = self.feature_cols[worst_idx]
                hint = SHAP_FEATURE_HINTS.get(
                    worst_feature,
                    f"The handoff scored low on '{worst_feature}'.",
                )
            except Exception as e:
                logger.warning(f"SHAP explanation computation error: {e}")

        return {
            "score": prob,
            "passed": passed,
            "decision": "PASS" if passed else "BLOCK",
            "features": feats,
            "worst_feature": worst_feature,
            "hint": hint,
        }

