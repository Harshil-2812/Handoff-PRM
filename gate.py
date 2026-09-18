"""
The real-time gate: given a live handoff message, scores it with the trained
Handoff PRM and decides pass/block. If blocked, uses SHAP to figure out which
feature dragged the score down and produces a plain-language reason that can
be sent back to Agent A.

Usage:
    from gate import HandoffGate
    gate = HandoffGate(threshold=0.7)
    result = gate.check(handoff_text, problem_text)
    if result["pass"]:
        # proceed to Agent B
    else:
        # send result["reason"] back to Agent A and retry
"""

import joblib
import pandas as pd
from feature_extraction import extract_features

MODEL_PATH = "handoff_prm.joblib"

# Plain-language hint templates keyed by which feature contributed most
# negatively to the score. Tune the wording as you see real failure cases.
FEATURE_HINTS = {
    "cosine_similarity": (
        "Your handoff doesn't align well with what the next agent needs -- "
        "make sure it directly addresses the task requirements."
    ),
    "entity_overlap": (
        "Key details (variable names, function signatures, specific facts) "
        "from the original task seem to be missing from your handoff -- "
        "include them explicitly."
    ),
    "length_ratio": (
        "Your handoff may be over-compressed -- you likely summarized away "
        "important detail. Consider including more of the plan/edge cases."
    ),
}


class HandoffGate:
    def __init__(self, threshold: float = 0.7, model_path: str = MODEL_PATH):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.explainer = bundle["explainer"]
        self.feature_cols = bundle["feature_cols"]
        self.threshold = threshold

    def check(self, handoff: str, reference_text: str) -> dict:
        feats = extract_features(handoff, reference_text)
        X = pd.DataFrame([feats])[self.feature_cols]

        score = float(self.model.predict_proba(X)[0, 1])
        passed = score >= self.threshold

        result = {
            "score": score,
            "pass": passed,
            "features": feats,
            "reason": None,
        }

        if not passed:
            shap_values = self.explainer.shap_values(X)[0]
            # most negative contribution = feature that hurt the score most
            worst_idx = shap_values.argmin()
            worst_feature = self.feature_cols[worst_idx]
            result["reason"] = FEATURE_HINTS.get(
                worst_feature,
                f"The handoff scored low on '{worst_feature}'.",
            )
            result["worst_feature"] = worst_feature

        return result
