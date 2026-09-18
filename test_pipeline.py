"""
Phase 11 -- Pipeline Verification & Test Suite

Executes automated checks to verify data integrity, feature extraction bounds,
gate operational pathways, and split safety (no task leakage).
"""

import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import corruption
from feature_extraction import extract_features
from gate import HandoffGate

class TestHandoffPRMPipeline(unittest.TestCase):

    def test_01_corruption_engine(self):
        text = (
            "We will solve this task by defining a helper function. "
            "def solve(x):\n    return x * 2\n"
            "This function handles basic multiplication. "
            "Edge cases include x=0 and negative inputs. "
            "Finally we print the output of solve(5)."
        )
        for name, fn in corruption.CORRUPTION_FUNCTIONS.items():
            corrupted = fn(text)
            self.assertIsInstance(corrupted, str, f"Corruption '{name}' did not return a string.")
            self.assertTrue(len(corrupted.strip()) > 0 or name in ("tool_result_drop", "entity_omission"), 
                            f"Corruption '{name}' returned empty output.")

    def test_02_feature_extraction(self):
        ref = "Build a function `def multiply(a, b): return a * b`."
        handoff = "Implementation plan: write `def multiply(a, b)` and return product of a and b."
        feats = extract_features(handoff, ref)
        
        self.assertIn("cosine_similarity", feats)
        self.assertIn("entity_overlap", feats)
        self.assertIn("length_ratio", feats)
        
        self.assertGreaterEqual(feats["cosine_similarity"], -1.0)
        self.assertLessEqual(feats["cosine_similarity"], 1.0)
        self.assertGreaterEqual(feats["entity_overlap"], 0.0)
        self.assertLessEqual(feats["entity_overlap"], 1.0)
        self.assertGreater(feats["length_ratio"], 0.0)

    def test_03_handoff_gate(self):
        gate = HandoffGate()
        self.assertIsNotNone(gate.threshold)
        self.assertGreater(gate.threshold, 0.0)
        self.assertLess(gate.threshold, 1.0)

        ref = "Write a function `def square(x): return x ** 2` handling edge case x=0."
        good_handoff = "Plan: Implement `def square(x): return x ** 2`. Explicitly handle x=0 as edge case."
        bad_handoff = "done"

        res_good = gate.check(good_handoff, ref)
        res_bad = gate.check(bad_handoff, ref)

        self.assertIn("score", res_good)
        self.assertIn("pass", res_good)
        self.assertIn("score", res_bad)
        self.assertIn("pass", res_bad)
        self.assertIsNotNone(res_bad["reason"])

    def test_04_split_leakage_safety(self):
        df = pd.read_csv("features.csv")
        gkf = GroupKFold(n_splits=5)
        X = df[["cosine_similarity", "entity_overlap", "length_ratio"]].values
        y = df["label"].values
        groups = df["task_id"].values

        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            train_tasks = set(groups[train_idx])
            val_tasks = set(groups[val_idx])
            intersection = train_tasks.intersection(val_tasks)
            self.assertEqual(len(intersection), 0, f"Task leakage detected in GroupKFold split: {intersection}")

if __name__ == "__main__":
    unittest.main()
