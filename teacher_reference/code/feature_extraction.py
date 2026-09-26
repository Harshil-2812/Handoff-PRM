"""
feature_extraction.py — Handoff-PRM Feature Extractor (Teacher Reference Copy)
================================================================================
This file re-exports the live feature_extraction.py module and adds paper-ready
documentation for all 11 gate features + 2 NLI features.

Paper section: §V (Feature Engineering), Table V

Feature taxonomy (11 gate features used in Stage-1, NLI excluded from Stage-1):
  Category 1 — Core Similarity (3 features):
    cosine_similarity        sentence-transformers all-MiniLM-L6-v2 cosine sim
    entity_overlap           spaCy NE + noun-chunk recall (F1-style)
    length_ratio             word count ratio handoff / problem

  Category 2 — Content Quality (6 features):
    sentence_count_ratio     handoff sentences / problem sentences
    section_coverage         fraction of expected handoff sections present
    verbatim_copy_rate       fraction of handoff 4-grams copied verbatim from problem
    trailing_specificity     information density of last 30% of handoff
    constraint_count         number of explicit constraint/requirement statements
    role_pronoun_rate        first-person pronoun rate (should be ~0)

  Category 3 — Lexical/Schema (2 features):
    novel_api_rate           fraction of backtick identifiers not in the problem
    function_name_preserved  1.0 if entry_point name appears in handoff, else 0.0

  Category 4 — NLI (2 features, Stage-2 only):
    nli_contradiction_max    max NLI contradiction score across sentence pairs
    nli_entailment_mean      mean NLI entailment score across sentence pairs

Extraction latency (from paper §V-D):
  Core features (cosine, entity, length): ~120ms (dominated by sentence-transformer)
  Full 11-feature set: ~135ms
  NLI features: +80ms (CrossEncoder; excluded from Stage-1)

Usage:
  from feature_extraction import extract_features
  feats = extract_features(handoff_text, problem_text)
  # feats is a dict with all 13 keys
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from feature_extraction import (
    extract_features,
    compute_nli_features,
)

__all__ = ["extract_features", "compute_nli_features"]

# ─── Feature documentation table (for reviewers) ─────────────────────────────
FEATURE_DOCS = {
    "cosine_similarity": {
        "category": "Core Similarity",
        "description": "Semantic similarity between handoff and problem using all-MiniLM-L6-v2",
        "range": "[0.0, 1.0]",
        "typical_clean": ">0.7",
        "typical_corrupted": {
            "invert_objective": "~0.6-0.65 (semantically inverted)",
            "truncation": "~0.55-0.65 (incomplete text)",
            "fake_completion": "~0.6-0.7 (generic filler replaces specific content)",
        },
        "shap_rank_primary": 4,
    },
    "entity_overlap": {
        "category": "Core Similarity",
        "description": "Recall of named entities and noun chunks from problem in handoff",
        "range": "[0.0, 1.0]",
        "typical_clean": ">0.5",
        "typical_corrupted": {
            "entity_omission": "~0.1-0.3 (NE sentences removed)",
            "over_summarization": "~0.2-0.4 (1-sentence summary loses entities)",
        },
        "shap_rank_primary": 7,
    },
    "length_ratio": {
        "category": "Core Similarity",
        "description": "Word count ratio: len(handoff_words) / len(problem_words)",
        "range": "[0.0, ∞) — typical range 0.5-2.0",
        "typical_clean": "0.8-1.5",
        "typical_corrupted": {
            "truncation": "~0.15-0.30 (30% word-boundary cut)",
            "over_summarization": "~0.1-0.2 (1-sentence TextRank summary)",
        },
        "shap_rank_primary": 1,
        "note": "Single strongest SHAP feature — structural corruptions trivially detectable",
    },
    "sentence_count_ratio": {
        "category": "Content Quality",
        "description": "Number of sentences in handoff / number of sentences in problem",
        "range": "[0.0, ∞)",
        "shap_rank_primary": 10,
    },
    "section_coverage": {
        "category": "Content Quality",
        "description": "Fraction of expected handoff sections present (constraints, approach, edge cases, signature)",
        "range": "[0.0, 1.0]",
        "shap_rank_primary": 3,
    },
    "verbatim_copy_rate": {
        "category": "Content Quality",
        "description": "Fraction of handoff 4-grams that appear verbatim in the problem text",
        "range": "[0.0, 1.0]",
        "shap_rank_primary": 5,
    },
    "trailing_specificity": {
        "category": "Content Quality",
        "description": "Type-token ratio of unique content words in last 30% of handoff",
        "range": "[0.0, 1.0]",
        "typical_corrupted": {
            "fake_completion": "~0.05-0.15 (generic filler has low TTR)",
        },
        "shap_rank_primary": 6,
    },
    "constraint_count": {
        "category": "Content Quality",
        "description": "Count of constraint/requirement statements (regex patterns: must, should, require, ensure, etc.)",
        "range": "[0, ∞)",
        "shap_rank_primary": 8,
    },
    "role_pronoun_rate": {
        "category": "Content Quality",
        "description": "First-person pronoun rate (I, me, my, we, our) — should be ~0 in a clean handoff",
        "range": "[0.0, 1.0]",
        "shap_rank_primary": 11,
    },
    "novel_api_rate": {
        "category": "Lexical/Schema",
        "description": "Fraction of backtick-enclosed identifiers in handoff not present in problem",
        "range": "[0.0, 1.0]",
        "typical_corrupted": {
            "signature_rename": "~0.5+ (_impl suffix creates novel identifiers)",
        },
        "shap_rank_primary": 9,
    },
    "function_name_preserved": {
        "category": "Lexical/Schema",
        "description": "Binary: 1.0 if entry_point name appears in handoff text, else 0.0",
        "range": "{0.0, 1.0}",
        "typical_corrupted": {
            "signature_rename": "0.0 (entry_point replaced with entry_point_impl)",
        },
        "shap_rank_primary": 2,
    },
    "nli_contradiction_max": {
        "category": "NLI (Stage-2 only)",
        "description": "Max NLI contradiction score across problem-sentence × handoff-sentence pairs (CrossEncoder nli-deberta-v3-small)",
        "range": "[0.0, 1.0]",
        "note": "Excluded from Stage-1 feature set due to 80ms latency per inference",
        "shap_rank_primary": "N/A (Stage-2 only)",
    },
    "nli_entailment_mean": {
        "category": "NLI (Stage-2 only)",
        "description": "Mean NLI entailment score across problem-sentence × handoff-sentence pairs",
        "range": "[0.0, 1.0]",
        "note": "Excluded from Stage-1 feature set due to 80ms latency per inference",
        "shap_rank_primary": "N/A (Stage-2 only)",
    },
}


if __name__ == "__main__":
    print("=== Feature Documentation ===\n")
    for feat, doc in FEATURE_DOCS.items():
        print(f"{feat} [{doc['category']}]")
        print(f"  {doc['description']}")
        print(f"  Range: {doc['range']}")
        if "shap_rank_primary" in doc:
            print(f"  SHAP rank (primary): {doc['shap_rank_primary']}")
        print()
