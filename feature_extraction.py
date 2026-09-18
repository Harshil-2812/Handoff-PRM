"""
Turns a handoff message into 3 numeric structural features:
  1. cosine similarity: how aligned is the handoff with the original problem
  2. entity overlap: what fraction of named entities/identifiers survived
  3. length ratio: how much the handoff was compressed relative to a baseline

Requires: sentence-transformers, spacy (+ `python -m spacy download en_core_web_sm`)
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import spacy

_embedder = None
_nlp = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def cosine_similarity_feature(handoff: str, reference_text: str) -> float:
    """Cosine similarity between the handoff and a reference (e.g. the original problem statement)."""
    embedder = _get_embedder()
    emb = embedder.encode([handoff, reference_text])
    a, b = emb[0], emb[1]
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def entity_overlap_feature(handoff: str, reference_text: str) -> float:
    """
    Fraction of named entities / noun chunks in the reference that still
    appear (as substrings) in the handoff. Low overlap = things got dropped.
    """
    nlp = _get_nlp()
    ref_doc = nlp(reference_text)
    ref_entities = set(chunk.text.lower().strip() for chunk in ref_doc.noun_chunks)
    ref_entities |= set(ent.text.lower().strip() for ent in ref_doc.ents)

    if not ref_entities:
        return 1.0  # nothing to lose, treat as fully preserved

    handoff_lower = handoff.lower()
    preserved = sum(1 for e in ref_entities if e in handoff_lower)
    return preserved / len(ref_entities)


def length_ratio_feature(handoff: str, reference_text: str) -> float:
    """Ratio of handoff length to reference length (word count). Low = heavy compression."""
    handoff_len = len(handoff.split())
    ref_len = len(reference_text.split())
    if ref_len == 0:
        return 1.0
    return handoff_len / ref_len


def extract_features(handoff: str, reference_text: str) -> dict:
    """Returns all 3 features as a dict, ready to be assembled into a training row."""
    return {
        "cosine_similarity": cosine_similarity_feature(handoff, reference_text),
        "entity_overlap": entity_overlap_feature(handoff, reference_text),
        "length_ratio": length_ratio_feature(handoff, reference_text),
    }
