"""
Feature extraction for Handoff PRM.

Extracts 13 features comparing a handoff message to its reference problem:

Existing (Category 1 - Core):
  cosine_similarity  : semantic similarity via sentence-transformers
  entity_overlap     : named-entity / noun-chunk recall via spaCy
  length_ratio       : word count ratio handoff / problem

New (Category 2 - Content Quality):
  sentence_count_ratio : handoff sentences / problem sentences
  section_coverage     : fraction of expected handoff sections present
  verbatim_copy_rate   : fraction of handoff 4-grams copied verbatim from problem
  trailing_specificity : information density of last 30% of handoff
  constraint_count     : number of explicit constraint/requirement statements
  role_pronoun_rate    : first-person pronoun rate (should be ~0 in a good handoff)

New (Category 3 - Lexical/Semantic):
  bleu_1             : BLEU-1 n-gram overlap
  rouge_l            : ROUGE-L F1
  novel_api_rate     : fraction of backtick identifiers not in the problem

New (Category 4 - Schema):
  function_name_preserved : 1.0 if entry_point name appears in handoff, else 0.0
"""

import re
from functools import lru_cache

import numpy as np
import spacy
from sentence_transformers import SentenceTransformer

# ── Model singletons ──────────────────────────────────────────────────────────
_st_model = None
_nlp       = None


_nli_model = None


def _get_st_model():
    global _st_model
    if _st_model is None:
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _st_model


def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp


def _get_nli_model():
    global _nli_model
    if _nli_model is None:
        from sentence_transformers import CrossEncoder
        _nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small")
    return _nli_model


# ── Category 5: NLI & Sentence-Level Semantics (Advanced) ────────────────────

def compute_nli_features(handoff: str, reference_text: str) -> tuple[float, float]:
    """Computes max NLI contradiction and mean entailment across sentence pairs."""
    try:
        nli = _get_nli_model()
        prob_sents = [s.strip() for s in re.split(r'[.!?]+', reference_text) if len(s.strip().split()) >= 4]
        hand_sents = [s.strip() for s in re.split(r'[.!?]+', handoff) if len(s.strip().split()) >= 4]

        if not prob_sents or not hand_sents:
            return 0.0, 0.5

        pairs = []
        for p_sent in prob_sents[:4]:
            for h_sent in hand_sents[:4]:
                pairs.append((p_sent, h_sent))

        scores = nli.predict(pairs)  # shape: (n_pairs, 3) -> [contradiction, entailment, neutral]
        if len(scores.shape) == 1:
            scores = np.expand_dims(scores, axis=0)

        # Softmax over logits
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        max_contradiction = float(np.max(probs[:, 0]))
        mean_entailment   = float(np.mean(probs[:, 1]))
        return max_contradiction, mean_entailment
    except Exception:
        return 0.0, 0.5


def compute_sentence_level_similarity(handoff: str, reference_text: str) -> tuple[float, float]:
    """Computes min & mean sentence-level cosine similarity."""
    st_model = _get_st_model()
    prob_sents = [s.strip() for s in re.split(r'[.!?]+', reference_text) if len(s.strip().split()) >= 3]
    hand_sents = [s.strip() for s in re.split(r'[.!?]+', handoff) if len(s.strip().split()) >= 3]

    if not prob_sents or not hand_sents:
        return 0.5, 0.5

    h_embs = st_model.encode(hand_sents, normalize_embeddings=True)
    p_embs = st_model.encode(prob_sents, normalize_embeddings=True)

    sim_matrix = np.dot(h_embs, p_embs.T)
    max_sim_per_hand_sent = np.max(sim_matrix, axis=1)

    min_sim = float(np.min(max_sim_per_hand_sent))
    mean_sim = float(np.mean(max_sim_per_hand_sent))
    return min_sim, mean_sim


def compute_signature_param_diff(handoff: str, reference_text: str) -> float:
    """Computes parameter count difference in function def lines."""
    m = re.search(r'def\s+(\w+)\s*\((.*?)\)', handoff)
    if not m:
        return 0.0
    params = [p.strip() for p in m.group(2).split(',') if p.strip()]

    ref_m = re.search(r'def\s+(\w+)\s*\((.*?)\)', reference_text)
    if ref_m:
        ref_params = [p.strip() for p in ref_m.group(2).split(',') if p.strip()]
        return float(abs(len(params) - len(ref_params)))
    return 0.0


# ── Category 1: Core (existing) ───────────────────────────────────────────────

def compute_cosine_similarity(handoff: str, reference_text: str) -> float:
    model = _get_st_model()
    emb = model.encode([handoff, reference_text], normalize_embeddings=True)
    return float(np.dot(emb[0], emb[1]))


def compute_entity_overlap(handoff: str, reference_text: str) -> float:
    nlp = _get_nlp()
    ref_doc = nlp(reference_text)
    ref_terms = {
        chunk.text.lower() for chunk in ref_doc.noun_chunks
    } | {
        ent.text.lower() for ent in ref_doc.ents
    }
    if not ref_terms:
        return 1.0
    handoff_lower = handoff.lower()
    found = sum(1 for term in ref_terms if term in handoff_lower)
    return found / len(ref_terms)


def compute_length_ratio(handoff: str, reference_text: str) -> float:
    ref_len = len(reference_text.split())
    if ref_len == 0:
        return 1.0
    return len(handoff.split()) / ref_len


# ── Category 2: Content Quality (new) ─────────────────────────────────────────

def compute_sentence_count_ratio(handoff: str, reference_text: str) -> float:
    """Ratio of sentence count handoff / problem."""
    def count_sents(text):
        return max(1, len([s for s in re.split(r'[.!?]+', text) if s.strip()]))
    return count_sents(handoff) / count_sents(reference_text)


def compute_section_coverage(handoff: str, reference_text: str,
                              entry_point: str = None) -> float:
    """What fraction of expected handoff sections are present.
    Sections: function_signature, edge_cases, algorithm_description,
              return_value, problem_restatement.
    """
    h = handoff.lower()
    # 1. Function signature present
    has_signature = bool(re.search(r'def\s+\w+\s*\(', handoff))
    # 2. Edge case language
    has_edge_cases = bool(re.search(
        r'\b(edge case|empty|none|null|raise|zero|negative|boundary)\b', h
    ))
    # 3. Algorithm / approach description
    has_algorithm = bool(re.search(
        r'\b(approach|algorithm|implement|use|strategy|method|steps?|procedure)\b', h
    ))
    # 4. Return value described
    has_return = 'return' in h
    # 5. Problem restatement (key problem words appear in handoff)
    prob_words = set(reference_text.lower().split()) - {
        'a', 'an', 'the', 'is', 'are', 'of', 'in', 'to', 'and', 'or', 'for'
    }
    h_words = set(h.split())
    has_restatement = len(prob_words & h_words) >= 3

    sections = [has_signature, has_edge_cases, has_algorithm,
                has_return, has_restatement]
    return sum(sections) / len(sections)


def compute_verbatim_copy_rate(handoff: str, reference_text: str) -> float:
    """Fraction of handoff 4-grams that appear verbatim in the problem.
    High value = Agent A is echoing the problem rather than specifying.
    """
    def get_ngrams(text, n=4):
        words = text.lower().split()
        if len(words) < n:
            return set()
        return set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))

    h_ngrams = get_ngrams(handoff)
    r_ngrams = get_ngrams(reference_text)
    if not h_ngrams:
        return 0.0
    return len(h_ngrams & r_ngrams) / len(h_ngrams)


def compute_trailing_specificity(handoff: str, reference_text: str) -> float:
    """Information density of last 30% of handoff.
    High = edge cases properly specified. Low = fake_completion or truncation.
    """
    words = handoff.split()
    if len(words) < 10:
        return 0.0
    tail = ' '.join(words[int(len(words) * 0.7):])
    # Count specific markers
    specifics = 0
    specifics += len(re.findall(r'\b\d+\b', tail))          # numbers
    specifics += len(re.findall(r'`[^`]+`', tail))           # code refs
    specifics += len(re.findall(
        r'\b(return|none|true|false|empty|null|raise|error|zero|negative)\b',
        tail.lower()
    ))
    tail_words = max(len(tail.split()), 1)
    return specifics / tail_words


def compute_constraint_count(handoff: str, reference_text: str) -> float:
    """Number of explicit constraint/requirement sentences (normalized by length)."""
    sentences = re.split(r'[.!?]+', handoff)
    constraint_words = [
        'must', 'should', 'require', 'ensure', 'handle',
        'edge case', 'assume', 'note:', 'constraint', 'guarantee'
    ]
    count = sum(
        1 for s in sentences
        if any(w in s.lower() for w in constraint_words)
    )
    # Normalize by problem length (longer problems should have more constraints)
    ref_words = max(len(reference_text.split()), 1)
    return count / (ref_words / 10)  # roughly: constraints per 10 problem words


def compute_role_pronoun_rate(handoff: str, reference_text: str) -> float:
    """First-person pronoun rate. Should be ~0 in a good handoff.
    High rate signals Agent A's internal monologue bleeding through.
    """
    words = handoff.lower().split()
    if not words:
        return 0.0
    first_person = sum(
        1 for w in words
        if w in {"i", "i've", "i'll", "i'm", "i'd", "my", "me",
                 "we", "we've", "we'll", "our", "ourselves"}
    )
    return first_person / len(words)


# ── Category 3: Lexical/Semantic (new) ────────────────────────────────────────

def compute_bleu_1(handoff: str, reference_text: str) -> float:
    """BLEU-1 unigram precision."""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        ref   = [reference_text.lower().split()]
        hyp   = handoff.lower().split()
        if not hyp:
            return 0.0
        return float(sentence_bleu(
            ref, hyp,
            weights=(1, 0, 0, 0),
            smoothing_function=SmoothingFunction().method1
        ))
    except Exception:
        return 0.0


def compute_rouge_l(handoff: str, reference_text: str) -> float:
    """ROUGE-L F1 score."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        result = scorer.score(reference_text, handoff)
        return float(result['rougeL'].fmeasure)
    except Exception:
        return 0.0


def compute_novel_api_rate(handoff: str, reference_text: str) -> float:
    """Fraction of backtick-wrapped identifiers in handoff not found in problem.
    High = fabricated API calls (hallucination risk).
    """
    h_apis = set(re.findall(r'`([^`\n]+)`', handoff))
    r_apis = set(re.findall(r'`([^`\n]+)`', reference_text))
    if not h_apis:
        return 0.0
    novel = h_apis - r_apis
    return len(novel) / len(h_apis)


# ── Category 4: Schema (new) ──────────────────────────────────────────────────

def compute_function_name_preserved(handoff: str, reference_text: str,
                                    entry_point: str = None) -> float:
    """Binary: 1.0 if entry_point function name appears in handoff, else 0.0.
    Returns 0.5 (neutral) if entry_point is unknown.
    """
    if not entry_point:
        return 0.5
    return 1.0 if entry_point in handoff else 0.0


# ── Master extractor ──────────────────────────────────────────────────────────

def extract_features(handoff: str, reference_text: str,
                     entry_point: str = None) -> dict:
    """Extract all 18 features for a (handoff, problem) pair."""
    nli_contra, nli_ent = compute_nli_features(handoff, reference_text)
    min_sim, mean_sim   = compute_sentence_level_similarity(handoff, reference_text)
    param_diff          = compute_signature_param_diff(handoff, reference_text)

    return {
        # Category 1: Core
        "cosine_similarity":        compute_cosine_similarity(handoff, reference_text),
        "entity_overlap":           compute_entity_overlap(handoff, reference_text),
        "length_ratio":             compute_length_ratio(handoff, reference_text),
        # Category 2: Content Quality
        "sentence_count_ratio":     compute_sentence_count_ratio(handoff, reference_text),
        "section_coverage":         compute_section_coverage(handoff, reference_text, entry_point),
        "verbatim_copy_rate":       compute_verbatim_copy_rate(handoff, reference_text),
        "trailing_specificity":     compute_trailing_specificity(handoff, reference_text),
        "constraint_count":         compute_constraint_count(handoff, reference_text),
        "role_pronoun_rate":        compute_role_pronoun_rate(handoff, reference_text),
        # Category 3: Lexical/Semantic
        "bleu_1":                   compute_bleu_1(handoff, reference_text),
        "rouge_l":                  compute_rouge_l(handoff, reference_text),
        "novel_api_rate":           compute_novel_api_rate(handoff, reference_text),
        # Category 4: Schema
        "function_name_preserved":  compute_function_name_preserved(handoff, reference_text, entry_point),
        "signature_param_diff":     param_diff,
        # Category 5: Advanced NLI & Sentence Semantics
        "nli_contradiction_max":    nli_contra,
        "nli_entailment_mean":      nli_ent,
        "min_sentence_cosine_sim":  min_sim,
        "mean_sentence_cosine_sim": mean_sim,
    }


def extract_features_batch(df, entry_point_lookup=None, include_nli=False):
    """Batched feature extraction over a DataFrame.
    Pre-computes sentence embeddings and spaCy parses in fast vector batches.
    Set include_nli=True only on GPU/Cloud environments to prevent CPU lag.
    """
    import pandas as pd
    from tqdm import tqdm
    if entry_point_lookup is None:
        entry_point_lookup = {}

    handoffs = [str(x) for x in df["handoff"].tolist()]
    problems = [str(x) for x in df["problem"].tolist()]

    print("1/3 Batch encoding text embeddings with SentenceTransformer...")
    st_model = _get_st_model()
    h_embs = st_model.encode(handoffs, batch_size=128, normalize_embeddings=True, show_progress_bar=True)
    p_embs = st_model.encode(problems, batch_size=128, normalize_embeddings=True, show_progress_bar=True)
    cosine_sims = np.sum(h_embs * p_embs, axis=1)

    print("2/3 Batch parsing problem entities with spaCy...")
    unique_problems = list(set(problems))
    nlp = _get_nlp()
    ref_terms_dict = {}
    for doc, prob_text in zip(nlp.pipe(unique_problems, batch_size=64), unique_problems):
        terms = {chunk.text.lower() for chunk in doc.noun_chunks} | {ent.text.lower() for ent in doc.ents}
        ref_terms_dict[prob_text] = terms

    row_nli_indices = []
    all_nli_pairs = []
    if include_nli:
        print("Scoring NLI contradiction pairs on GPU/Cloud...")
        for h_text, p_text in zip(handoffs, problems):
            p_sents = [s.strip() for s in re.split(r'[.!?]+', p_text) if len(s.strip().split()) >= 4][:3]
            h_sents = [s.strip() for s in re.split(r'[.!?]+', h_text) if len(s.strip().split()) >= 4][:3]

            start_idx = len(all_nli_pairs)
            for ps in p_sents:
                for hs in h_sents:
                    all_nli_pairs.append((ps, hs))
            end_idx = len(all_nli_pairs)
            row_nli_indices.append((start_idx, end_idx))

        nli_model = _get_nli_model()
        nli_scores = nli_model.predict(all_nli_pairs, batch_size=256, show_progress_bar=True)
        if len(nli_scores.shape) == 1:
            nli_scores = np.expand_dims(nli_scores, axis=0)
        exp_s = np.exp(nli_scores - np.max(nli_scores, axis=1, keepdims=True))
        nli_probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)
    else:
        nli_probs = np.zeros((0, 3))

    print("3/3 Assembling final feature matrix...")
    rows = []
    for idx, (_, row) in enumerate(df.iterrows()):
        handoff = handoffs[idx]
        problem = problems[idx]
        task_id = row["task_id"]
        entry_point = entry_point_lookup.get(task_id, "")

        ref_terms = ref_terms_dict.get(problem, set())
        if not ref_terms:
            entity_overlap = 1.0
        else:
            handoff_lower = handoff.lower()
            found = sum(1 for term in ref_terms if term in handoff_lower)
            entity_overlap = found / len(ref_terms)

        param_diff = compute_signature_param_diff(handoff, problem)

        feats = {
            "cosine_similarity":        float(cosine_sims[idx]),
            "entity_overlap":           float(entity_overlap),
            "length_ratio":             compute_length_ratio(handoff, problem),
            "sentence_count_ratio":     compute_sentence_count_ratio(handoff, problem),
            "section_coverage":         compute_section_coverage(handoff, problem, entry_point),
            "verbatim_copy_rate":       compute_verbatim_copy_rate(handoff, problem),
            "trailing_specificity":     compute_trailing_specificity(handoff, problem),
            "constraint_count":         compute_constraint_count(handoff, problem),
            "role_pronoun_rate":        compute_role_pronoun_rate(handoff, problem),
            "bleu_1":                   compute_bleu_1(handoff, problem),
            "rouge_l":                  compute_rouge_l(handoff, problem),
            "novel_api_rate":           compute_novel_api_rate(handoff, problem),
            "function_name_preserved":  compute_function_name_preserved(handoff, problem, entry_point),
            "signature_param_diff":     param_diff,
        }

        if include_nli and all_nli_pairs:
            s_idx, e_idx = row_nli_indices[idx]
            if e_idx > s_idx:
                row_probs = nli_probs[s_idx:e_idx]
                feats["nli_contradiction_max"] = float(np.max(row_probs[:, 0]))
                feats["nli_entailment_mean"]   = float(np.mean(row_probs[:, 1]))
            else:
                feats["nli_contradiction_max"] = 0.0
                feats["nli_entailment_mean"]   = 0.5

        feats["task_id"]         = task_id
        feats["source"]          = row.get("source", "unknown")
        feats["corruption_type"] = row["corruption_type"]
        feats["label"]           = row["label"]
        rows.append(feats)

    return pd.DataFrame(rows)

