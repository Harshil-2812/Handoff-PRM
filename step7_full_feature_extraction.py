"""
Handoff PRM — Step 7: Full 14-Feature Extraction (Kaggle GPU)
================================================================

Computes the full feature set on final_dataset.csv:

  Stage 1 — Fast surface features (12):
    1.  cosine_similarity        - MiniLM embedding similarity (handoff vs problem)
    2.  entity_overlap           - Jaccard overlap of spaCy entities/noun chunks
    3.  length_ratio             - len(handoff) / len(problem), word count
    4.  sentence_count_ratio     - sentences(handoff) / sentences(problem)
    5.  section_coverage         - fraction of expected structural sections present
    6.  verbatim_copy_rate       - fraction of handoff 4-grams appearing in problem
    7.  trailing_specificity     - noun/verb ratio of trailing 30% of handoff
    8.  constraint_count         - count of constraint tokens (must/shall/cannot/...)
    9.  role_pronoun_rate        - density of first-person pronouns
    10. novel_api_rate           - fraction of backtick identifiers absent from problem
    11. function_name_preserved  - binary: does entry_point appear verbatim in handoff
    12. signature_param_diff     - abs diff in argument counts, handoff vs problem

  Stage 2 — Deep semantic NLI features (2):
    13. nli_contradiction_max    - max P(contradiction | problem, sentence in handoff)
    14. nli_entailment_mean      - mean P(entailment | problem, sentence in handoff)

Run on Kaggle with GPU accelerator enabled (Settings > Accelerator > GPU T4 x2 or P100).

Install cell (run first in Kaggle):
    !pip install -q sentence-transformers spacy
    !python -m spacy download en_core_web_sm -q
"""

import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

# ----------------------------------------------------------------------
# 0. Config
# ----------------------------------------------------------------------

INPUT_PATH = "/kaggle/input/handoff-prm-data/final_dataset.csv"   # adjust to your Kaggle dataset path
ENTRY_POINT_MERGE_PATH = None  # e.g. "/kaggle/input/.../step5_corruption_results.csv" if you want to merge entry_point in
OUTPUT_PATH = "/kaggle/working/final_dataset_full_features.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

EXPECTED_SECTIONS = [
    r"\bobjective\b", r"\binput\b", r"\boutput\b",
    r"\bedge case", r"\balgorithm\b", r"\bconstraint",
]
CONSTRAINT_TOKENS = ["must", "shall", "cannot", "never", "strictly", "should", "require"]
FIRST_PERSON_PRONOUNS = {"i", "we", "my", "our", "me", "us"}


# ----------------------------------------------------------------------
# 1. Load models
# ----------------------------------------------------------------------

def load_models():
    print("Loading sentence-transformer (all-MiniLM-L6-v2)...")
    from sentence_transformers import SentenceTransformer
    embed_model = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)

    print("Loading spaCy (en_core_web_sm)...")
    import spacy
    nlp = spacy.load("en_core_web_sm")

    print("Loading NLI cross-encoder (cross-encoder/nli-deberta-v3-small)...")
    from sentence_transformers import CrossEncoder
    nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small", device=DEVICE)
    # label order for this checkpoint: 0=contradiction, 1=entailment, 2=neutral
    # (verify against model card if you swap checkpoints)

    return embed_model, nlp, nli_model


# ----------------------------------------------------------------------
# 2. Helper functions
# ----------------------------------------------------------------------

def extract_function_name(text: str) -> str:
    """Best-effort extraction of a function name from backtick-wrapped code or `def` statement."""
    m = re.search(r"`\s*def\s+(\w+)\s*\(", text)
    if m:
        return m.group(1)
    m = re.search(r"def\s+(\w+)\s*\(", text)
    if m:
        return m.group(1)
    m = re.search(r"`(\w+)\(", text)
    if m:
        return m.group(1)
    return None


def count_signature_params(text: str, func_name: str) -> int:
    """Counts comma-separated args in the first matching function signature."""
    if not func_name:
        return -1
    pattern = rf"{re.escape(func_name)}\s*\(([^)]*)\)"
    m = re.search(pattern, text)
    if not m:
        return -1
    args = [a for a in m.group(1).split(",") if a.strip()]
    return len(args)


def get_ngrams(text: str, n: int = 4) -> set:
    words = text.lower().split()
    return set(tuple(words[i:i + n]) for i in range(len(words) - n + 1)) if len(words) >= n else set()


def section_coverage(handoff: str) -> float:
    text = handoff.lower()
    hits = sum(1 for pat in EXPECTED_SECTIONS if re.search(pat, text))
    return hits / len(EXPECTED_SECTIONS)


def verbatim_copy_rate(handoff: str, problem: str) -> float:
    h_grams = get_ngrams(handoff, 4)
    p_grams = get_ngrams(problem, 4)
    if len(h_grams) == 0:
        return 0.0
    return len(h_grams & p_grams) / len(h_grams)


def trailing_specificity(handoff: str, nlp) -> float:
    words = handoff.split()
    if len(words) < 4:
        return 0.0
    tail = " ".join(words[int(len(words) * 0.7):])
    doc = nlp(tail)
    noun_verb = sum(1 for tok in doc if tok.pos_ in ("NOUN", "VERB", "PROPN"))
    total = max(len(doc), 1)
    return noun_verb / total


def constraint_count(handoff: str) -> int:
    text = handoff.lower()
    return sum(text.count(tok) for tok in CONSTRAINT_TOKENS)


def role_pronoun_rate(handoff: str) -> float:
    words = handoff.lower().split()
    if not words:
        return 0.0
    hits = sum(1 for w in words if w.strip(".,!?") in FIRST_PERSON_PRONOUNS)
    return hits / len(words)


def novel_api_rate(handoff: str, problem: str) -> float:
    h_idents = set(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", handoff))
    p_idents = set(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", problem)) | \
               set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", problem))
    if not h_idents:
        return 0.0
    novel = h_idents - p_idents
    return len(novel) / len(h_idents)


def split_sentences(text: str) -> list:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


# ----------------------------------------------------------------------
# 3. Main feature extraction
# ----------------------------------------------------------------------

def extract_all_features(df: pd.DataFrame, embed_model, nlp, nli_model) -> pd.DataFrame:
    df = df.copy()
    handoffs = df["handoff"].astype(str).tolist()
    problems = df["problem"].astype(str).tolist()

    # entry_point: use column if present, else regex-extract per row
    if "entry_point" in df.columns:
        entry_points = df["entry_point"].astype(str).tolist()
    else:
        print("No entry_point column found — extracting function names via regex from problem text.")
        entry_points = [extract_function_name(p) or extract_function_name(h)
                         for p, h in zip(problems, handoffs)]

    print(f"\nEmbedding {len(handoffs)} handoff/problem pairs (batched, GPU)...")
    handoff_emb = embed_model.encode(handoffs, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    problem_emb = embed_model.encode(problems, batch_size=64, show_progress_bar=True, convert_to_numpy=True)

    def cosine(a, b):
        num = np.sum(a * b, axis=1)
        denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        denom = np.where(denom == 0, 1e-8, denom)
        return num / denom

    cosine_similarity = cosine(handoff_emb, problem_emb)

    print("Computing spaCy-based + regex-based surface features (CPU loop)...")
    entity_overlap, length_ratio, sentence_count_ratio = [], [], []
    sec_cov, verb_copy, trail_spec, constr_cnt, pron_rate, novel_api = [], [], [], [], [], []
    func_preserved, sig_param_diff = [], []

    for h, p, ep in zip(handoffs, problems, entry_points):
        h_doc = nlp(h)
        p_doc = nlp(p)

        h_ents = set(e.text.lower().strip() for e in h_doc.ents) | \
                 set(c.text.lower().strip() for c in h_doc.noun_chunks)
        p_ents = set(e.text.lower().strip() for e in p_doc.ents) | \
                 set(c.text.lower().strip() for c in p_doc.noun_chunks)
        union = p_ents | h_ents
        entity_overlap.append(len(h_ents & p_ents) / len(union) if union else 0.0)

        h_words, p_words = len(h.split()), len(p.split())
        length_ratio.append(h_words / p_words if p_words > 0 else 0.0)

        h_sents, p_sents = split_sentences(h), split_sentences(p)
        sentence_count_ratio.append(len(h_sents) / len(p_sents) if p_sents else 0.0)

        sec_cov.append(section_coverage(h))
        verb_copy.append(verbatim_copy_rate(h, p))
        trail_spec.append(trailing_specificity(h, nlp))
        constr_cnt.append(constraint_count(h))
        pron_rate.append(role_pronoun_rate(h))
        novel_api.append(novel_api_rate(h, p))

        func_preserved.append(1.0 if (ep and ep in h) else 0.0)
        h_params = count_signature_params(h, ep)
        p_params = count_signature_params(p, ep)
        if h_params == -1 or p_params == -1:
            sig_param_diff.append(-1)
        else:
            sig_param_diff.append(abs(h_params - p_params))

    print("Computing NLI features (GPU, may take a while for long handoffs)...")
    nli_contradiction_max, nli_entailment_mean = [], []
    for h, p in zip(handoffs, problems):
        h_sents = split_sentences(h)
        if not h_sents:
            nli_contradiction_max.append(0.0)
            nli_entailment_mean.append(0.0)
            continue

        pairs = [[p, s] for s in h_sents]
        scores = nli_model.predict(pairs, batch_size=32, show_progress_bar=False)
        # scores shape: (n_sentences, 3) -> [contradiction, entailment, neutral]
        # softmax if raw logits returned
        scores = torch.softmax(torch.tensor(scores), dim=1).numpy()

        contradiction_probs = scores[:, 0]
        entailment_probs = scores[:, 1]

        nli_contradiction_max.append(float(np.max(contradiction_probs)))
        nli_entailment_mean.append(float(np.mean(entailment_probs)))

    df["cosine_similarity"] = cosine_similarity
    df["entity_overlap"] = entity_overlap
    df["length_ratio"] = length_ratio
    df["sentence_count_ratio"] = sentence_count_ratio
    df["section_coverage"] = sec_cov
    df["verbatim_copy_rate"] = verb_copy
    df["trailing_specificity"] = trail_spec
    df["constraint_count"] = constr_cnt
    df["role_pronoun_rate"] = pron_rate
    df["novel_api_rate"] = novel_api
    df["function_name_preserved"] = func_preserved
    df["signature_param_diff"] = sig_param_diff
    df["nli_contradiction_max"] = nli_contradiction_max
    df["nli_entailment_mean"] = nli_entailment_mean

    return df


# ----------------------------------------------------------------------
# 4. Main
# ----------------------------------------------------------------------

def main():
    print(f"Loading {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows, {df['task_id'].nunique()} unique tasks")

    if ENTRY_POINT_MERGE_PATH and "entry_point" not in df.columns:
        print(f"Merging entry_point from {ENTRY_POINT_MERGE_PATH}...")
        ep_df = pd.read_csv(ENTRY_POINT_MERGE_PATH)[["task_id", "entry_point"]].drop_duplicates("task_id")
        df = df.merge(ep_df, on="task_id", how="left")

    embed_model, nlp, nli_model = load_models()
    df_full = extract_all_features(df, embed_model, nlp, nli_model)

    df_full.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved full 14-feature dataset to {OUTPUT_PATH}")
    print(f"Final shape: {df_full.shape}")

    feature_cols = [
        "cosine_similarity", "entity_overlap", "length_ratio", "sentence_count_ratio",
        "section_coverage", "verbatim_copy_rate", "trailing_specificity", "constraint_count",
        "role_pronoun_rate", "novel_api_rate", "function_name_preserved", "signature_param_diff",
        "nli_contradiction_max", "nli_entailment_mean",
    ]
    print("\nFeature summary stats:")
    print(df_full[feature_cols].describe().T[["mean", "std", "min", "max"]])


if __name__ == "__main__":
    main()
