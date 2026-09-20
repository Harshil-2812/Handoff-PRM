"""
100% Self-Contained Cloud GPU Feature Extractor for Google Colab / Kaggle.
No other files or folders needed — only rollouts.csv!

Takes ~35-45 seconds on a free Google Colab T4 GPU.
"""

import os
import re
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

INPUT_CSV = "rollouts.csv"
OUTPUT_CSV = "features_with_nli.csv"


# ─── Model Singletons ─────────────────────────────────────────────────────────

_st_model = None
_nli_model = None
_nlp = None

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def print_hardware_info():
    cuda_avail = torch.cuda.is_available()
    print("=" * 65)
    print("Handoff-PRM Cloud GPU Feature Extractor")
    print(f"CUDA Available: {cuda_avail}")
    if cuda_avail:
        print(f"Hardware: GPU ({torch.cuda.get_device_name(0)})")
        print(f"VRAM Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("Hardware: CPU  ⚠️ (Colab session is not using GPU yet!)")
        print("Tip: Click 'Runtime' -> 'Restart session' in the top menu.")
def get_st_model():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        dev = get_device()
        print(f"Loading SentenceTransformer on {dev.upper()}...")
        _st_model = SentenceTransformer("all-MiniLM-L6-v2", device=dev)
    return _st_model

def get_nli_model():
    global _nli_model
    if _nli_model is None:
        from sentence_transformers import CrossEncoder
        dev = get_device()
        print(f"Loading NLI DeBERTa CrossEncoder on {dev.upper()}...")
        _nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-small", device=dev)
    return _nli_model

def get_nlp():
    global _nlp
    if _nlp is None:
        import spacy
        try:
            _nlp = spacy.load("en_core_web_sm")
        except Exception:
            from spacy.cli import download
            download("en_core_web_sm")
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


# ─── Feature Computation Functions ───────────────────────────────────────────

def compute_length_ratio(handoff: str, reference_text: str) -> float:
    ref_len = len(reference_text.split())
    return len(handoff.split()) / ref_len if ref_len > 0 else 1.0

def compute_sentence_count_ratio(handoff: str, reference_text: str) -> float:
    def count_sents(text):
        return max(1, len([s for s in re.split(r'[.!?]+', text) if s.strip()]))
    return count_sents(handoff) / count_sents(reference_text)

def compute_section_coverage(handoff: str, reference_text: str, entry_point: str = None) -> float:
    h = handoff.lower()
    has_signature = bool(re.search(r'def\s+\w+\s*\(', handoff))
    has_edge_cases = bool(re.search(r'\b(edge case|empty|none|null|raise|zero|negative|boundary)\b', h))
    has_algorithm = bool(re.search(r'\b(approach|algorithm|implement|use|strategy|method|steps?|procedure)\b', h))
    has_return = 'return' in h
    prob_words = set(reference_text.lower().split()) - {'a', 'an', 'the', 'is', 'are', 'of', 'in', 'to', 'and', 'or', 'for'}
    h_words = set(h.split())
    has_restatement = len(prob_words & h_words) >= 3
    return sum([has_signature, has_edge_cases, has_algorithm, has_return, has_restatement]) / 5.0

def compute_verbatim_copy_rate(handoff: str, reference_text: str) -> float:
    def get_ngrams(text, n=4):
        words = text.lower().split()
        return set(tuple(words[i:i+n]) for i in range(len(words) - n + 1)) if len(words) >= n else set()
    h_ngrams = get_ngrams(handoff)
    r_ngrams = get_ngrams(reference_text)
    return len(h_ngrams & r_ngrams) / len(h_ngrams) if h_ngrams else 0.0

def compute_trailing_specificity(handoff: str, reference_text: str) -> float:
    words = handoff.split()
    if len(words) < 10:
        return 0.0
    tail = ' '.join(words[int(len(words) * 0.7):])
    specifics = len(re.findall(r'\b\d+\b', tail)) + len(re.findall(r'`[^`]+`', tail)) + len(re.findall(r'\b(return|none|true|false|empty|null|raise|error|zero|negative)\b', tail.lower()))
    return specifics / max(len(tail.split()), 1)

def compute_constraint_count(handoff: str, reference_text: str) -> float:
    sentences = re.split(r'[.!?]+', handoff)
    words = ['must', 'should', 'require', 'ensure', 'handle', 'edge case', 'assume', 'note:', 'constraint', 'guarantee']
    count = sum(1 for s in sentences if any(w in s.lower() for w in words))
    return count / (max(len(reference_text.split()), 1) / 10.0)

def compute_role_pronoun_rate(handoff: str, reference_text: str) -> float:
    words = handoff.lower().split()
    if not words:
        return 0.0
    first_person = sum(1 for w in words if w in {"i", "i've", "i'll", "i'm", "i'd", "my", "me", "we", "we've", "we'll", "our", "ourselves"})
    return first_person / len(words)

def compute_bleu_1(handoff: str, reference_text: str) -> float:
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        return float(sentence_bleu([reference_text.lower().split()], handoff.lower().split(), weights=(1, 0, 0, 0), smoothing_function=SmoothingFunction().method1))
    except Exception:
        return 0.0

def compute_rouge_l(handoff: str, reference_text: str) -> float:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        return float(scorer.score(reference_text, handoff)['rougeL'].fmeasure)
    except Exception:
        return 0.0

def compute_novel_api_rate(handoff: str, reference_text: str) -> float:
    h_apis = set(re.findall(r'`([^`\n]+)`', handoff))
    r_apis = set(re.findall(r'`([^`\n]+)`', reference_text))
    return len(h_apis - r_apis) / len(h_apis) if h_apis else 0.0

def compute_function_name_preserved(handoff: str, reference_text: str, entry_point: str = None) -> float:
    if not entry_point:
        m = re.search(r'def\s+(\w+)\s*\(', reference_text)
        entry_point = m.group(1) if m else None
    if not entry_point:
        return 0.5
    return 1.0 if entry_point in handoff else 0.0

def compute_signature_param_diff(handoff: str, reference_text: str) -> float:
    m = re.search(r'def\s+(\w+)\s*\((.*?)\)', handoff)
    if not m:
        return 0.0
    params = [p.strip() for p in m.group(2).split(',') if p.strip()]
    ref_m = re.search(r'def\s+(\w+)\s*\((.*?)\)', reference_text)
    if ref_m:
        ref_params = [p.strip() for p in ref_m.group(2).split(',') if p.strip()]
        return float(abs(len(params) - len(ref_params)))
    return 0.0


# ─── Main Batch Extraction Pipeline ──────────────────────────────────────────

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: '{INPUT_CSV}' not found! Please upload rollouts.csv.")
        return

    print_hardware_info()

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rollouts from {INPUT_CSV}")

    handoffs = [str(x) for x in df["handoff"].tolist()]
    problems = [str(x) for x in df["problem"].tolist()]

    # 1. Batch SentenceTransformer embeddings (fast vector dot-product)
    print("\n[1/4] Batch encoding text embeddings with SentenceTransformer...")
    st = get_st_model()
    h_embs = st.encode(handoffs, batch_size=256, normalize_embeddings=True, show_progress_bar=True)
    p_embs = st.encode(problems, batch_size=256, normalize_embeddings=True, show_progress_bar=True)
    cosine_sims = np.sum(h_embs * p_embs, axis=1)

    # 2. Batch spaCy entity parsing on unique problems
    print("\n[2/4] Parsing entity terms with spaCy...")
    unique_problems = list(set(problems))
    nlp = get_nlp()
    ref_terms_dict = {}
    for doc, prob_text in zip(nlp.pipe(unique_problems, batch_size=64), unique_problems):
        terms = {chunk.text.lower() for chunk in doc.noun_chunks} | {ent.text.lower() for ent in doc.ents}
        ref_terms_dict[prob_text] = terms

    # 3. Batch NLI sentence pairs on GPU
    print("\n[3/4] Batch scoring NLI Contradiction & Entailment with DeBERTa...")
    row_nli_indices = []
    all_nli_pairs = []

    for h_text, p_text in zip(handoffs, problems):
        p_sents = [s.strip() for s in re.split(r'[.!?]+', p_text) if len(s.strip().split()) >= 4][:3]
        h_sents = [s.strip() for s in re.split(r'[.!?]+', h_text) if len(s.strip().split()) >= 4][:3]

        start_idx = len(all_nli_pairs)
        for ps in p_sents:
            for hs in h_sents:
                all_nli_pairs.append((ps, hs))
        end_idx = len(all_nli_pairs)
        row_nli_indices.append((start_idx, end_idx))

    print(f"Scoring {len(all_nli_pairs)} NLI sentence pairs across all rollouts...")
    import gc
    torch.cuda.empty_cache()
    gc.collect()

    nli = get_nli_model()
    nli_scores = nli.predict(all_nli_pairs, batch_size=64, show_progress_bar=True)
    if len(nli_scores.shape) == 1:
        nli_scores = np.expand_dims(nli_scores, axis=0)
    exp_s = np.exp(nli_scores - np.max(nli_scores, axis=1, keepdims=True))
    nli_probs = exp_s / np.sum(exp_s, axis=1, keepdims=True)

    # 4. Assemble final feature DataFrame
    print("\n[4/4] Assembling full 15-feature matrix...")
    rows = []
    for idx, (_, row) in tqdm(enumerate(df.iterrows()), total=len(df), desc="assembling features"):
        handoff = handoffs[idx]
        problem = problems[idx]
        task_id = row["task_id"]

        ref_terms = ref_terms_dict.get(problem, set())
        if not ref_terms:
            entity_overlap = 1.0
        else:
            handoff_lower = handoff.lower()
            entity_overlap = sum(1 for term in ref_terms if term in handoff_lower) / len(ref_terms)

        s_idx, e_idx = row_nli_indices[idx]
        if e_idx > s_idx:
            row_probs = nli_probs[s_idx:e_idx]
            nli_contra = float(np.max(row_probs[:, 0]))
            nli_ent    = float(np.mean(row_probs[:, 1]))
        else:
            nli_contra, nli_ent = 0.0, 0.5

        # Infer entry_point via regex
        m = re.search(r'def\s+(\w+)\s*\(', problem)
        entry_point = m.group(1) if m else None

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
            "signature_param_diff":     compute_signature_param_diff(handoff, problem),
            "nli_contradiction_max":    nli_contra,
            "nli_entailment_mean":      nli_ent,
            "task_id":                  task_id,
            "source":                   row.get("source", "unknown"),
            "corruption_type":          row["corruption_type"],
            "label":                    row["label"],
        }
        rows.append(feats)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print("\n" + "=" * 65)
    print(f"SUCCESS! Saved {len(out_df)} rows with 15 features to '{OUTPUT_CSV}'.")
    print("=" * 65)

    # Auto-trigger download in Google Colab if running in Colab
    try:
        from google.colab import files
        print("\nTriggering automatic download of features_with_nli.csv...")
        files.download(OUTPUT_CSV)
    except Exception:
        pass


if __name__ == "__main__":
    main()
