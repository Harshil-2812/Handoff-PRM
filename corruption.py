"""
Corruption engine: takes a successful handoff message and damages it in one
of four ways. This is how negative training labels get generated -- no
human annotation needed. Re-run Agent B on the corrupted version; if it now
fails, the corruption removed something load-bearing.
"""

import random
import re

import spacy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

_nlp = None

def _get_nlp():
    global _nlp
    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")
    return _nlp

def truncate(handoff: str, keep_fraction: float = 0.3) -> str:
    """Cuts the handoff at a fixed word-count fraction, mid-sentence if that's
    where the cutoff lands -- this is what real token-budget truncation does,
    unlike the earlier sentence-respecting version."""
    words = handoff.split()
    keep_n = max(1, int(len(words) * keep_fraction))
    return " ".join(words[:keep_n])



def omit_entity(handoff: str, fraction_to_remove: float = 0.5, seed: int = None) -> str:
    """Removes sentences that are either (a) genuine named entities per
    spaCy's NER (proper nouns, specific terms -- excludes bare numbers) or
    (b) structurally load-bearing: contain a function signature pattern
    (`def name(...)`) or an explicit edge-case marker. This targets content
    that's actually likely to matter, rather than any sentence with a digit."""
    nlp = _get_nlp()
    doc = nlp(handoff)
    sentences = list(doc.sents)
 
    def is_load_bearing(sent) -> bool:
        text = sent.text
        # real named entities (excludes plain numbers/quantities)
        has_real_entity = any(ent.label_ not in ("CARDINAL", "ORDINAL", "QUANTITY") for ent in sent.ents)
        # function signature pattern
        has_signature = bool(re.search(r"\bdef\s+\w+\s*\(", text))
        # explicit edge-case language
        has_edge_case_marker = bool(re.search(r"\bedge case|\bmust\b|\bshould\b|\brequire", text, re.IGNORECASE))
        return has_real_entity or has_signature or has_edge_case_marker
 
    entity_bearing_idx = [i for i, sent in enumerate(sentences) if is_load_bearing(sent)]
 
    if not entity_bearing_idx:
        return handoff
 
    rng = random.Random(seed if seed is not None else hash(handoff) % (2**32))
    n_to_remove = max(1, int(len(entity_bearing_idx) * fraction_to_remove))
    remove_idx = set(rng.sample(entity_bearing_idx, min(n_to_remove, len(entity_bearing_idx))))
 
    kept = [sent.text for i, sent in enumerate(sentences) if i not in remove_idx]
    return " ".join(kept)
 

def drop_tool_result(handoff: str) -> str:
    """Removes bullet/numbered lines AND fenced/inline code spans -- the
    places structured tool output or specific values are most likely to
    live in a handoff message."""
    text = re.sub(r"```.*?```", "", handoff, flags=re.DOTALL)  # fenced code blocks
    text = re.sub(r"`[^`]+`", "", text)                          # inline code spans
    lines = text.split("\n")
    filtered = [
        line for line in lines
        if not re.match(r"^\s*([-*•]|\d+[.)])\s+", line)
    ]
    return "\n".join(filtered)

def over_summarize(handoff: str, sentence_count: int = 1) -> str:
    """Extractive summarization via TextRank -- deterministic, picks the
    `sentence_count` most central sentences rather than always the first."""
    parser = PlaintextParser.from_string(handoff, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary_sentences = summarizer(parser.document, sentence_count)
    if not summary_sentences:
        # fallback for very short handoffs TextRank can't process
        sentences = re.split(r"(?<=[.!?])\s+", handoff)
        return sentences[0] if sentences else handoff
    return " ".join(str(s) for s in summary_sentences)


CORRUPTION_FUNCTIONS = {
    "truncation": truncate,
    "entity_omission": omit_entity,
    "tool_result_drop": drop_tool_result,
    "over_summarization": over_summarize,
}


def corrupt_handoff(handoff: str, corruption_type: str = None) -> tuple[str, str]:
    """
    Applies one corruption type to a handoff. If corruption_type is None,
    picks one at random -- useful for building a balanced dataset across
    all four types without manually looping.

    Returns (corrupted_text, corruption_type_used).
    """
    if corruption_type is None:
        corruption_type = random.choice(list(CORRUPTION_FUNCTIONS.keys()))
    fn = CORRUPTION_FUNCTIONS[corruption_type]
    return fn(handoff), corruption_type
