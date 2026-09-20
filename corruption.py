"""
Corruption engine: takes a successful handoff message and damages it in one
of several ways. Negative training labels get generated — no human annotation
needed. Re-run Agent B on the corrupted version; if it now fails, the
corruption removed something load-bearing.

Category 1 — Structural (length-reducing):
  truncation, over_summarization, tool_result_drop, entity_omission

Category 2 — Logic/Semantic (length-preserving):
  invert_objective, negate_edge_case, wrong_algorithm_name

Category 3 — Schema/Interface (length-preserving):
  signature_rename

Category 4 — Completeness (length-preserving):
  inject_false_constraint, fake_completion
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


# ─── Category 1: Structural (existing) ───────────────────────────────────────

def truncate(handoff: str, keep_fraction: float = 0.3) -> str:
    """Cuts the handoff at a fixed word-count fraction, mid-sentence."""
    words = handoff.split()
    keep_n = max(1, int(len(words) * keep_fraction))
    return " ".join(words[:keep_n])


def omit_entity(handoff: str, fraction_to_remove: float = 0.5, seed: int = None) -> str:
    """Removes sentences containing named entities or load-bearing patterns."""
    nlp = _get_nlp()
    doc = nlp(handoff)
    sentences = list(doc.sents)

    def is_load_bearing(sent) -> bool:
        text = sent.text
        has_real_entity = any(
            ent.label_ not in ("CARDINAL", "ORDINAL", "QUANTITY")
            for ent in sent.ents
        )
        has_signature = bool(re.search(r"\bdef\s+\w+\s*\(", text))
        has_edge_case_marker = bool(
            re.search(r"\bedge case|\bmust\b|\bshould\b|\brequire", text, re.IGNORECASE)
        )
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
    """Removes bullet/numbered lines and fenced/inline code spans."""
    text = re.sub(r"```.*?```", "", handoff, flags=re.DOTALL)
    text = re.sub(r"`[^`\n]+`", "", text)
    lines = text.split("\n")
    filtered = [
        line for line in lines
        if not re.match(r"^\s*([-*•]|\d+[.)])\s+", line)
    ]
    return "\n".join(filtered)


def over_summarize(handoff: str, sentence_count: int = 1) -> str:
    """Extractive summarization via TextRank."""
    parser = PlaintextParser.from_string(handoff, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    summary_sentences = summarizer(parser.document, sentence_count)
    if not summary_sentences:
        sentences = re.split(r"(?<=[.!?])\s+", handoff)
        return sentences[0] if sentences else handoff
    return " ".join(str(s) for s in summary_sentences)


# ─── Helpers for length-preserving corruptions ────────────────────────────────

def _split_code_prose(text: str):
    """Split handoff into alternating (prose, code) segments.
    Returns list of (segment_text, is_code) tuples."""
    segments = []
    pattern = re.compile(r"(```[\s\S]*?```|`[^`\n]+`)")
    last = 0
    for m in pattern.finditer(text):
        if m.start() > last:
            segments.append((text[last:m.start()], False))  # prose
        segments.append((m.group(), True))  # code
        last = m.end()
    if last < len(text):
        segments.append((text[last:], False))  # trailing prose
    return segments


def _rejoin_segments(segments):
    return "".join(s for s, _ in segments)


# ─── Category 2: Logic/Semantic (length-preserving) ──────────────────────────

# Ordered pairs: (pattern_to_find, replacement). Applied to PROSE only.
# Only the FIRST matching pair is used per handoff.
_INVERSION_PAIRS = [
    (r"\breturn\s+`?True`?\b",  "return False"),
    (r"\breturn\s+`?False`?\b", "return True"),
    (r"\bascending order\b",  "descending order"),
    (r"\bdescending order\b", "ascending order"),
    (r"\bascending\b",        "descending"),
    (r"\bdescending\b",       "ascending"),
    (r"\bmaximum\b",          "minimum"),
    (r"\bminimum\b",          "maximum"),
    (r"\blargest\b",          "smallest"),
    (r"\bsmallest\b",         "largest"),
    (r"\bgreatest\b",         "least"),
    (r"\bleast\b",            "greatest"),
    (r"\beven\b",             "odd"),
    (r"\bodd\b",              "even"),
    (r"\bsum\b",              "product"),
    (r"\bproduct\b",          "sum"),
    (r"\bincreasing\b",       "decreasing"),
    (r"\bdecreasing\b",       "increasing"),
    (r"\bTrue\b",             "False"),
    (r"\bFalse\b",            "True"),
]


def invert_objective(handoff: str) -> str:
    """Inverts the core return-value semantics of the handoff without
    changing text length. Applies to PROSE sections only (skips code blocks).
    Returns the original if no swappable term is found."""
    segments = _split_code_prose(handoff)
    for pattern, replacement in _INVERSION_PAIRS:
        # Check if pattern exists in any prose segment
        prose_text = "".join(s for s, is_code in segments if not is_code)
        if re.search(pattern, prose_text, flags=re.IGNORECASE):
            # Apply only to prose segments, first occurrence only
            replaced = False
            new_segments = []
            for seg_text, is_code in segments:
                if not is_code and not replaced:
                    new_text, count = re.subn(
                        pattern, replacement, seg_text,
                        count=1, flags=re.IGNORECASE
                    )
                    if count > 0:
                        replaced = True
                    new_segments.append((new_text, False))
                else:
                    new_segments.append((seg_text, is_code))
            return _rejoin_segments(new_segments)
    return handoff  # no applicable swap found → caller should skip


_ALGORITHM_SWAPS = [
    (r"\bbinary search\b",      "linear search"),
    (r"\blinear search\b",      "binary search"),
    (r"\bdynamic programming\b","greedy approach"),
    (r"\bgreedy\b",             "dynamic programming"),
    (r"\brecursive\b",          "iterative"),
    (r"\biterative\b",          "recursive"),
    (r"\bBFS\b",                "DFS"),
    (r"\bDFS\b",                "BFS"),
    (r"\bhash map\b",           "sorted list"),
    (r"\bhash set\b",           "sorted array"),
    (r"\btwo pointers\b",       "nested loops"),
    (r"\bsort.*ascending\b",    "sort in descending order"),
    (r"\bsort.*descending\b",   "sort in ascending order"),
]


def wrong_algorithm_name(handoff: str) -> str:
    """Swaps a key algorithm/data-structure keyword with a wrong alternative.
    Length-preserving (word-level swap). Returns original if no swap applies."""
    segments = _split_code_prose(handoff)
    for pattern, replacement in _ALGORITHM_SWAPS:
        prose_text = "".join(s for s, is_code in segments if not is_code)
        if re.search(pattern, prose_text, flags=re.IGNORECASE):
            replaced = False
            new_segments = []
            for seg_text, is_code in segments:
                if not is_code and not replaced:
                    new_text, count = re.subn(
                        pattern, replacement, seg_text,
                        count=1, flags=re.IGNORECASE
                    )
                    if count > 0:
                        replaced = True
                    new_segments.append((new_text, False))
                else:
                    new_segments.append((seg_text, is_code))
            return _rejoin_segments(new_segments)
    return handoff


_NEGATION_PATTERNS = [
    # (pattern_to_match, replacement_template)
    (r"\bmust handle\b",           "need not handle"),
    (r"\bshould handle\b",         "should not handle"),
    (r"\bmust return\b",           "must not return"),
    (r"\bshould return\b",         "should not return"),
    (r"\bensure that\b",           "do not ensure that"),
    (r"\bhandle the case\b",       "ignore the case"),
    (r"\bhandle empty\b",          "ignore empty"),
    (r"\bhandle None\b",           "ignore None"),
    (r"\bhandle negative\b",       "ignore negative"),
    (r"\bcase-insensitive\b",      "case-sensitive"),
    (r"\bignore.*spaces\b",        "preserve spaces"),
    (r"\bignore.*case\b",          "preserve case"),
]


def negate_edge_case(handoff: str) -> str:
    """Negates one explicit constraint sentence in the handoff.
    Length-preserving (word-level swap). Returns original if no pattern applies."""
    segments = _split_code_prose(handoff)
    for pattern, replacement in _NEGATION_PATTERNS:
        prose_text = "".join(s for s, is_code in segments if not is_code)
        if re.search(pattern, prose_text, flags=re.IGNORECASE):
            replaced = False
            new_segments = []
            for seg_text, is_code in segments:
                if not is_code and not replaced:
                    new_text, count = re.subn(
                        pattern, replacement, seg_text,
                        count=1, flags=re.IGNORECASE
                    )
                    if count > 0:
                        replaced = True
                    new_segments.append((new_text, False))
                else:
                    new_segments.append((seg_text, is_code))
            return _rejoin_segments(new_segments)
    return handoff


# ─── Category 3: Schema/Interface (length-preserving) ────────────────────────

def signature_rename(handoff: str, entry_point: str = None) -> str:
    """Renames the function defined in the handoff to a wrong name.
    Agent B implements the wrong function name → NameError in test harness.
    Requires entry_point to be passed; falls back to regex-detected name.
    ALWAYS produces a valid label=0 (test harness looks for original name)."""
    if entry_point:
        wrong_name = entry_point + "_impl"
        # Replace in def lines and inline references
        result = re.sub(
            r"\b" + re.escape(entry_point) + r"\b",
            wrong_name,
            handoff
        )
        if result != handoff:
            return result
    # Fallback: find any def line and rename
    m = re.search(r"(def\s+)(\w+)(\s*\()", handoff)
    if m:
        original_name = m.group(2)
        wrong_name = original_name + "_impl"
        return handoff.replace(original_name, wrong_name, 1)
    return handoff


# ─── Category 4: Completeness (length-preserving) ────────────────────────────

_FALSE_CONSTRAINTS = [
    "Note: The input is guaranteed to be non-empty; no empty-input handling is needed.",
    "Assumption: The function does not need to handle None or null inputs.",
    "Constraint: All inputs are guaranteed to be positive integers; negative values will not occur.",
    "Note: Duplicate values will not appear in the input; uniqueness is guaranteed.",
    "Assumption: The function should raise a ValueError for any edge-case input.",
    "Constraint: The input string will never contain spaces or special characters.",
]


def inject_false_constraint(handoff: str, seed: int = None) -> str:
    """Appends a false constraint and removes an equal number of words
    from the end of the handoff to preserve length.
    The false constraint contradicts typical test-harness edge cases."""
    rng = random.Random(seed if seed is not None else hash(handoff) % (2**32))
    constraint = rng.choice(_FALSE_CONSTRAINTS)
    constraint_words = constraint.split()
    handoff_words = handoff.split()

    if len(handoff_words) <= len(constraint_words) + 10:
        return handoff  # too short to safely modify

    # Remove words from end, add false constraint
    trimmed = handoff_words[: -len(constraint_words)]
    result_words = trimmed + constraint_words
    return " ".join(result_words)


_FILLER_SENTENCES = [
    "The implementation should follow standard Python coding conventions.",
    "Performance and readability should be balanced in the solution.",
    "Additional edge cases may be handled at the implementer's discretion.",
    "Standard library functions are available and may be used freely.",
    "The solution should be clean, well-organized, and maintainable.",
    "Error handling beyond the specified cases is left to the implementer.",
    "Python best practices and idiomatic style are encouraged throughout.",
    "The function signature must match the specification provided above.",
]


def fake_completion(handoff: str, seed: int = None) -> str:
    """Replaces the last ~25% of the handoff (where edge-case details live)
    with vague, non-informative completion language, preserving total length."""
    words = handoff.split()
    if len(words) < 20:
        return handoff

    cut = int(len(words) * 0.75)
    kept_words = words[:cut]
    n_to_fill = len(words) - cut

    rng = random.Random(seed if seed is not None else hash(handoff) % (2**32))
    filler_words = []
    while len(filler_words) < n_to_fill:
        sentence = rng.choice(_FILLER_SENTENCES)
        filler_words.extend(sentence.split())

    filler_words = filler_words[:n_to_fill]
    return " ".join(kept_words + filler_words)


# ─── Registry ─────────────────────────────────────────────────────────────────

CORRUPTION_FUNCTIONS = {
    # Category 1: Structural (length-reducing)
    "truncation":           truncate,
    "entity_omission":      omit_entity,
    "tool_result_drop":     drop_tool_result,
    "over_summarization":   over_summarize,
    # Category 2: Logic/Semantic (length-preserving)
    "invert_objective":     invert_objective,
    "negate_edge_case":     negate_edge_case,
    "wrong_algorithm_name": wrong_algorithm_name,
    # Category 3: Schema/Interface (length-preserving)
    "signature_rename":     signature_rename,
    # Category 4: Completeness (length-preserving)
    "inject_false_constraint": inject_false_constraint,
    "fake_completion":      fake_completion,
}

# Corruptions safe to use in random rollout sampling (exclude schema that needs entry_point arg)
RANDOM_SAMPLE_CORRUPTIONS = [
    "truncation", "entity_omission", "tool_result_drop", "over_summarization",
    "invert_objective", "negate_edge_case", "inject_false_constraint", "fake_completion",
]


def corrupt_handoff(handoff: str, corruption_type: str = None,
                   entry_point: str = None) -> tuple[str, str]:
    """
    Applies one corruption type to a handoff. Returns (corrupted_text, corruption_type_used).
    Pass entry_point for signature_rename to work correctly.
    """
    if corruption_type is None:
        corruption_type = random.choice(RANDOM_SAMPLE_CORRUPTIONS)
    fn = CORRUPTION_FUNCTIONS[corruption_type]
    if corruption_type == "signature_rename":
        return fn(handoff, entry_point=entry_point), corruption_type
    return fn(handoff), corruption_type