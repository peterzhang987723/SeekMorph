"""Query morphing, entity/code extraction, and adaptive intent routing for search + recommendation."""

from __future__ import annotations

import re
from dataclasses import dataclass


_WORD_RE = re.compile(r"[A-Za-z0-9_.\-]+")
_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")
_EXACT_CODE_RE = re.compile(r"\b(?:[A-Z]{2,}[-_][A-Z0-9_.\-]+|v?\d+\.\d+(?:\.\d+)?|#[0-9]+|[A-Za-z0-9_]+::[A-Za-z0-9_]+)\b")


def tokenize_query(text: str) -> list[str]:
    """Tokenize English identifiers and CJK unigrams + bigrams."""
    lowered = text.lower()
    tokens = [w.strip(".-") for w in _WORD_RE.findall(lowered) if w.strip(".-")]
    for run in _CJK_RUN_RE.findall(lowered):
        tokens.extend(list(run))
        if len(run) > 1:
            tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
    return tokens


@dataclass(frozen=True)
class QueryPlan:
    """Structured query plan with intent classification and adaptive channel weights."""

    raw_query: str
    tokens: tuple[str, ...]
    exact_identifiers: tuple[str, ...]
    intent: str  # "exact_lookup" | "semantic_qa" | "exploratory_rec"
    w_lexical: float
    w_semantic: float
    w_personal: float
    w_recency: float


def analyze_query(query: str, *, has_user_profile: bool = True) -> QueryPlan:
    """Analyze a raw search query and compute adaptive weights across recall and recommendation signals."""
    cleaned = query.strip()
    tokens = tuple(tokenize_query(cleaned))
    exact_ids = tuple(m.group(0).lower() for m in _EXACT_CODE_RE.finditer(cleaned))

    word_count = len([w.strip(".-") for w in _WORD_RE.findall(cleaned) if w.strip(".-")])
    cjk_char_count = sum(len(run) for run in _CJK_RUN_RE.findall(cleaned))
    is_short_exploratory = (word_count <= 2 and cjk_char_count == 0) or (cjk_char_count <= 4 and word_count == 0)

    if exact_ids:
        intent = "exact_lookup"
        w_lex, w_sem, w_pers, w_rec = (0.55, 0.20, 0.15 if has_user_profile else 0.0, 0.10)
    elif is_short_exploratory and has_user_profile:
        intent = "exploratory_rec"
        w_lex, w_sem, w_pers, w_rec = (0.25, 0.30, 0.30, 0.15)
    else:
        intent = "semantic_qa"
        w_lex, w_sem, w_pers, w_rec = (0.35, 0.40, 0.15 if has_user_profile else 0.0, 0.10)

    total = w_lex + w_sem + w_pers + w_rec
    return QueryPlan(
        raw_query=cleaned,
        tokens=tokens,
        exact_identifiers=exact_ids,
        intent=intent,
        w_lexical=round(w_lex / total, 4),
        w_semantic=round(w_sem / total, 4),
        w_personal=round(w_pers / total, 4),
        w_recency=round(w_rec / total, 4),
    )
