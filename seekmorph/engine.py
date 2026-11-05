"""Enterprise knowledge base search and personalized recommendation reranking engine."""

from __future__ import annotations

import json
import math
import threading
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from seekmorph.query import QueryPlan, analyze_query, tokenize_query


@dataclass(frozen=True)
class KnowledgeItem:
    """Retrievable knowledge base item with ACL tags, behavioral stats, and age."""

    item_id: str
    title: str
    content: str
    tags: tuple[str, ...] = ()
    tenant_id: str = "default"
    allowed_roles: tuple[str, ...] = ()
    age_days: float = 0.0
    impressions: int = 0
    clicks: int = 0
    embedding: tuple[float, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def smoothed_ctr(self, prior_clicks: float = 1.0, prior_impressions: float = 20.0) -> float:
        """Bayesian-smoothed click-through rate (CTR)."""
        return (max(0, self.clicks) + prior_clicks) / (max(0, self.impressions) + prior_impressions)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["tags"] = list(self.tags)
        data["allowed_roles"] = list(self.allowed_roles)
        data["embedding"] = list(self.embedding)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KnowledgeItem:
        """Deserialize a KnowledgeItem from SeekMorph JSON or a DocMorph chunk JSONL row."""
        item_id = str(data.get("item_id") or data.get("chunk_id") or "")
        doc_id = str(data.get("doc_id") or "")
        section_path = data.get("section_path") or ()
        if "title" in data and data["title"]:
            title = str(data["title"])
        elif doc_id and section_path:
            title = f"{doc_id} ({' > '.join(str(s) for s in section_path)})"
        else:
            title = doc_id or item_id

        raw_tags = list(data.get("tags") or [])
        if "kind" in data and data["kind"] and data["kind"] not in raw_tags:
            raw_tags.append(str(data["kind"]))
        for sec in section_path:
            sec_str = str(sec).strip().lower()
            if sec_str and sec_str not in raw_tags:
                raw_tags.append(sec_str)

        tenant_id = str(data.get("tenant_id") or data.get("namespace") or "default")
        return cls(
            item_id=item_id,
            title=title,
            content=str(data["content"]),
            tags=tuple(raw_tags),
            tenant_id=tenant_id,
            allowed_roles=tuple(data.get("allowed_roles") or ()),
            age_days=float(data.get("age_days", 0.0)),
            impressions=int(data.get("impressions", 0)),
            clicks=int(data.get("clicks", 0)),
            embedding=tuple(float(x) for x in (data.get("embedding") or ())),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(frozen=True)
class UserProfile:
    """User context and behavioral history for personalized search and recommendation."""

    user_id: str
    tenant_id: str = "default"
    roles: tuple[str, ...] = ()
    interest_tags: dict[str, float] = field(default_factory=dict)
    clicked_item_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RankedResult:
    """Ranked item returned by SeekMorphEngine with per-channel score breakdown."""

    item: KnowledgeItem
    final_score: float
    lexical_score: float
    semantic_score: float
    personal_score: float
    recency_score: float
    query_intent: str


class SeekMorphEngine:
    """Thread-safe inverted-index search and personalized recommendation engine with pre-retrieval ACL."""

    def __init__(self, rrf_k: int = 60, recency_half_life_days: float = 90.0) -> None:
        self.rrf_k = max(1, int(rrf_k))
        self.recency_half_life_days = max(1.0, float(recency_half_life_days))
        self._lock = threading.RLock()
        self._items: dict[str, KnowledgeItem] = {}
        self._doc_lens: dict[str, int] = {}
        self._postings: dict[str, dict[str, int]] = {}
        self._total_dl: int = 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def _unindex_item(self, item_id: str) -> None:
        old_len = self._doc_lens.pop(item_id, 0)
        self._total_dl = max(0, self._total_dl - old_len)
        empty_terms: list[str] = []
        for term, plist in self._postings.items():
            if item_id in plist:
                del plist[item_id]
                if not plist:
                    empty_terms.append(term)
        for term in empty_terms:
            del self._postings[term]

    def add(self, item: KnowledgeItem) -> None:
        """Insert or update a KnowledgeItem and update the inverted index in O(|tokens|) time."""
        with self._lock:
            if item.item_id in self._items:
                self._unindex_item(item.item_id)
            self._items[item.item_id] = item
            tokens = tokenize_query(f"{item.title} {' '.join(item.tags)} {item.content}")
            tf = Counter(tokens)
            dl = len(tokens)
            self._doc_lens[item.item_id] = dl
            self._total_dl += dl
            for term, freq in tf.items():
                self._postings.setdefault(term, {})[item.item_id] = freq

    def remove(self, item_id: str) -> bool:
        """Remove an item by ID from the engine."""
        with self._lock:
            if item_id not in self._items:
                return False
            del self._items[item_id]
            self._unindex_item(item_id)
            return True

    def add_many(self, items: Iterable[KnowledgeItem]) -> int:
        count = 0
        with self._lock:
            for item in items:
                self.add(item)
                count += 1
        return count

    def record_feedback(self, item_id: str, *, clicked: bool = False) -> None:
        """Record an impression and optional click event without re-tokenizing text."""
        with self._lock:
            curr = self._items.get(item_id)
            if curr is None:
                return
            self._items[item_id] = KnowledgeItem(
                item_id=curr.item_id,
                title=curr.title,
                content=curr.content,
                tags=curr.tags,
                tenant_id=curr.tenant_id,
                allowed_roles=curr.allowed_roles,
                age_days=curr.age_days,
                impressions=curr.impressions + 1,
                clicks=curr.clicks + (1 if clicked else 0),
                embedding=curr.embedding,
                metadata=curr.metadata,
            )

    def _filter_by_acl(self, user: UserProfile | None, tenant_id: str) -> list[KnowledgeItem]:
        """Pre-retrieval ACL gate: isolate tenant_id and enforce allowed_roles before scoring."""
        eff_tenant = user.tenant_id if user is not None else tenant_id
        user_roles = set(user.roles) if user is not None else set()

        authorized: list[KnowledgeItem] = []
        for item in self._items.values():
            if item.tenant_id != eff_tenant:
                continue
            if item.allowed_roles and not (user_roles & set(item.allowed_roles)):
                continue
            authorized.append(item)
        return authorized

    def search_and_recommend(
        self,
        query: str,
        *,
        user: UserProfile | None = None,
        tenant_id: str = "default",
        query_embedding: list[float] | tuple[float, ...] | None = None,
        top_k: int = 5,
    ) -> list[RankedResult]:
        """Execute ACL pre-filtering, inverted-index BM25 + dense recall, and personalized/temporal reranking."""
        with self._lock:
            candidates = self._filter_by_acl(user, tenant_id)
            if not candidates or top_k <= 0:
                return []

            allowed_ids = {it.item_id for it in candidates}
            n_docs = len(candidates)
            if n_docs == len(self._items):
                avgdl = self._total_dl / max(1, n_docs)
            else:
                avgdl = sum(self._doc_lens.get(cid, 0) for cid in allowed_ids) / max(1, n_docs)

            plan: QueryPlan = analyze_query(query, has_user_profile=user is not None)
            q_tokens = list(plan.tokens)

            raw_bm25: dict[str, float] = {cid: 0.0 for cid in allowed_ids}
            for q in q_tokens:
                plist = self._postings.get(q)
                if not plist:
                    continue
                if n_docs == len(self._items):
                    df = len(plist)
                else:
                    df = sum(1 for cid in plist if cid in allowed_ids)
                if df == 0:
                    continue
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                for cid, tf in plist.items():
                    if cid not in allowed_ids:
                        continue
                    dl = self._doc_lens.get(cid, 0)
                    denom = tf + 1.5 * (1.0 - 0.75 + 0.75 * (dl / max(1.0, avgdl)))
                    raw_bm25[cid] += idf * (tf * 2.5) / denom

            if plan.exact_identifiers:
                for it in candidates:
                    for exact_id in plan.exact_identifiers:
                        if exact_id in it.title.lower() or exact_id in it.content.lower():
                            raw_bm25[it.item_id] += 3.0

            max_bm25 = max(raw_bm25.values()) if raw_bm25 else 0.0
            norm_lexical = {
                cid: (val / max_bm25 if max_bm25 > 0.0 else 0.0) for cid, val in raw_bm25.items()
            }

            norm_semantic: dict[str, float] = {}
            if query_embedding:
                q_norm = math.sqrt(sum(x * x for x in query_embedding)) or 1.0
                for it in candidates:
                    if it.embedding and len(it.embedding) == len(query_embedding):
                        d_norm = math.sqrt(sum(x * x for x in it.embedding)) or 1.0
                        cos = sum(a * b for a, b in zip(query_embedding, it.embedding)) / (q_norm * d_norm)
                        norm_semantic[it.item_id] = max(0.0, cos)
                    else:
                        norm_semantic[it.item_id] = 0.0
            else:
                norm_semantic = dict(norm_lexical)

            clicked_tags: Counter[str] = Counter()
            if user is not None and user.clicked_item_ids:
                for cid in user.clicked_item_ids:
                    clicked_item = self._items.get(cid)
                    if clicked_item is not None:
                        for t in clicked_item.tags:
                            clicked_tags[t.lower()] += 1

            raw_personal: dict[str, float] = {}
            for it in candidates:
                p_score = it.smoothed_ctr()
                if user is not None:
                    for t in it.tags:
                        tl = t.lower()
                        p_score += float(user.interest_tags.get(tl, user.interest_tags.get(t, 0.0)))
                        if clicked_tags[tl] > 0:
                            p_score += 0.25 * math.log1p(clicked_tags[tl])
                raw_personal[it.item_id] = p_score

            max_pers = max(raw_personal.values()) if raw_personal else 0.0
            norm_personal = {
                cid: (val / max_pers if max_pers > 0.0 else 0.0) for cid, val in raw_personal.items()
            }

            decay_lambda = math.log(2.0) / self.recency_half_life_days
            norm_recency = {
                it.item_id: math.exp(-decay_lambda * max(0.0, it.age_days)) for it in candidates
            }

            results: list[RankedResult] = []
            for it in candidates:
                cid = it.item_id
                lex = norm_lexical[cid]
                sem = norm_semantic[cid]
                pers = norm_personal[cid]
                rec = norm_recency[cid]

                relevance_gate = max(lex, sem) if q_tokens else 1.0
                final = (
                    plan.w_lexical * lex
                    + plan.w_semantic * sem
                    + (plan.w_personal * pers + plan.w_recency * rec) * (0.25 + 0.75 * relevance_gate)
                )
                results.append(
                    RankedResult(
                        item=it,
                        final_score=round(final, 6),
                        lexical_score=round(lex, 4),
                        semantic_score=round(sem, 4),
                        personal_score=round(pers, 4),
                        recency_score=round(rec, 4),
                        query_intent=plan.intent,
                    )
                )

            results.sort(key=lambda r: (r.final_score, r.lexical_score, r.semantic_score), reverse=True)
            return results[:top_k]

    def save(self, path: str | Path) -> None:
        with self._lock:
            dest = Path(path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            with tmp.open("w", encoding="utf-8") as f:
                for item in self._items.values():
                    f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
            tmp.replace(dest)

    @classmethod
    def load(cls, path: str | Path) -> SeekMorphEngine:
        engine = cls()
        src = Path(path)
        for line in src.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and ("item_id" in row or "chunk_id" in row) and "content" in row:
                engine.add(KnowledgeItem.from_dict(row))
        return engine
