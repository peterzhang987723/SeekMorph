"""Standard Search & Recommendation Evaluation Metrics: Recall@K, MRR@K, and NDCG@K."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """Compute Recall@K given a ranked list of item IDs and the ground-truth relevant set."""
    if not relevant_ids or k <= 0:
        return 0.0
    top_k = set(ranked_ids[:k])
    return len(top_k & relevant_ids) / len(relevant_ids)


def mrr_at_k(ranked_ids: Sequence[str], relevant_ids: set[str], k: int) -> float:
    """Compute Mean Reciprocal Rank (MRR@K) for a single query."""
    if not relevant_ids or k <= 0:
        return 0.0
    for idx, item_id in enumerate(ranked_ids[:k]):
        if item_id in relevant_ids:
            return 1.0 / (idx + 1)
    return 0.0


def ndcg_at_k(ranked_ids: Sequence[str], graded_relevance: dict[str, float], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain (NDCG@K) with graded relevance labels."""
    if not graded_relevance or k <= 0:
        return 0.0

    def dcg(ids: Sequence[str]) -> float:
        total = 0.0
        for idx, item_id in enumerate(ids[:k]):
            rel = float(graded_relevance.get(item_id, 0.0))
            if rel > 0.0:
                total += (2.0**rel - 1.0) / math.log2(idx + 2.0)
        return total

    actual_dcg = dcg(ranked_ids)
    ideal_ids = sorted(graded_relevance.keys(), key=lambda x: graded_relevance[x], reverse=True)
    ideal_dcg = dcg(ideal_ids)
    return (actual_dcg / ideal_dcg) if ideal_dcg > 0.0 else 0.0


@dataclass(frozen=True)
class RankingEvalReport:
    """Aggregated evaluation report across a benchmark query set."""

    num_queries: int
    k: int
    mean_recall: float
    mean_mrr: float
    mean_ndcg: float


def evaluate_benchmark(
    predictions: list[Sequence[str]],
    ground_truth_graded: list[dict[str, float]],
    *,
    k: int = 5,
) -> RankingEvalReport:
    """Evaluate a batch of ranked predictions against graded relevance judgments."""
    if len(predictions) != len(ground_truth_graded) or not predictions:
        raise ValueError("predictions and ground_truth_graded must have the same non-zero length")

    n = len(predictions)
    rec_sum = 0.0
    mrr_sum = 0.0
    ndcg_sum = 0.0

    for preds, graded in zip(predictions, ground_truth_graded):
        rel_set = {cid for cid, grade in graded.items() if grade > 0.0}
        rec_sum += recall_at_k(preds, rel_set, k)
        mrr_sum += mrr_at_k(preds, rel_set, k)
        ndcg_sum += ndcg_at_k(preds, graded, k)

    return RankingEvalReport(
        num_queries=n,
        k=k,
        mean_recall=round(rec_sum / n, 4),
        mean_mrr=round(mrr_sum / n, 4),
        mean_ndcg=round(ndcg_sum / n, 4),
    )
