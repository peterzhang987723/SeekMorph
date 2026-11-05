"""SeekMorph: Hybrid AI search, personalized recommendation reranking, and Personal SuperAgent memory/mandate engine."""

from seekmorph.agent import (
    AgentTicket,
    Mandate,
    PersonalMemoryStore,
    ProactiveTicketRuntime,
    TriggerEvent,
)
from seekmorph.engine import (
    KnowledgeItem,
    RankedResult,
    SeekMorphEngine,
    UserProfile,
)
from seekmorph.metrics import (
    RankingEvalReport,
    evaluate_benchmark,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from seekmorph.query import (
    QueryPlan,
    analyze_query,
    tokenize_query,
)

__version__ = "0.2.0"

__all__ = [
    "AgentTicket",
    "KnowledgeItem",
    "Mandate",
    "PersonalMemoryStore",
    "ProactiveTicketRuntime",
    "QueryPlan",
    "RankedResult",
    "RankingEvalReport",
    "SeekMorphEngine",
    "TriggerEvent",
    "UserProfile",
    "analyze_query",
    "evaluate_benchmark",
    "mrr_at_k",
    "ndcg_at_k",
    "recall_at_k",
    "tokenize_query",
]
