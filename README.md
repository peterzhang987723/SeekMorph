# SeekMorph

Hybrid search, personalized recommendation reranking, RBAC/ACL knowledge base, and Personal SuperAgent memory/mandate runtime in pure Python.

`SeekMorph` pairs with [`DocMorph`](https://github.com/peterzhang987723/DocMorph) (or operates standalone) to index knowledge base items and personal agent memories, enforce pre-retrieval tenant and role access control, route query intent, combine inverted-index `BM25` and dense vector similarity with user interest profiles, Bayesian-smoothed CTR, and exponential recency decay, and gate proactive SuperAgent work items (`AgentTicket` / `Mandate`).

- **Interactive Demo (Hugging Face Space)**: [https://huggingface.co/spaces/zhangshoufa/DocMorph](https://huggingface.co/spaces/zhangshoufa/DocMorph)

## Features

- **Pre-retrieval ACL gate (`seekmorph.engine`)**: Filters items by `tenant_id` and `allowed_roles` before scoring so unauthorized documents never enter candidate sets.
- **Query intent routing (`seekmorph.query`)**: Classifies queries into `exact_lookup` (error codes, version strings, issue IDs), `exploratory_rec`, and `semantic_qa`, adjusting channel weights across lexical, semantic, personalization, and recency signals.
- **Thread-safe inverted index & personalized reranking**: Incremental posting lists (`_postings`) with `RLock` concurrency, user tag affinity, historical click log-boosting, Bayesian-smoothed CTR, and half-life time decay.
- **Personal Agent memory & SuperAgent mandate runtime (`seekmorph.agent`)**:
  - `PersonalMemoryStore`: User-isolated episodic and preference memory recall with automatic usage reinforcement.
  - `ProactiveTicketRuntime`: Host-enforced admission gate (`Mandate`) and proactive ticket lifecycle (`AgentTicket`) that revalidates sender authorization on both ticket creation and ticket resumption.
- **Native `DocMorph` JSONL compatibility**: Loads `DocMorph` chunk JSONL files directly without format conversion.
- **Offline IR & RecSys metrics (`seekmorph.metrics`)**: Built-in `recall_at_k`, `mrr_at_k`, `ndcg_at_k`, and batch `evaluate_benchmark`.

## Usage

### CLI

```bash
# Run personalized search & ACL demo
python -m seekmorph demo

# Run offline benchmark evaluation (Recall@K, MRR@K, NDCG@K)
python -m seekmorph eval

# Search a JSONL knowledge base (created by SeekMorph or DocMorph)
python -m docmorph ingest ./docs -o kb.jsonl --namespace acme
python -m seekmorph search kb.jsonl "deployment guide" --tenant acme --interest infra:1.2
```

### Python API: Search & Personalized Reranking

```python
from seekmorph import KnowledgeItem, SeekMorphEngine, UserProfile

engine = SeekMorphEngine(recency_half_life_days=90.0)
engine.add_many([
    KnowledgeItem(
        item_id="infra_k8s",
        title="Deployment Guide (vLLM & GPU Cluster)",
        content="Checklist for rolling out DeepSeek-V4 on MI308X nodes.",
        tags=("infra", "vllm"),
        tenant_id="acme",
        allowed_roles=("sre", "backend"),
        age_days=3.0,
        impressions=120,
        clicks=38,
    ),
    KnowledgeItem(
        item_id="fe_cdn",
        title="Deployment Guide (Web Console & CDN)",
        content="Checklist for rolling out static web assets.",
        tags=("frontend", "cdn"),
        tenant_id="acme",
        allowed_roles=("frontend", "sre"),
        age_days=14.0,
        impressions=100,
        clicks=19,
    ),
])

sre = UserProfile(user_id="u_1", tenant_id="acme", roles=("sre",), interest_tags={"infra": 1.2})
hits = engine.search_and_recommend("deployment checklist", user=sre, top_k=5)
```

### Python API: Personal Agent Memory & SuperAgent Mandate Gate

```python
from seekmorph import PersonalMemoryStore, ProactiveTicketRuntime, TriggerEvent, UserProfile

# 1. Personal Agent memory store isolated per user
memory = PersonalMemoryStore()
memory.remember(
    memory_id="mem-1",
    user_id="alice",
    tenant_id="acme",
    title="Testing Convention",
    content="Use pytest with deterministic fixtures and zero external runtime deps.",
    kind="preference",
    tags=("python", "testing"),
)
recalled = memory.recall(UserProfile(user_id="alice", tenant_id="acme"), "pytest fixtures", kind="preference")

# 2. Proactive SuperAgent admission gate & ticket lifecycle
runtime = ProactiveTicketRuntime(allowed_actors={"alice"}, tenant_id="acme", eval_mode="off")
ticket = runtime.receive(
    TriggerEvent(
        trigger_id="gh-101",
        source="github",
        thread_id="issue-42",
        actor_id="alice",
        text="@bot fix the memory leak in worker pool",
        addressed=True,
    )
)
```

## Tests

```bash
pytest -v
python3 tests/test_seekmorph_5layers.py
```

## License

Apache-2.0
