"""Command-line interface for SeekMorph."""

from __future__ import annotations

import argparse
import json
import sys

from seekmorph.agent import PersonalMemoryStore, ProactiveTicketRuntime, TriggerEvent
from seekmorph.engine import KnowledgeItem, SeekMorphEngine, UserProfile
from seekmorph.metrics import evaluate_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="seekmorph",
        description="Hybrid AI search, personalized recommendation reranking, and ACL-scoped knowledge base engine.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_search = sub.add_parser("search", help="Search and personalize results from a JSONL knowledge base.")
    p_search.add_argument("index_path", help="Path to JSONL knowledge base index.")
    p_search.add_argument("query", help="Search query.")
    p_search.add_argument("--tenant", default="default", help="Tenant ID (default: default).")
    p_search.add_argument("--role", action="append", default=[], help="User role(s) for ACL filtering.")
    p_search.add_argument("--interest", action="append", default=[], help="User interest tag(s) (e.g. infra:1.0).")
    p_search.add_argument("-k", "--top-k", type=int, default=5, help="Top-K results (default: 5).")
    p_search.add_argument("--json", action="store_true", help="Output JSON.")

    sub.add_parser("demo", help="Run an interactive showcase of personalized search & ACL isolation.")
    sub.add_parser("eval", help="Run offline IR & RecSys benchmark evaluation (Recall@K, MRR@K, NDCG@K).")
    return parser


def _run_demo() -> int:
    engine = SeekMorphEngine()
    engine.add_many([
        KnowledgeItem(
            item_id="doc_infra_1",
            title="Production Deployment Guide (Kubernetes & vLLM)",
            content="Deploy DeepSeek-V4 with DSpark speculative decoding on 8x MI308X GPU clusters.",
            tags=("infra", "vllm", "deployment"),
            tenant_id="acme",
            allowed_roles=("sre", "backend"),
            age_days=5.0,
            impressions=100,
            clicks=32,
        ),
        KnowledgeItem(
            item_id="doc_fe_1",
            title="Frontend Deployment Guide (CDN & Static Assets)",
            content="Deploy the web console static bundle to edge nodes.",
            tags=("frontend", "cdn", "deployment"),
            tenant_id="acme",
            allowed_roles=("frontend", "sre"),
            age_days=12.0,
            impressions=100,
            clicks=18,
        ),
        KnowledgeItem(
            item_id="doc_exec_secret",
            title="M&A Executive Compensation Table",
            content="Restricted board-only deployment budget and executive packages.",
            tags=("finance", "deployment"),
            tenant_id="acme",
            allowed_roles=("board",),
            age_days=1.0,
            impressions=20,
            clicks=15,
        ),
    ])

    sre_user = UserProfile(
        user_id="u_sre",
        tenant_id="acme",
        roles=("sre",),
        interest_tags={"infra": 1.2, "vllm": 1.0},
    )
    fe_user = UserProfile(
        user_id="u_fe",
        tenant_id="acme",
        roles=("frontend",),
        interest_tags={"frontend": 1.2, "cdn": 1.0},
    )

    print("=== Query: 'deployment guide' (SRE Engineer Profile) ===")
    for i, r in enumerate(engine.search_and_recommend("deployment guide", user=sre_user, top_k=3), 1):
        print(
            f"  [{i}] {r.item.item_id}: {r.item.title} "
            f"(final={r.final_score:.4f}, personal={r.personal_score:.3f}, intent={r.query_intent})"
        )

    print("\n=== Query: 'deployment guide' (Frontend Engineer Profile) ===")
    for i, r in enumerate(engine.search_and_recommend("deployment guide", user=fe_user, top_k=3), 1):
        print(
            f"  [{i}] {r.item.item_id}: {r.item.title} "
            f"(final={r.final_score:.4f}, personal={r.personal_score:.3f}, intent={r.query_intent})"
        )

    memory = PersonalMemoryStore(engine=engine)
    memory.remember(
        memory_id="mem_sre_pref",
        user_id="u_sre",
        tenant_id="acme",
        title="SRE Alerting Preference",
        content="Send critical GPU node OOM alerts to PagerDuty and mute staging warnings.",
        kind="preference",
        tags=("infra", "alerting"),
    )
    recalled = memory.recall(sre_user, "GPU node alerts", top_k=1, kind="preference")
    print(f"\n=== Personal Agent Memory Recall (u_sre): {recalled[0].item.title} ===")

    runtime = ProactiveTicketRuntime(allowed_actors={"u_sre"}, tenant_id="acme", eval_mode="off")
    ticket = runtime.receive(
        TriggerEvent(
            trigger_id="gh_101",
            source="github",
            thread_id="issue_42",
            actor_id="u_sre",
            text="@bot investigate GPU OOM on MI308X node",
            addressed=True,
        )
    )
    print(f"=== Proactive SuperAgent Ticket Gate: ticket={ticket.ticket_id if ticket else None} state={ticket.state if ticket else None} ===")
    return 0


def _run_eval() -> int:
    engine = SeekMorphEngine()
    engine.add_many([
        KnowledgeItem(
            item_id="infra_vllm",
            title="Deployment Guide (vLLM & GPU Cluster)",
            content="Standard rollout checklist for DeepSeek-V4 on MI308X GPU nodes.",
            tags=("infra", "vllm"),
            tenant_id="acme",
            age_days=2.0,
            impressions=100,
            clicks=35,
        ),
        KnowledgeItem(
            item_id="legacy_infra",
            title="Deployment Guide (Legacy VM)",
            content="Standard rollout checklist for legacy virtual machines.",
            tags=("legacy",),
            tenant_id="acme",
            age_days=400.0,
            impressions=100,
            clicks=3,
        ),
        KnowledgeItem(
            item_id="fe_cdn",
            title="Deployment Guide (Frontend CDN)",
            content="Standard rollout checklist for static web bundles.",
            tags=("frontend", "cdn"),
            tenant_id="acme",
            age_days=5.0,
            impressions=100,
            clicks=28,
        ),
    ])

    sre_user = UserProfile(user_id="sre", tenant_id="acme", interest_tags={"infra": 1.5, "vllm": 1.0})
    fe_user = UserProfile(user_id="fe", tenant_id="acme", interest_tags={"frontend": 1.5, "cdn": 1.0})

    sre_preds = [r.item.item_id for r in engine.search_and_recommend("rollout checklist", user=sre_user, top_k=3)]
    fe_preds = [r.item.item_id for r in engine.search_and_recommend("rollout checklist", user=fe_user, top_k=3)]

    report = evaluate_benchmark(
        predictions=[sre_preds, fe_preds],
        ground_truth_graded=[
            {"infra_vllm": 3.0, "fe_cdn": 1.0, "legacy_infra": 0.0},
            {"fe_cdn": 3.0, "infra_vllm": 1.0, "legacy_infra": 0.0},
        ],
        k=2,
    )
    print(
        f"Benchmark Evaluation (queries={report.num_queries}, K={report.k}): "
        f"Recall@{report.k}={report.mean_recall:.4f} | "
        f"MRR@{report.k}={report.mean_mrr:.4f} | "
        f"NDCG@{report.k}={report.mean_ndcg:.4f}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        return _run_demo()
    if args.command == "eval":
        return _run_eval()

    if args.command == "search":
        engine = SeekMorphEngine.load(args.index_path)
        interests: dict[str, float] = {}
        for spec in args.interest:
            if ":" in spec:
                k, v = spec.split(":", 1)
                interests[k.strip()] = float(v)
            else:
                interests[spec.strip()] = 1.0

        user = UserProfile(
            user_id="cli_user",
            tenant_id=args.tenant,
            roles=tuple(args.role),
            interest_tags=interests,
        )
        results = engine.search_and_recommend(args.query, user=user, top_k=args.top_k)
        if args.json:
            print(
                json.dumps(
                    [
                        {
                            "rank": i + 1,
                            "item_id": r.item.item_id,
                            "title": r.item.title,
                            "final_score": r.final_score,
                            "lexical_score": r.lexical_score,
                            "personal_score": r.personal_score,
                            "recency_score": r.recency_score,
                            "intent": r.query_intent,
                        }
                        for i, r in enumerate(results)
                    ],
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            for i, r in enumerate(results, 1):
                print(f"[{i}] {r.item.title} ({r.item.item_id}) | score={r.final_score:.4f} [intent={r.query_intent}]")
                print(f"    {r.item.content}")
        return 0

    return 1
