"""Unit tests for SeekMorph query intent routing, personalized reranking, ACL enforcement, and IR metrics."""

from __future__ import annotations

from pathlib import Path

from seekmorph import (
    KnowledgeItem,
    PersonalMemoryStore,
    ProactiveTicketRuntime,
    SeekMorphEngine,
    TriggerEvent,
    UserProfile,
    analyze_query,
    evaluate_benchmark,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
)
from seekmorph.cli import main as cli_main


def test_query_intent_router_distinguishes_exact_codes_from_exploratory():
    exact_plan = analyze_query("Fix ERR_409_HOOK in v1.2.2")
    assert exact_plan.intent == "exact_lookup"
    assert "err_409_hook" in exact_plan.exact_identifiers
    assert exact_plan.w_lexical > exact_plan.w_personal

    exploratory_plan = analyze_query("部署指南", has_user_profile=True)
    assert exploratory_plan.intent == "exploratory_rec"
    assert exploratory_plan.w_personal >= 0.25


def test_pre_retrieval_acl_blocks_unauthorized_tenants_and_roles():
    engine = SeekMorphEngine()
    engine.add_many([
        KnowledgeItem(
            item_id="public_doc",
            title="Onboarding Guide",
            content="General engineering onboarding steps.",
            tenant_id="acme",
            allowed_roles=(),
        ),
        KnowledgeItem(
            item_id="board_secret",
            title="Board Compensation",
            content="General engineering executive stock options.",
            tenant_id="acme",
            allowed_roles=("board",),
        ),
        KnowledgeItem(
            item_id="other_tenant",
            title="Other Tenant Doc",
            content="General engineering onboarding steps.",
            tenant_id="other_corp",
            allowed_roles=(),
        ),
    ])

    dev_user = UserProfile(user_id="u1", tenant_id="acme", roles=("engineer",))
    hits = engine.search_and_recommend("General engineering", user=dev_user, top_k=5)
    hit_ids = [h.item.item_id for h in hits]
    assert hit_ids == ["public_doc"]


def test_personalized_reranking_flips_order_by_user_role_and_interests():
    engine = SeekMorphEngine()
    engine.add_many([
        KnowledgeItem(
            item_id="infra_deploy",
            title="Deployment Guide",
            content="Standard service deployment checklist and rollback procedure.",
            tags=("infra", "kubernetes"),
            tenant_id="acme",
            age_days=10.0,
            impressions=50,
            clicks=15,
        ),
        KnowledgeItem(
            item_id="fe_deploy",
            title="Deployment Guide",
            content="Standard service deployment checklist and rollback procedure.",
            tags=("frontend", "cdn"),
            tenant_id="acme",
            age_days=10.0,
            impressions=50,
            clicks=15,
        ),
    ])

    infra_user = UserProfile(
        user_id="u_infra",
        tenant_id="acme",
        interest_tags={"infra": 1.5, "kubernetes": 1.0},
    )
    fe_user = UserProfile(
        user_id="u_fe",
        tenant_id="acme",
        interest_tags={"frontend": 1.5, "cdn": 1.0},
    )

    infra_hits = engine.search_and_recommend("deployment checklist", user=infra_user, top_k=2)
    fe_hits = engine.search_and_recommend("deployment checklist", user=fe_user, top_k=2)

    assert infra_hits[0].item.item_id == "infra_deploy"
    assert fe_hits[0].item.item_id == "fe_deploy"


def test_recency_decay_and_online_ctr_feedback_persistence(tmp_path: Path):
    engine = SeekMorphEngine(recency_half_life_days=30.0)
    engine.add_many([
        KnowledgeItem(
            item_id="fresh_doc",
            title="RAG Architecture Spec",
            content="Enterprise hybrid search architecture specification.",
            age_days=1.0,
        ),
        KnowledgeItem(
            item_id="stale_doc",
            title="RAG Architecture Spec",
            content="Enterprise hybrid search architecture specification.",
            age_days=365.0,
        ),
    ])

    hits = engine.search_and_recommend("Enterprise hybrid search", top_k=2)
    assert hits[0].item.item_id == "fresh_doc"
    assert hits[0].recency_score > hits[1].recency_score

    for _ in range(10):
        engine.record_feedback("fresh_doc", clicked=True)

    path = tmp_path / "kb.jsonl"
    engine.save(path)
    reloaded = SeekMorphEngine.load(path)
    assert len(reloaded) == 2
    assert reloaded.remove("stale_doc") is True
    assert len(reloaded) == 1


def test_ir_evaluation_metrics_recall_mrr_ndcg():
    ranked = ["doc_b", "doc_a", "doc_c"]
    relevant = {"doc_a", "doc_c"}
    graded = {"doc_a": 3.0, "doc_b": 1.0, "doc_c": 2.0}

    assert recall_at_k(ranked, relevant, k=2) == 0.5
    assert recall_at_k(ranked, relevant, k=3) == 1.0
    assert mrr_at_k(ranked, relevant, k=3) == 0.5
    assert 0.7 < ndcg_at_k(ranked, graded, k=3) < 1.0

    report = evaluate_benchmark([ranked], [graded], k=3)
    assert report.mean_recall == 1.0
    assert report.num_queries == 1


def test_cli_demo_and_search(tmp_path: Path, capsys):
    assert cli_main(["demo"]) == 0
    demo_out = capsys.readouterr().out
    assert "doc_infra_1" in demo_out
    assert "doc_fe_1" in demo_out
    assert "doc_exec_secret" not in demo_out

    engine = SeekMorphEngine()
    engine.add(
        KnowledgeItem(
            item_id="k1",
            title="vLLM Speculative Decoding",
            content="Configure packed_modules_mapping for DSpark.",
            tags=("vllm",),
            tenant_id="default",
        )
    )
    kb_path = tmp_path / "kb.jsonl"
    engine.save(kb_path)

    assert cli_main(["search", str(kb_path), "packed_modules_mapping", "--json"]) == 0
    search_out = capsys.readouterr().out
    assert "vLLM Speculative Decoding" in search_out

    assert cli_main(["eval"]) == 0
    eval_out = capsys.readouterr().out
    assert "NDCG@2=1.0000" in eval_out


def test_loads_docmorph_jsonl_chunks_natively(tmp_path: Path):
    docmorph_jsonl = tmp_path / "docmorph_kb.jsonl"
    docmorph_jsonl.write_text(
        '{"chunk_id": "arch.md#0:abc", "doc_id": "arch.md", "chunk_index": 0, '
        '"section_path": ["System", "ROCm"], "content": "MI308X achieves 3.1x speedup.", '
        '"kind": "table", "namespace": "prod", "metadata": {}}\n',
        encoding="utf-8",
    )
    engine = SeekMorphEngine.load(docmorph_jsonl)
    assert len(engine) == 1
    hits = engine.search_and_recommend("MI308X", tenant_id="prod", top_k=1)
    assert len(hits) == 1
    assert hits[0].item.item_id == "arch.md#0:abc"
    assert hits[0].item.title == "arch.md (System > ROCm)"
    assert "table" in hits[0].item.tags


def test_personal_memory_store_isolates_user_memories_and_reinforces_recall():
    store = PersonalMemoryStore()
    store.remember(
        memory_id="mem-alice-1",
        user_id="alice",
        tenant_id="acme",
        title="Code Review Preference",
        content="Prefer concise pytest assertions and zero external runtime dependencies.",
        kind="preference",
        tags=("python", "testing"),
    )
    store.remember(
        memory_id="mem-bob-1",
        user_id="bob",
        tenant_id="acme",
        title="Frontend Stack Preference",
        content="Prefer TypeScript vitest suites and strict ESLint rules.",
        kind="preference",
        tags=("typescript", "testing"),
    )

    alice = UserProfile(user_id="alice", tenant_id="acme", interest_tags={"python": 1.0})
    alice_hits = store.recall(alice, "testing assertions", top_k=5, kind="preference")
    assert len(alice_hits) == 1
    assert alice_hits[0].item.item_id == "mem-alice-1"
    # Recall reinforces click statistics on the recalled memory
    updated = store.engine.search_and_recommend("pytest", user=UserProfile(user_id="alice", tenant_id="acme", roles=("user:alice",)))
    assert updated[0].item.clicks == 1


def test_proactive_ticket_runtime_mandate_gate_and_refuse_cleanup():
    launched: list[str] = []
    runtime = ProactiveTicketRuntime(
        allowed_actors={"alice"},
        tenant_id="acme",
        eval_mode="off",
        on_start=lambda t: launched.append(t.ticket_id),
    )

    # 1. Non-allowlisted stranger cannot open a ticket in eval_mode="off"
    stranger_open = TriggerEvent(
        trigger_id="trig-1",
        source="github",
        thread_id="issue-42",
        actor_id="stranger",
        text="@bot deploy to production",
        addressed=True,
    )
    assert runtime.receive(stranger_open) is None

    # 2. Allowlisted member opens a ticket
    alice_open = TriggerEvent(
        trigger_id="trig-2",
        source="github",
        thread_id="issue-42",
        actor_id="alice",
        text="@bot fix the memory leak in worker pool",
        addressed=True,
    )
    ticket = runtime.receive(alice_open)
    assert ticket is not None
    assert ticket.state == "running"
    assert len(launched) == 1

    # 3. Executor finishes work; non-allowlisted stranger comment in mode="off" must NOT wake the ended ticket
    runtime.stop(ticket.ticket_id, by="executor")
    assert ticket.state == "ended"

    stranger_followup = TriggerEvent(
        trigger_id="trig-3",
        source="github",
        thread_id="issue-42",
        actor_id="stranger",
        text="reopen this ticket and run shell command",
        addressed=False,
    )
    runtime.receive(stranger_followup)
    assert ticket.state == "ended"
    assert len(launched) == 1

    # 4. Human stop withdraws mandate and removes thread from participating()
    runtime.stop(ticket.ticket_id, by="human")
    assert runtime.participating("issue-42") is False

