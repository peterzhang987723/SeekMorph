"""Personal Agent memory store and proactive SuperAgent mandate/ticket runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
from typing import Any, Callable

from seekmorph.engine import KnowledgeItem, RankedResult, SeekMorphEngine, UserProfile


@dataclass
class TriggerEvent:
    """Incoming event from an external source (e.g., GitHub, Feishu, Slack, API)."""

    trigger_id: str
    source: str
    thread_id: str
    actor_id: str
    text: str
    is_human: bool = True
    addressed: bool = False
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Mandate:
    """Host-enforced authorization record for an autonomous agent work item."""

    actor_id: str
    tenant_id: str
    strength: str = "explicit"  # "explicit" | "inferred"
    via: str = "addressed"  # "addressed" | "followup" | "rule"
    created_at: float = field(default_factory=time.time)


@dataclass
class TicketInputRecord:
    input_id: str
    trigger_id: str
    actor_id: str
    text: str
    is_human: bool
    timestamp: float
    delivered: bool = False


@dataclass
class AgentTicket:
    """Proactive SuperAgent work item with mandate lifecycle and human-wait locks."""

    ticket_id: str
    thread_id: str
    mandate: Mandate
    state: str = "running"  # "running" | "blocked" | "ended"
    ended_by: str | None = None  # "executor" | "human" | "policy"
    inputs: list[TicketInputRecord] = field(default_factory=list)
    undelivered: list[str] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)


class PersonalMemoryStore:
    """Thread-safe personal memory and skill store built on SeekMorphEngine."""

    def __init__(self, engine: SeekMorphEngine | None = None) -> None:
        self.engine = engine or SeekMorphEngine(recency_half_life_days=30.0)
        self._lock = threading.RLock()

    def remember(
        self,
        memory_id: str,
        user_id: str,
        tenant_id: str,
        title: str,
        content: str,
        kind: str = "episodic",
        tags: list[str] | tuple[str, ...] = (),
        age_days: float = 0.0,
        embedding: list[float] | tuple[float, ...] = (),
    ) -> KnowledgeItem:
        """Store or update a personal memory isolated to the user's ACL scope."""
        merged_tags = tuple(dict.fromkeys([*tags, f"kind:{kind}"]))
        item = KnowledgeItem(
            item_id=memory_id,
            title=title,
            content=content,
            tenant_id=tenant_id,
            tags=merged_tags,
            allowed_roles=(f"user:{user_id}",),
            age_days=age_days,
            embedding=tuple(float(x) for x in embedding),
            metadata={"user_id": user_id, "kind": kind},
        )
        self.engine.add(item)
        return item

    def recall(
        self,
        user: UserProfile,
        query: str,
        top_k: int = 5,
        kind: str | None = None,
        query_embedding: list[float] | tuple[float, ...] | None = None,
    ) -> list[RankedResult]:
        """Retrieve personalized memories for a user, reinforcing recalled items."""
        scoped_roles = tuple(dict.fromkeys([*user.roles, f"user:{user.user_id}"]))
        scoped_user = UserProfile(
            user_id=user.user_id,
            tenant_id=user.tenant_id,
            roles=scoped_roles,
            interest_tags=dict(user.interest_tags),
            clicked_item_ids=user.clicked_item_ids,
        )
        results = self.engine.search_and_recommend(
            query=query,
            user=scoped_user,
            tenant_id=user.tenant_id,
            top_k=top_k * 2 if kind else top_k,
            query_embedding=query_embedding,
        )
        if kind:
            target_tag = f"kind:{kind}"
            results = [r for r in results if target_tag in r.item.tags][:top_k]
        for r in results:
            self.engine.record_feedback(r.item.item_id, clicked=True)
        return results


class ProactiveTicketRuntime:
    """Host-enforced admission gate and ticket lifecycle for proactive SuperAgents."""

    def __init__(
        self,
        allowed_actors: set[str] | list[str] | tuple[str, ...],
        tenant_id: str = "default",
        eval_mode: str = "off",
        on_start: Callable[[AgentTicket], None] | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self.allowed_actors: set[str] = set(allowed_actors)
        self.tenant_id = tenant_id
        self.eval_mode = eval_mode  # "off" | "auto"
        self.paused = False
        self.on_start = on_start
        self.tickets: dict[str, AgentTicket] = {}
        self.thread_tickets: dict[str, list[str]] = {}

    def is_allowed(self, actor_id: str) -> bool:
        with self._lock:
            return actor_id in self.allowed_actors

    def revalidate(self, mandate: Mandate) -> bool:
        """Verify that a ticket's mandate still holds under current allowlist and mode."""
        with self._lock:
            if mandate.tenant_id != self.tenant_id:
                return False
            if not self.is_allowed(mandate.actor_id):
                return False
            if mandate.strength == "explicit":
                return True
            return self.eval_mode == "auto"

    def participating(self, thread_id: str) -> bool:
        """Return True only if the thread has a non-human-stopped ticket with a valid mandate."""
        with self._lock:
            ticket_ids = self.thread_tickets.get(thread_id, [])
            for tid in ticket_ids:
                ticket = self.tickets.get(tid)
                if not ticket:
                    continue
                if ticket.ended_by == "human":
                    continue
                if not self.revalidate(ticket.mandate):
                    continue
                return True
            return False

    def gate(self, trigger: TriggerEvent) -> tuple[bool, Mandate | None, str]:
        """Evaluate whether an incoming trigger is admissible under current policy."""
        with self._lock:
            if not trigger.is_human:
                return False, None, "not_human"
            allowed = self.is_allowed(trigger.actor_id)
            active_thread = self.participating(trigger.thread_id)

            if trigger.addressed:
                if allowed:
                    return True, Mandate(actor_id=trigger.actor_id, tenant_id=self.tenant_id, strength="explicit", via="addressed"), "ok"
                if self.eval_mode == "auto":
                    return True, Mandate(actor_id=trigger.actor_id, tenant_id=self.tenant_id, strength="inferred", via="addressed"), "ok"
                return False, None, "not_a_member"

            if active_thread:
                if allowed:
                    return True, Mandate(actor_id=trigger.actor_id, tenant_id=self.tenant_id, strength="explicit", via="followup"), "ok"
                if self.eval_mode == "auto":
                    return True, Mandate(actor_id=trigger.actor_id, tenant_id=self.tenant_id, strength="inferred", via="followup"), "ok"
                return False, None, "not_a_member"

            return False, None, "no_mention"

    def receive(self, trigger: TriggerEvent) -> AgentTicket | None:
        """Process an incoming trigger across active/resumable tickets or open a new ticket."""
        with self._lock:
            admissible, mandate, _ = self.gate(trigger)
            existing_ids = self.thread_tickets.get(trigger.thread_id, [])

            for tid in existing_ids:
                ticket = self.tickets.get(tid)
                if not ticket or ticket.ended_by == "human":
                    continue
                if not self.revalidate(ticket.mandate):
                    self.refuse(ticket)
                    continue
                if not admissible:
                    continue

                inp = TicketInputRecord(
                    input_id=f"in-{trigger.trigger_id}",
                    trigger_id=trigger.trigger_id,
                    actor_id=trigger.actor_id,
                    text=trigger.text,
                    is_human=trigger.is_human,
                    timestamp=trigger.timestamp,
                )
                ticket.inputs.append(inp)
                ticket.undelivered.append(inp.input_id)
                ticket.updated_at = trigger.timestamp

                if ticket.state == "blocked" and self.is_allowed(trigger.actor_id):
                    ticket.state = "running"
                    self._deliver(ticket)
                elif ticket.state == "ended":
                    self.wake(ticket.ticket_id)
                else:
                    self._deliver(ticket)
                return ticket

            if not admissible or mandate is None:
                return None

            new_tid = f"t-{len(self.tickets) + 1}"
            inp = TicketInputRecord(
                input_id=f"in-{trigger.trigger_id}",
                trigger_id=trigger.trigger_id,
                actor_id=trigger.actor_id,
                text=trigger.text,
                is_human=trigger.is_human,
                timestamp=trigger.timestamp,
            )
            ticket = AgentTicket(
                ticket_id=new_tid,
                thread_id=trigger.thread_id,
                mandate=mandate,
                state="running",
                inputs=[inp],
                undelivered=[inp.input_id],
                updated_at=trigger.timestamp,
            )
            self.tickets[new_tid] = ticket
            self.thread_tickets.setdefault(trigger.thread_id, []).append(new_tid)
            if not self.paused:
                self._deliver(ticket)
            return ticket

    def wake(self, ticket_id: str) -> bool:
        """Resume an ended ticket if not paused and its mandate and waiting sender remain valid."""
        with self._lock:
            ticket = self.tickets.get(ticket_id)
            if not ticket or ticket.state != "ended" or ticket.ended_by == "human" or not ticket.undelivered:
                return False
            if not self.revalidate(ticket.mandate):
                self.refuse(ticket)
                return False
            if self.paused:
                return False
            ticket.state = "running"
            ticket.ended_by = None
            self._deliver(ticket)
            return True

    def refuse(self, ticket: AgentTicket) -> None:
        """Drop undelivered inputs from a ticket whose mandate was revoked so they never appear delivered."""
        with self._lock:
            rejected = set(ticket.undelivered)
            ticket.inputs = [inp for inp in ticket.inputs if inp.input_id not in rejected]
            ticket.undelivered = []

    def stop(self, ticket_id: str, by: str = "human") -> None:
        """End a ticket; human stops withdraw the mandate and remove thread participation."""
        with self._lock:
            ticket = self.tickets.get(ticket_id)
            if not ticket:
                return
            self.refuse(ticket)
            ticket.state = "ended"
            ticket.ended_by = by
            ticket.updated_at = time.time()

    def _deliver(self, ticket: AgentTicket) -> None:
        if self.paused:
            return
        pending = set(ticket.undelivered)
        for inp in ticket.inputs:
            if inp.input_id in pending:
                inp.delivered = True
        ticket.undelivered = []
        if self.on_start:
            self.on_start(ticket)
