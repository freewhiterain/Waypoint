"""Bounded Deep Search subgraph for domain subagents."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, model_validator

from app.schemas.tool_policy import ToolPolicy
from app.rag.evidence import detect_fact_conflicts, is_evidence_fresh, merge_fact_conflicts
from app.schemas.planning import Evidence, ResearchTask, TaskType
from app.schemas.research import Claim, ResearchConflict, ResearchReport


SearchFunction = Callable[[str, int], Awaitable[list[Evidence]]]
EventCallback = Callable[[str, dict[str, Any]], Awaitable[None]]

HARD_MAX_ROUNDS = 3
HARD_MAX_TOOL_CALLS = 5
HARD_MAX_TIMEOUT_SECONDS = 15.0


class DeepSearchRequest(BaseModel):
    """Input for one bounded Deep Search run."""

    model_config = ConfigDict(arbitrary_types_allowed=True, str_strip_whitespace=True)

    query: str = Field(min_length=1)
    worker: TaskType
    task: ResearchTask | None = None
    tool_policy: ToolPolicy | None = None
    max_rounds: int = Field(default=2, ge=1)
    max_tool_calls: int = Field(default=3, ge=1)
    timeout_seconds: float = Field(default=10.0, gt=0)
    results_per_query: int = Field(default=3, ge=1, le=20)
    require_fresh: bool = True

    @classmethod
    def from_task(
        cls,
        task: ResearchTask,
        *,
        max_rounds: int = 2,
        max_tool_calls: int = 3,
        timeout_seconds: float = 10.0,
        results_per_query: int = 3,
        tool_policy: ToolPolicy | None = None,
    ) -> "DeepSearchRequest":
        return cls(
            query=task.query,
            worker=task.task_type,
            task=task,
            tool_policy=tool_policy,
            max_rounds=max_rounds,
            max_tool_calls=max_tool_calls,
            timeout_seconds=timeout_seconds,
            results_per_query=results_per_query,
        )

    @model_validator(mode="after")
    def validate_contract_alignment(self) -> "DeepSearchRequest":
        if self.task is not None:
            if self.task.task_type != self.worker:
                raise ValueError("Deep Search task type must match worker")
            if not self.query:
                self.query = self.task.query
        if self.tool_policy is not None and self.tool_policy.worker != self.worker:
            raise ValueError("Tool policy worker must match Deep Search worker")
        return self


class DeepSearchEvaluation(BaseModel):
    """Typed evaluator output used for routing transitions."""

    needs_follow_up: StrictBool = False
    follow_up_query: str | None = Field(default=None, min_length=1)
    missing_facts: list[str] = Field(default_factory=list)
    conflicts: list[ResearchConflict] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    summary: str = ""


class DeepSearchState(BaseModel):
    """Mutable state snapshot for the bounded Deep Search loop."""

    request: DeepSearchRequest
    planned_queries: list[str] = Field(default_factory=list)
    completed_queries: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    conflicts: list[ResearchConflict] = Field(default_factory=list)
    missing_facts: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    rounds: int = 0
    tool_calls: int = 0
    summary: str = ""


class DeepSearchReport(ResearchReport):
    """ResearchReport-compatible result with Deep Search telemetry."""

    rounds: int = 0
    tool_calls: int = 0
    missing_facts: list[str] = Field(default_factory=list)
    planned_queries: list[str] = Field(default_factory=list)


def _policy_for(request: DeepSearchRequest) -> ToolPolicy:
    return request.tool_policy or ToolPolicy.for_worker(request.worker)


def _limit_warnings(request: DeepSearchRequest) -> tuple[int, int, float, list[str]]:
    warnings: list[str] = []
    max_rounds = min(request.max_rounds, HARD_MAX_ROUNDS)
    max_tool_calls = min(request.max_tool_calls, HARD_MAX_TOOL_CALLS)
    timeout_seconds = min(request.timeout_seconds, HARD_MAX_TIMEOUT_SECONDS)
    if request.max_rounds > HARD_MAX_ROUNDS:
        warnings.append(f"Deep Search max rounds capped at {HARD_MAX_ROUNDS}.")
    if request.max_tool_calls > HARD_MAX_TOOL_CALLS:
        warnings.append(f"Deep Search tool call limit capped at {HARD_MAX_TOOL_CALLS}.")
    if request.timeout_seconds > HARD_MAX_TIMEOUT_SECONDS:
        warnings.append(f"Deep Search timeout capped at {HARD_MAX_TIMEOUT_SECONDS:g} seconds.")
    return max_rounds, max_tool_calls, timeout_seconds, warnings


def _dedupe_and_filter(
    existing: list[Evidence],
    incoming: list[Evidence],
    *,
    query: str,
    round_number: int,
    require_fresh: bool,
    now: datetime,
) -> tuple[list[Evidence], list[str]]:
    warnings: list[str] = []
    evidence: list[Evidence] = list(existing)
    seen = {_dedupe_key(item) for item in evidence}
    duplicate_count = 0
    stale_count = 0

    for item in incoming:
        if require_fresh and not is_evidence_fresh(item, now):
            stale_count += 1
            continue
        key = _dedupe_key(item)
        if key in seen:
            duplicate_count += 1
            continue
        if not item.id:
            seed = f"{query.strip()}\n{item.content.strip()}\n{round_number}"
            item = item.model_copy(
                update={"id": f"deep-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:24]}"}
            )
        evidence.append(item)
        seen.add(key)

    if duplicate_count:
        warnings.append(f"Filtered {duplicate_count} duplicate evidence item(s).")
    if stale_count:
        warnings.append(f"Filtered {stale_count} stale evidence item(s).")
    return evidence, warnings


def _dedupe_key(item: Evidence) -> tuple[str, str, str]:
    if item.id:
        return ("id", item.id, "")
    if item.source_url:
        return ("url", item.source_url, "")
    return ("content", item.source, item.content)


def _metadata_conflicts(items: list[Evidence]) -> list[ResearchConflict]:
    """委托给 app.rag.evidence 的唯一实现，保持既有内部调用点不变。"""
    return detect_fact_conflicts(items)


def _next_query(request: DeepSearchRequest, evaluation: DeepSearchEvaluation, round_number: int) -> str:
    if evaluation.follow_up_query:
        return evaluation.follow_up_query
    if evaluation.conflicts:
        # 冲突驱动的补搜：把冲突事实和它们的取值一起写进查询，让下一轮
        # 去找能裁决分歧的权威来源，而不是泛泛地重复原查询。
        conflict_terms = " ".join(
            f"{conflict.fact_key} {' '.join(conflict.values)}".strip()
            for conflict in evaluation.conflicts
        )
        return f"{request.query} {conflict_terms} 官方 最新 核实 第{round_number + 1}轮"
    if evaluation.missing_facts:
        missing = " ".join(evaluation.missing_facts)
        return f"{request.query} {missing} 官方 最新 第{round_number + 1}轮"
    return f"{request.query} 补充检索 第{round_number + 1}轮"


async def _call_search(
    search: SearchFunction,
    query: str,
    limit: int,
    timeout_seconds: float,
) -> tuple[list[Evidence], str | None]:
    try:
        result = await asyncio.wait_for(search(query, limit), timeout=timeout_seconds)
    except TimeoutError:
        return [], "Search timed out."
    except Exception as exc:
        return [], f"Search failed: {type(exc).__name__}."

    try:
        evidence = [Evidence.model_validate(item) for item in result]
    except Exception:
        return [], "Search returned invalid evidence."
    return evidence, None


async def _call_evaluator(evaluator: Any, state: DeepSearchState) -> DeepSearchEvaluation:
    if evaluator is None:
        return _default_evaluation(state)
    method = getattr(evaluator, "evaluate", None)
    candidate = method(state) if method is not None else evaluator(state)
    if inspect.isawaitable(candidate):
        candidate = await candidate
    return DeepSearchEvaluation.model_validate(candidate)


def _default_evaluation(state: DeepSearchState) -> DeepSearchEvaluation:
    conflicts = _metadata_conflicts(state.evidence)
    needs_follow_up = not state.evidence
    missing_facts = ["supporting evidence"] if needs_follow_up else []
    return DeepSearchEvaluation(
        needs_follow_up=needs_follow_up,
        missing_facts=missing_facts,
        conflicts=conflicts,
        claims=[
            Claim(
                text=item.content,
                evidence_ids=[item.id] if item.id else [],
            )
            for item in state.evidence
        ],
        summary="; ".join(item.content for item in state.evidence[:3]),
    )


def _merge_conflicts(
    evaluator_conflicts: list[ResearchConflict],
    evidence: list[Evidence],
) -> list[ResearchConflict]:
    return merge_fact_conflicts(evaluator_conflicts, evidence)


def _escalate_conflicts(
    evaluation: DeepSearchEvaluation,
    conflicts: list[ResearchConflict],
    chased_keys: set[str],
) -> DeepSearchEvaluation:
    """证据互相冲突时强制再补搜一轮。

    改动前只有"完全没有证据"或评估器自己要求时才会进入下一轮，于是冲突
    只被 _final_report 写成一条 warning 就收尾——Deep Search 在最需要它的
    场景（来源互相打架、需要权威来源裁决）里恰恰不跑。

    只追此前没追过的 fact_key：同一冲突追过一轮仍未消解，说明补搜裁决
    不了它，再发同样的查询只是白烧一次工具调用；此时把冲突留在报告里
    交给治理层降级。轮次和工具调用上限仍由调用方的硬上限兜住。
    """
    fresh_keys = [
        conflict.fact_key for conflict in conflicts if conflict.fact_key not in chased_keys
    ]
    if not fresh_keys or evaluation.needs_follow_up:
        return evaluation
    return evaluation.model_copy(
        update={
            "needs_follow_up": True,
            "conflicts": conflicts,
            "missing_facts": list(dict.fromkeys([*evaluation.missing_facts, *fresh_keys])),
        }
    )


def _ground_claims(claims: list[Claim], evidence: list[Evidence]) -> list[Claim]:
    evidence_ids = {item.id for item in evidence if item.id}
    return [
        claim
        for claim in claims
        if claim.evidence_ids and set(claim.evidence_ids).issubset(evidence_ids)
    ]


def _status_for(state: DeepSearchState) -> str:
    if not state.evidence and state.warnings:
        return "partial"
    if any(
        marker in warning.lower()
        for warning in state.warnings
        for marker in ("timeout", "stopped", "max rounds", "tool call limit")
    ):
        return "partial"
    if state.missing_facts or state.conflicts:
        return "partial"
    return "completed"


def _final_report(state: DeepSearchState) -> DeepSearchReport:
    warnings = list(state.warnings)
    if state.conflicts and not any("conflict" in warning.lower() for warning in warnings):
        warnings.append("Deep Search found typed evidence conflicts.")
    return DeepSearchReport(
        status=_status_for(state),
        summary=state.summary,
        claims=state.claims,
        conflicts=state.conflicts,
        evidence=state.evidence,
        warnings=warnings,
        rounds=state.rounds,
        tool_calls=state.tool_calls,
        missing_facts=state.missing_facts,
        planned_queries=state.planned_queries,
    )


async def run_deep_search(
    request: DeepSearchRequest,
    *,
    search: SearchFunction,
    evaluator: Any | None = None,
    now: datetime | None = None,
    event_callback: EventCallback | None = None,
) -> ResearchReport:
    """Run bounded Deep Search with typed evaluator-controlled transitions."""

    state = DeepSearchState(request=request)
    max_rounds, max_tool_calls, timeout_seconds, limit_warnings = _limit_warnings(request)
    state.warnings.extend(limit_warnings)
    now = now or datetime.now(timezone.utc)
    deadline = asyncio.get_running_loop().time() + timeout_seconds

    policy = _policy_for(request)
    if not policy.allow_deep_research:
        state.warnings.append(f"Deep Search is not allowed for worker {request.worker}.")
        return DeepSearchReport(
            status="unavailable",
            warnings=state.warnings,
            rounds=0,
            tool_calls=0,
        )

    query = request.query
    completed_query_set: set[str] = set()
    # 已经驱动过一轮补搜的冲突 fact_key，避免同一冲突反复触发同样的查询。
    chased_conflict_keys: set[str] = set()
    last_evaluation = DeepSearchEvaluation()

    while state.rounds < max_rounds:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            state.warnings.append("Deep Search total timeout reached.")
            break
        if state.tool_calls >= max_tool_calls:
            state.warnings.append("Deep Search stopped before follow-up because tool call limit was reached.")
            break
        if query in completed_query_set:
            state.warnings.append("Deep Search stopped because the follow-up query duplicated a completed query.")
            break

        round_number = state.rounds + 1
        if event_callback is not None:
            await event_callback(
                "subagent_tool_called",
                {"round_number": round_number},
            )
            if round_number > 1:
                await event_callback(
                    "follow_up_search",
                    {"round_number": round_number},
                )

        state.planned_queries.append(query)
        found, warning = await _call_search(
            search,
            query,
            request.results_per_query,
            min(timeout_seconds, remaining),
        )
        state.tool_calls += 1
        state.rounds += 1
        state.completed_queries.append(query)
        completed_query_set.add(query)
        if event_callback is not None:
            await event_callback(
                "subagent_tool_completed",
                {
                    "round_number": round_number,
                    "status": "sufficient" if found else "failed" if warning else "empty",
                    "evidence_count": len(found),
                },
            )
        if warning:
            state.warnings.append(warning)

        state.evidence, normalize_warnings = _dedupe_and_filter(
            state.evidence,
            found,
            query=query,
            round_number=round_number,
            require_fresh=request.require_fresh,
            now=now,
        )
        state.warnings.extend(normalize_warnings)

        try:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                state.warnings.append("Deep Search total timeout reached.")
                break
            last_evaluation = await asyncio.wait_for(
                _call_evaluator(evaluator, state),
                timeout=remaining,
            )
        except TimeoutError:
            state.warnings.append("Deep Search total timeout reached.")
            break
        except Exception as exc:
            state.warnings.append(f"Evaluation failed: {type(exc).__name__}.")
            last_evaluation = DeepSearchEvaluation()

        state.claims = _ground_claims(last_evaluation.claims, state.evidence)
        dropped_claims = len(last_evaluation.claims) - len(state.claims)
        if dropped_claims:
            state.warnings.append(f"Dropped {dropped_claims} unbound claim(s).")
        state.summary = last_evaluation.summary
        state.missing_facts = last_evaluation.missing_facts
        state.conflicts = _merge_conflicts(last_evaluation.conflicts, state.evidence)
        last_evaluation = _escalate_conflicts(last_evaluation, state.conflicts, chased_conflict_keys)
        if last_evaluation.needs_follow_up:
            state.missing_facts = last_evaluation.missing_facts
            chased_conflict_keys.update(conflict.fact_key for conflict in state.conflicts)

        if not last_evaluation.needs_follow_up:
            break
        if state.rounds >= max_rounds:
            state.warnings.append("Deep Search stopped because max rounds was reached.")
            break
        if state.tool_calls >= max_tool_calls:
            state.warnings.append("Deep Search stopped because tool call limit was reached.")
            break
        query = _next_query(request, last_evaluation, state.rounds)

    if last_evaluation.needs_follow_up and state.rounds >= max_rounds:
        state.missing_facts = last_evaluation.missing_facts
    return _final_report(state)


__all__ = [
    "DeepSearchEvaluation",
    "DeepSearchReport",
    "DeepSearchRequest",
    "DeepSearchState",
    "run_deep_search",
]
