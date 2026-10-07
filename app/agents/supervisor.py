"""Supervisor + Planner-Worker 的旅行规划主流程。

图结构（确定性 DAG）：
    planner → dispatch ⇄ run_worker（按依赖分组扇出，组内并行）
            → route_planner → budget → synthesize → END

只有 synthesize 节点使用 LLM（可用时），其余全部是确定性函数。
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field

from app.agents.planner import create_research_plan, parallel_groups
from app.agents.scheduling import calculate_budget, schedule_itinerary
from app.agents.subagents.registry import create_default_subagent_registry
from app.config import settings
from app.governance.evidence import EvidenceGovernanceService, ReviewedResearch
from app.governance.events import TaskEventService
from app.rag.evidence import evidence_is_mock
from app.rag.identifiers import stable_hash
from app.schemas.planning import (
    BudgetSummary,
    CandidateOption,
    ItineraryDay,
    ResearchTask,
    TravelPlanDraft,
    TravelRequirement,
    WorkerResult,
)
from app.schemas.research import EvidenceBoundCandidate, SubagentResponse
from app.utils.callables import supports_keyword
from app.utils.logger import app_logger


def _worker_result_mapping(value: Any) -> dict[str, dict[str, Any]]:
    if not value:
        return {}
    if isinstance(value, list):
        mapped: dict[str, dict[str, Any]] = {}
        for item in value:
            data = item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
            task_id = data.get("task_id")
            if task_id:
                mapped[task_id] = data
        return mapped
    if isinstance(value, dict) and "task_id" in value and "worker" in value:
        return {str(value["task_id"]): dict(value)}
    return {str(task_id): (payload.model_dump(mode="json") if hasattr(payload, "model_dump") else dict(payload)) for task_id, payload in dict(value).items()}


def merge_worker_results(current: Any, incoming: Any) -> dict[str, dict[str, Any]]:
    """LangGraph reducer that deterministically merges worker results by task id."""
    merged = _worker_result_mapping(current)
    merged.update(_worker_result_mapping(incoming))
    return merged


class SupervisorState(TypedDict, total=False):
    requirement: dict[str, Any]
    tasks: list[dict[str, Any]]
    groups: list[list[str]]
    group_index: int
    task: dict[str, Any]
    worker_results: Annotated[dict[str, dict[str, Any]], merge_worker_results]
    subagent_responses: Annotated[dict[str, dict[str, Any]], merge_worker_results]
    itinerary: list[dict[str, Any]]
    budget: dict[str, Any]
    draft: dict[str, Any]
    status: str
    warnings: list[str]
    task_id: str
    user_id: str
    conversation_id: str | None


def _candidate_to_option(candidate: EvidenceBoundCandidate, worker: str) -> CandidateOption:
    return CandidateOption(
        id=candidate.id or uuid4().hex,
        name=candidate.name,
        category=candidate.category or worker,
        description=candidate.description,
        estimated_cost=candidate.estimated_cost,
        attributes=dict(candidate.attributes),
        evidence_ids=candidate.evidence_ids,
    )


def _legacy_evidence_with_ids(result: WorkerResult):
    id_remap: dict[str, str] = {}
    normalized = []
    for index, item in enumerate(result.evidence):
        evidence_id = item.id or (
            f"{result.task_id}-ev-"
            f"{stable_hash(result.task_id, item.source, item.source_url or '', item.content, index)[:16]}"
        )
        if item.id:
            id_remap[item.id] = evidence_id
        normalized.append(item.model_copy(update={"id": evidence_id}))
    return normalized, id_remap


def _infer_legacy_option_evidence_ids(option: CandidateOption, evidence) -> list[str]:
    if option.evidence_ids:
        return option.evidence_ids
    matches = []
    option_name = option.name.strip().casefold()
    option_source = str(option.attributes.get("source", "")).strip().casefold()
    description = option.description.strip()
    for item in evidence:
        if item.id is None:
            continue
        source_matches = option_source and option_source == item.source.strip().casefold()
        content_matches = (
            (description and item.content.startswith(description))
            or (len(option_name) >= 4 and option_name in item.content.casefold())
        )
        if source_matches and content_matches:
            matches.append(item.id)
    if matches:
        return list(dict.fromkeys(matches))
    if len(evidence) == 1 and evidence[0].id is not None:
        return [evidence[0].id]
    return []


def _worker_result_to_subagent_response(result: WorkerResult) -> SubagentResponse:
    evidence, id_remap = _legacy_evidence_with_ids(result)
    return SubagentResponse(
        task_id=result.task_id,
        worker=result.worker,
        status=result.status,
        summary=result.summary,
        candidates=[
            EvidenceBoundCandidate(
                id=option.id,
                name=option.name,
                category=option.category,
                description=option.description,
                estimated_cost=option.estimated_cost,
                attributes=option.attributes,
                evidence_ids=[
                    id_remap.get(evidence_id, evidence_id)
                    for evidence_id in _infer_legacy_option_evidence_ids(option, evidence)
                ],
            )
            for option in result.options
        ],
        evidence=evidence,
        warnings=result.warnings,
    )


def _subagent_response_to_worker_result(
    response: SubagentResponse,
    governance: EvidenceGovernanceService | None = None,
    *,
    is_mock: bool = False,
) -> WorkerResult:
    reviewed = (governance or EvidenceGovernanceService()).review([response])
    summary = _governed_summary(response, reviewed)
    return WorkerResult(
        task_id=response.task_id,
        worker=response.worker,
        status=response.status,
        summary=summary,
        options=[_candidate_to_option(candidate, response.worker) for candidate in reviewed.candidates],
        evidence=reviewed.evidence,
        warnings=reviewed.warnings,
        is_mock=is_mock,
    )


def _reviewed_response_to_worker_result(
    response: SubagentResponse,
    *,
    is_mock: bool = False,
) -> WorkerResult:
    reviewed = ReviewedResearch(
        claims=response.claims,
        candidates=response.candidates,
        evidence=response.evidence,
        warnings=response.warnings,
    )
    return WorkerResult(
        task_id=response.task_id,
        worker=response.worker,
        status=response.status,
        summary=_governed_summary(response, reviewed),
        options=[_candidate_to_option(candidate, response.worker) for candidate in response.candidates],
        evidence=response.evidence,
        warnings=response.warnings,
        is_mock=is_mock,
    )


def _governed_summary(response: SubagentResponse, reviewed) -> str:
    if response.status == "failed":
        return "Domain subagent execution failed."
    if reviewed.claims:
        return " ".join(claim.text for claim in reviewed.claims)
    if reviewed.candidates:
        candidate_count = len(reviewed.candidates)
        noun = "candidate" if candidate_count == 1 else "candidates"
        return f"{candidate_count} evidence-governed {noun} retained."
    if reviewed.evidence:
        evidence_count = len(reviewed.evidence)
        noun = "evidence item" if evidence_count == 1 else "evidence items"
        return f"{evidence_count} {noun} retained; no evidence-bound claims retained."
    return "No evidence-governed claims retained."


def _coerce_subagent_response(result: Any, task: ResearchTask) -> SubagentResponse:
    if isinstance(result, SubagentResponse):
        return result
    if isinstance(result, WorkerResult):
        return _worker_result_to_subagent_response(result)
    try:
        return SubagentResponse.model_validate(result)
    except Exception:
        return _worker_result_to_subagent_response(WorkerResult.model_validate(result))


def _worker_results_from_state(state: SupervisorState) -> list[WorkerResult]:
    values = _worker_result_mapping(state.get("worker_results", {})).values()
    return [WorkerResult.model_validate(value) for value in values]


def _subagent_responses_from_state(state: SupervisorState) -> list[SubagentResponse]:
    response_values = _worker_result_mapping(state.get("subagent_responses", {}))
    if response_values:
        return [SubagentResponse.model_validate(value) for value in response_values.values()]
    return [_worker_result_to_subagent_response(result) for result in _worker_results_from_state(state)]


class _LLMItinerary(BaseModel):
    """LLM 综合输出的结构化行程。"""

    days: list[ItineraryDay] = Field(default_factory=list)


async def synthesize_itinerary_with_llm(
    requirement: TravelRequirement,
    results: list[WorkerResult],
    template: list[ItineraryDay],
) -> list[ItineraryDay]:
    """用 LLM 基于 Worker 证据润色行程；失败或无 Key 时退回模板。"""
    if not settings.llm_api_key:
        return template
    try:
        from app.agents.llm import get_llm

        evidence_digest = "\n".join(
            f"[{result.worker}] {result.summary} "
            + "；".join(option.name for option in result.options[:5])
            + ("；证据：" + "；".join(item.content[:120] for item in result.evidence[:3]) if result.evidence else "")
            for result in results
        )
        structured = get_llm(temperature=0.3).with_structured_output(_LLMItinerary)
        response = await structured.ainvoke(
            [
                {
                    "role": "system",
                    "content": (
                        "你是行程编排专家，以 JSON 格式输出。只能使用给定研究结果中出现过的景点、餐厅和事实，"
                        "不得虚构任何班次、价格、营业时间或天气结论；"
                        "缺乏依据的时段保持通用描述并注明需实时确认。"
                        "保持天数、日期和 morning/afternoon/evening 三时段结构与模板一致。"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"需求：{requirement.model_dump_json()}\n"
                        f"研究结果：\n{evidence_digest}\n"
                        f"模板行程：{[day.model_dump_json() for day in template]}\n"
                        "请在模板基础上，用研究结果改写每个时段的标题和描述。"
                    ),
                },
            ]
        )
        if len(response.days) == len(template):
            return response.days
        app_logger.warning("LLM 综合返回的天数与模板不一致，退回模板行程")
        return template
    except Exception:
        app_logger.warning("LLM itinerary synthesis failed; using deterministic template: llm_synthesis_failed")
        return template


def assemble_draft(
    requirement: TravelRequirement,
    results: list[WorkerResult],
    itinerary: list[ItineraryDay],
    budget: BudgetSummary,
    *,
    governance_warnings: list[str] | None = None,
) -> TravelPlanDraft:
    evidence_by_id: dict[str, Any] = {}
    for result in results:
        for item in result.evidence:
            key = item.id or f"{item.source}:{item.content}"
            evidence_by_id.setdefault(key, item)
    evidence = list(evidence_by_id.values())
    warnings = [*(governance_warnings or []), *(warning for result in results for warning in result.warnings)]
    degraded_reason = None
    runtime_results = [result for result in results if not result.is_mock]
    if runtime_results and all(result.status in {"unavailable", "failed"} for result in runtime_results):
        degraded_reason = "worker_unavailable"
    elif any(result.status in {"partial", "unavailable", "failed"} for result in runtime_results):
        degraded_reason = "provider_degraded"
    if degraded_reason is not None:
        warnings.append(f"planning_degraded:{degraded_reason}")
    return TravelPlanDraft(
        requirement=requirement,
        itinerary=itinerary,
        budget=budget,
        worker_results=results,
        evidence=evidence,
        warnings=list(dict.fromkeys(warnings)),
        status="degraded" if degraded_reason else "draft",
        degraded_reason=degraded_reason,
    )


def create_supervisor_graph(
    registry: Any | None = None,
    *,
    checkpointer=None,
    event_service: TaskEventService | None = None,
):
    """创建按依赖分组、组内并行的 LangGraph。"""
    registry = registry or create_default_subagent_registry()
    governance = EvidenceGovernanceService()

    async def emit(state: SupervisorState, event_type: str, payload: dict | None = None):
        if event_service is not None:
            await event_service.emit(
                task_id=state["task_id"],
                user_id=state["user_id"],
                conversation_id=state.get("conversation_id"),
                event_type=event_type,
                payload=payload,
            )

    async def planner_node(state: SupervisorState) -> dict[str, Any]:
        requirement = TravelRequirement.model_validate(state["requirement"])
        tasks = create_research_plan(requirement)
        groups = [[task.id for task in group] for group in parallel_groups(tasks)]
        await emit(state, "task_created", {"destination": requirement.destination})
        await emit(state, "plan_created", {"tasks": [task.model_dump(mode="json") for task in tasks]})
        return {
            "tasks": [task.model_dump(mode="json") for task in tasks],
            "groups": groups,
            "group_index": 0,
            "status": "planned",
        }

    async def dispatch_node(state: SupervisorState) -> dict[str, Any]:
        return {}

    def route_group(state: SupervisorState):
        groups = state.get("groups", [])
        index = state.get("group_index", 0)
        if index >= len(groups):
            return "route_planner"
        task_ids = set(groups[index])
        tasks = [task for task in state.get("tasks", []) if task["id"] in task_ids]
        completed = _worker_result_mapping(state.get("worker_results", {}))
        return [
            Send(
                "run_worker",
                {
                    "requirement": state["requirement"],
                    "task": task,
                    # 只把依赖任务的结果交给这个 Worker（如交通要用景点/住宿/美食候选）；
                    # 这些结果已经过 worker 节点的证据治理。
                    "worker_results": {
                        dependency: completed[dependency]
                        for dependency in task.get("dependencies", [])
                        if dependency in completed
                    },
                    "task_id": state["task_id"],
                    "user_id": state["user_id"],
                    "conversation_id": state.get("conversation_id"),
                },
            )
            for task in tasks
        ]

    async def worker_node(state: SupervisorState) -> dict[str, Any]:
        requirement = TravelRequirement.model_validate(state["requirement"])
        task = ResearchTask.model_validate(state["task"])
        await emit(state, "worker_started", {"task_id": task.id, "worker": task.task_type})

        async def emit_worker_event(event_type: str, payload: dict[str, Any] | None = None):
            await emit(
                state,
                event_type,
                {
                    "task_id": task.id,
                    "worker": task.task_type,
                    **(payload or {}),
                },
            )

        is_mock = False
        response: SubagentResponse | None = None
        run_kwargs: dict[str, Any] = {}
        if supports_keyword(registry.run, "event_callback"):
            run_kwargs["event_callback"] = emit_worker_event
        if task.dependencies and supports_keyword(registry.run, "prior_results"):
            run_kwargs["prior_results"] = _worker_results_from_state(state)
        try:
            raw_result = await registry.run(task, requirement, **run_kwargs)
            # 兼容旧 WorkerResult 的显式标记，同时按证据来源兜底：subagent 返回的
            # 是 SubagentResponse，只看 isinstance 会让本地模拟资料丢失 mock 披露。
            is_mock = isinstance(raw_result, WorkerResult) and raw_result.is_mock
            response = _coerce_subagent_response(raw_result, task)
            is_mock = is_mock or evidence_is_mock(response.evidence)
            if response.research_report and response.research_report.conflicts:
                await emit_worker_event(
                    "research_conflict",
                    {"conflict_count": len(response.research_report.conflicts)},
                )
            result = _subagent_response_to_worker_result(response, governance, is_mock=is_mock)
        except Exception:
            response = None
            result = WorkerResult(
                task_id=task.id,
                worker=task.task_type,
                status="failed",
                summary="Domain subagent execution failed.",
                warnings=["subagent_error:supervisor_worker_failed"],
                is_mock=is_mock,
            )
        await emit(state, "worker_completed", result.model_dump(mode="json"))
        if result.evidence:
            await emit(state, "evidence_collected", {"task_id": task.id, "count": len(result.evidence)})
        return {
            "worker_results": {task.id: result.model_dump(mode="json")},
            "subagent_responses": {task.id: response.model_dump(mode="json")} if response else {},
        }

    async def advance_node(state: SupervisorState) -> dict[str, Any]:
        return {"group_index": state.get("group_index", 0) + 1}

    async def route_planner_node(state: SupervisorState) -> dict[str, Any]:
        requirement = TravelRequirement.model_validate(state["requirement"])
        reviewed = governance.review(_subagent_responses_from_state(state))
        preliminary = _worker_result_mapping(state.get("worker_results", {}))
        results = [
            _reviewed_response_to_worker_result(
                response,
                # 治理复核后重建结果时同样要按证据兜底，否则 worker 节点推导出的
                # mock 标记会在这一步被 preliminary 里的缺省值悄悄抹掉。
                is_mock=bool(preliminary.get(response.task_id, {}).get("is_mock", False))
                or evidence_is_mock(response.evidence),
            )
            for response in reviewed.responses
        ]
        itinerary, scheduling_warnings = schedule_itinerary(requirement, results)
        await emit(state, "route_planned", {"days": len(itinerary)})
        return {
            "itinerary": [day.model_dump(mode="json") for day in itinerary],
            "worker_results": {result.task_id: result.model_dump(mode="json") for result in results},
            "warnings": [*reviewed.warnings, *scheduling_warnings],
        }

    async def budget_node(state: SupervisorState) -> dict[str, Any]:
        requirement = TravelRequirement.model_validate(state["requirement"])
        itinerary = [ItineraryDay.model_validate(value) for value in state.get("itinerary", [])]
        budget = calculate_budget(requirement, _worker_results_from_state(state), itinerary)
        await emit(state, "budget_estimated", {"total_estimate": budget.total_estimate})
        return {"budget": budget.model_dump(mode="json")}

    async def synthesizer_node(state: SupervisorState) -> dict[str, Any]:
        requirement = TravelRequirement.model_validate(state["requirement"])
        results = _worker_results_from_state(state)
        template = [ItineraryDay.model_validate(value) for value in state.get("itinerary", [])]
        budget = BudgetSummary.model_validate(state["budget"])
        itinerary = await synthesize_itinerary_with_llm(requirement, results, template)
        draft = assemble_draft(
            requirement,
            results,
            itinerary,
            budget,
            governance_warnings=state.get("warnings", []),
        )
        await emit(state, "plan_generated", {"days": len(draft.itinerary), "warnings": len(draft.warnings)})
        await emit(state, "task_completed", {"status": draft.status})
        return {
            "draft": draft.model_dump(mode="json"),
            "status": "completed",
            "warnings": draft.warnings,
        }

    workflow = StateGraph(SupervisorState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("dispatch", dispatch_node)
    workflow.add_node("run_worker", worker_node)
    workflow.add_node("advance", advance_node)
    workflow.add_node("route_planner", route_planner_node)
    workflow.add_node("budget", budget_node)
    workflow.add_node("synthesize", synthesizer_node)
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "dispatch")
    workflow.add_conditional_edges("dispatch", route_group, ["run_worker", "route_planner"])
    workflow.add_edge("run_worker", "advance")
    workflow.add_edge("advance", "dispatch")
    workflow.add_edge("route_planner", "budget")
    workflow.add_edge("budget", "synthesize")
    workflow.add_edge("synthesize", END)
    return workflow.compile(checkpointer=checkpointer)


async def run_travel_planning(
    requirement: TravelRequirement,
    registry: Any | None = None,
    *,
    checkpointer=None,
    event_service: TaskEventService | None = None,
    task_id: str | None = None,
    user_id: str = "anonymous",
    conversation_id: str | None = None,
) -> TravelPlanDraft:
    """供 API、测试和后台任务调用的结构化规划入口。"""
    task_id = task_id or uuid4().hex
    graph = create_supervisor_graph(registry, checkpointer=checkpointer, event_service=event_service)
    try:
        result = await graph.ainvoke(
            {
                "requirement": requirement.model_dump(mode="json"),
                "worker_results": {},
                "warnings": [],
                "task_id": task_id,
                "user_id": user_id,
                "conversation_id": conversation_id,
            },
            config={"configurable": {"thread_id": task_id}},
        )
    except Exception as exc:
        if event_service is not None:
            await event_service.emit(
                task_id=task_id,
                user_id=user_id,
                conversation_id=conversation_id,
                event_type="task_failed",
                payload={"error": type(exc).__name__},
            )
        raise
    return TravelPlanDraft.model_validate(result["draft"])


async def create_supervisor_agent():
    """兼容 Agent 工厂的异步构造入口。"""
    return create_supervisor_graph()


def build_itinerary(
    requirement: TravelRequirement,
    results: list[WorkerResult],
) -> list[ItineraryDay]:
    """Compatibility wrapper for the constraint-aware scheduler."""

    itinerary, _warnings = schedule_itinerary(requirement, results)
    return itinerary


def build_budget(
    requirement: TravelRequirement,
    results: list[WorkerResult] | None = None,
) -> BudgetSummary:
    """Compatibility wrapper for evidence-backed budget calculation."""

    return calculate_budget(requirement, results or [])
