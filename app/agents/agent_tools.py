"""Agent-as-tool adapters for delegating focused research to domain subagents."""

from __future__ import annotations

from datetime import date
from typing import Any

from app.agents.factory import create_planning_registry
from app.schemas.planning import ResearchTask, TaskType, TravelRequirement
from app.schemas.research import SubagentResponse
from app.schemas.tools import AgentToolCall, AgentToolName, AgentToolResult
from app.utils.callables import supported_kwargs
from app.utils.logger import app_logger


TOOL_WORKERS: dict[AgentToolName, TaskType] = {
    "research_attractions": "attractions",
    "research_weather": "weather",
    "research_transport": "transport",
    "research_hotel": "hotel",
    "research_food": "food",
}

_UNKNOWN_TOOL_ANSWER = "这个研究工具不存在，请换一个具体的研究方向。"
_EXECUTION_FAILED_ANSWER = "这次研究没有完成，请稍后再试。"
_NO_EVIDENCE_ANSWER = "没有查到可以支撑结论的资料，暂时无法给出这个方向的研究结果。"
_MAX_LINES = 5


class AgentToolRegistry:
    """Expose the existing domain subagents through explicit, read-only tool names."""

    def __init__(self, *, registry: Any | None = None):
        self.registry = registry or create_planning_registry()[0]

    async def invoke(self, call: AgentToolCall, *, event_callback=None) -> AgentToolResult:
        """Run one domain subagent and return a sanitized, evidence-grounded result."""

        worker = TOOL_WORKERS.get(call.name)
        if worker is None:
            return AgentToolResult(
                tool_name=call.name,
                worker=None,
                status="failed",
                answer=_UNKNOWN_TOOL_ANSWER,
                warnings=["agent_tool_error:unknown_tool"],
            )

        arguments = call.arguments
        task = ResearchTask(
            task_type=worker,
            query=arguments.query,
            research_mode=arguments.research_mode,
        )
        requirement = TravelRequirement(
            destination=arguments.destination,
            departure_date=arguments.departure_date or date.today(),
            days=arguments.days,
        )

        try:
            response = await self.registry.run(
                task,
                requirement,
                **supported_kwargs(self.registry.run, event_callback=event_callback),
            )
        except Exception as exc:
            app_logger.warning(
                f"Agent tool {call.name} failed: {type(exc).__name__}: {exc}"
            )
            return AgentToolResult(
                tool_name=call.name,
                worker=worker,
                status="failed",
                answer=_EXECUTION_FAILED_ANSWER,
                warnings=["agent_tool_error:execution_failed"],
            )

        return AgentToolResult(
            tool_name=call.name,
            worker=response.worker,
            status=response.status,
            answer=_render_response(response),
            evidence_count=len(response.evidence),
            warnings=list(response.warnings),
        )


def _render_response(response: SubagentResponse) -> str:
    """Render only grounded subagent output; never invent facts."""

    lines: list[str] = []
    if response.summary:
        lines.append(response.summary)

    seen: set[str] = set()
    for claim in response.claims[:_MAX_LINES]:
        if claim.text not in seen:
            seen.add(claim.text)
            lines.append(f"- {claim.text}")

    if not response.claims:
        for candidate in response.candidates[:_MAX_LINES]:
            description = f"：{candidate.description}" if candidate.description else ""
            lines.append(f"- {candidate.name}{description}")

    sources = list(
        dict.fromkeys(
            item.source_url or item.source for item in response.evidence if item.source_url or item.source
        )
    )[:_MAX_LINES]
    if sources:
        lines.append("")
        lines.append("资料来源：")
        lines.extend(f"- {source}" for source in sources)

    if not lines:
        return _NO_EVIDENCE_ANSWER
    return "\n".join(lines)


async def run_agent_tool(
    call: AgentToolCall,
    *,
    registry: Any | None = None,
    event_callback=None,
) -> AgentToolResult:
    """Default executor seam used by the chat stream."""

    return await AgentToolRegistry(registry=registry).invoke(
        call,
        event_callback=event_callback,
    )


__all__ = ["AgentToolRegistry", "TOOL_WORKERS", "run_agent_tool"]
