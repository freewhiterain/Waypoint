"""Weather domain subagent."""

from __future__ import annotations

from app.agents.subagents.base import DomainSubagent, SubagentAnalysis
from app.schemas.planning import Evidence, ResearchTask, TravelRequirement
from app.schemas.research import Claim


class WeatherSubagent(DomainSubagent):
    def __init__(self, **kwargs):
        super().__init__(
            worker="weather",
            provider_order=("weather_mcp", "weather_fallback_api"),
            **kwargs,
        )

    def build_query(self, task: ResearchTask, requirement: TravelRequirement) -> str:
        return (
            f"{task.query} {requirement.destination} "
            f"{requirement.departure_date.isoformat()} {requirement.days} days"
        )

    async def _analyze(
        self,
        task: ResearchTask,
        requirement: TravelRequirement,
        evidence: list[Evidence],
    ) -> SubagentAnalysis:
        # 每条预报本身就是一句完整的事实，逐条引用即可。让模型转述一遍反而常被证据校验
        # 丢掉，结果是天气明明查到了，这个子 Agent 却被标成"部分完成"。
        return SubagentAnalysis(
            claims=[Claim(text=item.content, evidence_ids=[item.id]) for item in evidence if item.id]
        )


__all__ = ["WeatherSubagent"]
