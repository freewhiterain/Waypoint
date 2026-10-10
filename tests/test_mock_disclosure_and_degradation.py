"""is_mock 必须反映证据的真实来源，而不是执行路径。

`assemble_draft` 用 `is_mock` 把结果排除在降级统计之外（mock 数据不代表
线上 provider 不可用），因此 `is_mock` 一旦失真，会同时损坏两件事：

1. 向用户披露"这条建议来自本地模拟资料"；
2. `planning_degraded` 的触发。
"""

from datetime import date

import pytest

from app.agents.subagents.registry import SubagentRegistry
from app.agents.supervisor import assemble_draft, run_travel_planning
from app.agents.workers.rag_analysis import WorkerAnalysis, worker_result_from_analysis
from app.schemas.planning import (
    Evidence,
    ResearchTask,
    TravelRequirement,
    WorkerResult,
)
from app.schemas.research import SubagentResponse


WORKERS = ("attractions", "weather", "transport", "hotel", "food")


def _requirement() -> TravelRequirement:
    return TravelRequirement(
        origin="上海",
        destination="成都",
        departure_date=date(2026, 9, 1),
        days=3,
    )


def _mock_evidence(task_id: str) -> Evidence:
    return Evidence(
        id=f"{task_id}-evidence",
        content="熊猫基地上午开放。",
        source="attractions/chengdu.md",
        metadata={"source_type": "mock_markdown", "category": "attractions"},
    )


class _MockBackedSubagent:
    """返回本地模拟 Markdown 证据的 subagent。"""

    async def run(self, task, requirement):
        return SubagentResponse(
            task_id=task.id,
            worker=task.task_type,
            status="completed",
            evidence=[_mock_evidence(task.id)],
        )


@pytest.mark.asyncio
async def test_subagent_results_backed_by_mock_markdown_are_disclosed_as_mock():
    """subagent 路径上的模拟证据此前完全丢失 is_mock 标记。

    Supervisor 按 `isinstance(raw_result, WorkerResult)` 推导 is_mock，而
    subagent 返回的是 SubagentResponse，于是本地模拟资料被当成线上结果，
    用户界面上再也看不到"这是模拟数据"的披露。
    """
    registry = SubagentRegistry({name: _MockBackedSubagent() for name in WORKERS})

    draft = await run_travel_planning(_requirement(), registry=registry)

    assert all(result.is_mock for result in draft.worker_results)


def test_local_worker_keeps_mock_disclosure_even_without_evidence():
    """本地 Worker 一律披露 is_mock=True，与是否检索到证据无关。

    is_mock 表示"数据来源是本地模拟资料"，不是健康度信号；降级由
    assemble_draft 按状态另行判定。这条守住数据披露契约不被后续改动放宽。
    """
    task = ResearchTask(id="transport-task", task_type="transport", query="成都交通")
    analysis = WorkerAnalysis(
        summary="没有检索到交通证据。",
        options=[],
        warnings=["No evidence is available for this analysis."],
        used_mock_data=False,
    )

    result = worker_result_from_analysis(task, "transport", [], analysis)

    assert result.status == "unavailable"
    assert result.is_mock is True


def test_runtime_workers_without_evidence_still_degrade_the_draft():
    """真实 provider 全部不可用时必须降级。"""
    results = [
        WorkerResult(
            task_id=f"{name}-task",
            worker=name,
            status="unavailable",
            summary="没有可用证据。",
            evidence=[],
            is_mock=False,
        )
        for name in WORKERS
    ]

    draft = assemble_draft(_requirement(), results, [])

    assert draft.degraded_reason == "worker_unavailable"
    assert draft.status == "degraded"
    assert "planning_degraded:worker_unavailable" in draft.warnings


def test_mock_backed_results_still_keep_the_draft_out_of_provider_degraded():
    """模拟证据支撑的结果不应被当成 provider 故障。"""
    results = [
        WorkerResult(
            task_id=f"{name}-task",
            worker=name,
            status="completed",
            summary="根据本地资料整理。",
            evidence=[_mock_evidence(f"{name}-task")],
            is_mock=True,
        )
        for name in WORKERS
    ]

    draft = assemble_draft(_requirement(), results, [])

    assert draft.degraded_reason is None
    assert draft.status == "draft"
