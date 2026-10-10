from datetime import date

import pytest
from langchain_core.documents import Document

from app.agents.workers.rag_analysis import WorkerAnalysis, analyze_worker_evidence
from app.agents.workers.local_knowledge import LocalKnowledgeService
from app.agents.workers.attractions import AttractionsWorker
from app.agents.workers.food import FoodWorker
from app.agents.workers.hotel import HotelWorker
from app.agents.workers.transport import TransportWorker
from app.agents.workers.weather import WeatherWorker
from app.schemas.planning import CandidateOption, Evidence, ResearchTask, TravelRequirement, WorkerResult
from app.agents.supervisor import assemble_draft


def make_requirement() -> TravelRequirement:
    return TravelRequirement(
        origin="Shanghai",
        destination="Chengdu",
        departure_date=date(2026, 8, 1),
        days=3,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("worker_name", "worker_class"),
    [
        ("attractions", AttractionsWorker),
        ("weather", WeatherWorker),
        ("transport", TransportWorker),
        ("hotel", HotelWorker),
        ("food", FoodWorker),
    ],
)
async def test_local_workers_use_category_scoped_rag(worker_name, worker_class):
    result = await worker_class(knowledge=LocalKnowledgeService()).run(
        ResearchTask(task_type=worker_name, query=f"Chengdu {worker_name}"),
        make_requirement(),
    )

    assert result.is_mock is True
    assert result.evidence or result.status == "unavailable"
    assert all(item.metadata.get("category") == worker_name for item in result.evidence)


def test_supervisor_assembly_preserves_mock_evidence_and_warnings():
    evidence = Evidence(
        content="### Panda Base\nSuitable for a morning visit.",
        source="attractions/chengdu.md",
        metadata={"source_type": "mock_markdown", "category": "attractions"},
    )
    results = [
        WorkerResult(
            task_id="a",
            worker="attractions",
            status="completed",
            summary="evidence-backed",
            evidence=[evidence],
            warnings=["Local mock data."],
            is_mock=True,
        ),
        WorkerResult(
            task_id="w",
            worker="weather",
            status="unavailable",
            summary="No evidence.",
            warnings=["Local mock data."],
            is_mock=True,
        ),
    ]

    draft = assemble_draft(make_requirement(), results, [])

    assert draft.evidence == [evidence]
    assert draft.warnings == ["Local mock data."]
    assert all(result.is_mock for result in draft.worker_results)


@pytest.mark.asyncio
async def test_analyze_worker_evidence_uses_structured_llm_with_supplied_evidence_only():
    class FakeStructuredLlm:
        def __init__(self):
            self.messages = None

        def with_structured_output(self, schema):
            assert schema is WorkerAnalysis
            return self

        async def ainvoke(self, messages):
            self.messages = messages
            return WorkerAnalysis(
                summary="The evidence says advance reservations are required.",
                options=[
                    CandidateOption(
                        name="Panda Base",
                        category="attractions",
                        description="Advance reservations are required.",
                    )
                ],
                warnings=[],
                used_mock_data=True,
            )

    llm = FakeStructuredLlm()
    evidence = [
        Evidence(
            content="Panda Base requires advance reservations.",
            source="attractions/chengdu.md",
            metadata={"source_type": "mock_markdown"},
        )
    ]

    analysis = await analyze_worker_evidence(
        "attractions",
        ResearchTask(task_type="attractions", query="Panda Base"),
        make_requirement(),
        evidence,
        llm=llm,
    )

    assert analysis.summary == "根据检索到的 attractions 证据整理了 1 条候选信息。"
    assert analysis.options[0].name == "Panda Base"
    assert analysis.used_mock_data is True
    prompt = "\n".join(message["content"] for message in llm.messages)
    assert "Panda Base requires advance reservations." in prompt
    assert "attractions/chengdu.md" in prompt
    assert "Do not invent prices, schedules, operating status, inventory, or weather facts." in prompt
    assert "hidden chain-of-thought" in prompt


@pytest.mark.asyncio
async def test_analyze_worker_evidence_returns_no_options_without_evidence():
    analysis = await analyze_worker_evidence(
        "weather",
        ResearchTask(task_type="weather", query="forecast"),
        make_requirement(),
        [],
    )

    assert analysis.options == []
    assert analysis.used_mock_data is False
    assert analysis.warnings == ["No evidence is available for this analysis."]
    assert "No evidence is available" in analysis.summary


@pytest.mark.asyncio
async def test_no_evidence_short_circuits_even_when_llm_is_configured():
    class ExplodingLlm:
        def with_structured_output(self, _schema):
            raise AssertionError("LLM must not run without evidence")

    analysis = await analyze_worker_evidence(
        "weather",
        ResearchTask(task_type="weather", query="forecast"),
        make_requirement(),
        [],
        llm=ExplodingLlm(),
    )

    assert analysis.options == []
    assert analysis.warnings == ["No evidence is available for this analysis."]


@pytest.mark.asyncio
async def test_structured_llm_failure_falls_back_to_evidence_summary():
    class FailingLlm:
        def with_structured_output(self, _schema):
            return self

        async def ainvoke(self, _messages):
            raise RuntimeError("model unavailable")

    evidence = [Evidence(content="### 熊猫基地\n适合上午游览", source="attractions/chengdu.md")]
    analysis = await analyze_worker_evidence(
        "attractions",
        ResearchTask(task_type="attractions", query="成都景点"),
        make_requirement(),
        evidence,
        llm=FailingLlm(),
    )

    assert len(analysis.options) == 1
    assert "降级为证据摘要" in analysis.warnings[0]
    assert analysis.options[0].attributes["source"] == "attractions/chengdu.md"


@pytest.mark.asyncio
async def test_structured_options_without_evidence_are_discarded():
    class FabricatingLlm:
        def with_structured_output(self, _schema):
            return self

        async def ainvoke(self, _messages):
            return WorkerAnalysis(
                summary="Unsupported recommendation with price 999.",
                options=[
                    CandidateOption(name="不存在的景点", category="attractions"),
                    CandidateOption(name="", category="attractions"),
                ],
                used_mock_data=False,
            )

    evidence = [Evidence(content="### 熊猫基地\n适合上午游览", source="attractions/chengdu.md")]
    analysis = await analyze_worker_evidence(
        "attractions",
        ResearchTask(task_type="attractions", query="成都景点"),
        make_requirement(),
        evidence,
        llm=FabricatingLlm(),
    )

    assert analysis.options == []
    assert any("缺少证据支持" in warning for warning in analysis.warnings)
    assert "price" not in analysis.summary


def test_worker_result_supports_unavailable_status_and_mock_default():
    result = WorkerResult(
        task_id="weather-1",
        worker="weather",
        status="unavailable",
        summary="Weather source is unavailable.",
    )

    assert result.is_mock is False
    assert result.model_dump()["is_mock"] is False


def test_search_destination_returns_evidence_only_from_requested_category():
    service = LocalKnowledgeService(
        documents=[
            Document(
                page_content="Chengdu Panda Base requires advance attraction reservations.",
                metadata={"source": "attractions/chengdu.md", "city": " 成都 ", "category": "attractions"},
            ),
            Document(
                page_content="Chengdu Metro Line 3 connects major transport hubs.",
                metadata={"source": "transport/chengdu.md", "city": "成都", "category": "transport"},
            ),
        ]
    )

    evidence = service.search_destination("成都", "attractions", "Panda Base reservations")

    assert [item.content for item in evidence] == ["Chengdu Panda Base requires advance attraction reservations."]
    assert evidence[0].source == "attractions/chengdu.md"
    assert evidence[0].metadata["city"] == " 成都 "
    assert evidence[0].metadata["category"] == "attractions"


def test_search_destination_returns_empty_when_category_has_no_city_documents():
    service = LocalKnowledgeService(
        documents=[
            Document(
                page_content="Chengdu Panda Base requires advance attraction reservations.",
                metadata={"source": "attractions/chengdu.md", "city": "成都", "category": "attractions"},
            ),
        ]
    )

    evidence = service.search_destination("成都", "weather", "forecast")

    assert evidence == []


def test_search_keeps_global_retrieval_compatible():
    service = LocalKnowledgeService(
        documents=[
            Document(
                page_content="Chengdu Panda Base requires advance attraction reservations.",
                metadata={"source": "attractions/chengdu.md", "city": "成都", "category": "attractions"},
            ),
            Document(
                page_content="Chengdu Metro Line 3 connects major transport hubs.",
                metadata={"source": "transport/chengdu.md", "city": "成都", "category": "transport"},
            ),
        ]
    )

    evidence = service.search("Metro Line 3")

    assert evidence[0].source == "transport/chengdu.md"
    assert evidence[0].metadata["category"] == "transport"


def test_search_destination_returns_empty_for_an_empty_injected_corpus():
    assert LocalKnowledgeService(documents=[]).search_destination("成都", "weather", "forecast") == []


def test_search_destination_returns_empty_when_matching_document_has_no_content():
    service = LocalKnowledgeService(
        documents=[
            Document(
                page_content="",
                metadata={"source": "weather/chengdu.md", "city": "成都", "category": "weather"},
            ),
        ]
    )

    assert service.search_destination("成都", "weather", "forecast") == []
