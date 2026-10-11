from datetime import date

import pytest
from langchain_core.documents import Document

from app.rag.local_knowledge import LocalKnowledgeService
from app.schemas.planning import CandidateOption, Evidence, ResearchTask, TravelRequirement, WorkerResult
from app.agents.supervisor import assemble_draft


def make_requirement() -> TravelRequirement:
    return TravelRequirement(
        origin="Shanghai",
        destination="Chengdu",
        departure_date=date(2026, 8, 1),
        days=3,
    )



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
