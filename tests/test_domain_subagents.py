import pytest

from app.agents.subagents.attractions import AttractionsSubagent
from app.agents.subagents.food import FoodSubagent
from app.agents.subagents.hotel import HotelSubagent
from app.agents.subagents.registry import SubagentRegistry, create_default_subagent_registry
from app.agents.subagents.transport import TransportSubagent
from app.agents.subagents.weather import WeatherSubagent
from app.schemas.planning import Evidence, ResearchTask, TravelRequirement
from app.schemas.research import Claim, EvidenceBoundCandidate, ResearchReport
from app.agents.subagents.base import SubagentAnalysis


def requirement() -> TravelRequirement:
    return TravelRequirement(
        origin="Shanghai",
        destination="Chengdu",
        departure_date="2026-08-01",
        days=3,
        adults=2,
        styles=["family"],
        accommodation_preferences=["quiet"],
        food_preferences=["local"],
        transport_preferences=["train"],
    )


def task(
    worker: str,
    query: str | None = None,
    research_mode: str = "normal",
) -> ResearchTask:
    return ResearchTask(
        id=f"{worker}-task",
        task_type=worker,
        query=query or f"Find {worker} options in Chengdu",
        research_mode=research_mode,
    )


class FakeProvider:
    def __init__(self, provider: str, result):
        self.provider = provider
        self.calls = []
        self._result = result

    async def ainvoke(self, payload):
        self.calls.append(payload)
        if isinstance(self._result, BaseException):
            raise self._result
        return self._result


class FailIfCalled:
    calls = []

    async def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        raise AssertionError("Deep Search must not be called for this worker")


class RecordingDeepSearch:
    def __init__(self, report: ResearchReport):
        self.calls = []
        self._report = report

    async def __call__(self, request, **kwargs):
        self.calls.append((request, kwargs))
        return self._report


@pytest.mark.asyncio
async def test_attractions_subagent_returns_evidence_bound_candidates():
    agent = AttractionsSubagent(
        rag=FakeProvider(
            "local_rag",
            [Evidence(id="ev-rag", content="Panda Base is a family-friendly attraction.", source="local")],
        ),
        search=FakeProvider("search_mcp", []),
    )

    result = await agent.run(task("attractions"), requirement())

    assert result.worker == "attractions"
    assert result.status == "completed"
    assert result.candidates
    assert all(item.evidence_ids for item in result.candidates)
    assert {eid for item in result.candidates for eid in item.evidence_ids} <= {"ev-rag"}


@pytest.mark.asyncio
async def test_weather_subagent_does_not_call_deep_search():
    deep_search = FailIfCalled()
    agent = WeatherSubagent(
        weather_mcp=FakeProvider(
            "weather_mcp",
            [Evidence(id="ev-weather", content="Warm with afternoon rain.", source="weather")],
        ),
        deep_search=deep_search,
    )

    result = await agent.run(task("weather"), requirement())

    assert result.status == "completed"
    assert result.worker == "weather"
    assert deep_search.calls == []


@pytest.mark.asyncio
async def test_transport_subagent_uses_search_fallback_without_deep_search():
    deep_search = FailIfCalled()
    transport = FakeProvider("transport_mcp", [])
    search = FakeProvider(
        "search_mcp",
        [Evidence(id="ev-search", content="High-speed train is available.", source="search")],
    )
    agent = TransportSubagent(
        transport_mcp=transport,
        search=search,
        deep_search=deep_search,
    )

    result = await agent.run(task("transport"), requirement())

    assert result.status == "completed"
    assert [payload["provider"] for payload in (transport.calls + search.calls)] == [
        "transport_mcp",
        "search_mcp",
    ]
    assert [item.id for item in result.evidence] == ["ev-search"]
    assert deep_search.calls == []


@pytest.mark.asyncio
async def test_hotel_subagent_falls_back_from_empty_rag_to_hotel_provider_before_search():
    rag = FakeProvider("local_rag", [])
    hotel = FakeProvider(
        "hotel_mcp",
        [Evidence(id="ev-hotel", content="Hotel near Tianfu Square has family rooms.", source="hotel")],
    )
    search = FakeProvider("search_mcp", [])
    agent = HotelSubagent(rag=rag, hotel_mcp=hotel, search=search)

    result = await agent.run(task("hotel"), requirement())

    assert result.status == "completed"
    assert [payload["provider"] for payload in (rag.calls + hotel.calls)] == [
        "local_rag",
        "hotel_mcp",
    ]
    assert search.calls == []
    assert result.candidates[0].evidence_ids == ["ev-hotel"]


@pytest.mark.asyncio
async def test_food_subagent_uses_bounded_deep_search_only_after_rag_and_search_are_empty():
    deep_report = ResearchReport(
        status="completed",
        summary="Found current food evidence.",
        evidence=[Evidence(id="ev-deep", content="Try mapo tofu at a current listed restaurant.", source="web")],
    )
    deep_search = RecordingDeepSearch(deep_report)
    agent = FoodSubagent(
        rag=FakeProvider("local_rag", []),
        search=FakeProvider("search_mcp", []),
        deep_search=deep_search,
    )

    result = await agent.run(task("food"), requirement())

    assert result.status == "completed"
    assert [call[0].worker for call in deep_search.calls] == ["food"]
    assert result.research_report == deep_report
    assert result.candidates[0].evidence_ids == ["ev-deep"]


@pytest.mark.asyncio
async def test_explicit_deep_research_runs_even_when_rag_is_sufficient():
    deep_report = ResearchReport(
        status="completed",
        summary="Deep Search found current food evidence.",
        evidence=[Evidence(id="ev-deep-explicit", content="Current food listing.", source="web")],
    )
    deep_search = RecordingDeepSearch(deep_report)
    agent = FoodSubagent(
        rag=FakeProvider(
            "local_rag",
            [Evidence(id="ev-rag", content="Local food guide listing.", source="local_rag")],
        ),
        search=FakeProvider("search_mcp", []),
        deep_search=deep_search,
    )

    result = await agent.run(task("food", research_mode="deep"), requirement())

    assert len(deep_search.calls) == 1
    assert result.research_report == deep_report


@pytest.mark.asyncio
async def test_conflicting_provider_evidence_triggers_deep_search_even_when_a_provider_was_sufficient():
    # 用户报的核心问题：deep search 在信息冲突时不触发。改动前只要某个
    # provider 报 sufficient 就直接收尾，两个来源给出矛盾事实时也不补搜。
    deep_report = ResearchReport(
        status="completed",
        summary="Official source resolves the conflict.",
        evidence=[Evidence(id="ev-official", content="Official notice: open.", source="official")],
    )
    deep_search = RecordingDeepSearch(deep_report)
    agent = AttractionsSubagent(
        rag=FakeProvider(
            "local_rag",
            [
                Evidence(
                    id="ev-open",
                    content="Panda Base is open.",
                    source="guide-a",
                    metadata={"fact_key": "opening_status", "fact_value": "open"},
                ),
                Evidence(
                    id="ev-closed",
                    content="Panda Base is closed.",
                    source="guide-b",
                    metadata={"fact_key": "opening_status", "fact_value": "closed"},
                ),
            ],
        ),
        search=FakeProvider("search_mcp", []),
        deep_search=deep_search,
    )

    result = await agent.run(task("attractions"), requirement())

    assert len(deep_search.calls) == 1
    # 第一轮查询就要带上冲突事实，而不是白跑一轮原查询。
    assert "opening_status" in deep_search.calls[0][0].query
    assert any("disagree on opening_status" in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_conflicting_evidence_is_reported_when_the_worker_may_not_deep_search():
    # weather 的 ToolPolicy 不允许 deep research，冲突不能悄悄咽下去。
    deep_search = FailIfCalled()
    agent = WeatherSubagent(
        weather_mcp=FakeProvider(
            "weather_mcp",
            [
                Evidence(
                    id="ev-rain",
                    content="Rain is expected.",
                    source="api-a",
                    metadata={"fact_key": "precipitation", "fact_value": "rain"},
                ),
                Evidence(
                    id="ev-clear",
                    content="Clear skies are expected.",
                    source="api-b",
                    metadata={"fact_key": "precipitation", "fact_value": "clear"},
                ),
            ],
        ),
        deep_search=deep_search,
    )

    result = await agent.run(task("weather"), requirement())

    assert deep_search.calls == []
    assert any("disagree on precipitation" in warning for warning in result.warnings)
    assert any("unresolved" in warning.lower() for warning in result.warnings)


@pytest.mark.asyncio
async def test_agreeing_provider_evidence_does_not_trigger_deep_search():
    # 同一 fact_key 上取值一致时不算冲突，不能因此白烧一次补搜。
    deep_search = FailIfCalled()
    agent = AttractionsSubagent(
        rag=FakeProvider(
            "local_rag",
            [
                Evidence(
                    id="ev-a",
                    content="Panda Base is open.",
                    source="guide-a",
                    metadata={"fact_key": "opening_status", "fact_value": "open"},
                ),
                Evidence(
                    id="ev-b",
                    content="Panda Base is open in the morning.",
                    source="guide-b",
                    metadata={"fact_key": "opening_status", "fact_value": "open"},
                ),
            ],
        ),
        search=FakeProvider("search_mcp", []),
        deep_search=deep_search,
    )

    result = await agent.run(task("attractions"), requirement())

    assert deep_search.calls == []
    assert result.status == "completed"


def test_subagent_keeps_chinese_claims_that_reword_the_evidence():
    # 改动前 _text_supported 用整串子串匹配，而 re.findall(r"\w+") 不切中文，
    # 整段中文塌成一个 token。于是模型只要调整语序或加标点，claims 和
    # candidates 就会被全部丢弃——中文路径上的 grounding 等于删掉模型全部输出。
    agent = AttractionsSubagent()
    analysis = SubagentAnalysis(
        summary="上午开放的熊猫基地，需要预约门票。",
        claims=[Claim(text="熊猫基地上午开放", evidence_ids=["ev-1"])],
        candidates=[
            EvidenceBoundCandidate(
                name="熊猫基地",
                description="上午开放，门票需要预约",
                evidence_ids=["ev-1"],
            )
        ],
    )

    grounded, warnings = agent._ground_analysis(
        analysis,
        [
            Evidence(
                id="ev-1",
                content="熊猫基地上午开放，门票需要提前预约。",
                source="official",
            )
        ],
    )

    assert [claim.text for claim in grounded.claims] == ["熊猫基地上午开放"]
    assert [candidate.name for candidate in grounded.candidates] == ["熊猫基地"]
    assert grounded.summary
    assert warnings == []


def test_subagent_still_drops_chinese_content_introducing_facts_not_in_evidence():
    # 放宽到词级覆盖不能变成放弃校验：证据里没有的实词（价格、"免费"）必须拦住。
    agent = AttractionsSubagent()
    analysis = SubagentAnalysis(
        claims=[Claim(text="熊猫基地免费开放", evidence_ids=["ev-1"])],
        candidates=[
            EvidenceBoundCandidate(
                name="熊猫基地",
                description="门票 200 元",
                evidence_ids=["ev-1"],
            )
        ],
    )

    grounded, warnings = agent._ground_analysis(
        analysis,
        [Evidence(id="ev-1", content="熊猫基地上午开放，门票需要提前预约。", source="official")],
    )

    assert grounded.claims == []
    # 熊猫基地本身有证据，保留地点；编造的"200 元"被换成证据原文。
    assert [candidate.name for candidate in grounded.candidates] == ["熊猫基地"]
    assert "200" not in grounded.candidates[0].description
    assert grounded.candidates[0].description == "熊猫基地上午开放，门票需要提前预约。"
    assert len(warnings) >= 2


def test_subagent_keeps_a_real_place_but_strips_commentary_price_and_attributes():
    agent = AttractionsSubagent()
    analysis = SubagentAnalysis(
        candidates=[
            EvidenceBoundCandidate(
                name="成都大熊猫繁育研究基地",
                description="位于成华区，与本次熊猫偏好直接匹配",
                estimated_cost=58,
                attributes={"区域": "成华区", "适配偏好": ["熊猫", "亲子"]},
                evidence_ids=["ev-1"],
            )
        ],
    )

    grounded, warnings = agent._ground_analysis(
        analysis,
        [Evidence(id="ev-1", content="### 成都大熊猫繁育研究基地\n位于成华区。是熊猫文化主题下的代表性自然教育地点。",
                  source="local")],
    )

    candidate = grounded.candidates[0]
    assert candidate.description == "位于成华区。是熊猫文化主题下的代表性自然教育地点。"
    assert candidate.estimated_cost is None
    assert candidate.attributes == {"区域": "成华区"}
    assert "Removed unsupported details from a attractions candidate." in warnings


def test_estimated_cost_matches_an_integer_price_written_in_the_evidence():
    # estimated_cost 是 float，"55.0" 与证据里的 "55" 此前永远对不上，
    # 任何带价格的候选都会被丢弃。
    agent = AttractionsSubagent()
    analysis = SubagentAnalysis(
        candidates=[
            EvidenceBoundCandidate(
                name="熊猫基地",
                description="门票 55 元",
                estimated_cost=55,
                evidence_ids=["ev-1"],
            )
        ],
    )

    grounded, _ = agent._ground_analysis(
        analysis,
        [Evidence(id="ev-1", content="熊猫基地门票 55 元。", source="official")],
    )

    assert [candidate.estimated_cost for candidate in grounded.candidates] == [55]


def test_subagent_drops_content_not_supported_by_referenced_evidence():
    agent = AttractionsSubagent()
    analysis = SubagentAnalysis(
        summary="invented summary",
        claims=[Claim(text="invented claim", evidence_ids=["ev-1"])],
        candidates=[EvidenceBoundCandidate(
            name="invented place",
            description="invented details",
            estimated_cost=9999,
            attributes={"availability": "guaranteed"},
            evidence_ids=["ev-1"],
        )],
    )

    grounded, warnings = agent._ground_analysis(
        analysis,
        [Evidence(id="ev-1", content="Panda Base is open in the morning.", source="official")],
    )

    assert grounded.claims == []
    assert grounded.candidates == []
    assert grounded.summary == ""
    assert len(warnings) >= 2


def test_subagent_does_not_use_unreferenced_evidence_to_ground_a_claim():
    agent = AttractionsSubagent()
    analysis = SubagentAnalysis(
        claims=[Claim(text="Rain is expected.", evidence_ids=["ev-1"])],
    )

    grounded, warnings = agent._ground_analysis(
        analysis,
        [
            Evidence(id="ev-1", content="Panda Base is open.", source="official"),
            Evidence(id="ev-2", content="Rain is expected.", source="weather"),
        ],
    )

    assert grounded.claims == []
    assert any("unbound claim" in warning.lower() for warning in warnings)


@pytest.mark.asyncio
async def test_deep_search_failure_returns_structured_unavailable_response():
    async def failing_deep_search(*args, **kwargs):
        raise RuntimeError("provider secret should not escape")

    agent = FoodSubagent(
        rag=FakeProvider("local_rag", []),
        search=FakeProvider("search_mcp", []),
        deep_search=failing_deep_search,
    )

    result = await agent.run(task("food"), requirement())

    assert result.status == "unavailable"
    assert result.evidence == []
    assert any("deep search" in warning.lower() for warning in result.warnings)
    assert all("secret" not in warning for warning in result.warnings)


@pytest.mark.asyncio
async def test_subagent_registry_returns_structured_failure_for_unregistered_worker():
    registry = SubagentRegistry({})

    result = await registry.run(task("food"), requirement())

    assert result.status == "failed"
    assert result.worker == "food"
    assert result.candidates == []
    assert result.evidence == []


def test_default_subagent_registry_keeps_all_domain_workers_registered():
    registry = create_default_subagent_registry(build_tools=lambda worker: [])

    assert set(registry.workers) == {"attractions", "weather", "transport", "hotel", "food"}
