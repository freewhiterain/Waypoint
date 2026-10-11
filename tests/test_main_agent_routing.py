import pytest

from app.services.main_agent import MainAgentService


@pytest.mark.asyncio
async def test_affirmation_after_offer_opens_form():
    decision = await MainAgentService(use_llm=False).decide(
        "好的",
        [{"role": "assistant", "content": "需要我帮你规划一下旅行吗？"}],
    )

    assert decision.action == "collect_trip_requirements"


@pytest.mark.asyncio
async def test_direct_plan_request_opens_prefilled_form():
    decision = await MainAgentService(use_llm=False).decide("帮我规划一次成都旅行", [])

    assert decision.action == "collect_trip_requirements"
    assert decision.slot_update.destination == "成都"
    assert decision.slot_update.departure_date is None
    assert decision.slot_update.days is None


@pytest.mark.asyncio
async def test_explicit_planning_takes_precedence_over_open_question_markers():
    decision = await MainAgentService(use_llm=False).decide("我最近想规划一次成都旅行", [])

    assert decision.action == "collect_trip_requirements"


@pytest.mark.asyncio
async def test_open_question_stays_rag_even_with_old_destination():
    context = [{"role": "tool", "content": '{"destination":"成都"}'}]

    decision = await MainAgentService(use_llm=False).decide("最近成都有什么好玩的？", context)

    assert decision.action == "answer_open_question"


@pytest.mark.asyncio
async def test_destination_recommendation_is_separate_action():
    decision = await MainAgentService(use_llm=False).decide("还没想好去哪，帮我推荐", [])

    assert decision.action == "recommend_destination"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    ["帮我推荐适合亲子游的地方", "不知道去哪", "去哪玩比较好"],
)
async def test_destination_recommendation_covers_uncertain_destination_requests(message):
    decision = await MainAgentService(use_llm=False).decide(message, [])

    assert decision.action == "recommend_destination"


@pytest.mark.asyncio
async def test_known_city_attraction_question_is_not_destination_recommendation():
    decision = await MainAgentService(use_llm=False).decide("成都有哪些适合亲子游的地方？", [])

    assert decision.action == "answer_open_question"


@pytest.mark.asyncio
async def test_affirmation_without_offer_is_direct_response():
    decision = await MainAgentService(use_llm=False).decide("好的", [])

    assert decision.action == "direct_response"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "context",
    [
        [
            {"role": "assistant", "content": "需要我帮你规划一下旅行吗？"},
            {"role": "assistant", "content": "也可以先问我一个具体问题。"},
        ],
        [{"role": "assistant", "content": "需要我帮你规划一下旅行吗？现在开始吧。"}],
    ],
)
async def test_affirmation_requires_latest_exact_proactive_offer(context):
    decision = await MainAgentService(use_llm=False).decide("好的", context)

    assert decision.action == "direct_response"




@pytest.mark.asyncio
async def test_enabled_llm_without_key_does_not_attempt_routing_model(monkeypatch):
    calls = []

    def unexpected_llm():
        calls.append(True)
        raise AssertionError("routing must not call get_llm without a key")

    monkeypatch.setattr("app.services.main_agent.settings.llm_api_key", "")
    monkeypatch.setattr("app.agents.llm.get_llm", unexpected_llm)

    decision = await MainAgentService(use_llm=True).decide("我想想看", [])

    assert decision.action == "direct_response"
    assert calls == []


class _UnderstandingLlm:
    """假的结构化输出模型：记录收到的消息，返回预设的 TurnUnderstanding。"""

    def __init__(self, result):
        self.result = result
        self.messages = None

    def with_structured_output(self, schema):
        assert schema.__name__ == "TurnUnderstanding"
        return self

    async def ainvoke(self, messages):
        self.messages = messages
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def _use_llm(monkeypatch, result):
    llm = _UnderstandingLlm(result)
    monkeypatch.setattr("app.services.main_agent.settings.llm_api_key", "configured")
    monkeypatch.setattr("app.agents.llm.get_llm", lambda: llm)
    return llm


@pytest.mark.asyncio
async def test_model_understands_every_turn_and_returns_intent_with_slot_changes(monkeypatch):
    from app.schemas.slots import TripSlots

    llm = _use_llm(monkeypatch, {"intent": "open_question", "slots": {"destination": "成都", "days": 5}})

    decision = await MainAgentService().decide(
        "成都玩5天够吗", [{"role": "assistant", "content": "你好"}], TripSlots(adults=2)
    )

    # 开放问题也照样带回槽位变化：填槽每一轮都做，与走哪条线路无关。
    assert decision.action == "answer_open_question"
    assert (decision.slot_update.destination, decision.slot_update.days) == ("成都", 5)
    prompt = llm.messages[-1]["content"]
    assert '"adults":2' in prompt  # 当前槽位进了模型上下文
    assert "成都玩5天够吗" in prompt and "你好" in prompt


@pytest.mark.asyncio
async def test_planning_intent_comes_from_the_model_not_keywords(monkeypatch):
    _use_llm(monkeypatch, {"intent": "plan", "slots": {}})

    # 这句话里没有任何"规划""行程"关键词，旧的关键词规则识别不出来。
    decision = await MainAgentService().decide("那就这么定了，帮我弄一下吧", [])

    assert decision.action == "collect_trip_requirements"


@pytest.mark.asyncio
async def test_research_intent_needs_a_destination(monkeypatch):
    from app.schemas.slots import TripSlots

    _use_llm(monkeypatch, {"intent": "research", "slots": {}, "research_topic": "food", "deep_research": True})

    with_destination = await MainAgentService().decide("仔细查查当地吃的", [], TripSlots(destination="成都"))
    without_destination = await MainAgentService().decide("仔细查查当地吃的", [], TripSlots())

    assert with_destination.action == "invoke_agent_tool"
    assert with_destination.tool_call.name == "research_food"
    assert with_destination.tool_call.arguments.destination == "成都"
    assert with_destination.tool_call.arguments.research_mode == "deep"
    assert without_destination.action == "answer_open_question"


@pytest.mark.asyncio
async def test_chat_intent_uses_the_models_reply(monkeypatch):
    _use_llm(monkeypatch, {"intent": "chat", "slots": {}, "reply": "我只能帮忙处理旅行相关的事。"})

    decision = await MainAgentService().decide("帮我写个冒泡排序", [])

    assert decision.action == "direct_response"
    assert decision.response == "我只能帮忙处理旅行相关的事。"


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_result", [None, {"intent": "not-a-real-intent"}, TimeoutError("model down")])
async def test_falls_back_to_keyword_rules_when_the_model_gives_nothing_usable(monkeypatch, bad_result):
    _use_llm(monkeypatch, bad_result)

    decision = await MainAgentService().decide("帮我规划一次成都旅行", [])

    assert decision.action == "collect_trip_requirements"
    assert decision.slot_update.destination == "成都"
