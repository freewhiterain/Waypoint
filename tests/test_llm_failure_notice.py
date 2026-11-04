"""模型调用失败时给用户的说明：余额、密钥、限流、网络分别说清楚，而不是悄悄降级。"""

import httpx
import openai
import pytest

from app.core.llm import describe_llm_failure

_REQUEST = httpx.Request("POST", "http://llm.test/v1/chat/completions")


def _status_error(code):
    return openai.APIStatusError("x", response=httpx.Response(code, request=_REQUEST), body=None)


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (_status_error(402), "余额不足"),
        (_status_error(401), "密钥无效"),
        (_status_error(429), "繁忙"),
        (_status_error(500), "暂时出错"),
        (openai.APIConnectionError(request=_REQUEST), "连不上"),
        (ValueError("bad json"), "没有给出有效结果"),
    ],
)
def test_provider_errors_get_a_plain_explanation(exc, expected):
    assert expected in describe_llm_failure(exc)


@pytest.mark.asyncio
async def test_open_question_fallback_says_why_the_answer_is_raw(monkeypatch):
    from app.core import llm as llm_module
    from app.schemas.planning import Evidence
    from app.services import open_qa

    class Broken:
        async def ainvoke(self, _messages):
            raise _status_error(402)

    class Knowledge:
        def search(self, _query):
            return [Evidence(content="宽窄巷子全天开放", source="攻略", source_url="https://a.test")]

    monkeypatch.setattr(open_qa.settings, "llm_api_key", "k")
    monkeypatch.setattr(open_qa, "get_local_knowledge_service", lambda: Knowledge())
    monkeypatch.setattr(llm_module, "get_llm", lambda **_: Broken())

    answer = await open_qa.answer_open_question("宽窄巷子几点开")

    assert answer.startswith("（模型服务余额不足") and "宽窄巷子全天开放" in answer


def test_plan_notice_names_the_reason_the_model_was_skipped():
    from app.agents.itinerary_planner import FALLBACK_REASON_PREFIX
    from app.services.plan_render import user_facing_warnings

    notices = user_facing_warnings(
        ["itinerary_selection_fallback:llm_failed", f"{FALLBACK_REASON_PREFIX}模型服务余额不足，暂时无法调用"]
    )

    assert notices == ["智能选点暂时不可用（模型服务余额不足，暂时无法调用），本次按固定规则选点。"]
    assert user_facing_warnings(["itinerary_selection_fallback:llm_failed"]) == ["智能选点暂时不可用，本次按固定规则选点。"]


@pytest.mark.asyncio
async def test_open_question_drops_other_cities_and_local_paths(monkeypatch):
    from app.schemas.planning import Evidence
    from app.services import open_qa

    class Knowledge:
        def search(self, _query):
            return [
                Evidence(content="肉夹馍推荐赵记", source=r"D:\data\destinations\xian.md", source_url="https://a.test"),
                Evidence(content="杭州名菜西湖醋鱼", source=r"D:\data\destinations\hangzhou.md", source_url="https://a.test"),
            ]

    monkeypatch.setattr(open_qa.settings, "llm_api_key", "")
    monkeypatch.setattr(open_qa, "get_local_knowledge_service", lambda: Knowledge())

    answer = await open_qa.answer_open_question("杭州有什么特色菜推荐")

    assert "西湖醋鱼" in answer and "肉夹馍" not in answer
    assert "D:" not in answer and "hangzhou.md" in answer


@pytest.mark.asyncio
async def test_open_question_says_when_the_city_has_no_local_material(monkeypatch):
    from app.schemas.planning import Evidence
    from app.services import open_qa

    class Knowledge:
        def search(self, _query):
            return [Evidence(content="肉夹馍推荐赵记", source="xian.md", source_url="https://a.test")]

    monkeypatch.setattr(open_qa, "get_local_knowledge_service", lambda: Knowledge())

    answer = await open_qa.answer_open_question("杭州有什么特色菜推荐")

    assert "没有杭州" in answer and "肉夹馍" not in answer


def test_source_name_strips_windows_and_posix_directories():
    from app.services.open_qa import source_name

    assert source_name(r"D:\data\destinations\xian.md") == "xian.md"
    assert source_name("/srv/data/xian.md") == "xian.md"
    assert source_name("攻略") == "攻略"
