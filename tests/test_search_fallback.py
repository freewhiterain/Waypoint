"""联网搜索降级链：Tavily → SearXNG；没有搜索 MCP 时直接给子 Agent 用。"""

import pytest

from app.agents.subagents.tools import build_subagent_tools
from app.mcp_core.adapters import search as search_module
from app.mcp_core.reliability import ExternalServiceError
from app.schemas.planning import Evidence


class _Adapter:
    def __init__(self, name, fail=False):
        self.name, self.fail, self.calls = name, fail, []

    async def search(self, query, max_results=5):
        self.calls.append((query, max_results))
        if self.fail:
            raise ExternalServiceError("quota exceeded")
        return [Evidence(content=f"{self.name}:{query}", source=self.name, source_url=f"https://{self.name}.test")]


class _NoTools:
    async def get_allowed_tools(self, _allowed):
        return []


@pytest.mark.asyncio
async def test_falls_back_to_the_next_source_when_one_fails():
    tavily, searxng = _Adapter("tavily", fail=True), _Adapter("searxng")

    results = await search_module.FallbackSearchAdapter([tavily, searxng]).search("西安 景点", 3)

    assert [item.content for item in results] == ["searxng:西安 景点"]
    assert tavily.calls and searxng.calls == [("西安 景点", 3)]


@pytest.mark.asyncio
async def test_all_sources_failing_raises_a_service_error():
    with pytest.raises(ExternalServiceError):
        await search_module.FallbackSearchAdapter([_Adapter("a", fail=True)]).search("q")


@pytest.mark.parametrize(
    ("tavily", "searxng", "expected"),
    [("", "", None), ("key", "", ["TavilySearchAdapter"]), ("key", "http://s", ["TavilySearchAdapter", "SearxngSearchAdapter"])],
)
def test_default_chain_follows_configuration(monkeypatch, tavily, searxng, expected):
    monkeypatch.setattr(search_module.settings, "tavily_api_key", tavily)
    monkeypatch.setattr(search_module.settings, "searxng_url", searxng)

    adapter = search_module.default_search_adapter()

    assert (None if adapter is None else [type(item).__name__ for item in adapter.adapters]) == expected


@pytest.mark.asyncio
async def test_search_adapter_is_registered_as_web_search_when_no_search_mcp_exists():
    adapter = _Adapter("searxng")

    tools = await build_subagent_tools(
        "attractions", mcp_manager=_NoTools(), knowledge=object(), graph=object(), search_adapter=adapter
    )
    web = next(tool for tool in tools if tool.name == "web_search")
    result = await web._handler({"query": "西安 景点", "limit": 4})

    assert web.provider == "search_mcp"
    assert result[0].content == "searxng:西安 景点" and adapter.calls == [("西安 景点", 4)]


def test_planning_search_uses_engine_friendly_keywords_but_deep_search_queries_pass_through():
    from app.agents.subagents.tools import web_search_queries

    planning = {"query": "研究西安本地美食与用户饮食偏好匹配情况 西安", "destination": "西安", "category": "food"}
    deep = {**planning, "query": "西安 回民街 老字号", "limit": 3}

    assert web_search_queries(planning) == ["西安 必吃 美食 餐厅 推荐", "西安 美食攻略 老字号"]
    assert web_search_queries(deep) == ["西安 回民街 老字号"]


@pytest.mark.asyncio
async def test_planning_search_merges_queries_and_drops_results_not_about_the_city():
    from app.agents.subagents.tools import search_for_planning

    class Noisy:
        async def search(self, query, max_results=5):
            return [
                Evidence(content=f"厦门 鼓浪屿 {query}", source="攻略", source_url="https://a.test"),
                Evidence(content="一首老歌的歌词", source="歌词网", source_url="https://lyrics.test"),
                Evidence(content="厦门 鼓浪屿 重复", source="攻略", source_url="https://a.test"),
            ]

    results = await search_for_planning(Noisy(), {"query": "x", "destination": "厦门", "category": "attractions"})

    assert [item.source_url for item in results] == ["https://a.test"]


def test_places_the_user_named_get_their_own_search():
    from app.agents.subagents.tools import web_search_queries

    payload = {"query": "x", "destination": "大理", "category": "attractions", "must_visit": ["喜洲古镇"]}

    assert web_search_queries(payload)[-1] == "大理 喜洲古镇"
    assert "大理 喜洲古镇" not in web_search_queries({**payload, "category": "weather"})


@pytest.mark.asyncio
async def test_searxng_uses_preferred_engines_and_falls_back_to_defaults_when_they_are_blocked(monkeypatch):
    calls = []

    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self.payload

    class Client:
        def __init__(self, **_):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return False

        async def get(self, _url, params):
            calls.append(params.get("engines"))
            # 搜狗被验证码暂停：没有结果；默认引擎有结果
            results = [] if params.get("engines") else [{"content": "西安 兵马俑", "url": "https://x.test", "title": "t"}]
            return Response({"results": results})

    monkeypatch.setattr(search_module.httpx, "AsyncClient", Client)
    adapter = search_module.SearxngSearchAdapter(base_url="http://s", engines="sogou,quark")

    results = await adapter.search("西安 兵马俑")

    assert calls == ["sogou,quark", None]
    assert [item.content for item in results] == ["西安 兵马俑"]
