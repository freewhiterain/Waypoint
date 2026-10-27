import pytest

from app.schemas.tool_policy import ToolPolicy
from app.agents.subagents.tools import (
    ProviderError,
    build_subagent_tools,
    normalize_tool_result,
)
from app.mcp_core.client import MCPClientManager
from app.schemas.planning import Evidence


def test_weather_policy_excludes_rag_and_deep_research():
    policy = ToolPolicy.for_worker("weather")

    assert policy.allowed_tools == {"weather_mcp", "weather_fallback_api"}
    assert policy.allow_deep_research is False


def test_attractions_policy_allows_rag_search_and_deep_research():
    policy = ToolPolicy.for_worker("attractions")

    assert {"local_rag", "search_mcp"}.issubset(policy.allowed_tools)
    assert policy.allow_deep_research is True


def test_policy_matrix_is_explicit_for_all_domain_workers():
    assert ToolPolicy.for_worker("transport").allowed_tools == {"transport_mcp", "search_mcp"}
    assert ToolPolicy.for_worker("hotel").allowed_tools == {
        "local_rag", "hotel_mcp", "search_mcp", "deep_research"
    }
    assert ToolPolicy.for_worker("food").allowed_tools == {
        "local_rag", "search_mcp", "deep_research"
    }


def test_policy_rejects_unknown_workers():
    with pytest.raises(ValueError, match="Unsupported subagent worker"):
        ToolPolicy.for_worker("payments")


@pytest.mark.asyncio
async def test_mcp_manager_exposes_only_allowlisted_read_only_tools():
    class FakeTool:
        def __init__(self, name: str):
            self.name = name

    class FakeClient:
        async def get_tools(self):
            return [
                FakeTool("get_weather"),
                FakeTool("web_search"),
                FakeTool("delete_all_trips"),
            ]

    manager = MCPClientManager()
    manager.client = FakeClient()

    tools = await manager.get_allowed_tools({"weather_mcp", "delete_all_trips"})

    assert [tool.name for tool in tools] == ["get_weather"]


@pytest.mark.asyncio
async def test_build_tools_wraps_mcp_and_non_mcp_providers_without_write_tools():
    class FakeTool:
        name = "get_weather"

        async def ainvoke(self, payload):
            return {
                "status": "completed",
                "evidence": [{"content": payload["city"], "source": "weather"}],
                "internal_trace": "hidden",
            }

    class FakeManager:
        async def get_allowed_tools(self, allowed_providers):
            return [FakeTool()]

    class FakeWeather:
        async def query(self, city, forecast=True):
            return [Evidence(content=city, source="fallback")]

    tools = await build_subagent_tools(
        "weather",
        mcp_manager=FakeManager(),
        weather_adapter=FakeWeather(),
    )

    assert {tool.name for tool in tools} == {"get_weather", "weather_fallback_api"}
    result = await next(tool for tool in tools if tool.name == "get_weather").ainvoke({"city": "成都"})
    assert result[0].content == "成都"
    assert not hasattr(result[0], "internal_trace")


def test_normalizer_strips_hidden_metadata_from_evidence():
    result = normalize_tool_result(
        "search_mcp",
        {
            "evidence": [{
                "content": "public result",
                "source": "search",
                "metadata": {"raw": {"api_key": "secret"}, "provider": "tavily"},
            }]
        },
    )

    assert result[0].metadata == {"provider": "tavily"}


def test_normalizer_returns_evidence_without_the_hidden_provider_payload():
    result = normalize_tool_result(
        "weather_mcp",
        {
            "status": "completed",
            "evidence": [{"content": "Clear skies", "source": "weather provider"}],
            "internal_trace": "do-not-return",
        },
    )

    assert len(result) == 1
    assert isinstance(result[0], Evidence)
    assert result[0].content == "Clear skies"
    assert result[0].source == "weather provider"
    assert not hasattr(result[0], "internal_trace")


def test_normalizer_converts_provider_exceptions_to_sanitized_typed_errors():
    result = normalize_tool_result(
        "weather_mcp",
        RuntimeError("request failed api_key=super-secret-token"),
    )

    assert len(result) == 1
    assert isinstance(result[0], ProviderError)
    assert result[0].provider == "weather_mcp"
    assert result[0].code == "unavailable"
    assert "super-secret-token" not in result[0].message
