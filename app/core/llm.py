"""共享的 LLM 构造入口（OpenAI 兼容接口，供应商由 .env 决定）。"""

from typing import Any

import httpx
from langchain_openai import ChatOpenAI

from app.config import settings


# 跨调用复用的 httpx 客户端。get_llm 在每一次对话轮次都会被调用
# （main_agent 路由、planning 抽取、open_qa、supervisor 行程合成），
# 原先每次都新建一对客户端且从不 close，连接池随请求量单调增长直到
# 耗尽文件描述符。httpx 客户端本身可并发复用，连接池正是为此设计的。
_http_client: httpx.Client | None = None
_http_async_client: httpx.AsyncClient | None = None


def _shared_http_clients() -> tuple[httpx.Client, httpx.AsyncClient]:
    """惰性创建并复用同一对 httpx 客户端。

    环境兼容性说明：本机开代理时,openai SDK 内部的 httpx 客户端会读 HTTPS_PROXY,
    即使设了 NO_PROXY 在异步场景也可能不生效,因此显式传入不走代理的 httpx 客户端。
    """
    global _http_client, _http_async_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.Client(trust_env=False, timeout=60.0)
    if _http_async_client is None or _http_async_client.is_closed:
        _http_async_client = httpx.AsyncClient(trust_env=False, timeout=60.0)
    return _http_client, _http_async_client


def reset_http_clients() -> None:
    """关闭并丢弃共享客户端。供应用关闭和测试隔离使用。

    异步客户端的 aclose 需要事件循环，这里只做同步 close；未关闭的异步
    连接会在进程退出时由 GC 回收，不影响运行期的池复用。
    """
    global _http_client, _http_async_client
    if _http_client is not None and not _http_client.is_closed:
        _http_client.close()
    _http_client = None
    _http_async_client = None


async def aclose_http_clients() -> None:
    """在事件循环内彻底关闭共享客户端，供 FastAPI lifespan 收尾调用。"""
    global _http_client, _http_async_client
    if _http_async_client is not None and not _http_async_client.is_closed:
        await _http_async_client.aclose()
    _http_async_client = None
    if _http_client is not None and not _http_client.is_closed:
        _http_client.close()
    _http_client = None


class StructuredChatOpenAI(ChatOpenAI):
    """把结构化输出固定走 function_calling 的 ChatOpenAI。

    langchain 默认用 `json_schema`（response_format），但 DeepSeek 对此返回
    400 "This response_format type is unavailable now"。`json_mode` 又不校验
    schema——实测会返回 action="plan_trip" 这种不在枚举里的值，等于把校验推给
    下游。只有 function_calling 既被支持、又能保住 Pydantic 契约。

    调用方仍可显式传 method 覆盖，换回支持 json_schema 的供应商时不用改这里。
    """

    def with_structured_output(self, schema=None, *, method: str = "function_calling", **kwargs: Any):
        return super().with_structured_output(schema, method=method, **kwargs)


def get_llm(temperature: float | None = None) -> StructuredChatOpenAI:
    """获取配置好的对话模型（OpenAI 兼容接口）。

    每次返回新的实例（temperature 等参数按调用方变化），但底层 httpx
    客户端是共享的。
    """
    http_client_sync, http_async_client = _shared_http_clients()

    extra_body: dict[str, Any] = {}
    if settings.llm_disable_thinking:
        # DeepSeek 专用开关；不认识这个字段的供应商会忽略它，认识但不接受的
        # 会在第一次调用时直接报错，不会静默降级。
        extra_body["thinking"] = {"type": "disabled"}

    return StructuredChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        temperature=settings.llm_temperature if temperature is None else temperature,
        max_tokens=settings.llm_max_tokens,
        streaming=True,
        http_client=http_client_sync,
        http_async_client=http_async_client,
        extra_body=extra_body or None,
    )
