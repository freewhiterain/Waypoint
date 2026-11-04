"""无规划意图的开放式提问：直接查通用知识库并回答，不建行程草稿。"""

from __future__ import annotations

from pathlib import Path

from app.rag.local_knowledge import get_local_knowledge_service
from app.config import settings
from app.core.llm import describe_llm_failure
from app.utils.logger import app_logger
from app.services.requirements import KNOWN_DESTINATIONS


def _source_name(source: str) -> str:
    """只给文件名，不把本机绝对路径展示给用户。"""
    return Path(source).name if source else ""


def _about(item, cities: list[str]) -> bool:
    text = f"{item.content or ''} {item.source or ''}"
    return any(city in text for city in cities)


def _line(item) -> str:
    return f"{item.content}（来源：{_source_name(item.source)}）"


async def answer_open_question(query: str) -> str:
    """检索通用旅行知识库，用 LLM 组织成一段回答；无 Key 或调用失败时直接列证据摘要。"""
    evidence = get_local_knowledge_service().search(query)
    cities = [city for city in KNOWN_DESTINATIONS if city in query]
    if cities:
        # 本地资料只有少数城市，语义检索会把成都、西安的资料当成杭州的返回
        evidence = [item for item in evidence if _about(item, cities)]
    if not evidence:
        where = "、".join(cities)
        missing = f"本地资料里暂时没有{where}的相关内容。" if where else "暂时没有查到相关的本地资料。"
        return missing + "告诉我想去的城市、天数和出发日期，我可以直接帮你排行程。"

    if not settings.llm_api_key:
        lines = ["为你找到以下相关资料：", ""]
        lines.extend(f"- {_line(item)}" for item in evidence[:4])
        return "\n".join(lines)

    try:
        from app.core.llm import get_llm

        context = "\n".join(f"[{i + 1}] {_line(item)}" for i, item in enumerate(evidence))
        response = await get_llm(temperature=0.5).ainvoke(
            [
                {
                    "role": "system",
                    "content": (
                        "你是旅行知识助手。只能使用下方资料回答，不得编造资料之外的景点、价格或时间信息；"
                        "资料不足以支撑的部分要明确说明。回答简洁，适合直接展示给用户。"
                    ),
                },
                {"role": "user", "content": f"资料：\n{context}\n\n用户问题：{query}"},
            ]
        )
        return response.content
    except Exception as exc:
        app_logger.warning(f"开放问答 LLM 组织失败，退回证据摘要: {type(exc).__name__}: {exc}")
        lines = [f"（{describe_llm_failure(exc)}，下面是查到的原始资料，没有经过整理。）", ""]
        lines.extend(f"- {_line(item)}" for item in evidence[:4])
        return "\n".join(lines)
