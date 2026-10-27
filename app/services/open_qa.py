"""无规划意图的开放式提问：直接查通用知识库并回答，不建行程草稿。"""

from __future__ import annotations

from app.rag.local_knowledge import get_local_knowledge_service
from app.config import settings
from app.utils.logger import app_logger


async def answer_open_question(query: str) -> str:
    """检索通用旅行知识库，用 LLM 组织成一段回答；无 Key 或调用失败时直接列证据摘要。"""
    evidence = get_local_knowledge_service().search(query)
    if not evidence:
        return "暂时没有查到相关的本地资料，告诉我具体想去的城市，我可以帮你展开研究。"

    if not settings.llm_api_key:
        lines = ["为你找到以下相关资料：", ""]
        lines.extend(f"- {item.content}（来源：{item.source}）" for item in evidence[:4])
        return "\n".join(lines)

    try:
        from app.core.llm import get_llm

        context = "\n".join(f"[{i + 1}] {item.content}（来源：{item.source}）" for i, item in enumerate(evidence))
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
        lines = ["为你找到以下相关资料：", ""]
        lines.extend(f"- {item.content}（来源：{item.source}）" for item in evidence[:4])
        return "\n".join(lines)
