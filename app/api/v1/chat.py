"""对话接口：SSE 流式对话和会话历史。流程本身在 app/services/chat_flow.py。"""

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import ConversationId, get_current_user
from app.models.base import get_db
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.user import User
from app.schemas.message import MessageCreate
from app.services.chat_flow import generate_sse_stream


router = APIRouter(prefix="/chat", tags=["对话"])

# 建会话时的默认标题；还是这些值时用第一句话代替，侧边栏才分得清各个会话。
_DEFAULT_TITLES = {"新对话", "新行程", "新的旅程", ""}
_TITLE_LENGTH = 20


def conversation_title_from(message: str) -> str:
    """取第一句话的开头作标题：去掉换行和多余空白，过长时截断加省略号。"""
    text = " ".join(message.split())
    return text if len(text) <= _TITLE_LENGTH else text[:_TITLE_LENGTH] + "…"
_DIRECT_RESPONSE = "我可以解答旅行问题，也可以在你准备好时开始规划行程。"


@router.post("/stream/{conversation_id}")
async def stream_chat(
    conversation_id: ConversationId,
    data: MessageCreate,
    request: Request = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Conversation)
        .where(Conversation.id == conversation_id)
        .where(Conversation.user_id == user.id)
    )
    conversation = result.scalar_one_or_none()
    if not conversation:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")
    if (conversation.title or "").strip() in _DEFAULT_TITLES and data.content.strip():
        conversation.title = conversation_title_from(data.content)
        await db.commit()

    app_state = getattr(getattr(request, "app", None), "state", None)
    registry = getattr(app_state, "planning_registry", None)
    draft_repository = getattr(app_state, "draft_repository", None)
    return StreamingResponse(
        generate_sse_stream(
            conversation_id,
            data.content,
            str(user.id),
            registry=registry,
            draft_repository=draft_repository,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/history/{conversation_id}")
async def get_chat_history(
    conversation_id: ConversationId,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(Conversation)
        .where(Conversation.id == conversation_id)
        .where(Conversation.user_id == user.id)
    )
    conversation = result.scalar_one_or_none()
    if not conversation:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="会话不存在")

    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at)
    )
    messages = result.scalars().all()
    return {
        "conversation": conversation.to_dict(),
        "messages": [message.to_dict() for message in messages],
    }
