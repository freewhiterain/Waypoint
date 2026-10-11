"""Streaming chat API."""

import json
import asyncio
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import ConversationId, get_current_user
from app.config import settings
from app.api.sse import SSEEventBuilder, sse
from app.agents.agent_tools import run_agent_tool
from app.governance.drafts import load_trip_draft_context
from app.governance.tool_invocations import PostgresToolInvocationRepository, ToolInvocationRecord
from app.models.base import async_session_maker, get_db
from app.models.conversation import Conversation
from app.models.message import Message
from app.models.user import User
from app.schemas.message import MessageCreate
from app.schemas.tools import AgentToolResult, ToolCallPayload
from app.services.main_agent import MainAgentService
from app.services.planning import KNOWN_DESTINATIONS
from app.services.react_agent import build_react_tools, run_react_agent
from app.services.slots import (
    PostgresSlotRepository,
    apply_slot_update,
    load_slots,
    remember_suggestions,
    store_slots,
)
from app.services.open_qa import answer_open_question
from app.utils.logger import app_logger


router = APIRouter(prefix="/chat", tags=["对话"])

_RECENT_CONTEXT_LIMIT = 12
_DIRECT_RESPONSE = "我可以解答旅行问题，也可以在你准备好时开始规划行程。"


async def save_message(
    db: AsyncSession,
    conversation_id: str,
    role: str,
    content: str,
    extra_info: dict | None = None,
) -> Message:
    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
        extra_info=extra_info or {},
    )
    db.add(message)
    await db.commit()
    await db.refresh(message)
    return message


def _public_error(exc: Exception) -> dict:
    if isinstance(exc, ValueError):
        return {"code": "validation_error", "message": str(exc), "retryable": False}
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return {"code": "timeout", "message": "外部服务响应超时，请稍后重试。", "retryable": True}
    return {"code": "internal_error", "message": "旅行规划暂时无法完成，请稍后重试。", "retryable": True}


async def _load_recent_context(
    db: AsyncSession,
    conversation_id: str,
    excluded_message_id,
) -> list[dict[str, str]]:
    result = await db.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .where(Message.id != excluded_message_id)
        .order_by(Message.created_at.desc(), Message.id.desc())
        .limit(_RECENT_CONTEXT_LIMIT)
    )
    messages = list(reversed(result.scalars().all()))
    return [_context_item(message) for message in messages]


def _context_item(message: Message) -> dict[str, str]:
    """把一条历史消息压成路由器要看的最小结构。

    带上 extra_info["kind"]：主动邀请的识别靠这个结构化标记，只传 content 的话
    MainAgentService 只能拿问候语文案做全等比较，改一个字整条"好的 → 开始规划"
    就静默失效。extra_info 在模型上是 Python 侧 default=dict，历史行可能是
    NULL，所以要判类型而不是直接 .get。
    """
    item = {"role": message.role, "content": message.content}
    extra_info = getattr(message, "extra_info", None)
    if isinstance(extra_info, dict) and extra_info.get("kind"):
        item["kind"] = str(extra_info["kind"])
    return item


_ROUTE_ONE_ACTIONS = {"answer_open_question", "recommend_destination", "invoke_agent_tool"}


def _react_llm():
    """线路 1 的 ReAct 用的模型；没配 Key 或初始化失败时返回 None，改走固定流程。"""
    if not settings.llm_api_key.strip():
        return None
    try:
        from app.agents.llm import get_llm

        return get_llm(temperature=0.3)
    except Exception:
        return None


async def _react_events(question, context, slots, llm, registry):
    """跑 ReAct，边跑边产出 ("tool_call" | "tool_result", payload)，最后产出 ("answer", ReactResult)。"""
    queue: asyncio.Queue = asyncio.Queue()

    async def on_tool_call(payload):
        await queue.put(("tool_call", payload))

    async def on_tool_result(payload):
        await queue.put(("tool_result", payload))

    task = asyncio.create_task(
        run_react_agent(
            question,
            context,
            slots,
            llm=llm,
            tools=build_react_tools(registry),
            on_tool_call=on_tool_call,
            on_tool_result=on_tool_result,
        )
    )
    try:
        while True:
            getter = asyncio.create_task(queue.get())
            done, _pending = await asyncio.wait({task, getter}, return_when=asyncio.FIRST_COMPLETED)
            if getter in done:
                yield getter.result()
                continue
            getter.cancel()
            break
        while not queue.empty():
            yield queue.get_nowait()
        yield "answer", task.result()
    finally:
        if not task.done():
            task.cancel()


async def generate_sse_stream(
    conversation_id: str,
    user_message: str,
    user_id: str,
    registry=None,
    draft_repository=None,
    slot_repository=None,
):
    """Save one user turn, understand it once, update slots, and stream its action.

    模型只给出"意图 + 槽位变化"。槽位并进会话状态后，由这里的固定规则决定走哪条线路：
    想规划就发出表单工具调用，预填已有的槽位——表单只问缺的项，三项都齐时由前端
    显示确认。规划本身只有在表单（或确认）提交、三个槽都有值之后才会开始。
    """
    slot_repository = slot_repository or PostgresSlotRepository()
    task_id = uuid4().hex
    event = SSEEventBuilder(task_id, conversation_id)

    try:
        async with async_session_maker() as db:
            saved_user_message = await save_message(db, conversation_id, "user", user_message)
            context = await _load_recent_context(db, conversation_id, saved_user_message.id)

        if draft_repository is not None:
            try:
                draft_context = await load_trip_draft_context(
                    draft_repository,
                    user_id,
                    conversation_id,
                )
            except Exception as exc:
                app_logger.warning("Trip draft context unavailable: %s", type(exc).__name__)
                draft_context = None
            if draft_context is not None:
                context.append(
                    {
                        "role": "system",
                        "content": json.dumps(
                            {"trip_workspace": draft_context},
                            ensure_ascii=False,
                        ),
                    }
                )

        slots = await load_slots(slot_repository, conversation_id)
        decision = await MainAgentService().decide(user_message, context, slots)
        # 填槽每一轮都做，与这一轮走哪条线路无关。
        slots, dropped = apply_slot_update(slots, decision.slot_update, user_text=user_message)
        if dropped:
            app_logger.info(f"槽位更新缺少依据，已丢弃: {dropped}")
        await store_slots(slot_repository, conversation_id, slots)

        if decision.action == "collect_trip_requirements":
            call_id = uuid4().hex
            arguments: dict[str, Any] = {"initial_values": slots.form_values()}
            requirement_context = slots.requirement_context()
            if requirement_context:
                arguments["requirement_context"] = requirement_context
            payload = ToolCallPayload(
                call_id=call_id,
                tool="collect_trip_requirements",
                arguments=arguments,
            )
            async with async_session_maker() as db:
                async with db.begin():
                    await PostgresToolInvocationRepository().create_in_session(
                        db,
                        ToolInvocationRecord(
                            call_id=call_id,
                            user_id=user_id,
                            conversation_id=conversation_id,
                            tool=payload.tool,
                            arguments=payload.arguments,
                        ),
                    )
                    db.add(
                        Message(
                            conversation_id=conversation_id,
                            role="assistant",
                            content="正在收集旅行需求。",
                            extra_info={"action": decision.action, "tool_call": payload.model_dump()},
                        )
                    )
            yield sse(event("tool_call", payload.model_dump()))
            yield sse(event("done"))
            return

        react_llm = _react_llm() if decision.action in _ROUTE_ONE_ACTIONS else None
        if react_llm is not None:
            # 线路 1：开放问题、推荐目的地、单项研究都交给 ReAct，由模型自己决定查什么。
            try:
                react_result = None
                async for kind, payload in _react_events(user_message, context, slots, react_llm, registry):
                    if kind == "answer":
                        react_result = payload
                    else:
                        yield sse(event(kind, payload))
                answer = react_result.answer
                if decision.action == "recommend_destination":
                    slots = remember_suggestions(slots, answer, KNOWN_DESTINATIONS)
                    await store_slots(slot_repository, conversation_id, slots)
                async with async_session_maker() as db:
                    await save_message(
                        db,
                        conversation_id,
                        "assistant",
                        answer,
                        {
                            "action": decision.action,
                            "react_steps": [
                                {"tool": step.tool, "arguments": step.arguments, "status": step.outcome.status}
                                for step in react_result.steps
                            ],
                        },
                    )
                yield sse(event("token", {"content": answer, "action": decision.action}))
                yield sse(event("done"))
                return
            except Exception as exc:
                # ReAct 出错不该让这一轮没有回答：退回下面的固定流程。
                app_logger.warning(f"ReAct 问答失败，改走固定流程: {type(exc).__name__}: {exc}")

        if decision.action == "invoke_agent_tool" and decision.tool_call is not None:
            call_id = uuid4().hex
            tool_call = decision.tool_call
            call_payload = {
                "call_id": call_id,
                "tool": tool_call.name,
                "arguments": tool_call.arguments.model_dump(mode="json"),
            }
            yield sse(event("tool_call", call_payload))

            if registry is None:
                result: AgentToolResult = await run_agent_tool(tool_call)
            else:
                result = await run_agent_tool(tool_call, registry=registry)
            result_payload = {
                "call_id": call_id,
                "tool": result.tool_name,
                "status": result.status,
                "result": result.model_dump(mode="json"),
            }
            yield sse(event("tool_result", result_payload))

            async with async_session_maker() as db:
                await save_message(
                    db,
                    conversation_id,
                    "assistant",
                    result.answer,
                    {
                        "action": decision.action,
                        "tool_call": call_payload,
                        "tool_result": result.model_dump(mode="json"),
                    },
                )
            yield sse(event("token", {"content": result.answer, "action": decision.action}))
            yield sse(event("done"))
            return

        if decision.action in {"answer_open_question", "recommend_destination"}:
            answer = await answer_open_question(user_message)
            if decision.action == "recommend_destination":
                # 推荐到的城市只是候选，用户接受之后才会填进目的地。
                slots = remember_suggestions(slots, answer, KNOWN_DESTINATIONS)
                await store_slots(slot_repository, conversation_id, slots)
            async with async_session_maker() as db:
                await save_message(
                    db,
                    conversation_id,
                    "assistant",
                    answer,
                    {"action": decision.action},
                )
            yield sse(event("token", {"content": answer, "action": decision.action}))
            yield sse(event("done"))
            return

        response = decision.response or _DIRECT_RESPONSE
        async with async_session_maker() as db:
            await save_message(
                db,
                conversation_id,
                "assistant",
                response,
                {"action": decision.action},
            )
        yield sse(event("token", {"content": response, "action": decision.action}))
        yield sse(event("done"))
    except Exception as exc:
        app_logger.exception("SSE chat routing error")
        yield sse(event("error", _public_error(exc)))
        yield sse(event("done"))


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
