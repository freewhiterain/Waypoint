"""Tool-result SSE API."""

import asyncio
import json
from contextlib import suppress
from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from app.agents import factory as agent_factory
from app.api.dependencies import get_current_user
from app.core.checkpointer import get_checkpointer
from app.governance.drafts import PostgresDraftRepository, save_trip_draft
from app.governance.events import PublishingEventRepository, TaskEventService, task_event_to_sse_event
from app.governance.postgres import PostgresEventRepository
from app.governance.tool_invocations import PostgresToolInvocationRepository
from app.models.base import async_session_maker
from app.models.message import Message
from app.models.user import User
from app.schemas.events import SSEEvent
from app.schemas.planning import TravelRequirement
from app.schemas.tools import ToolResultRequest
from app.services.open_qa import answer_open_question
from app.services.planning import render_plan_markdown
from app.utils.logger import app_logger


router = APIRouter(prefix="/chat/tools", tags=["chat tools"])
PROCESSING_LEASE_TIMEOUT = timedelta(minutes=2)


async def run_travel_planning(*args, **kwargs):
    """Compatibility seam for tests and callers replacing the planning runner."""
    return await agent_factory.run_travel_planning(*args, **kwargs)


class ProcessingLeaseLostError(RuntimeError):
    pass


def sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


async def save_assistant_message(conversation_id: str, content: str, extra_info: dict) -> None:
    async with async_session_maker() as db:
        message = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=content,
            extra_info=extra_info,
        )
        db.add(message)
        await db.commit()
        await db.refresh(message)


def build_requirement(form_result: dict, arguments: dict | None) -> TravelRequirement:
    """表单确认值 + 主 Agent 从原话里提取的人数/偏好。

    表单字段优先：用户在表单里改过的值必须覆盖原话推断。上下文里有任何字段
    导致校验失败（如出发地与目的地相同），就退回只用表单值，不让规划因为
    推断出的附加信息而整体失败。
    """
    context = (arguments or {}).get("requirement_context") or {}
    allowed = set(TravelRequirement.model_fields)
    merged = {key: value for key, value in context.items() if key in allowed}
    merged.update(form_result)
    try:
        return TravelRequirement(**merged)
    except ValueError as exc:
        app_logger.warning(f"需求上下文与表单冲突，只使用表单值: {exc}")
        return TravelRequirement(**form_result)


def recommendation_query(partial_values: dict) -> str:
    values = json.dumps(partial_values, ensure_ascii=False, sort_keys=True)
    return f"Recommend a travel destination based on these confirmed preferences: {values}"


async def processing_heartbeat(
    repository,
    call_id: str,
    user_id: str,
    claim_version: int,
    lease_timeout: timedelta,
    lease_lost: asyncio.Event,
) -> None:
    interval = max(lease_timeout.total_seconds() / 3, 0.01)
    try:
        while True:
            await asyncio.sleep(interval)
            if not await repository.renew_processing(call_id, user_id, claim_version):
                lease_lost.set()
                return
    except asyncio.CancelledError:
        raise
    except Exception:
        lease_lost.set()


async def stop_processing_heartbeat(heartbeat_task: asyncio.Task | None) -> None:
    if heartbeat_task is None:
        return
    heartbeat_task.cancel()
    with suppress(asyncio.CancelledError):
        await heartbeat_task


async def cancel_and_await_task(task: asyncio.Task | None) -> None:
    if task is None:
        return
    if not task.done():
        task.cancel()
    with suppress(asyncio.CancelledError, Exception):
        await task


async def existing_completion_stream(call_id: str, conversation_id: str, result: dict | None):
    sequence = 0

    def event(event_type: str, payload: dict | None = None) -> dict:
        nonlocal sequence
        sequence += 1
        return SSEEvent(
            type=event_type,
            task_id=call_id,
            conversation_id=conversation_id,
            sequence=sequence,
            payload=payload or {},
        ).legacy_payload()

    durable = result if isinstance(result, dict) else {}
    task_id = durable.get("task_id", call_id)
    assistant_result = durable.get("assistant_result", result)
    assistant_markdown = durable.get(
        "assistant_markdown", "A travel-planning task has already been submitted."
    )
    yield sse(
        event(
            "result",
            {"task_id": task_id, "status": "completed", "result": assistant_result},
        )
    )
    yield sse(event("token", {"content": assistant_markdown}))
    yield sse(event("done"))


async def processing_stream(call_id: str, conversation_id: str):
    sequence = 0

    def event(event_type: str, payload: dict | None = None) -> dict:
        nonlocal sequence
        sequence += 1
        return SSEEvent(
            type=event_type,
            task_id=call_id,
            conversation_id=conversation_id,
            sequence=sequence,
            payload=payload or {},
        ).legacy_payload()

    yield sse(
        event(
            "tool_result",
            {
                "tool": "collect_trip_requirements",
                "status": "processing",
                "terminal": False,
            },
        )
    )
    yield sse(event("done"))


async def tool_result_stream(call_id: str, data: ToolResultRequest, user_id: str, record):
    sequence = 0

    def event(event_type: str, payload: dict | None = None) -> dict:
        nonlocal sequence
        sequence += 1
        return SSEEvent(
            type=event_type,
            task_id=call_id,
            conversation_id=record.conversation_id,
            sequence=sequence,
            payload=payload or {},
        ).legacy_payload()

    def research_event_frame(research_event: SSEEvent) -> str:
        nonlocal sequence
        sequence += 1
        research_event.sequence = sequence
        return sse(research_event.legacy_payload())

    claim_version = None
    repository = None
    heartbeat_task = None
    planning_task = None
    research_event_task = None
    processing_guard = None
    planning_completed = False
    try:
        repository = PostgresToolInvocationRepository()

        if data.status == "recommend_destination":
            if record.status == "processing":
                async for frame in processing_stream(call_id, record.conversation_id):
                    yield frame
                return
            updated = await repository.update_partial(call_id, user_id, data.partial_values)
            if updated is None:
                raise ValueError("Tool invocation is unavailable")

            answer = await answer_open_question(recommendation_query(updated.partial_values))
            tool_result = {
                "tool": data.tool,
                "status": "awaiting_destination",
                "partial_values": updated.partial_values,
            }
            await save_assistant_message(
                record.conversation_id,
                answer,
                {"tool_result": tool_result},
            )
            yield sse(event("token", {"content": answer}))
            yield sse(event("tool_result", tool_result))
            yield sse(event("done"))
            return

        if data.status != "completed" or data.result is None:
            raise ValueError("A completed tool result requires destination, departure_date, and days")

        confirmed_result = data.result.model_dump(mode="json")
        acquire_guard = getattr(repository, "acquire_processing_guard", None)
        if acquire_guard is not None:
            processing_guard = await acquire_guard()
        claim = await repository.claim_processing(
            call_id, user_id, PROCESSING_LEASE_TIMEOUT
        )
        if claim is None:
            raise ValueError("Tool invocation is unavailable")
        if not claim.claimed:
            if claim.record.status == "completed":
                async for frame in existing_completion_stream(
                    call_id, claim.record.conversation_id, claim.record.result
                ):
                    yield frame
            elif claim.record.status == "processing":
                async for frame in processing_stream(call_id, claim.record.conversation_id):
                    yield frame
            else:
                raise ValueError("Tool invocation is not available for processing")
            return

        claim_version = claim.claim_version
        record = claim.record
        requirement = build_requirement(data.result.model_dump(), record.arguments)
        research_events: asyncio.Queue[SSEEvent] = asyncio.Queue()

        async def publish_research_event(task_event):
            research_event = task_event_to_sse_event(task_event)
            if research_event is not None:
                await research_events.put(research_event)

        event_service = TaskEventService(
            PublishingEventRepository(PostgresEventRepository(), publish_research_event)
        )
        lease_lost = asyncio.Event()
        heartbeat_task = asyncio.create_task(
            processing_heartbeat(
                repository,
                call_id,
                user_id,
                claim_version,
                PROCESSING_LEASE_TIMEOUT,
                lease_lost,
            ),
            name=f"tool-result-heartbeat:{call_id}",
        )
        planning_task = asyncio.create_task(
            run_travel_planning(
                requirement,
                checkpointer=await get_checkpointer(),
                event_service=event_service,
                task_id=call_id,
                user_id=user_id,
                conversation_id=record.conversation_id,
            ),
            name=f"tool-result-planning:{call_id}",
        )
        try:
            while True:
                research_event_task = asyncio.create_task(
                    research_events.get(),
                    name=f"tool-result-research-event:{call_id}",
                )
                done, _pending = await asyncio.wait(
                    {planning_task, heartbeat_task, research_event_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if research_event_task in done:
                    yield research_event_frame(research_event_task.result())
                    research_event_task = None
                    continue

                await cancel_and_await_task(research_event_task)
                research_event_task = None
                if planning_task in done:
                    draft = planning_task.result()
                    planning_completed = True
                    break
                if heartbeat_task in done and lease_lost.is_set():
                    await cancel_and_await_task(planning_task)
                    planning_task = None
                    await stop_processing_heartbeat(heartbeat_task)
                    heartbeat_task = None
                    raise ProcessingLeaseLostError("processing lease lost")
            while not research_events.empty():
                yield research_event_frame(research_events.get_nowait())
        finally:
            await cancel_and_await_task(research_event_task)
            research_event_task = None
            if planning_task is not None and planning_task.done():
                planning_task = None
            await stop_processing_heartbeat(heartbeat_task)
            heartbeat_task = None
        if lease_lost.is_set() and not planning_completed:
            raise ProcessingLeaseLostError("processing lease lost")
        assistant_result = draft.model_dump(mode="json")
        assistant_content = json.dumps(assistant_result, ensure_ascii=False)
        if hasattr(draft, "itinerary") and hasattr(draft, "requirement"):
            assistant_content = render_plan_markdown(draft)
        durable_result = {
            "confirmed_result": confirmed_result,
            "task_id": call_id,
            "assistant_result": assistant_result,
            "assistant_markdown": assistant_content,
            "draft": assistant_result,
            "route": getattr(draft, "route", None),
        }
        tool_result = {
            "tool": data.tool,
            "status": "completed",
            "result": confirmed_result,
            "task_id": call_id,
        }
        async with async_session_maker() as db:
            async with db.begin():
                finished = await repository.finish_processing(
                    call_id,
                    user_id,
                    claim_version,
                    durable_result,
                    session=db,
                )
                if finished is None:
                    raise ValueError("Tool invocation processing claim was lost")
                db.add(
                    Message(
                        conversation_id=finished.conversation_id,
                        role="assistant",
                        content=assistant_content,
                        extra_info={
                            "tool_result": tool_result,
                            "assistant_result": assistant_result,
                        },
                    )
                )
        claim_version = None
        # 前端只走这条链路，草稿必须在这里落库：/planning/tasks 才是原先唯一
        # 调用 save_trip_draft 的地方，而 UI 从不访问它，于是
        # GET /planning/drafts/{cid} 永远 404，chat 的多轮上下文也拿不到
        # 上一版行程。落库失败不能吞掉已经算好的结果，因此单独 try。
        if finished.conversation_id:
            try:
                await save_trip_draft(
                    PostgresDraftRepository(),
                    user_id,
                    finished.conversation_id,
                    draft,
                )
            except Exception as exc:
                app_logger.warning(
                    f"行程草稿持久化失败，本次结果仍照常返回: task_id={call_id} "
                    f"error={type(exc).__name__}: {exc}"
                )
        yield sse(event("result", {"task_id": call_id, "status": "completed", "result": assistant_result}))
        yield sse(event("token", {"content": assistant_content}))
        yield sse(event("done"))
    except asyncio.CancelledError:
        await cancel_and_await_task(planning_task)
        await stop_processing_heartbeat(heartbeat_task)
        if repository is not None and claim_version is not None:
            with suppress(Exception):
                await repository.release_processing(call_id, user_id, claim_version)
        raise
    except Exception as exc:
        await cancel_and_await_task(planning_task)
        await stop_processing_heartbeat(heartbeat_task)
        if repository is not None and claim_version is not None:
            try:
                await repository.release_processing(call_id, user_id, claim_version)
            except Exception:
                pass
        yield sse(
            event(
                "error",
                {
                    "code": (
                        "processing_conflict"
                        if isinstance(exc, ProcessingLeaseLostError)
                        else "internal_error"
                    ),
                    "message": "Travel planning is temporarily unavailable. Please retry.",
                    "retryable": True,
                },
            )
        )
        yield sse(event("done"))
    finally:
        if repository is not None and processing_guard is not None:
            release_guard = getattr(repository, "release_processing_guard", None)
            if release_guard is not None:
                with suppress(Exception):
                    await release_guard(processing_guard)


@router.post("/{call_id}/result")
async def submit_tool_result(
    call_id: str,
    data: ToolResultRequest,
    user: User = Depends(get_current_user),
):
    user_id = str(user.id)
    repository = PostgresToolInvocationRepository()
    record = await repository.get_for_user(call_id, user_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tool invocation not found")
    if record.tool != data.tool:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Tool does not match invocation")
    if data.status == "completed" and data.result is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Completed results require destination, departure_date, and days",
        )
    if data.status == "cancelled":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Cancelled tool results are not supported",
        )
    if record.status == "completed" and data.status == "completed":
        return StreamingResponse(
            existing_completion_stream(call_id, record.conversation_id, record.result),
            media_type="text/event-stream",
        )
    if record.status not in {"pending", "processing"}:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Tool invocation is not pending")

    return StreamingResponse(
        tool_result_stream(call_id, data, user_id, record),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
