"""
FastAPI 应用入口
"""
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from app.agents.factory import create_planning_registry
from app.config import settings
from app.governance.drafts import PostgresDraftRepository
from app.utils.logger import app_logger
from app.api.v1 import conversations, chat, health, planning, tools, users


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    import asyncio

    settings.validate_security()

    # LangSmith 只读 os.environ，pydantic-settings 不会写回去，必须显式导出。
    if settings.apply_langsmith_env():
        app_logger.info(f"LangSmith 追踪已启用: project={settings.langsmith_project}")
    elif settings.langsmith_tracing:
        app_logger.warning("LANGSMITH_TRACING=true 但缺少 LANGSMITH_API_KEY，追踪未启用")

    loop = asyncio.get_running_loop()
    app_logger.info(f"FastAPI 使用的事件循环: {type(loop).__name__}")

    from app.core.checkpointer import checkpointer_lifespan
    from app.governance.startup_recovery import recover_orphaned_tool_invocations
    from app.mcp_core.client import MCPClientManager
    from app.core.store import store_lifespan

    app_logger.info("启动应用...")

    async with checkpointer_lifespan():
        app_logger.info("Checkpointer 已就绪")

        async with store_lifespan():
            app_logger.info("Store 已就绪")

            # 上一次进程若被强杀，认领会滞留在 processing，这里自动归还。
            await recover_orphaned_tool_invocations()

            # 初始化 MCP（如果配置了的话）
            mcp = await MCPClientManager.get_instance()
            planning_registry, planning_fallback_reason = create_planning_registry()
            app.state.planning_registry = planning_registry
            app.state.planning_fallback_reason = planning_fallback_reason
            app.state.draft_repository = PostgresDraftRepository()
            app_logger.info("MCP 服务初始化完成")

            yield

            # 关闭 MCP
            try:
                if hasattr(mcp, "close"):
                    await mcp.close()
                app_logger.info("MCP 服务已关闭")
            except Exception as e:
                app_logger.warning(f"MCP 关闭异常: {e}")

            # 归还共享的 LLM httpx 连接池。
            try:
                from app.agents.llm import aclose_http_clients

                await aclose_http_clients()
            except Exception as e:
                app_logger.warning(f"LLM HTTP 客户端关闭异常: {e}")

    app_logger.info("应用已关闭")


app = FastAPI(
    title="WayPoint 旅行规划 API",
    description="基于 LangGraph 的多 Agent 旅行规划服务",
    version="1.0.0",
    lifespan=lifespan
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 前端（web/，Vite + React）构建产物。没构建时 /ui 给出构建提示而不是 404。
_WEB_DIST = _PROJECT_ROOT / "web" / "dist"
_WEB_MISSING = """<!doctype html><meta charset="utf-8"><title>WayPoint</title>
<body style="font-family:system-ui;padding:48px;max-width:640px;margin:auto;line-height:1.7">
<h2>前端还没有构建</h2><p>在项目根目录执行：</p>
<pre style="background:#f4f4f4;padding:12px;border-radius:8px">cd web
npm install
npm run build</pre>
<p>开发时也可以 <code>npm run dev</code>，访问 http://localhost:5173/ui/ 。</p></body>"""


@app.get("/ui", include_in_schema=False)
async def ui_redirect():
    return RedirectResponse("/ui/")


if (_WEB_DIST / "index.html").exists():
    app.mount("/ui", StaticFiles(directory=_WEB_DIST, html=True), name="ui")
else:
    @app.get("/ui/", include_in_schema=False)
    async def ui_missing():
        return HTMLResponse(_WEB_MISSING)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册 API 路由
app.include_router(health.router, prefix="/api/v1")
app.include_router(users.router, prefix="/api/v1")
app.include_router(conversations.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")
app.include_router(tools.router, prefix="/api/v1")
app.include_router(planning.router, prefix="/api/v1")


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/ui/")
