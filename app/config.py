"""
配置管理模块
使用 pydantic-settings 管理环境变量
"""
import os
from urllib.parse import quote
from sqlalchemy.engine import URL
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from functools import lru_cache

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEGRADED_PLANNING_REASON = "no_llm_or_provider"
DEGRADED_PLANNING_MARKER = "planning_degraded:no_llm_or_provider"

# ======== 环境兼容性修复（课件没有，但本地环境必需） ========
# 问题：本机开了代理（如 Clash 端口 7897）时，发往本机服务（Ollama、数据库）的请求
#       也会被送进代理，导致连接失败。
# 修复：把本机地址加入 NO_PROXY，让这些请求绕过代理直连。
_no_proxy = os.environ.get("NO_PROXY", "")
_local_hosts = "localhost,127.0.0.1"
if _local_hosts not in _no_proxy:
    new_value = f"{_no_proxy},{_local_hosts}" if _no_proxy else _local_hosts
    os.environ["NO_PROXY"] = new_value
    os.environ["no_proxy"] = new_value
# ======== 修复结束 ========


class Settings(BaseSettings):
    """应用配置"""

    # ============== 应用基础配置 ==============
    app_env: str = Field(default="development", alias="APP_ENV")
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=18000, alias="APP_PORT")
    debug: bool = Field(default=False, alias="DEBUG")
    jwt_secret_key: str = Field(
        default="development-only-change-me",
        alias="JWT_SECRET_KEY",
    )
    enable_external_tools: bool = Field(default=False, alias="ENABLE_EXTERNAL_TOOLS")
    cors_origins: str = Field(
        default="http://localhost:18000,http://127.0.0.1:18000",
        alias="CORS_ORIGINS",
    )

    # ============== LLM 配置 ==============
    # 供应商无关：换模型只改 .env，不动代码。当前用 DeepSeek 的 OpenAI 兼容接口。
    llm_api_key: str = Field(default="", alias="LLM_API_KEY")
    llm_model: str = Field(default="deepseek-v4-flash", alias="LLM_MODEL")
    llm_base_url: str = Field(
        default="https://api.deepseek.com",
        alias="LLM_BASE_URL"
    )
    llm_temperature: float = 0.7
    llm_max_tokens: int = 8000
    # DeepSeek 的 thinking 模式不支持强制 tool_choice，而 with_structured_output
    # 恰恰靠强制 tool_choice 取结构化结果。不关掉它，全部结构化调用都会返回
    # 400 "Thinking mode does not support this tool_choice"，路由、需求抽取、
    # Worker 分析和行程合成会一起静默退化成兜底路径。
    llm_disable_thinking: bool = Field(default=True, alias="LLM_DISABLE_THINKING")

    # ============== Embedding 配置 ==============
    # 走本地 Ollama，不依赖任何外部 API Key，也不受 LLM 供应商切换影响。
    embedding_base_url: str = Field(
        default="http://127.0.0.1:11434/v1",
        alias="EMBEDDING_BASE_URL",
    )
    embedding_model: str = Field(default="qwen3-embedding:4b", alias="EMBEDDING_MODEL")

    # ============== LangSmith 配置 ==============
    langsmith_api_key: str = Field(default="", alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(default="travel-planner-dev", alias="LANGSMITH_PROJECT")
    langsmith_tracing: bool = Field(default=False, alias="LANGSMITH_TRACING")
    langsmith_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        alias="LANGSMITH_ENDPOINT"
    )

    # ============== 数据库配置 ==============
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_db: str = Field(default="ai_travel_db", alias="POSTGRES_DB")
    postgres_user: str = Field(default="travel_user", alias="POSTGRES_USER")
    postgres_password: str = Field(default="travel123456", alias="POSTGRES_PASSWORD")

    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_db: int = Field(default=0, alias="REDIS_DB")
    redis_password: str = Field(default="", alias="REDIS_PASSWORD")

    # ============== MCP 服务配置 ==============
    amap_api_key: str = Field(default="", alias="AMAP_API_KEY")
    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")
    # 自建 SearXNG 元搜索（如 http://127.0.0.1:8080），Tavily 没配或额度用完时作为联网搜索兜底。
    searxng_url: str = Field(default="", alias="SEARXNG_URL")
    # 指定 SearXNG 用哪些引擎。默认组合里谷歌常被限流、必应对中文旅游查询结果很差
    # （"兵马俑 门票"会搜出无关网页），搜狗和夸克的中文结果最相关。留空则用 SearXNG 自己的默认。
    searxng_engines: str = Field(default="sogou,quark", alias="SEARXNG_ENGINES")
    mcp_weather_url: str = Field(default="", alias="MCP_WEATHER_URL")
    mcp_search_url: str = Field(default="", alias="MCP_SEARCH_URL")
    external_timeout_seconds: float = Field(default=10.0, alias="EXTERNAL_TIMEOUT_SECONDS")
    external_max_retries: int = Field(default=2, alias="EXTERNAL_MAX_RETRIES")

    # ============== RAG 检索配置 ==============
    enable_cross_encoder_rerank: bool = Field(default=False, alias="ENABLE_CROSS_ENCODER_RERANK")
    cross_encoder_model: str = Field(default="BAAI/bge-reranker-base", alias="CROSS_ENCODER_MODEL")

    def validate_security(self) -> None:
        """拒绝在非开发环境使用公开的默认 JWT 密钥。"""
        if (
            self.app_env.lower() not in {"development", "dev", "test"}
            and self.jwt_secret_key == "development-only-change-me"
        ):
            raise RuntimeError("生产环境必须配置独立的 JWT_SECRET_KEY")

    def apply_langsmith_env(self) -> bool:
        """把 LangSmith 配置导出到 os.environ，返回追踪是否真的开启。

        pydantic-settings 只把 .env 读进 Settings 对象，不写回 os.environ；而
        langsmith SDK 只认 os.environ。所以在这之前这四个配置项是死的——
        .env 里写了 LANGSMITH_TRACING=true 也不会有任何一条 trace 上报，
        排查链路时看不到 Agent 调用轨迹却完全没有报错提示。

        没有 API Key 就不开：开了只会让每次 LLM 调用多一轮注定 401 的上报。
        """
        if not (self.langsmith_tracing and self.langsmith_api_key):
            return False
        # LANGSMITH_* 是新名字，LANGCHAIN_* 是 langchain-core 仍在读的旧名字，
        # 两套都写以免版本差异导致静默不生效。
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_API_KEY"] = self.langsmith_api_key
        os.environ["LANGCHAIN_API_KEY"] = self.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = self.langsmith_project
        os.environ["LANGCHAIN_PROJECT"] = self.langsmith_project
        os.environ["LANGSMITH_ENDPOINT"] = self.langsmith_endpoint
        os.environ["LANGCHAIN_ENDPOINT"] = self.langsmith_endpoint
        return True

    model_config = SettingsConfigDict(
        env_file=os.path.join(BASE_DIR, ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    @property
    def database_url(self) -> str:
        """生成 PostgreSQL 连接字符串"""
        return URL.create(
            "postgresql",
            username=self.postgres_user,
            password=self.postgres_password,
            host=self.postgres_host,
            port=self.postgres_port,
            database=self.postgres_db,
        ).render_as_string(hide_password=False)

    @property
    def redis_url(self) -> str:
        """生成 Redis 连接字符串"""
        if self.redis_password:
            password = quote(self.redis_password, safe="")
            return f"redis://:{password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """获取配置单例（缓存）"""
    return Settings()


settings = get_settings()
