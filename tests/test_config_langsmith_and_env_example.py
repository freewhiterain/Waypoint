"""LangSmith 配置必须真的生效，.env.example 必须和 Settings 对得上。

两个此前静默失效的问题：

1. langsmith_api_key / project / tracing / endpoint 四个配置项从来没人读过。
   pydantic-settings 只把 .env 读进 Settings 对象，而 langsmith SDK 只认
   os.environ——.env 里写 LANGSMITH_TRACING=true 也一条 trace 都不会上报，
   且没有任何报错。默认值还是 True，暗示"默认开着"，更容易误判。

2. .env.example 漏了 CORS_ORIGINS 和两个 RAG 开关，照抄它起服务时
   浏览器访问 /ui 会撞 CORS，而用户无从知道该加哪一项。
"""

import re
from pathlib import Path

from app.config import Settings

_ENV_EXAMPLE = Path(__file__).resolve().parent.parent / ".env.example"


def _example_keys() -> set[str]:
    keys = set()
    for line in _ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        keys.add(line.split("=", 1)[0].strip())
    return keys


def _settings_aliases() -> set[str]:
    aliases = set()
    for field in Settings.model_fields.values():
        if field.alias:
            aliases.add(field.alias)
    return aliases


def test_langsmith_env_is_exported_when_key_and_flag_are_both_present(monkeypatch):
    settings = Settings(
        LANGSMITH_TRACING=True,
        LANGSMITH_API_KEY="ls-test-key",
        LANGSMITH_PROJECT="trip-test",
        LANGSMITH_ENDPOINT="https://example.invalid",
    )
    for name in [
        "LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2",
        "LANGSMITH_API_KEY", "LANGCHAIN_API_KEY",
        "LANGSMITH_PROJECT", "LANGCHAIN_PROJECT",
        "LANGSMITH_ENDPOINT", "LANGCHAIN_ENDPOINT",
    ]:
        monkeypatch.delenv(name, raising=False)

    assert settings.apply_langsmith_env() is True

    import os

    # 新旧两套变量名都要写：langchain-core 仍在读 LANGCHAIN_*。
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
    assert os.environ["LANGSMITH_API_KEY"] == "ls-test-key"
    assert os.environ["LANGCHAIN_API_KEY"] == "ls-test-key"
    assert os.environ["LANGSMITH_PROJECT"] == "trip-test"
    assert os.environ["LANGCHAIN_PROJECT"] == "trip-test"
    assert os.environ["LANGSMITH_ENDPOINT"] == "https://example.invalid"


def test_tracing_without_an_api_key_stays_off(monkeypatch):
    """开了开关但没 Key 时不要开：只会给每次 LLM 调用加一轮注定 401 的上报。"""
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)

    settings = Settings(LANGSMITH_TRACING=True, LANGSMITH_API_KEY="")

    assert settings.apply_langsmith_env() is False

    import os

    assert os.environ.get("LANGSMITH_TRACING") is None
    assert os.environ.get("LANGCHAIN_TRACING_V2") is None


def test_tracing_defaults_to_off():
    """默认 True 会让人以为追踪开着，而实际上它从未生效过。

    这里查字段声明的默认值而不是 Settings().langsmith_tracing：后者会读开发者
    本机真实的 .env（其中就设了 LANGSMITH_TRACING=true），测的就不是默认值了。
    """
    assert Settings.model_fields["langsmith_tracing"].default is False


def test_lifespan_applies_langsmith_env():
    """契约测试：真的接进了启动流程，而不是写了个没人调的方法。"""
    import inspect

    from app import main

    assert "apply_langsmith_env" in inspect.getsource(main.lifespan)


def test_env_example_covers_every_documented_setting():
    """照抄 .env.example 起服务不该缺关键项（漏 CORS_ORIGINS 会撞 CORS）。"""
    example = _example_keys()

    for required in [
        "CORS_ORIGINS",
        "ENABLE_CROSS_ENCODER_RERANK",
        "CROSS_ENCODER_MODEL",
        "JWT_SECRET_KEY",
        "POSTGRES_PORT",
        "LANGSMITH_TRACING",
    ]:
        assert required in example, f".env.example 缺少 {required}"


def test_env_example_has_no_keys_that_settings_would_ignore():
    """写了却没人读的项必须显式标注，否则用户填了以为生效。"""
    unknown = _example_keys() - _settings_aliases()
    body = _ENV_EXAMPLE.read_text(encoding="utf-8")

    for key in unknown:
        # 允许存在预留项，但必须在同一文件里说明"暂不生效"。
        assert re.search(r"没有代码读取|暂时不会生效|预留", body), (
            f".env.example 里的 {key} 没有对应的 Settings 字段，且没有标注为预留项"
        )
