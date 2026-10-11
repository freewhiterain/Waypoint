"""event_callback 探测只能有一份实现。

supports_keyword 决定"要不要把 event_callback 往下传"，也就是研究进度事件
能不能流到前端。它原先在 supervisor、agent_tools、subagents/base、
subagents/registry 四个文件里各抄了一份完全相同的代码：任何一份漂移，
对应那条链路就会静默丢掉进度事件——不报错，只是前端不再更新。

这里既锁行为，也锁"不再出现第五份拷贝"。
"""

from pathlib import Path

import pytest

from app.utils.callables import supported_kwargs, supports_keyword


_CALL_SITES = [
    "app/agents/supervisor.py",
    "app/agents/agent_tools.py",
    "app/agents/subagents/base.py",
    "app/agents/subagents/registry.py",
]


def test_detects_explicit_keyword_only_parameter():
    async def worker(task, requirement, *, event_callback=None):
        return None

    assert supports_keyword(worker, "event_callback") is True


def test_detects_var_keyword_catch_all():
    async def worker(task, requirement, **kwargs):
        return None

    assert supports_keyword(worker, "event_callback") is True


def test_rejects_callable_without_the_keyword():
    async def worker(task, requirement):
        return None

    assert supports_keyword(worker, "event_callback") is False


def test_unintrospectable_callable_is_conservative():
    """拿不到签名时返回 False：少传一个可选回调，不要抛 TypeError 断链。"""
    assert supports_keyword(len, "event_callback") is False


def test_bound_method_ignores_self():
    class Registry:
        async def run(self, task, requirement, *, event_callback=None):
            return None

    assert supports_keyword(Registry().run, "event_callback") is True
    assert supports_keyword(Registry().run, "self") is False


@pytest.mark.parametrize("relative_path", _CALL_SITES)
def test_call_sites_do_not_redefine_the_helper(relative_path):
    """没人再私藏一份实现——四份拷贝的历史不要重演。"""
    source = (Path(__file__).resolve().parent.parent / relative_path).read_text(encoding="utf-8")

    assert "def _supports_keyword" not in source, (
        f"{relative_path} 又定义了本地副本；请 import app.utils.callables.supports_keyword"
    )
    assert "from app.utils.callables import supported_kwargs" in source


def test_supported_kwargs_keeps_only_accepted_non_none_values():
    async def narrow(task, requirement):
        return None

    async def wide(task, requirement, *, event_callback=None, prior_results=None):
        return None

    callback = object()

    assert supported_kwargs(narrow, event_callback=callback) == {}
    assert supported_kwargs(wide, event_callback=callback, prior_results=None) == {"event_callback": callback}
    assert supported_kwargs(wide, event_callback=callback, prior_results=[]) == {
        "event_callback": callback,
        "prior_results": [],
    }
