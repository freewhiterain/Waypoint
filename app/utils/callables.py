"""可调用对象的签名探测工具。

`supports_keyword` 原先在四个文件里各抄了一份完全相同的实现
（supervisor、agent_tools、subagents/base、subagents/registry）。它决定的是
"要不要把 event_callback 传下去"，也就是研究进度事件能不能流到前端——
四份实现只要有一份漂移，某条链路就会静默丢掉进度事件而不报错。
放在 app/utils 下而不是任一 agent 模块里，是为了避开 agents 包内部
（registry → base、agent_tools → registry）的相互 import。
"""

from __future__ import annotations

import inspect
from typing import Any


def supports_keyword(callable_obj: Any, keyword: str) -> bool:
    """判断 callable 能否接收指定的关键字参数（含 **kwargs 兜底）。

    取不到签名（C 实现的内置函数、部分 Mock）时保守返回 False：
    宁可少传一个可选回调，也不要抛 TypeError 打断整条规划链路。
    """
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    return keyword in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def supported_kwargs(callable_obj: Any, **optional: Any) -> dict[str, Any]:
    """从可选关键字参数里挑出 callable 接受、且值不为 None 的那些。

    调用方写成 `await fn(a, b, **supported_kwargs(fn, event_callback=cb))`，
    不用在每个调用点各写一遍"先探测再决定怎么调"的分支。
    """
    return {
        name: value
        for name, value in optional.items()
        if value is not None and supports_keyword(callable_obj, name)
    }
