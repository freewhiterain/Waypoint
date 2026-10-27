"""
测试对话模型连接与结构化输出。

运行方式：python scripts/test_llm.py

直接复用 app.core.llm.get_llm()，所以这里测到的就是应用真正会用的
供应商、模型和参数——包括结构化输出走哪种 method。单测普通对话是不够的：
本项目的路由、需求抽取、Worker 分析和行程合成全部依赖结构化输出，
普通对话通、结构化不通时，系统会静默退化成兜底路径而不报错。
"""
# === 兼容性修复：让脚本能直接运行（python scripts/test_llm.py） ===
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# === 修复结束 ===

# Windows 控制台默认 GBK，编码不了下面的符号会抛 UnicodeEncodeError，
# 把真正的错误信息盖掉。显式换成 UTF-8 并对无法编码的字符降级。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pydantic import BaseModel, Field

from app.core.llm import get_llm
from app.config import settings


class _Probe(BaseModel):
    """用于验证结构化输出链路的最小 schema。"""

    city: str = Field(description="用户提到的城市")
    days: int = Field(description="用户提到的天数")


def main() -> int:
    print(f"供应商入口: {settings.llm_base_url}")
    print(f"模型:       {settings.llm_model}")
    print(f"关闭思考:   {settings.llm_disable_thinking}")
    if not settings.llm_api_key:
        print("[失败] 未配置 LLM_API_KEY")
        return 1

    llm = get_llm()

    try:
        response = llm.invoke([{"role": "user", "content": "你好，请用一句话介绍你自己。"}])
        print(f"[通过] 普通对话 -> {response.content.strip()[:60]}")
    except Exception as exc:
        print(f"[失败] 普通对话 -> {type(exc).__name__}: {exc}")
        return 1

    try:
        result = llm.with_structured_output(_Probe).invoke(
            [{"role": "user", "content": "我想去成都玩三天"}]
        )
        print(f"[通过] 结构化输出 -> {_Probe.model_validate(result).model_dump_json()}")
    except Exception as exc:
        print(f"[失败] 结构化输出 -> {type(exc).__name__}: {exc}")
        print("       结构化不通时，路由/需求抽取/Worker 分析/行程合成都会退化成兜底路径。")
        return 1

    print("全部通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
