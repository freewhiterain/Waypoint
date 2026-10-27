"""端到端对话评测：对着正在运行的服务跑一组多轮对话，检查"输入 → 输出"是否对得上。

运行方式（先启动服务）：
    python scripts/eval_dialogues.py                 # 跑全部场景
    python scripts/eval_dialogues.py -k chengdu      # 只跑名字包含 chengdu 的场景
    python scripts/eval_dialogues.py --no-plan       # 不提交表单，只看路由和槽位

账号从 .env 里的 AUTO_TEST_USERNAME / AUTO_TEST_PASSWORD 读取（允许以 "# " 注释形式出现），
也可以用环境变量 EVAL_USERNAME / EVAL_PASSWORD 覆盖。

每个场景声明期望：第一轮该走哪个动作、表单预填了哪些槽位、最终行程里必须出现/
不能出现的文字。每份行程还要过一组质量底线（没有退回规则选点、没有误导性的空时段、
晚餐人均不超过 400 元）。脚本逐条断言，最后打印通过率，并把完整输出写到 logs/eval_dialogues.json。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


@dataclass
class Scenario:
    name: str
    turns: list[str]
    # 最后一轮期望的路由动作：tool_call(=进入规划表单) / token(=直接回答)
    expect_route: str = "tool_call"
    # 表单预填值里必须出现的键值（最后一轮的 tool_call）
    expect_prefill: dict[str, Any] = field(default_factory=dict)
    # 表单缺项时用来补齐的值
    fill: dict[str, Any] = field(default_factory=dict)
    # 最终输出（行程 markdown 或回答）必须包含 / 不能包含的文字
    must_contain: list[str] = field(default_factory=list)
    must_not_contain: list[str] = field(default_factory=list)
    # 拿到行程后再说的修改意见：每句都应再次出现表单，提交后以最后一版行程做断言。
    revise: list[str] = field(default_factory=list)


def _next_weekday(weekday: int) -> date:
    today = date.today()
    return today + timedelta(days=(weekday - today.weekday()) % 7 or 7)


def scenarios() -> list[Scenario]:
    soon = (date.today() + timedelta(days=5)).isoformat()
    return [
        Scenario(
            name="chengdu_full_slots",
            turns=[f"帮我规划成都三天的行程，{soon}出发，喜欢美食"],
            expect_prefill={"destination": "成都", "days": 3, "departure_date": soon},
            # 说了喜欢美食：每天要有午餐
            must_contain=["成都", "第1天", "第3天", "**中午**"],
            must_not_contain=["第4天"],
        ),
        Scenario(
            name="hangzhou_missing_date",
            turns=["我想去杭州玩两天，帮我安排一下"],
            expect_prefill={"destination": "杭州", "days": 2},
            fill={"departure_date": soon},
            must_contain=["杭州", "第1天", "第2天"],
            must_not_contain=["第3天"],
        ),
        Scenario(
            name="beijing_multi_turn",
            turns=["下周六想去北京", "玩4天，带孩子", "开始规划吧"],
            # "下周六"指下一周的周六，不是最近的这个周六
            expect_prefill={
                "destination": "北京",
                "days": 4,
                "departure_date": (_next_weekday(0) + timedelta(days=5)).isoformat(),
            },
            must_contain=["北京", "第4天"],
            must_not_contain=["第5天"],
        ),
        Scenario(
            name="hangzhou_next_weekend",
            turns=["下周末去杭州玩两天，帮我规划一下"],
            expect_prefill={
                "destination": "杭州",
                "days": 2,
                "departure_date": (_next_weekday(0) + timedelta(days=5)).isoformat(),
            },
            must_contain=["杭州", "第2天"],
        ),
        Scenario(
            name="nanjing_from_shanghai",
            turns=[f"{soon}从上海出发去南京玩两天，帮我规划"],
            expect_prefill={"destination": "南京", "days": 2, "departure_date": soon},
            must_contain=["从上海出发", "返回上海"],
            must_not_contain=["第3天"],
        ),
        Scenario(
            name="change_destination",
            turns=["帮我规划去上海玩2天", "算了，改成南京吧"],
            expect_prefill={"destination": "南京", "days": 2},
            fill={"departure_date": soon},
            must_contain=["南京", "第2天"],
            must_not_contain=["上海", "第3天"],
        ),
        Scenario(
            name="chengdu_no_spicy",
            turns=[f"带爸妈去成都玩两天，{soon}出发，他们不吃辣，帮我规划一下"],
            expect_prefill={"destination": "成都", "days": 2, "departure_date": soon},
            must_contain=["成都", "第2天"],
            must_not_contain=["麻辣", "火锅", "第3天"],
        ),
        Scenario(
            name="shanghai_museum_lover",
            turns=[f"{soon}去上海玩两天，喜欢博物馆和美术馆，帮我做个攻略"],
            expect_prefill={"destination": "上海", "days": 2, "departure_date": soon},
            must_contain=["上海", "博物馆"],
        ),
        Scenario(
            name="chengdu_revise_after_plan",
            turns=[f"帮我规划成都两天的行程，{soon}出发"],
            revise=["把熊猫基地换成别的，第二天轻松一点"],
            expect_prefill={"destination": "成都", "days": 2, "departure_date": soon},
            must_contain=["第2天", "轻松"],
            must_not_contain=["熊猫", "第3天"],
        ),
        Scenario(
            name="open_question_weather",
            turns=["成都最近天气怎么样"],
            expect_route="token",
            must_contain=["成都"],
        ),
        Scenario(
            name="budget_question",
            turns=["成都玩三天两个人大概要花多少钱"],
            expect_route="token",
            must_contain=["元"],
            must_not_contain=["没有任何具体价格"],
        ),
        Scenario(
            name="recommend_destination",
            turns=["国庆不知道去哪玩，推荐几个城市"],
            expect_route="token",
        ),
        Scenario(
            name="chitchat",
            turns=["你好"],
            expect_route="token",
        ),
        Scenario(
            name="off_topic",
            turns=["帮我写一段 Python 排序代码"],
            expect_route="token",
            must_not_contain=["def "],
        ),
    ]


def _load_credentials() -> tuple[str, str]:
    username = os.environ.get("EVAL_USERNAME")
    password = os.environ.get("EVAL_PASSWORD")
    env_file = ROOT / ".env"
    if (not username or not password) and env_file.exists():
        for raw in env_file.read_text(encoding="utf-8").splitlines():
            line = raw.lstrip("# ").strip()
            if line.startswith("AUTO_TEST_USERNAME=") and not username:
                username = line.split("=", 1)[1]
            if line.startswith("AUTO_TEST_PASSWORD=") and not password:
                password = line.split("=", 1)[1]
    if not username or not password:
        raise SystemExit("缺少评测账号：设置 EVAL_USERNAME / EVAL_PASSWORD 或在 .env 里写 AUTO_TEST_*")
    return username, password


def _read_sse(response: httpx.Response) -> list[dict[str, Any]]:
    events = []
    for line in response.iter_lines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))
    return events


class Client:
    def __init__(self, base_url: str, timeout: float) -> None:
        self.http = httpx.Client(base_url=base_url, trust_env=False, timeout=timeout)

    def login(self) -> None:
        username, password = _load_credentials()
        response = self.http.post("/api/v1/users/login", json={"username": username, "password": password})
        response.raise_for_status()
        self.http.headers["Authorization"] = f"Bearer {response.json()['access_token']}"

    def new_conversation(self) -> str:
        response = self.http.post("/api/v1/conversations", json={})
        response.raise_for_status()
        body = response.json()
        return body.get("id") or body["conversation"]["id"]

    def send(self, conversation_id: str, content: str) -> list[dict[str, Any]]:
        with self.http.stream(
            "POST", f"/api/v1/chat/stream/{conversation_id}", json={"content": content}
        ) as response:
            response.raise_for_status()
            return _read_sse(response)

    def submit_form(self, call_id: str, values: dict[str, Any]) -> list[dict[str, Any]]:
        body = {"tool": "collect_trip_requirements", "status": "completed", "result": values}
        with self.http.stream("POST", f"/api/v1/chat/tools/{call_id}/result", json=body) as response:
            response.raise_for_status()
            return _read_sse(response)


# 每份行程都要满足的质量底线（与具体场景无关）。
MAX_DINNER_PRICE = 400
_PRICE = re.compile(r"人均 (\d+(?:\.\d+)?) 元")


def plan_quality_failures(output: str) -> list[str]:
    failures = []
    if "智能选点暂时不可用" in output:
        failures.append("LLM 选点失败，退回了规则选点")
    if "没有第二个带证据的景点候选" in output:
        failures.append("出现了误导性的空时段文案")
    expensive = [float(value) for value in _PRICE.findall(output) if float(value) > MAX_DINNER_PRICE]
    if expensive:
        failures.append(f"晚餐人均超过 {MAX_DINNER_PRICE} 元：{expensive}")
    return failures


def _text_of(events: list[dict[str, Any]]) -> str:
    """按前端的方式拼回答：token 追加，token_reset 清空已显示的内容。"""
    text = ""
    for event in events:
        if event.get("type") == "token":
            text += event.get("content", "")
        elif event.get("type") == "token_reset":
            text = ""
    return text


def run_scenario(client: Client, scenario: Scenario, submit_plan: bool) -> dict[str, Any]:
    started = time.perf_counter()
    conversation_id = client.new_conversation()
    transcript = []
    last_events: list[dict[str, Any]] = []
    for turn in scenario.turns:
        last_events = client.send(conversation_id, turn)
        transcript.append({"user": turn, "events": last_events})

    failures: list[str] = []
    # 只有规划表单算"进入规划"；线路 1 的 ReAct 也会发 tool_call（查资料），那仍是直接回答。
    tool_call = next(
        (
            e
            for e in last_events
            if e.get("type") == "tool_call" and e.get("tool") == "collect_trip_requirements"
        ),
        None,
    )
    route = "tool_call" if tool_call else "token"
    if route != scenario.expect_route:
        failures.append(f"路由期望 {scenario.expect_route}，实际 {route}")

    output = _text_of(last_events)
    if tool_call is not None:
        prefill = tool_call.get("arguments", {}).get("initial_values", {})
        for key, expected in scenario.expect_prefill.items():
            if prefill.get(key) != expected:
                failures.append(f"预填 {key} 期望 {expected!r}，实际 {prefill.get(key)!r}")
        if submit_plan:
            values = {**scenario.fill, **prefill}
            missing = [k for k in ("destination", "departure_date", "days") if k not in values]
            if missing:
                failures.append(f"表单仍缺 {missing}，无法提交")
            else:
                plan_events = client.submit_form(tool_call["call_id"], values)
                transcript.append({"form": values, "events": plan_events})
                errors = [e for e in plan_events if e.get("type") == "error"]
                if errors:
                    failures.append(f"规划报错：{errors[0].get('message')}")
                output = _text_of(plan_events)
                failures.extend(plan_quality_failures(output))
                for turn in scenario.revise:
                    revise_events = client.send(conversation_id, turn)
                    transcript.append({"user": turn, "events": revise_events})
                    form = next(
                        (e for e in revise_events if e.get("tool") == "collect_trip_requirements"), None
                    )
                    if form is None:
                        failures.append(f"修改意见「{turn}」没有触发重新规划")
                        break
                    replan = client.submit_form(form["call_id"], form["arguments"]["initial_values"])
                    transcript.append({"form": form["arguments"], "events": replan})
                    output = _text_of(replan)
                    failures.extend(plan_quality_failures(output))

    if submit_plan or tool_call is None:
        for needle in scenario.must_contain:
            if needle not in output:
                failures.append(f"输出缺少「{needle}」")
        for needle in scenario.must_not_contain:
            if needle in output:
                failures.append(f"输出不应出现「{needle}」")

    return {
        "name": scenario.name,
        "passed": not failures,
        "failures": failures,
        "seconds": round(time.perf_counter() - started, 1),
        "output": output,
        "transcript": transcript,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("EVAL_BASE_URL", "http://127.0.0.1:18000"))
    parser.add_argument("-k", default="", help="只跑名字包含该子串的场景")
    parser.add_argument("--no-plan", action="store_true", help="不提交表单")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    client = Client(args.base_url, args.timeout)
    client.login()
    results = []
    for scenario in scenarios():
        if args.k and args.k not in scenario.name:
            continue
        try:
            result = run_scenario(client, scenario, submit_plan=not args.no_plan)
        except Exception as exc:  # 单个场景失败不影响其余场景
            result = {"name": scenario.name, "passed": False, "failures": [f"{type(exc).__name__}: {exc}"]}
        results.append(result)
        mark = "PASS" if result["passed"] else "FAIL"
        print(f"[{mark}] {scenario.name} ({result.get('seconds', '-')}s)")
        for failure in result["failures"]:
            print(f"       - {failure}")

    report = ROOT / "logs" / "eval_dialogues.json"
    report.parent.mkdir(exist_ok=True)
    report.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    passed = sum(r["passed"] for r in results)
    print(f"\n{passed}/{len(results)} 通过，完整输出：{report}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
