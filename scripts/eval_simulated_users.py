"""模拟用户评测（参照 τ-bench）：让模型扮演带着隐藏目标的用户，和服务多轮对话直到拿到行程。

eval_dialogues.py 的每句输入是写死的，测的是"这句话进来、那份行程出去"；真实用户
会一点点透露信息、答非所问、中途补充。这里每个场景只给模拟用户一个目标和说话风格，
台词由模型现编，所以每次对话都不一样，更接近线上的真实分布。

运行方式（先启动服务）：
    python scripts/eval_simulated_users.py              # 跑全部用户
    python scripts/eval_simulated_users.py -k xian      # 只跑名字包含 xian 的
    python scripts/eval_simulated_users.py --repeat 3   # 每个用户跑 3 次，看 pass^3

判定（全部通过才算成功）：
- 交付：最多 MAX_TURNS 轮内弹出规划表单，并拿到结构化行程
- 理解：表单预填和最终需求里的目的地、天数、出发日期和目标一致，点名的地点、忌口记下了
- 行程：过 app/evaluation/plan_checks 的环境层和逻辑层检查
完整对话写到 logs/eval_simulated_users.json。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from app.core.llm import get_llm  # noqa: E402
from eval_dialogues import LAYERS, Client, _structured_issues, _text_of, summarize  # noqa: E402

MAX_TURNS = 6
FORM_TOOL = "collect_trip_requirements"


@dataclass
class SimulatedUser:
    name: str
    # 交给模拟用户的目标，用自然语言写；模拟用户只会按它说话
    goal: str
    persona: str
    # 用来判定的结构化期望
    destination: str
    days: int
    departure_date: date
    must_visit: list[str] = field(default_factory=list)
    food_avoid: list[str] = field(default_factory=list)
    # 看完第一版行程后想改的地方（自然语言），以及改完后行程里不能再出现的地点、要轻松的天
    after_plan: str = ""
    avoid: list[str] = field(default_factory=list)
    relaxed_days: list[int] = field(default_factory=list)


def _next_weekday(weekday: int) -> date:
    today = date.today()
    return today + timedelta(days=(weekday - today.weekday()) % 7 or 7)


def users() -> list[SimulatedUser]:
    friday = _next_weekday(0) + timedelta(days=4)  # 下周五 = 下周一 + 4
    saturday = _next_weekday(5)
    soon = date.today() + timedelta(days=6)
    return [
        SimulatedUser(
            name="nanjing_history_terse",
            goal=f"{friday.isoformat()}（下周五）去南京玩 3 天，喜欢历史，中山陵一定要去。",
            persona="说话很短，一次只说一件事，助手问什么才答什么。",
            destination="南京", days=3, departure_date=friday, must_visit=["中山陵"],
        ),
        SimulatedUser(
            name="guangzhou_family_no_spicy",
            goal=f"{soon.isoformat()} 带老婆和 6 岁的孩子去广州玩 2 天，全家都不吃辣。",
            persona="说话随意，第一句只说想带家人出去玩，没说去哪；之后慢慢补充。",
            destination="广州", days=2, departure_date=soon, food_avoid=["辣"],
        ),
        SimulatedUser(
            name="beijing_parents_relaxed",
            goal=f"{saturday.isoformat()}（这周六）陪爸妈去北京玩 3 天，爸妈年纪大了，行程别太赶。",
            persona="说话客气，喜欢先寒暄一句再说正事。",
            destination="北京", days=3, departure_date=saturday,
        ),
        SimulatedUser(
            name="chongqing_couple_named_spot",
            goal=f"{soon.isoformat()} 和女朋友去重庆玩 2 天，想吃火锅，洪崖洞和李子坝轻轨站必须去。",
            persona="说话活泼，第一句就把城市和天数说了，日期和想去的地方后面再说。",
            destination="重庆", days=2, departure_date=soon, must_visit=["洪崖洞", "李子坝轻轨站"],
        ),
        SimulatedUser(
            name="suzhou_changes_mind",
            goal=f"一开始说想去杭州，聊一句后改主意，最终去苏州玩 2 天，{soon.isoformat()} 出发。",
            persona="犹豫不决，会先提杭州，下一句说算了改去苏州。",
            destination="苏州", days=2, departure_date=soon,
        ),
        SimulatedUser(
            name="xian_outside_catalog",
            # 西安不在地点库里：候选全靠联网搜索
            goal=f"{soon.isoformat()} 去西安玩 3 天，兵马俑一定要去，喜欢历史。",
            persona="说话直接，第一句就说城市、天数和日期。",
            destination="西安", days=3, departure_date=soon, must_visit=["兵马俑"],
        ),
        SimulatedUser(
            name="chengdu_revise_after_plan",
            goal=f"{soon.isoformat()} 去成都玩 2 天。",
            persona="说话简短。",
            destination="成都", days=2, departure_date=soon,
            after_plan="看完行程后不想去熊猫基地，并且希望第二天轻松一点",
            avoid=["熊猫"], relaxed_days=[2],
        ),
        SimulatedUser(
            name="hangzhou_date_later",
            goal=f"去杭州玩 2 天，喜欢自然风光。日期一开始没定，被问到时说 {soon.isoformat()} 出发。",
            persona="说话简洁，第一句只说城市和天数。",
            destination="杭州", days=2, departure_date=soon,
        ),
    ]


_USER_PROMPT = """你在扮演一个正在用旅行助手 App 规划行程的普通用户，用中文和助手对话。
你的目标：{goal}
你的说话风格：{persona}
规则：
- 每次只输出你要对助手说的一句话，不加引号、不加旁白。
- 只表达目标里有的信息，不要编造目标以外的要求，也不要一次把所有信息倒出来（除非风格要求）。
- 助手问到什么就如实回答；助手确认信息或问要不要开始规划时，说"开始规划吧"。
- 不要扮演助手，不要替助手回答。"""


def next_user_line(llm: Any, user: SimulatedUser, history: list[tuple[str, str]]) -> str:
    dialogue = "\n".join(f"{'我' if role == 'user' else '助手'}：{text}" for role, text in history) or "（还没开始，你先开口）"
    response = llm.invoke([
        SystemMessage(content=_USER_PROMPT.format(goal=user.goal, persona=user.persona)),
        HumanMessage(content=f"到目前为止的对话：\n{dialogue}\n\n你接下来说："),
    ])
    return str(response.content).strip().strip("“”\"").splitlines()[0][:200]


_NOTES_PROMPT = """助手弹出了行程表单，目的地、日期、天数已经填好，最下面有一栏“还有什么要求？（可选）”。
你的目标：{goal}
到目前为止你在对话里说过的话：{said}
逐条对照目标：想去的具体地方、忌口、同行的人、节奏、兴趣。目标里有、但你在对话里还没说过的，写进这一栏，用一句话。
只能写目标里明确有的内容，不要编造（目标没提老人就不要写老人，没提忌口就不要写忌口）。
确实都说过了才只输出“无”。"""


def form_notes(llm: Any, user: SimulatedUser, history: list[tuple[str, str]]) -> str:
    said = "；".join(text for role, text in history if role == "user")
    response = llm.invoke([HumanMessage(content=_NOTES_PROMPT.format(goal=user.goal, said=said))])
    text = str(response.content).strip().strip("“”\"")
    return "" if text in ("无", "没有", "") else text[:300]


_REVISE_PROMPT = """你刚收到旅行助手给你做的行程，内容如下：
{plan}
你想这样修改：{change}
用一句口语化的话告诉助手，不加引号、不加旁白。"""


def revision_line(llm: Any, user: SimulatedUser, plan_text: str) -> str:
    response = llm.invoke([HumanMessage(content=_REVISE_PROMPT.format(plan=plan_text[:1500], change=user.after_plan))])
    return str(response.content).strip().strip("“”\"").splitlines()[0][:200]


def _revision_issues(user: SimulatedUser, result: dict[str, Any]) -> list[str]:
    issues = []
    titles = [slot["title"] for day in result.get("itinerary") or [] for slot in day["slots"]]
    for place in user.avoid:
        if any(place in title for title in titles):
            issues.append(f"用户说不去「{place}」，改完的行程里还有")
    relaxed = set((result.get("requirement") or {}).get("relaxed_days") or [])
    for day in user.relaxed_days:
        if day not in relaxed:
            issues.append(f"没记下第{day}天要轻松")
    return issues


def _form_of(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    return next((e for e in events if e.get("type") == "tool_call" and e.get("tool") == FORM_TOOL), None)


def _understanding_issues(user: SimulatedUser, requirement: dict[str, Any]) -> list[str]:
    issues = []
    expected = {"destination": user.destination, "days": user.days, "departure_date": user.departure_date.isoformat()}
    for key, value in expected.items():
        if requirement.get(key) != value:
            issues.append(f"需求 {key} 是 {requirement.get(key)!r}，用户要的是 {value!r}")
    remembered = "".join(requirement.get("must_visit") or [])
    for place in user.must_visit:
        if place not in remembered:
            issues.append(f"没记下用户点名的「{place}」")
    food = "".join(requirement.get("food_preferences") or []) + "".join(requirement.get("special_needs") or [])
    for word in user.food_avoid:
        if word not in food:
            issues.append(f"没记下忌口「{word}」")
    return issues


def _revise(
    client: Client, llm: Any, user: SimulatedUser, conversation_id: str, plan_text: str,
    history: list[tuple[str, str]], transcript: list[dict[str, Any]], issues: dict[str, list[str]],
) -> dict[str, list[str]]:
    """看完行程提修改：应再次弹出表单，提交后以改完的那版判定行程。"""
    history.append(("assistant", "（给出了第一版行程）"))
    line = revision_line(llm, user, plan_text)
    history.append(("user", line))
    events = client.send(conversation_id, line)
    transcript.append({"user": line, "events": events})
    form = _form_of(events)
    if form is None:
        issues["delivery"].append(f"修改意见「{line}」没有触发重新规划")
        return issues
    replan = client.submit_form(form["call_id"], form["arguments"]["initial_values"])
    transcript.append({"form": form["arguments"], "events": replan})
    # 第一版的问题被改掉才算数：只按改完的这版判定行程
    issues = {"delivery": issues["delivery"], "environment": [], "logic": []}
    _structured_issues(replan, issues)
    result = next((e["payload"].get("result") for e in replan if e.get("type") == "result"), None)
    if isinstance(result, dict):
        issues["logic"].extend(_revision_issues(user, result))
    return issues


def run_user(client: Client, llm: Any, user: SimulatedUser) -> dict[str, Any]:
    started = time.perf_counter()
    conversation_id = client.new_conversation()
    history: list[tuple[str, str]] = []
    transcript: list[dict[str, Any]] = []
    issues: dict[str, list[str]] = {layer: [] for layer in LAYERS}
    form = None
    for _ in range(MAX_TURNS):
        line = next_user_line(llm, user, history)
        events = client.send(conversation_id, line)
        transcript.append({"user": line, "events": events})
        history.append(("user", line))
        form = _form_of(events)
        if form is not None:
            break
        history.append(("assistant", _text_of(events)[:600] or "（无回复）"))

    if form is None:
        issues["delivery"].append(f"{MAX_TURNS} 轮内没有进入规划")
    else:
        # 用户在表单里会把错的改对、缺的补上：按目标提交，预填错了记一笔理解问题
        prefill = form.get("arguments", {}).get("initial_values", {})
        values = {"destination": user.destination, "days": user.days, "departure_date": user.departure_date.isoformat()}
        # 预填错了才算理解问题；用户还没说到正确的目的地（比如说了杭州、正要改主意）时
        # 预填成杭州是对的，用户会在表单里改。
        said = "".join(text for role, text in history if role == "user")
        for key, value in values.items():
            stated = user.destination in said if key == "destination" else True
            if key in prefill and prefill[key] != value and stated:
                issues["logic"].append(f"表单预填 {key}={prefill[key]!r}，用户要的是 {value!r}")
        notes = form_notes(llm, user, history)
        if notes:
            values["notes"] = notes
            history.append(("form", notes))
        plan_events = client.submit_form(form["call_id"], values)
        transcript.append({"form": values, "events": plan_events})
        _structured_issues(plan_events, issues)
        result = next((e["payload"].get("result") for e in plan_events if e.get("type") == "result"), None)
        if isinstance(result, dict) and result.get("requirement"):
            issues["logic"].extend(_understanding_issues(user, result["requirement"]))
        if user.after_plan and isinstance(result, dict):
            issues = _revise(client, llm, user, conversation_id, _text_of(plan_events), history, transcript, issues)

    failures = [f"[{layer}] {message}" for layer in LAYERS for message in issues[layer]]
    return {
        "name": user.name,
        "plans": True,
        "passed": not failures,
        "layers": {layer: not issues[layer] for layer in LAYERS},
        "failures": failures,
        "turns": sum(role == "user" for role, _ in history),
        "seconds": round(time.perf_counter() - started, 1),
        "dialogue": history,
        "transcript": transcript,
    }


_SPEAKERS = {"user": "用户", "assistant": "助手", "form": "表单补充"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.environ.get("EVAL_BASE_URL", "http://127.0.0.1:18000"))
    parser.add_argument("-k", default="", help="只跑名字包含该子串的用户")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    client = Client(args.base_url, args.timeout)
    client.login()
    llm = get_llm(temperature=0.8)
    results = []
    for trial in range(1, args.repeat + 1):
        for user in users():
            if args.k and args.k not in user.name:
                continue
            try:
                result = run_user(client, llm, user)
            except Exception as exc:  # 单个用户失败不影响其余
                result = {"name": user.name, "plans": True, "passed": False,
                          "failures": [f"[delivery] {type(exc).__name__}: {exc}"]}
            result["trial"] = trial
            results.append(result)
            prefix = f"#{trial} " if args.repeat > 1 else ""
            print(f"{prefix}[{'PASS' if result['passed'] else 'FAIL'}] {user.name} "
                  f"({result.get('turns', '-')} 轮, {result.get('seconds', '-')}s)")
            for role, text in result.get("dialogue", []):
                print(f"       {_SPEAKERS[role]}：{text[:80]}")
            for failure in result["failures"]:
                print(f"       - {failure}")

    report = ROOT / "logs" / "eval_simulated_users.json"
    report.parent.mkdir(exist_ok=True)
    report.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print()
    for line in summarize(results, args.repeat):
        print(line)
    print(f"完整输出：{report}")
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
