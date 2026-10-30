"""用模型给评测产出的行程打"软质量"分，帮错误分析找该细看的行程。

plan_checks 只能判对错（地点在不在库里、天数对不对、超没超预算），判断不了"安排得好不好"：
一天里从城东跑到城西再回来、描述空洞、和用户说的兴趣不搭。这里让模型按固定量表逐项打分
并写出理由，按总分从低到高列出来——低分的那几份再由人去读完整对话。

分数没有和人工标注对齐过，不作为通过/失败的门槛（参照 Hamel Husain 的做法：先人工标几十份，
确认评分模型和人的判断一致，才能拿它当自动化指标）。

运行方式（先跑过 eval_dialogues.py 或 eval_simulated_users.py）：
    python scripts/judge_plans.py                          # 评 logs/eval_dialogues.json
    python scripts/judge_plans.py logs/eval_simulated_users.json
结果写到 logs/judge_plans.json。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.core.llm import get_llm  # noqa: E402
from eval_dialogues import _text_of  # noqa: E402

RUBRIC = {
    "route": "路线：同一天的地点彼此近、顺序不绕路，没有一天里来回横穿城市",
    "pace": "节奏：每天安排的量合适，符合用户要的轻松或紧凑，吃饭时间合理",
    "fit": "贴合：选的景点和餐厅符合用户说的兴趣、同行人、忌口、预算",
    "writing": "描述：具体、有用（怎么去、开放时间、注意事项），没有空话套话和自相矛盾",
}


class Score(BaseModel):
    score: int = Field(ge=1, le=5, description="1=很差，3=能用但有明显问题，5=很好")
    reason: str = Field(description="一句话说明扣分点；满分就说好在哪")


class Judgement(BaseModel):
    route: Score
    pace: Score
    fit: Score
    writing: Score


_PROMPT = """你是严格的旅行行程评审。下面是用户和旅行助手的对话，以及助手最终给出的行程。
按下面四项分别打 1~5 分并给出一句理由，只根据行程内容和用户说过的话判断，不要因为格式好看加分：
{rubric}

用户说过的话：
{said}

行程：
{plan}"""


def plans_from_log(path: Path) -> list[dict[str, Any]]:
    """从评测日志里取出每份最终行程（规划后又修改的取最后一版）和用户说过的话。"""
    plans = []
    for result in json.loads(path.read_text(encoding="utf-8")):
        said = [turn["user"] for turn in result.get("transcript", []) if "user" in turn]
        forms = [turn for turn in result.get("transcript", []) if "form" in turn]
        notes = [turn["form"].get("notes") for turn in forms if isinstance(turn.get("form"), dict) and turn["form"].get("notes")]
        if not forms:
            continue
        text = _text_of(forms[-1]["events"])
        if "## 第1天" not in text:
            continue
        plans.append({"name": result["name"], "trial": result.get("trial", 1), "said": said + notes, "plan": text})
    return plans


def judge(llm: Any, item: dict[str, Any]) -> Judgement:
    prompt = _PROMPT.format(
        rubric="\n".join(f"- {key}：{text}" for key, text in RUBRIC.items()),
        said="\n".join(f"- {line}" for line in item["said"]),
        plan=item["plan"][:6000],
    )
    return Judgement.model_validate(llm.with_structured_output(Judgement).invoke([{"role": "user", "content": prompt}]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", nargs="?", default=str(ROOT / "logs" / "eval_dialogues.json"))
    args = parser.parse_args()

    llm = get_llm(temperature=0)
    rows = []
    for item in plans_from_log(Path(args.log)):
        try:
            judgement = judge(llm, item)
        except Exception as exc:  # 单份评审失败不影响其余
            print(f"[跳过] {item['name']}: {type(exc).__name__}")
            continue
        scores = {key: getattr(judgement, key) for key in RUBRIC}
        rows.append({
            "name": item["name"], "trial": item["trial"],
            "total": sum(score.score for score in scores.values()),
            "scores": {key: score.model_dump() for key, score in scores.items()},
        })

    rows.sort(key=lambda row: row["total"])
    for row in rows:
        print(f"{row['total']:>2}/20  {row['name']} #{row['trial']}")
        for key, score in row["scores"].items():
            if score["score"] < 4:
                print(f"         {key} {score['score']}：{score['reason']}")
    if rows:
        averages = {key: sum(row["scores"][key]["score"] for row in rows) / len(rows) for key in RUBRIC}
        print("\n平均分：" + " ｜ ".join(f"{key} {value:.1f}" for key, value in averages.items()))
    report = ROOT / "logs" / "judge_plans.json"
    report.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完整结果：{report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
