"""评测汇总：分层通过率（DR/EPR/LPR/FPR）和重复跑的 pass^N。"""

import importlib.util
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "eval_dialogues", Path(__file__).resolve().parents[1] / "scripts" / "eval_dialogues.py"
)
eval_dialogues = importlib.util.module_from_spec(_spec)
sys.modules["eval_dialogues"] = eval_dialogues  # dataclass 需要能按模块名找到自己
_spec.loader.exec_module(eval_dialogues)


def result(name, delivery=True, environment=True, logic=True, plans=True):
    layers = {"delivery": delivery, "environment": environment, "logic": logic}
    return {"name": name, "plans": plans, "layers": layers, "passed": all(layers.values())}


def test_layered_rates_only_count_delivered_plans():
    lines = eval_dialogues.summarize(
        [result("a"), result("b", environment=False), result("c", delivery=False, logic=False), result("qa", plans=False)],
        repeat=1,
    )

    assert lines[0] == "pass@1：2/4 = 50%"
    assert "交付 DR 2/3" in lines[1] and "环境 EPR 1/3" in lines[1]
    assert "逻辑 LPR 2/3" in lines[1] and "最终 FPR 1/3" in lines[1]


def test_pass_hat_k_needs_every_trial_to_pass():
    lines = eval_dialogues.summarize(
        [result("a"), result("b"), result("a"), result("b", logic=False)], repeat=2
    )

    assert "pass^2：1/2 = 50%" in lines
    assert lines[-1] == "时好时坏：b(1/2)"
