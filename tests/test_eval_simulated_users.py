"""模拟用户评测的"理解"判定：目的地、天数、日期、必去、忌口都要记对。"""

import importlib.util
import sys
from datetime import date
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
_spec = importlib.util.spec_from_file_location("eval_simulated_users", SCRIPTS / "eval_simulated_users.py")
sim = importlib.util.module_from_spec(_spec)
sys.modules["eval_simulated_users"] = sim
_spec.loader.exec_module(sim)

USER = sim.SimulatedUser(
    name="t", goal="", persona="", destination="重庆", days=2, departure_date=date(2026, 11, 3),
    must_visit=["洪崖洞"], food_avoid=["辣"],
)


def test_a_fully_understood_requirement_has_no_issues():
    requirement = {"destination": "重庆", "days": 2, "departure_date": "2026-11-03",
                   "must_visit": ["洪崖洞"], "food_preferences": ["不吃辣"]}

    assert sim._understanding_issues(USER, requirement) == []


def test_wrong_slots_and_forgotten_constraints_are_reported():
    requirement = {"destination": "成都", "days": 2, "departure_date": "2026-11-04", "must_visit": []}

    issues = sim._understanding_issues(USER, requirement)

    assert len(issues) == 4
    assert any("洪崖洞" in issue for issue in issues) and any("忌口" in issue for issue in issues)


def test_every_simulated_user_has_a_reachable_goal():
    for user in sim.users():
        assert user.departure_date >= date.today()
        assert user.destination in user.goal
