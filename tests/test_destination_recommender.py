"""目的地推荐只在地点库覆盖的城市里挑，亮点只能是该城市真有的热门景点。"""

import pytest

from app.services import destination_recommender as rec


PROFILES = {"西安": ["兵马俑", "大雁塔", "古城墙"], "成都": ["熊猫基地", "武侯祠", "宽窄巷子"], "杭州": ["西湖"]}


class _FakeLLM:
    def __init__(self, picks):
        self.picks = picks
        self.messages = None

    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, messages):
        self.messages = messages
        return rec._Picks(cities=self.picks)


@pytest.mark.asyncio
async def test_unknown_cities_and_made_up_highlights_are_dropped(monkeypatch):
    llm = _FakeLLM([
        rec.CityPick(city="西安", reason="古都", highlights=["兵马俑", "华山"]),
        rec.CityPick(city="大理", reason="风景好", highlights=["洱海"]),
        rec.CityPick(city="成都", reason="美食", highlights=[]),
    ])
    monkeypatch.setattr(rec, "city_profiles", lambda: PROFILES)
    monkeypatch.setattr(rec.settings, "llm_api_key", "test-key")
    monkeypatch.setattr("app.core.llm.get_llm", lambda temperature=None: llm)

    result = await rec.recommend_destinations({"days": 2}, {"styles": ["美食"]})

    assert result.cities == ["西安", "成都"]
    assert "兵马俑" in result.answer and "华山" not in result.answer and "大理" not in result.answer
    assert "天数 2" in result.answer and "喜欢 美食" in result.answer
    assert "宽窄巷子" in result.answer or "熊猫基地" in result.answer


@pytest.mark.asyncio
async def test_without_llm_recommends_the_best_covered_cities(monkeypatch):
    monkeypatch.setattr(rec, "city_profiles", lambda: PROFILES)
    monkeypatch.setattr(rec.settings, "llm_api_key", "")

    result = await rec.recommend_destinations({})

    assert result.cities == ["西安", "成都", "杭州"]


def test_catalog_lists_its_covered_cities():
    from app.places.catalog import get_place_catalog

    cities = get_place_catalog().cities()
    assert len(cities) == 10 and {"成都", "北京", "上海"} <= set(cities)
