"""Open-Meteo 预报适配器，以及"近期高德、远期 Open-Meteo"的合并（不访问网络）。"""

import httpx
import pytest

from app.mcp_core.adapters.weather import CombinedWeatherAdapter, OpenMeteoWeatherAdapter
from app.mcp_core.reliability import ExternalServiceError, ResilientExecutor
from app.schemas.planning import Evidence


def _executor():
    return ResilientExecutor(timeout_seconds=1, max_retries=0)


def _handler(requests, *, geocode_results=None):
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "geocoding-api.open-meteo.com":
            results = geocode_results if geocode_results is not None else [{"latitude": 30.67, "longitude": 104.07}]
            return httpx.Response(200, json={"results": results})
        return httpx.Response(200, json={"daily": {
            "time": ["2026-10-15", "2026-10-16"],
            "weather_code": [61, 0],
            "temperature_2m_max": [18.4, 24.6],
            "temperature_2m_min": [13.2, 15.5],
            "precipitation_probability_max": [80, None],
        }})

    return handler


@pytest.mark.asyncio
async def test_daily_forecast_becomes_one_dated_chinese_evidence_per_day():
    requests = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(_handler(requests))) as client:
        evidence = await OpenMeteoWeatherAdapter(executor=_executor(), client=client).query("成都")

    assert [item.content for item in evidence] == [
        "成都 2026-10-15：小雨 13~18℃；降水概率 80%",
        "成都 2026-10-16：晴 16~25℃",
    ]
    assert evidence[0].source_url and evidence[0].metadata["source_type"] == "api"
    assert requests[0].url.params["countryCode"] == "CN"
    assert requests[1].url.params["forecast_days"] == "16"


@pytest.mark.asyncio
async def test_unknown_city_raises_instead_of_guessing_a_location():
    requests = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(_handler(requests, geocode_results=[]))) as client:
        with pytest.raises(ExternalServiceError):
            await OpenMeteoWeatherAdapter(executor=_executor(), client=client).query("不存在的地方")

    assert len(requests) == 2  # 国内找不到后放开再找一次，都没有就放弃


class _Fixed:
    def __init__(self, contents=None, error=None):
        self.contents, self.error = contents or [], error

    async def query(self, city, forecast=True):
        if self.error:
            raise self.error
        return [Evidence(content=text, source="s") for text in self.contents]


@pytest.mark.asyncio
async def test_near_dates_come_from_amap_and_later_dates_from_open_meteo():
    adapter = CombinedWeatherAdapter(
        near=_Fixed(["成都 2026-10-11：白天晴 22℃；夜间多云 17℃", "成都 2026-10-12：白天晴 25℃；夜间多云 16℃"]),
        far=_Fixed(["成都 2026-10-12：晴 16~25℃", "成都 2026-10-13：小雨 13~18℃；降水概率 80%"]),
    )

    evidence = await adapter.query("成都")

    assert [item.content[3:13] for item in evidence] == ["2026-10-11", "2026-10-12", "2026-10-13"]
    assert "白天晴 25℃" in evidence[1].content  # 重叠的日期保留高德的


@pytest.mark.asyncio
async def test_one_source_failing_still_returns_the_other_and_both_failing_raises():
    far_only = CombinedWeatherAdapter(near=_Fixed(error=ExternalServiceError("未配置 AMAP_API_KEY")),
                                      far=_Fixed(["成都 2026-10-13：小雨 13~18℃"]))
    assert len(await far_only.query("成都")) == 1

    near_only = CombinedWeatherAdapter(near=_Fixed(["成都 2026-10-11：白天晴 22℃"]), far=_Fixed(error=TimeoutError()))
    assert len(await near_only.query("成都")) == 1

    with pytest.raises(ExternalServiceError):
        await CombinedWeatherAdapter(near=_Fixed(error=ExternalServiceError("x")), far=_Fixed(error=TimeoutError())).query("成都")
