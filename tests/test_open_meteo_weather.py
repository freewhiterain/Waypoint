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
async def test_open_meteo_is_used_for_every_day_and_amap_only_when_it_fails():
    amap = _Fixed(["成都 2026-10-11：晴转多云 17~22℃"])

    primary = CombinedWeatherAdapter(primary=_Fixed(["成都 2026-10-11：晴 16~25℃", "成都 2026-10-12：小雨 13~18℃"]), fallback=amap)
    assert [item.content for item in await primary.query("成都")] == ["成都 2026-10-11：晴 16~25℃", "成都 2026-10-12：小雨 13~18℃"]

    fallback = CombinedWeatherAdapter(primary=_Fixed(error=TimeoutError()), fallback=amap)
    assert [item.content for item in await fallback.query("成都")] == ["成都 2026-10-11：晴转多云 17~22℃"]

    with pytest.raises(ExternalServiceError):
        await CombinedWeatherAdapter(primary=_Fixed(error=TimeoutError()), fallback=_Fixed(error=ExternalServiceError("x"))).query("成都")


@pytest.mark.asyncio
async def test_amap_forecast_is_written_in_the_same_format_as_open_meteo():
    from app.mcp_core.adapters.weather import AmapWeatherAdapter

    def handler(request: httpx.Request) -> httpx.Response:
        if "geocode" in request.url.path:
            return httpx.Response(200, json={"status": "1", "geocodes": [{"adcode": "510100"}]})
        return httpx.Response(200, json={"status": "1", "forecasts": [{"casts": [
            {"date": "2026-10-12", "dayweather": "晴", "nightweather": "多云", "daytemp": "25", "nighttemp": "16"},
            {"date": "2026-10-13", "dayweather": "阴", "nightweather": "阴", "daytemp": "22", "nighttemp": "17"},
        ]}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        evidence = await AmapWeatherAdapter(api_key="test", executor=_executor(), client=client).query("成都")

    assert [item.content for item in evidence] == ["成都 2026-10-12：晴转多云 16~25℃", "成都 2026-10-13：阴 17~22℃"]
