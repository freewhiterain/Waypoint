"""高德天气 Source-native 适配器。"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import httpx

from app.config import settings
from app.mcp_core.reliability import ExternalServiceError, ResilientExecutor
from app.schemas.planning import Evidence


class AmapWeatherAdapter:
    def __init__(
        self,
        api_key: str | None = None,
        executor: ResilientExecutor | None = None,
        client: httpx.AsyncClient | None = None,
    ):
        self.api_key = api_key if api_key is not None else settings.amap_api_key
        self.executor = executor or ResilientExecutor(
            timeout_seconds=settings.external_timeout_seconds,
            max_retries=settings.external_max_retries,
        )
        self.client = client

    async def _request(self, url: str, params: dict) -> dict:
        if self.client is not None:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        async with httpx.AsyncClient(timeout=settings.external_timeout_seconds, trust_env=False) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()

    async def _resolve_adcode(self, city: str) -> str:
        async def operation():
            data = await self._request(
                "https://restapi.amap.com/v3/geocode/geo",
                {"key": self.api_key, "address": city, "output": "JSON"},
            )
            if data.get("status") != "1" or not data.get("geocodes"):
                raise ExternalServiceError(f"无法解析城市：{city}")
            return str(data["geocodes"][0]["adcode"])

        return await self.executor.execute(f"amap:adcode:{city}", operation, ttl_seconds=86400)

    async def query(self, city: str, forecast: bool = True) -> list[Evidence]:
        if not self.api_key:
            raise ExternalServiceError("未配置 AMAP_API_KEY")
        adcode = await self._resolve_adcode(city)
        extension = "all" if forecast else "base"

        async def operation():
            data = await self._request(
                "https://restapi.amap.com/v3/weather/weatherInfo",
                {"key": self.api_key, "city": adcode, "extensions": extension, "output": "JSON"},
            )
            if data.get("status") != "1":
                raise ExternalServiceError(f"高德天气查询失败：{data.get('info', '未知错误')}")
            return data

        data = await self.executor.execute(
            f"amap:weather:{adcode}:{extension}",
            operation,
            ttl_seconds=900 if forecast else 300,
        )
        now = datetime.now(timezone.utc)
        rows = data.get("forecasts", [{}])[0].get("casts", []) if forecast else data.get("lives", [])
        return [
            Evidence(
                content=(
                    f"{city} {row.get('date', row.get('reporttime', ''))}："
                    f"白天{row.get('dayweather', row.get('weather', '未知'))} "
                    f"{row.get('daytemp', row.get('temperature', '未知'))}℃；"
                    f"夜间{row.get('nightweather', '未知')} {row.get('nighttemp', '未知')}℃"
                ),
                source="高德开放平台天气服务",
                source_url="https://lbs.amap.com/api/webservice/guide/api/weatherinfo",
                retrieved_at=now,
                valid_from=now,
                valid_until=now + timedelta(hours=1),
                confidence=0.95,
                metadata={"provider": "amap", "city": city, "raw": row, "evidence_type": "weather"},
            )
            for row in rows
        ]


OPEN_METEO_GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_MAX_DAYS = 16
_DATE_IN_CONTENT = re.compile(r"(\d{4}-\d{2}-\d{2})：")

# WMO 天气代码 → 中文。Open-Meteo 的 daily.weather_code 是当天最严重的天气现象。
_WMO_TEXT = {
    0: "晴", 1: "晴间多云", 2: "多云", 3: "阴", 45: "雾", 48: "雾凇",
    51: "零星小雨", 53: "小雨", 55: "小到中雨", 56: "冻雨", 57: "冻雨",
    61: "小雨", 63: "中雨", 65: "大雨", 66: "冻雨", 67: "冻雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "米雪",
    80: "阵雨", 81: "阵雨", 82: "强阵雨", 85: "阵雪", 86: "强阵雪",
    95: "雷阵雨", 96: "雷阵雨伴冰雹", 99: "强雷阵雨伴冰雹",
}


class OpenMeteoWeatherAdapter:
    """Open-Meteo 每日预报：最长 16 天，不需要 Key（免费版仅限非商业用途）。"""

    def __init__(self, executor: ResilientExecutor | None = None, client: httpx.AsyncClient | None = None):
        self.executor = executor or ResilientExecutor(
            timeout_seconds=settings.external_timeout_seconds,
            max_retries=settings.external_max_retries,
        )
        self.client = client

    async def _request(self, url: str, params: dict) -> dict:
        if self.client is not None:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        async with httpx.AsyncClient(timeout=settings.external_timeout_seconds, trust_env=False) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()

    async def _resolve_location(self, city: str) -> tuple[float, float]:
        async def operation():
            # 先限定在国内找，避免同名的海外地点；找不到再放开。
            for extra in ({"countryCode": "CN"}, {}):
                data = await self._request(
                    OPEN_METEO_GEOCODE_URL, {"name": city, "count": 1, "language": "zh", "format": "json", **extra}
                )
                results = data.get("results") or []
                if results:
                    return float(results[0]["latitude"]), float(results[0]["longitude"])
            raise ExternalServiceError(f"无法解析城市：{city}")

        return await self.executor.execute(f"open_meteo:geo:{city}", operation, ttl_seconds=86400)

    async def query(self, city: str, forecast: bool = True) -> list[Evidence]:
        latitude, longitude = await self._resolve_location(city)
        days = OPEN_METEO_MAX_DAYS if forecast else 1

        async def operation():
            return await self._request(
                OPEN_METEO_FORECAST_URL,
                {
                    "latitude": latitude,
                    "longitude": longitude,
                    "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                    "timezone": "auto",
                    "forecast_days": days,
                },
            )

        data = await self.executor.execute(
            f"open_meteo:daily:{latitude:.3f},{longitude:.3f}:{days}", operation, ttl_seconds=1800
        )
        daily = data.get("daily") or {}
        dates = daily.get("time") or []
        if not dates:
            raise ExternalServiceError("Open-Meteo 没有返回预报数据")
        now = datetime.now(timezone.utc)
        evidence: list[Evidence] = []
        for index, day in enumerate(dates):
            row = {key: (values[index] if index < len(values) else None) for key, values in daily.items()}
            sky = _WMO_TEXT.get(row.get("weather_code"), "未知")
            low, high = row.get("temperature_2m_min"), row.get("temperature_2m_max")
            content = f"{city} {day}：{sky}"
            if low is not None and high is not None:
                content += f" {round(low)}~{round(high)}℃"
            if row.get("precipitation_probability_max") is not None:
                content += f"；降水概率 {round(row['precipitation_probability_max'])}%"
            evidence.append(
                Evidence(
                    content=content,
                    source="Open-Meteo 天气预报",
                    source_url="https://open-meteo.com/en/docs",
                    retrieved_at=now,
                    valid_from=now,
                    valid_until=now + timedelta(hours=3),
                    # 越远的预报越不准。
                    confidence=0.9 if index < 7 else 0.6,
                    metadata={"provider": "open_meteo", "source_type": "api", "city": city, "raw": row,
                              "evidence_type": "weather"},
                )
            )
        return evidence


class CombinedWeatherAdapter:
    """近几天用高德（国内官方预报，4 天），之后的日期用 Open-Meteo 补到 16 天。"""

    def __init__(self, near: AmapWeatherAdapter | None = None, far: OpenMeteoWeatherAdapter | None = None):
        self.near = near if near is not None else AmapWeatherAdapter()
        self.far = far if far is not None else OpenMeteoWeatherAdapter()

    async def query(self, city: str, forecast: bool = True) -> list[Evidence]:
        errors: list[Exception] = []
        near: list[Evidence] = []
        try:
            near = await self.near.query(city, forecast)
        except Exception as exc:
            errors.append(exc)
        if not forecast and near:
            return near
        far: list[Evidence] = []
        try:
            far = await self.far.query(city, forecast)
        except Exception as exc:
            errors.append(exc)
        if not near and not far:
            raise errors[0]
        covered = {match.group(1) for item in near if (match := _DATE_IN_CONTENT.search(item.content))}
        return near + [
            item for item in far
            if not ((match := _DATE_IN_CONTENT.search(item.content)) and match.group(1) in covered)
        ]
