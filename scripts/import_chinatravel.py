"""把 ChinaTravel 沙盒数据转成本项目的地点库（data/places/*.csv）。

数据来源：https://huggingface.co/datasets/LAMDA-NeSy/ChinaTravel-Sandbox （CC BY 4.0）
下载 raw/ChinaTravel_sandbox_zh.zip 并解压后，把解压目录里的 database 目录传进来：

    python scripts/import_chinatravel.py D:/path/to/zh/database

只读 CSV、只写 data/places 下的三个文件，可以重复执行。
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

CITIES = {
    "beijing": "北京", "shanghai": "上海", "guangzhou": "广州", "shenzhen": "深圳", "chengdu": "成都",
    "chongqing": "重庆", "hangzhou": "杭州", "nanjing": "南京", "suzhou": "苏州", "wuhan": "武汉",
}
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "places"

# 目标列 → 源列。rank 是源文件里的行号，源数据按热门程度排列。
COLUMNS = {
    "attractions": {
        "name": "name", "type": "type", "latitude": "lat", "longitude": "lon", "open_time": "opentime",
        "close_time": "endtime", "price": "price", "min_hours": "recommendmintime", "max_hours": "recommendmaxtime",
    },
    "restaurants": {
        "name": "name", "type": "cuisine", "latitude": "lat", "longitude": "lon", "open_time": "opentime",
        "close_time": "endtime", "price": "price", "recommended_food": "recommendedfood",
    },
    "accommodations": {
        "name": "name", "type": "featurehoteltype", "latitude": "lat", "longitude": "lon", "price": "price",
        "beds": "numbed",
    },
}


def convert(source: Path, kind: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for folder, city in CITIES.items():
        files = sorted((source / kind / folder).glob("*.csv"))
        if not files:
            print(f"跳过 {kind}/{folder}：没有找到 CSV")
            continue
        with files[0].open(encoding="utf-8-sig", newline="") as handle:
            for rank, raw in enumerate(csv.DictReader(handle)):
                name = (raw.get("name") or "").strip()
                if not name:
                    continue
                row = {"city": city, "rank": str(rank)}
                row.update({target: (raw.get(column) or "").strip() for target, column in COLUMNS[kind].items()})
                rows.append(row)
    return rows


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    source = Path(sys.argv[1])
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for kind in COLUMNS:
        rows = convert(source, kind)
        with (OUTPUT / f"{kind}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["city", "rank", *COLUMNS[kind]], lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        print(f"{kind}: {len(rows)} 条 → {OUTPUT / (kind + '.csv')}")


if __name__ == "__main__":
    main()
