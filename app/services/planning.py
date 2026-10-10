"""自然语言需求提取与行程草稿展示。"""

from __future__ import annotations

import re
from datetime import date, timedelta

from app.config import settings
from app.schemas.planning import TravelPlanDraft, TravelRequirementDraft
from app.utils.logger import app_logger


CHINESE_NUMBERS = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}

RELATIVE_DATES = [("大后天", 3), ("后天", 2), ("明天", 1), ("今天", 0)]

# 人数里常见“两个人”“俩人”，天数正则不认这两个字，所以单独一张表。
PEOPLE_NUMBERS = {**CHINESE_NUMBERS, "两": 2, "俩": 2}
_PEOPLE_NUM = r"(\d{1,2}|[一两俩二三四五六七八九十])"
ADULTS_PATTERN = re.compile(_PEOPLE_NUM + r"\s*个?(?:大人|成人)")
CHILDREN_PATTERN = re.compile(_PEOPLE_NUM + r"\s*个?(?:孩子|小孩|儿童|宝宝)")
PEOPLE_PATTERN = re.compile(_PEOPLE_NUM + r"\s*个?(?:人|位)")
FAMILY_PATTERN = re.compile(r"一家([三四五])口")
INTERESTS_PATTERN = re.compile(r"喜欢([^，,。；;！!？?\n]{1,30})")
INTEREST_SEPARATORS = re.compile(r"和|、|及|与|跟|还有|以及|,|，")
# 顺序有意义：“不吃辣”必须先于“吃辣”判断，否则会被误记成能吃辣。
FOOD_PREFERENCE_KEYWORDS = ("清淡", "素食", "火锅", "小吃", "海鲜")

GREETING_STOPWORDS = {"你好", "您好", "在吗", "谢谢", "好的", "可以", "没有", "不用", "嗯嗯", "哈喽", "再见"}

KNOWN_DESTINATIONS = {
    "北京", "上海", "广州", "深圳", "杭州", "南京", "苏州", "成都", "重庆", "西安",
    "厦门", "青岛", "昆明", "丽江", "大理", "桂林", "阳朔", "三亚", "海口", "武汉",
    "长沙", "郑州", "天津", "沈阳", "长春", "哈尔滨", "乌鲁木齐", "拉萨", "西宁", "兰州",
    "银川", "呼和浩特", "贵阳", "南宁", "福州", "南昌", "合肥", "石家庄", "太原", "济南",
    "大连", "宁波", "无锡", "洛阳", "开封", "敦煌", "张家界", "黄山", "九寨沟", "威海",
    "烟台", "秦皇岛", "北戴河", "承德", "大同", "平遥", "凤凰", "香格里拉", "西双版纳",
    "稻城", "泸沽湖", "青海湖", "呼伦贝尔", "漠河", "延吉", "珠海", "汕头", "潮州", "顺德",
}


class RequirementExtractor:
    async def extract(
        self,
        text: str,
        today: date | None = None,
        use_llm: bool = True,
    ) -> TravelRequirementDraft:
        today = today or date.today()
        draft = self._extract_rules(text, today)
        if not draft.missing_fields() or not use_llm or not settings.llm_api_key:
            return draft
        try:
            from app.agents.llm import get_llm

            structured = get_llm().with_structured_output(TravelRequirementDraft)
            llm_draft = await structured.ainvoke(
                [
                    {
                        "role": "system",
                        "content": (
                            f"今天是 {today.isoformat()}。从用户文本提取国内旅行需求，以 JSON 格式输出。"
                            "相对日期（如明天、下周六）换算为具体日期。"
                            "未明确的信息必须返回 null，不得猜测日期、城市或人数。"
                        ),
                    },
                    {"role": "user", "content": text},
                ]
            )
            return self._merge(draft, llm_draft)
        except Exception as exc:
            app_logger.warning(f"需求提取 LLM 兜底失败，退回规则结果: {type(exc).__name__}: {exc}")
            return draft

    @staticmethod
    def _merge(rules: TravelRequirementDraft, llm: TravelRequirementDraft) -> TravelRequirementDraft:
        """规则命中的字段优先（确定性高），LLM 只补规则漏掉的。"""
        values = rules.model_dump()
        llm_values = llm.model_dump()
        for field in ("origin", "destination", "departure_date", "days"):
            if values[field] is None:
                values[field] = llm_values[field]
        for field in ("styles", "special_needs", "food_preferences"):
            merged = values[field] + [item for item in llm_values[field] if item not in values[field]]
            values[field] = merged
        return TravelRequirementDraft(**values)

    @staticmethod
    def _bare_destination(text: str) -> str | None:
        """用户单发一个地名（如“哈尔滨”）时直接识别为目的地。

        只认白名单或“××市/州/县”后缀，避免把“重新推荐”这类指令误判为地名；
        白名单外的城市由 LLM 兜底识别。
        """
        for line in reversed(text.splitlines()):
            candidate = line.strip().strip("，,。.!！?？~～ ")
            if not (2 <= len(candidate) <= 8 and re.fullmatch(r"[一-鿿]+", candidate)):
                continue
            if candidate in KNOWN_DESTINATIONS:
                return candidate
            if len(candidate) >= 3 and candidate[-1] in "市州县" and candidate not in GREETING_STOPWORDS:
                return candidate[:-1] if candidate[-1] == "市" else candidate
        return None

    @staticmethod
    def _extract_rules(text: str, today: date) -> TravelRequirementDraft:
        origin_match = re.search(r"(?:出发地|出发城市)[:：]?\s*([^，,。\s]{2,12})", text)
        if origin_match is None:
            origin_match = re.search(r"从([^，,。\s]{2,12}?)(?:出发|去)", text)
        destination_match = re.search(r"(?:目的地)[:：]?\s*([^，,。\s]{2,12})", text)
        if destination_match is None:
            destination_match = re.search(
                r"(?:规划|安排)(?:一次|一趟)?([^，,。\s\d一二三四五六七八九十]{2,12}?)(?:旅行|旅游|行程|之旅)",
                text,
            )
        if destination_match is None:
            destination_match = re.search(r"(?:去|规划|前往)([^，,。\s\d一二三四五六七八九十]{2,12}?)(?=\d|[一二三四五六七八九十]|旅|游|玩|，|,|。|$)", text)
        destination = destination_match.group(1) if destination_match else RequirementExtractor._bare_destination(text)
        days_match = re.search(r"(\d{1,2}|[一二三四五六七八九十])(?:天|日)(?:游|行程)", text)
        if days_match is None:
            days_match = re.search(r"(?:游玩|旅游|旅行|行程|玩)(\d{1,2}|[一二三四五六七八九十])天", text)
        date_match = re.search(r"(20\d{2})[-年/](\d{1,2})[-月/](\d{1,2})日?", text)
        days = None
        if days_match:
            raw = days_match.group(1)
            days = int(raw) if raw.isdigit() else CHINESE_NUMBERS.get(raw)
        departure_date = None
        if date_match:
            departure_date = date(*(int(value) for value in date_match.groups()))
        else:
            for keyword, offset in RELATIVE_DATES:
                if keyword in text:
                    departure_date = today + timedelta(days=offset)
                    break
        styles = [keyword for keyword in ["文化", "美食", "亲子", "户外", "休闲", "自然"] if keyword in text]
        for interest in RequirementExtractor._extract_interests(text):
            if interest not in styles:
                styles.append(interest)
        adults, children = RequirementExtractor._extract_party(text)
        return TravelRequirementDraft(
            origin=origin_match.group(1) if origin_match else None,
            destination=destination,
            departure_date=departure_date,
            days=days,
            adults=adults,
            children=children,
            styles=styles,
            food_preferences=RequirementExtractor._extract_food_preferences(text),
        )

    @staticmethod
    def _people_number(raw: str) -> int | None:
        return int(raw) if raw.isdigit() else PEOPLE_NUMBERS.get(raw)

    @staticmethod
    def _extract_party(text: str) -> tuple[int, int]:
        """只认明确写出的人数；没写就保持 1 个大人，不猜同行人数。"""
        family = FAMILY_PATTERN.search(text)
        if family:
            total = CHINESE_NUMBERS[family.group(1)]
            return 2, total - 2

        children_match = CHILDREN_PATTERN.search(text)
        children = RequirementExtractor._people_number(children_match.group(1)) if children_match else 0
        children = children or 0

        adults_match = ADULTS_PATTERN.search(text)
        if adults_match:
            adults = RequirementExtractor._people_number(adults_match.group(1)) or 1
            return max(adults, 1), children

        # “两个人”“3位”是总人数，里面若另写了孩子数，大人数要扣掉孩子。
        people_match = PEOPLE_PATTERN.search(text)
        if people_match:
            total = RequirementExtractor._people_number(people_match.group(1)) or 1
            return max(total - children, 1), children
        return 1, children

    @staticmethod
    def _extract_interests(text: str) -> list[str]:
        """“喜欢美食和熊猫”→ ["美食", "熊猫"]，作为风格偏好交给子 Agent。"""
        interests: list[str] = []
        for match in INTERESTS_PATTERN.finditer(text):
            for part in INTEREST_SEPARATORS.split(match.group(1)):
                item = part.strip().removesuffix("的")
                # 口味（“喜欢吃辣”）归 food_preferences，不重复记成游玩风格。
                if "辣" in item:
                    continue
                if 1 <= len(item) <= 10 and item not in interests:
                    interests.append(item)
        return interests

    @staticmethod
    def _extract_food_preferences(text: str) -> list[str]:
        preferences: list[str] = []
        if "不吃辣" in text or "不能吃辣" in text:
            preferences.append("不吃辣")
        elif any(marker in text for marker in ("能吃辣", "爱吃辣", "喜欢吃辣", "无辣不欢")):
            preferences.append("能吃辣")
        preferences.extend(keyword for keyword in FOOD_PREFERENCE_KEYWORDS if keyword in text)
        return preferences

def render_plan_markdown(draft: TravelPlanDraft) -> str:
    lines = [f"# {draft.requirement.destination}{draft.requirement.days}日行程草稿", ""]
    if draft.requirement.assumptions:
        lines.append("## 假设说明")
        lines.extend(f"- {assumption}" for assumption in draft.requirement.assumptions)
        lines.append("")
    for day in draft.itinerary:
        lines.append(f"## 第{day.day}天 · {day.date.isoformat()}")
        labels = {"morning": "上午", "afternoon": "下午", "evening": "晚上"}
        for slot in day.slots:
            lines.append(f"- **{labels[slot.period]}**：{slot.title}。{slot.description}")
        lines.append("")
    if draft.warnings:
        lines.append("## 数据说明")
        lines.extend(f"- {warning}" for warning in draft.warnings)
    lines.append("\n当前为可编辑规划草稿，不包含下单、预订或支付服务。")
    return "\n".join(lines)
