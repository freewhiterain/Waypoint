"""2. 排程：把选中的景点分到每天、排好上下午，再配午餐、晚饭和夜景。"""

from __future__ import annotations

from itertools import permutations

from app.places.catalog import is_local_food
from app.schemas.planning import CandidateOption, ItineraryDay, TimeSlot, TravelRequirement, WorkerResult
from app.agents.itinerary.rules import (
    ATTRACTIONS_PER_DAY,
    _avoided,
    _base_name,
    _candidates,
    day_capacities,
    _DINNER_WINDOW,
    _fits,
    FULL_DAY_NOTE,
    is_excursion,
    is_full_day,
    is_night_spot,
    MAX_DINNER_MINUTES,
    MAX_HOP_MINUTES,
    MAX_LUNCH_MINUTES,
    MAX_NIGHT_MINUTES,
    _morning_best,
    _open_during,
    _open_for_lunch,
    _options,
    outskirts,
    PlaceSelection,
    same_place,
    _serves,
    wanted_dishes,
    wants_lunch,
    wants_night,
)
from app.agents.itinerary.travel import (
    DRY_MARGIN,
    _OUTDOOR_TYPES,
    _rain_chance,
    RAINY_CHANCE,
    TravelCosts,
    _trip_weather,
)


def dodge_rain(
    day_plans: list[list[str]],
    options: dict[str, CandidateOption],
    weather: dict[int, str],
    capacities: list[int],
) -> list[list[str]]:
    """雨天那天户外景点多、另一天明显更干且户外少时，两天的景点整组对调。

    只换日子不换地点：每天内部的顺路关系不变；只在容量相同的两天之间换，
    "第二天轻松点"这类按天的要求不受影响。
    """
    plans = [list(plan) for plan in day_plans]

    def outdoor(plan: list[str]) -> int:
        return sum(str(options[place_id].attributes.get("type")) in _OUTDOOR_TYPES for place_id in plan)

    chance = {day: _rain_chance(text) for day, text in weather.items()}
    rainy = sorted((day for day, value in chance.items() if value is not None and value >= RAINY_CHANCE),
                   key=lambda day: -(chance[day] or 0))
    swapped: set[int] = set()
    for wet in rainy:
        if wet in swapped:
            continue
        candidates = [
            day for day in range(1, len(plans) + 1)
            if day != wet and day not in swapped and chance.get(day) is not None
            and (chance[day] or 0) + DRY_MARGIN <= (chance[wet] or 0)
            and capacities[day - 1] == capacities[wet - 1] and outdoor(plans[day - 1]) < outdoor(plans[wet - 1])
        ]
        if candidates:
            dry = min(candidates, key=lambda day: chance[day] or 0)
            plans[wet - 1], plans[dry - 1] = plans[dry - 1], plans[wet - 1]
            swapped.update((wet, dry))
    return plans


def assign_days(
    attraction_ids: list[str],
    preferred: dict[str, int],
    days: int,
    hotel_id: str | None,
    costs: TravelCosts,
    capacities: list[int] | None = None,
) -> list[list[str]]:
    """把景点分到每天（每天最多 capacities[i] 个，默认 ATTRACTIONS_PER_DAY），并排好每天内部顺序。

    先放用户/LLM 指定了日期的景点；剩下的按"离酒店最近的作为当天起点，再补离它
    最近的"贪心成组，让同一天的景点彼此靠近。
    """
    capacities = capacities or [ATTRACTIONS_PER_DAY] * days
    plan: list[list[str]] = [[] for _ in range(days)]
    remaining = list(attraction_ids)
    for place_id in list(remaining):
        day = preferred.get(place_id)
        if day is not None and len(plan[day - 1]) < capacities[day - 1]:
            plan[day - 1].append(place_id)
            remaining.remove(place_id)

    for day_places, capacity in zip(plan, capacities):
        while remaining and len(day_places) < capacity:
            anchor = day_places[-1] if day_places else hotel_id
            nearest = min(remaining, key=lambda place_id: (costs.cost(anchor, place_id), remaining.index(place_id)))
            # 当天已经有景点时，离它太远的不硬凑进同一天：这个时段宁可空着，远的留给后面的日子。
            if day_places and costs.cost(anchor, nearest) > MAX_HOP_MINUTES:
                break
            day_places.append(nearest)
            remaining.remove(nearest)

    # 贪心到最后一天时，被"太远"挡下的景点没有后面的日子可去，原先会被悄悄丢掉，
    # 行程里却出现"没有第二个景点候选"的空时段。这里把它们补进还有空位、
    # 增加路程最少的那一天；真的太远时由约束检查报出来，交给 LLM 重选。
    for place_id in list(remaining):
        open_days = [day_places for day_places, capacity in zip(plan, capacities) if len(day_places) < capacity]
        if not open_days:
            break
        target = min(
            open_days,
            key=lambda day_places: costs.cost(day_places[-1] if day_places else hotel_id, place_id),
        )
        target.append(place_id)
        remaining.remove(place_id)

    return [_best_order(day_places, hotel_id, costs) for day_places in plan]


def _day_cost(plan: list[str], hotel_id: str | None, costs: TravelCosts) -> int:
    """从住处出发、按顺序走完当天景点的路程（取两种走法里短的那个）。"""
    def walk(order: list[str]) -> int:
        stops = [hotel_id, *order] if hotel_id else order
        return sum(costs.cost(first, second) for first, second in zip(stops, stops[1:]))

    return min(walk(plan), walk(plan[::-1])) if plan else 0


def fill_empty_days(day_plans: list[list[str]], whole_day: list[CandidateOption]) -> list[list[str]]:
    """补完空位后仍有一整天空着（能凑普通一天的候选不够了），就排一个还没去的远郊或乐园，便宜的优先。

    上海学生预算行程：候选里一半是乐园和远郊，普通景点只够两天，第三天整天"自由活动"。
    """
    used = {place_id for plan in day_plans for place_id in plan}
    spare = sorted((option for option in whole_day if option.id not in used),
                   key=lambda option: option.estimated_cost if option.estimated_cost is not None else float("inf"))
    plans = [list(plan) for plan in day_plans]
    for plan in plans:
        if not plan and spare:
            plan.append(spare.pop(0).id)
    return plans


def regroup_days(
    day_plans: list[list[str]], fixed: set[str], hotel_id: str | None, costs: TravelCosts, max_swaps: int = 20
) -> list[list[str]]:
    """贪心分天之后做局部调整：两天之间对调景点，总路程变短就保留，直到找不到更好的对调。

    贪心从离住处最近的开始一天天凑，排在后面的几天只能拿剩下的，常常横跨半个城。
    指定了日子的（preferred_day、整天的乐园）不动；只做对调，每天的景点数不变。
    """
    plans = [list(plan) for plan in day_plans]

    def better_swap() -> tuple[int, int, int, int] | None:
        for first in range(len(plans)):
            for second in range(first + 1, len(plans)):
                before = _day_cost(plans[first], hotel_id, costs) + _day_cost(plans[second], hotel_id, costs)
                for i, a in enumerate(plans[first]):
                    for j, b in enumerate(plans[second]):
                        if a in fixed or b in fixed:
                            continue
                        new_first = plans[first][:i] + [b] + plans[first][i + 1:]
                        new_second = plans[second][:j] + [a] + plans[second][j + 1:]
                        if _day_cost(new_first, hotel_id, costs) + _day_cost(new_second, hotel_id, costs) < before:
                            return first, i, second, j
        return None

    for _ in range(max_swaps):
        swap = better_swap()
        if swap is None:
            break
        first, i, second, j = swap
        plans[first][i], plans[second][j] = plans[second][j], plans[first][i]
    # 每天内部取路程更短的那个方向
    return [
        plan if _day_cost(plan, hotel_id, costs) <= _day_cost(plan[::-1], hotel_id, costs) else plan[::-1]
        for plan in plans
    ]


def top_up_days(
    day_plans: list[list[str]],
    candidate_ids: list[str],
    hotel_id: str | None,
    costs: TravelCosts,
    capacities: list[int] | None = None,
    names: dict[str, str] | None = None,
    city: str = "",
) -> list[list[str]]:
    """选点不够时，从没选中的候选里给空时段补一个顺路的景点。

    LLM 常常只挑出 天数×2 少一两个的景点，原先空出来的半天会写成"没有第二个
    带证据的景点候选"，可候选其实还有几十个。候选顺序已经是"命中偏好的在前、
    其余按热门程度"，这里在离当天已有景点（没有时离酒店）不超过 MAX_HOP_MINUTES
    的候选里取最近的；附近确实没有时才让时段空着。
    """
    capacities = capacities or [ATTRACTIONS_PER_DAY] * len(day_plans)
    used = {place_id for day in day_plans for place_id in day}
    used_names = [names[place_id] for place_id in used if place_id in names] if names else []
    filled: list[list[str]] = []
    for day_places, capacity in zip(day_plans, capacities):
        day_places = list(day_places)
        while len(day_places) < capacity:
            anchor = day_places[-1] if day_places else hotel_id
            pool = [
                place_id
                for place_id in candidate_ids
                if place_id not in used and not any(same_place(names.get(place_id, ""), name, city) for name in used_names)
            ] if names else [place_id for place_id in candidate_ids if place_id not in used]
            if not pool:
                break
            nearest = min(pool, key=lambda place_id: (costs.cost(anchor, place_id), pool.index(place_id)))
            if costs.cost(anchor, nearest) > MAX_HOP_MINUTES:
                break
            day_places.append(nearest)
            used.add(nearest)
            if names and nearest in names:
                used_names.append(names[nearest])
        filled.append(_best_order(day_places, hotel_id, costs))
    return filled


def _best_order(place_ids: list[str], hotel_id: str | None, costs: TravelCosts) -> list[str]:
    """每天只有两三个点，直接枚举所有顺序取总路程最短的。"""
    if len(place_ids) < 2:
        return list(place_ids)

    def route_cost(order: tuple[str, ...]) -> int:
        stops = [hotel_id, *order]
        return sum(costs.cost(a, b) for a, b in zip(stops, stops[1:]))

    return list(min(permutations(place_ids), key=route_cost))


def _pick_nearby(pool: list[str], used: set[str], near: str, costs: TravelCosts, max_minutes: int) -> str | None:
    """午餐、夜景这类"顺路加一站"的时段：在离 near 不超过 max_minutes 的候选里取最近的。"""
    candidates = [place_id for place_id in pool if place_id not in used]
    if not candidates:
        return None
    best = min(candidates, key=lambda place_id: (costs.cost(near, place_id), candidates.index(place_id)))
    return best if costs.cost(near, best) <= max_minutes else None


def _pick_lunch(
    pool: list[str], used: set[str], near: str, costs: TravelCosts, options: dict[str, CandidateOption], city: str,
    after: str | None = None,
) -> str | None:
    """午餐在上午景点附近（MAX_LUNCH_MINUTES 内）挑：当地菜和小吃优先，其次看绕路多少。

    原来只挑最近的，成都中午吃粤菜、重庆中午吃 186 元的西餐，评审说不像专程来吃当地菜的。
    绕路 = 上午景点 → 午餐 → 下午景点 比直接过去多走的路：离上午近、却在反方向的店
    （熊猫基地附近的东北郊火锅，下午再横穿回市区）会被排到后面。
    """
    nearby = [
        place_id for place_id in pool
        if place_id not in used and costs.cost(near, place_id) <= MAX_LUNCH_MINUTES
    ]
    if not nearby:
        return None
    def detour(place_id: str) -> int:
        if after is None:
            return costs.cost(near, place_id)
        return costs.cost(near, place_id) + costs.cost(place_id, after) - costs.cost(near, after)

    return min(nearby, key=lambda place_id: (
        not is_local_food(city, str(options[place_id].attributes.get("type") or "")), detour(place_id),
    ))


def build_itinerary(
    requirement: TravelRequirement,
    results: list[WorkerResult],
    selection: PlaceSelection,
    costs: TravelCosts,
) -> list[ItineraryDay]:
    options = {option.id: option for worker in ("attractions", "food") for option in _options(results, worker)}
    preferred = {item.id: item.preferred_day for item in selection.attractions if item.preferred_day}
    capacities = day_capacities(requirement)
    remote = outskirts(results)

    def whole(option: CandidateOption) -> bool:
        return is_full_day(option) or is_excursion(option, remote)

    # 乐园独占一天；远郊景点也各给一天，彼此挨着的（都江堰和青城山）可以同一天，但不和市区景点混排。
    # 先定好日子、那天的容量只留给它们，排在最前面先放。
    full_day: list[str] = []
    excursion_days: dict[str, int] = {}
    for item in selection.attractions:
        option = options[item.id]
        if not whole(option):
            continue
        taken = {preferred[place_id] for place_id in full_day}
        if not is_full_day(option):
            near = next((day for other, day in excursion_days.items()
                         if costs.cost(other, item.id) <= MAX_HOP_MINUTES), None)
            if near is not None:
                if capacities[near - 1] >= ATTRACTIONS_PER_DAY:
                    continue
                preferred[item.id] = near
                capacities[near - 1] += 1
                full_day.append(item.id)
                continue
        # 第一天多半要赶路，整天的安排尽量放在行程中间
        middle = (requirement.days + 1) / 2
        free = sorted((day for day in range(1, requirement.days + 1) if day not in taken),
                      key=lambda day: (day == 1 and requirement.days > 1, abs(day - middle)))
        if not free:
            continue
        day = item.preferred_day if item.preferred_day in free else free[0]
        preferred[item.id] = day
        full_day.append(item.id)
        # 那天的容量 = 排进去的远郊/乐园数：市区景点即使真实车程刚好压线，也不会被凑进来
        capacities[day - 1] = 1
        if not is_full_day(option):
            excursion_days[item.id] = day
    attraction_ids = full_day + [
        item.id for item in selection.attractions if item.id not in full_day and not whole(options[item.id])
    ]
    day_plans = assign_days(attraction_ids, preferred, requirement.days, selection.hotel_id, costs, capacities)
    day_plans = top_up_days(
        day_plans,
        [option.id for option in _candidates(results, "attractions", requirement) if not whole(option)],
        selection.hotel_id,
        costs,
        capacities,
        {option.id: option.name for option in _options(results, "attractions")},
        requirement.destination,
    )
    day_plans = fill_empty_days(day_plans, [option for option in _candidates(results, "attractions", requirement)
                                            if whole(option)])
    day_plans = regroup_days(day_plans, set(preferred), selection.hotel_id, costs)
    day_plans = dodge_rain(day_plans, options, _trip_weather(requirement, results), capacities)
    open_for_dinner = [
        option.id for option in _candidates(results, "food", requirement) if _open_during(option, _DINNER_WINDOW)
    ]
    dinner_pool = [item.id for item in selection.dinners if item.id in open_for_dinner]
    backup_dinners = [place_id for place_id in open_for_dinner if place_id not in dinner_pool]
    used_dinners: set[str] = set()
    food_names = {option.id: option.name for option in _options(results, "food")}

    def eat_at(place_id: str) -> None:
        """记下吃过的餐厅；同一家连锁的其他分店（括号外名字相同）也不再排。"""
        base = _base_name(food_names.get(place_id, ""))
        used_dinners.add(place_id)
        used_dinners.update(other for other, name in food_names.items() if base and _base_name(name) == base)

    # 联网搜索的城市，同一个名字常同时出现在景点和美食候选里（西安的"回民街"）：
    # 已经作为景点去了的，不再当饭店排。只比括号外的名字是否完全相同，"外滩酒店·西餐厅"不受影响。
    sight_bases = {_base_name(options[place_id].name) for plan in day_plans for place_id in plan}
    used_dinners.update(other for other, name in food_names.items() if _base_name(name) in sight_bases)
    night_options = {
        option.id: option
        for option in _options(results, "attractions")
        if is_night_spot(option) and not _avoided(option, requirement)
    }
    night_pool = list(night_options) if wants_night(requirement) else []
    used_nights: set[str] = set()
    # 午餐从没被选作晚餐的餐饮候选里就近挑，不交给 LLM：它只关心"离上午景点近、中午开门"。
    lunch_pool = (
        [
            option.id
            for option in _candidates(results, "food", requirement)
            if option.id not in dinner_pool and _open_for_lunch(option)
        ]
        if wants_lunch(requirement)
        else []
    )

    itinerary: list[ItineraryDay] = []
    for index, place_ids in enumerate(day_plans):
        slots: list[TimeSlot] = []
        previous = selection.hotel_id
        periods = _assign_periods([options[place_id] for place_id in place_ids])
        whole_day = len(place_ids) == 1 and whole(options[place_ids[0]])
        for period in ("morning", "afternoon"):
            option = periods.get(period)
            if whole_day:
                if option is not None:
                    slot = _slot(period, option, *costs.leg(previous, option.id))
                    slots.append(slot.model_copy(update={"description": f"{FULL_DAY_NOTE}{slot.description}"}))
                    previous = option.id
                continue
            if period == "afternoon" and previous is not None and periods.get("morning") is not None:
                after = option.id if option is not None else None
                lunch_id = _pick_lunch(lunch_pool, used_dinners, previous, costs, options, requirement.destination, after)
                if lunch_id is not None:
                    eat_at(lunch_id)
                    slots.append(_slot("lunch", options[lunch_id], *costs.leg(previous, lunch_id)))
                    previous = lunch_id
            if option is not None:
                slots.append(_slot(period, option, *costs.leg(previous, option.id)))
                previous = option.id
            elif period == "morning" and not periods:
                slots.append(TimeSlot(period="morning", title=f"{requirement.destination}分区自由活动",
                                      description="该时段没有留下带证据的景点候选。"))
            elif period == "morning":
                slots.append(TimeSlot(period="morning", title="自由活动",
                                      description="当天的景点开门较晚，安排在下午；上午可以自由安排。"))
            elif capacities[index] < ATTRACTIONS_PER_DAY:
                slots.append(TimeSlot(period="afternoon", title="自由活动",
                                      description="按你的要求这天安排得轻松些，下午留给休息或在周边随意逛逛。"))
            else:
                slots.append(TimeSlot(period="afternoon", title="自由活动",
                                      description="附近没有能顺路安排的景点，下午留给休息或在周边随意逛逛。"))

        # 选中的晚餐用完了就从其余餐饮候选里就近补，不再让同一家餐厅连吃几晚。
        # 选中的那家离当天最后一站太远时，附近有备选就换成备选：玩完再跑一个多小时去吃饭不合理。
        dinner_id = _pick_dinner(dinner_pool, used_dinners, previous, costs) or _pick_dinner(
            backup_dinners, used_dinners, previous, costs
        )
        if dinner_id is not None and previous is not None and costs.cost(previous, dinner_id) > MAX_DINNER_MINUTES:
            nearby = _pick_dinner(backup_dinners, used_dinners, previous, costs)
            if nearby is not None and costs.cost(previous, nearby) <= MAX_DINNER_MINUTES:
                dinner_id = nearby
        if dinner_id is not None:
            eat_at(dinner_id)
            slots.append(_slot("evening", options[dinner_id], *costs.leg(previous, dinner_id)))
            previous = dinner_id
        else:
            reason = "能查到的餐厅前几晚已经安排过，这晚可以自由选择。" if used_dinners else "没有带证据的餐饮候选可安排。"
            slots.append(TimeSlot(period="evening", title="自由安排晚餐", description=reason))

        # 想看夜景的，晚饭后就近安排一个夜景点。
        night_id = _pick_nearby(night_pool, used_nights, previous, costs, MAX_NIGHT_MINUTES) if previous else None
        if night_id is not None:
            used_nights.add(night_id)
            # "重庆夜游""夜游两江"其实是同一种活动，排过一个就把同名的都排除掉。
            used_nights.update(
                other
                for other, candidate in night_options.items()
                if same_place(candidate.name, night_options[night_id].name, requirement.destination)
            )
            slots.append(_slot("night", night_options[night_id], *costs.leg(previous, night_id)))

        notes = [] if place_ids else [f"第{index + 1}天已无未使用的景点候选。"]
        itinerary.append(
            ItineraryDay(
                day=index + 1,
                date=requirement.departure_date.fromordinal(requirement.departure_date.toordinal() + index),
                slots=slots,
                notes=notes,
            )
        )
    return ensure_wanted_food(requirement, results, itinerary, costs)


def ensure_wanted_food(
    requirement: TravelRequirement, results: list[WorkerResult], itinerary: list[ItineraryDay], costs: TravelCosts
) -> list[ItineraryDay]:
    """说了"想吃火锅"，整趟却一顿火锅都没有时，把一顿晚饭换成附近能吃到的。

    重庆情侣两天吃了两顿西餐、没有火锅，评审直接判"严重偏离"。换哪一顿：在 45 分钟内
    找得到、晚饭时段营业、没吃过的，挑离上一站最近的那一顿。
    """
    dishes = wanted_dishes(requirement)
    if not dishes:
        return itinerary
    by_name = {option.name: option for worker in ("attractions", "food", "hotel") for option in _options(results, worker)}
    food = _candidates(results, "food", requirement)
    days = [day.model_copy(deep=True) for day in itinerary]
    for dish in dishes:
        meals = [slot for day in days for slot in day.slots if slot.period in ("lunch", "evening")]
        if any(slot.title in by_name and _serves(by_name[slot.title], dish) for slot in meals):
            continue
        eaten = {_base_name(slot.title) for slot in meals}
        best: tuple[int, int, int, CandidateOption] | None = None
        for day_index, day in enumerate(days):
            for slot_index, slot in enumerate(day.slots):
                if slot.period != "evening":
                    continue
                previous_title = day.slots[slot_index - 1].title if slot_index else None
                previous = by_name[previous_title].id if previous_title in by_name else None
                for option in food:
                    if not _serves(option, dish) or _base_name(option.name) in eaten:
                        continue
                    if not _open_during(option, _DINNER_WINDOW):
                        continue
                    minutes = costs.cost(previous, option.id) if previous else 0
                    if minutes <= MAX_DINNER_MINUTES and (best is None or minutes < best[2]):
                        best = (day_index, slot_index, minutes, option)
        if best is not None:
            day_index, slot_index, _, option = best
            previous_title = days[day_index].slots[slot_index - 1].title if slot_index else None
            previous = by_name[previous_title].id if previous_title in by_name else None
            days[day_index].slots[slot_index] = _slot("evening", option, *costs.leg(previous, option.id))
    return days


def _assign_periods(places: list[CandidateOption]) -> dict[str, CandidateOption]:
    """按路程排好顺序的当天景点 → 上午 / 下午。开放时间对不上时调换，或者把唯一的景点放到下午。"""
    if len(places) >= 2:
        first, second = places[0], places[1]
        if not (_fits(first, "morning") and _fits(second, "afternoon")) and _fits(second, "morning") and _fits(first, "afternoon"):
            first, second = second, first
        elif _morning_best(second) and not _morning_best(first) and _fits(second, "morning") and _fits(first, "afternoon"):
            first, second = second, first
        return {"morning": first, "afternoon": second}
    if len(places) == 1:
        only = places[0]
        late = not _fits(only, "morning") and _fits(only, "afternoon")
        return {"afternoon" if late else "morning": only}
    return {}


def _pick_dinner(pool: list[str], used: set[str], previous: str | None, costs: TravelCosts) -> str | None:
    if not pool:
        return None
    candidates = [dinner for dinner in pool if dinner not in used]
    if not candidates:
        return None
    return min(candidates, key=lambda dinner: (costs.cost(previous, dinner), candidates.index(dinner)))


def _slot(
    period: str, option: CandidateOption, travel_minutes: int | None, travel_mode: str | None = None
) -> TimeSlot:
    opening = option.attributes.get("opening_hours")
    location = option.attributes.get("location") or option.attributes.get("district")
    return TimeSlot(
        period=period,
        title=option.name,
        description=option.description,
        location=str(location) if location else None,
        travel_minutes=travel_minutes,
        travel_mode=travel_mode,
        opening_window=str(opening) if opening else None,
        estimated_cost=option.estimated_cost,
    )
