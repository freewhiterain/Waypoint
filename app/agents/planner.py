"""确定性的旅行研究 Planner。"""

from app.schemas.planning import ResearchTask, TravelRequirement


def create_research_plan(requirement: TravelRequirement) -> list[ResearchTask]:
    """Generate five research tasks in two dependency groups.

    景点、住宿、美食、天气互不依赖，第一批并行；交通查的是这些地点之间的
    市内交通，必须等景点、住宿、美食出了候选才能算，所以作为第二批。
    """
    destination = requirement.destination
    common_criteria = ["返回结构化候选项", "事实结论附带 Evidence", "缺失实时数据时明确降级"]

    def task(task_type, query, tools, dependencies=()):
        return ResearchTask(
            task_type=task_type,
            query=query,
            required_tools=tools,
            completion_criteria=common_criteria,
            dependencies=[dependency.id for dependency in dependencies],
        )

    attractions_task = task(
        "attractions", f"研究{destination}的景点、文化和适合的游览区域", ["hybrid_rag", "search"]
    )
    hotel_task = task("hotel", f"研究{destination}适合本次行程的住宿区域与住宿类型", ["hotel_api", "map"])
    food_task = task("food", f"研究{destination}本地美食与用户饮食偏好匹配情况", ["hybrid_rag", "search"])
    weather_task = task(
        "weather", f"查询{destination}在{requirement.departure_date}前后的天气与出行条件", ["weather_api"]
    )
    transport_task = task(
        "transport",
        f"计算{destination}市内景点、住宿与餐饮之间的交通方式和耗时",
        ["route_graph", "map"],
        dependencies=(attractions_task, hotel_task, food_task),
    )
    return [attractions_task, hotel_task, food_task, weather_task, transport_task]


def parallel_groups(tasks: list[ResearchTask]) -> list[list[ResearchTask]]:
    """按依赖关系生成可并行执行的任务组，并检测循环依赖。"""
    remaining = {task.id: task for task in tasks}
    completed: set[str] = set()
    groups: list[list[ResearchTask]] = []

    while remaining:
        ready = [
            task
            for task in remaining.values()
            if set(task.dependencies).issubset(completed)
        ]
        if not ready:
            raise ValueError("研究任务存在循环依赖或引用了不存在的依赖")
        groups.append(ready)
        for task in ready:
            completed.add(task.id)
            remaining.pop(task.id)

    return groups
