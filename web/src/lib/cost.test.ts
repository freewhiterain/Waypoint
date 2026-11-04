import { describe, expect, it } from "vitest";
import { estimateCost } from "./cost";
import { day, hotels, plan, slot } from "@/test/fixtures";

// 规则与 app/agents/trip_cost.py 一致，用例对应 tests/test_budget.py。
describe("estimateCost", () => {
  it("counts tickets and meals per person and the hotel per night", () => {
    const p = plan(
      [
        day(1, [slot("morning", "武侯祠", 50), slot("evening", "火锅", 100)], ["住宿：桔子酒店。近地铁"]),
        day(2, [slot("morning", "锦里"), slot("evening", "自由安排晚餐")]),
      ],
      { adults: 2, worker_results: [hotels(["桔子酒店", 300])] },
    );
    expect(estimateCost(p)).toEqual({ tickets: 100, meals: 200, hotel: 300, total: 600, people: 2, nights: 1, unknown: 1 });
  });

  it("reads the hotel price from the first day's hotel note, name before the first 。", () => {
    const p = plan([day(1, [], ["天气：晴", "住宿：桔子酒店。近地铁"]), day(2, []), day(3, [])], {
      worker_results: [hotels(["别家", 100], ["桔子酒店", 388])],
    });
    const cost = estimateCost(p);
    expect(cost.nights).toBe(2);
    expect(cost.hotel).toBe(776);
  });

  it("children count as people; party size is at least one", () => {
    const p = plan([day(1, [slot("afternoon", "熊猫基地", 55), slot("lunch", "串串", 40)])], { adults: 2, children: 1 });
    const cost = estimateCost(p);
    expect(cost.people).toBe(3);
    expect(cost.tickets).toBe(165);
    expect(cost.meals).toBe(120);
    expect(estimateCost(plan([day(1, [])], { adults: 0 })).people).toBe(1);
  });

  it("lunch and evening are meals; morning, afternoon and night are tickets", () => {
    const p = plan([
      day(1, [
        slot("morning", "A", 1),
        slot("lunch", "B", 10),
        slot("afternoon", "C", 100),
        slot("evening", "D", 1000),
        slot("night", "E", 10000),
      ]),
    ]);
    const cost = estimateCost(p);
    expect(cost.tickets).toBe(10101);
    expect(cost.meals).toBe(1010);
  });

  it("free-time slots are skipped, not counted as unknown; free entries (0) count as known", () => {
    const p = plan([
      day(1, [slot("morning", "春熙路分区自由活动"), slot("afternoon", "自由安排"), slot("morning", "人民公园", 0)]),
    ]);
    const cost = estimateCost(p);
    expect(cost.unknown).toBe(0);
    expect(cost.total).toBe(0);
  });

  it("no hotel note, unknown hotel or a one-day trip costs nothing for lodging", () => {
    const results = [hotels(["桔子酒店", 300])];
    expect(estimateCost(plan([day(1, []), day(2, [])], { worker_results: results })).hotel).toBe(0);
    expect(estimateCost(plan([day(1, [], ["住宿：别家。"]), day(2, [])], { worker_results: results })).hotel).toBe(0);
    expect(estimateCost(plan([day(1, [], ["住宿：桔子酒店"])], { worker_results: results })).hotel).toBe(0);
    expect(estimateCost(plan([day(1, [], ["住宿：无价酒店"]), day(2, [])], { worker_results: [hotels(["无价酒店", null])] })).hotel).toBe(0);
  });

  it("an empty itinerary is all zeros", () => {
    expect(estimateCost(plan([]))).toMatchObject({ total: 0, nights: 0, unknown: 0 });
  });
});
