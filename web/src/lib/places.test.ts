import { describe, expect, it } from "vitest";
import { dayRoutes, hasPlaces, hotelName } from "./places";
import { day, plan, slot } from "@/test/fixtures";
import type { WorkerResult } from "@/api/types";

function located(worker: WorkerResult["worker"], name: string, lat: number, lng: number): WorkerResult {
  return {
    worker,
    status: "completed",
    summary: "",
    options: [{ id: name, name, category: worker, description: "", attributes: { latitude: lat, longitude: lng } }],
    evidence: [],
    warnings: [],
  };
}

describe("hasPlaces", () => {
  it("is false when every slot is free time", () => {
    expect(hasPlaces(plan([day(1, [slot("morning", "自由活动"), slot("afternoon", "春熙路分区自由活动")])]))).toBe(false);
    expect(hasPlaces(plan([]))).toBe(false);
  });
  it("is true with at least one concrete place", () => {
    expect(hasPlaces(plan([day(1, [slot("morning", "自由活动"), slot("afternoon", "武侯祠")])]))).toBe(true);
  });
});

describe("hotelName / dayRoutes", () => {
  it("reads the hotel name and starts each day's route there", () => {
    const p = plan(
      [day(1, [slot("morning", "武侯祠"), slot("afternoon", "没坐标")], ["住宿：桔子酒店。近地铁"]), day(2, [slot("morning", "锦里")])],
      { worker_results: [located("hotel", "桔子酒店", 30, 104), located("attractions", "武侯祠", 30.6, 104.0), located("attractions", "锦里", 30.6, 104.1)] },
    );
    expect(hotelName(p.itinerary[0])).toBe("桔子酒店");
    const routes = dayRoutes(p);
    expect(routes.map((r) => r.points.map((pt) => pt.name))).toEqual([
      ["桔子酒店", "武侯祠"],
      ["桔子酒店", "锦里"],
    ]);
    expect(hotelName(undefined)).toBeUndefined();
  });
});
