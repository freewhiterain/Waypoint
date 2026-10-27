import type { ItineraryDay, TravelPlanDraft } from "@/api/types";
import { splitNotes } from "./format";

export interface Point {
  name: string;
  lat: number;
  lng: number;
  category: string;
}

export const DAY_COLORS = ["#fb6110", "#2563eb", "#14a37c", "#9333ea", "#e11d48", "#0891b2", "#ca8a04"];

/** 行程里的地点名 → 候选上的坐标（地点库城市才有）。 */
export function pointIndex(plan: TravelPlanDraft): Map<string, Point> {
  const index = new Map<string, Point>();
  for (const result of plan.worker_results || []) {
    for (const option of result.options || []) {
      const lat = option.attributes?.latitude;
      const lng = option.attributes?.longitude;
      if (typeof lat === "number" && typeof lng === "number" && !index.has(option.name)) {
        index.set(option.name, { name: option.name, lat, lng, category: result.worker });
      }
    }
  }
  return index;
}

export function hotelName(day: ItineraryDay | undefined): string | undefined {
  const hotel = splitNotes(day?.notes || []).hotel;
  return hotel?.split("。")[0];
}

/** 每天的路线：住处 → 上午 → 下午 → 晚上，只保留有坐标的点。 */
export function dayRoutes(plan: TravelPlanDraft): { day: number; points: Point[]; hotel?: Point }[] {
  const index = pointIndex(plan);
  const hotel = index.get(hotelName(plan.itinerary[0]) || "");
  return plan.itinerary.map((day) => {
    const stops = day.slots.map((slot) => index.get(slot.title)).filter((p): p is Point => !!p);
    return { day: day.day, points: hotel ? [hotel, ...stops] : stops, hotel };
  });
}
