import type { TravelPlanDraft } from "@/api/types";
import { hotelName } from "./places";

export interface CostEstimate {
  tickets: number;
  meals: number;
  hotel: number;
  total: number;
  people: number;
  nights: number;
  /** 有多少个时段没有价格数据，没算进去。 */
  unknown: number;
}

/**
 * 按行程里已有的价格直接算，不经过模型：门票和餐饮按人头，住宿按晚（一间房）。
 * 价格来自地点库快照，只作参考。
 */
export function estimateCost(plan: TravelPlanDraft): CostEstimate {
  const people = Math.max(1, (plan.requirement.adults || 1) + (plan.requirement.children || 0));
  let tickets = 0;
  let meals = 0;
  let unknown = 0;
  for (const day of plan.itinerary) {
    for (const slot of day.slots) {
      if (slot.title.endsWith("自由活动") || slot.title.startsWith("自由安排")) continue;
      if (slot.estimated_cost == null) {
        unknown += 1;
        continue;
      }
      if (slot.period === "lunch" || slot.period === "evening") meals += slot.estimated_cost * people;
      else tickets += slot.estimated_cost * people;
    }
  }
  const nights = Math.max(0, plan.itinerary.length - 1);
  const name = hotelName(plan.itinerary[0]);
  const hotelOption = plan.worker_results
    .find((result) => result.worker === "hotel")
    ?.options.find((option) => option.name === name);
  const hotel = (hotelOption?.estimated_cost || 0) * nights;
  return { tickets, meals, hotel, total: tickets + meals + hotel, people, nights, unknown };
}
