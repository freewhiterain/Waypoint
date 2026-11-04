import type { ItineraryDay, TimeSlot, TravelPlanDraft, WorkerResult } from "@/api/types";

export function slot(period: TimeSlot["period"], title: string, estimated_cost: number | null = null): TimeSlot {
  return { period, title, description: "", estimated_cost };
}

export function day(n: number, slots: TimeSlot[], notes: string[] = []): ItineraryDay {
  return { day: n, date: `2026-12-0${n}`, slots, notes };
}

export function hotels(...options: [string, number | null][]): WorkerResult {
  return {
    worker: "hotel",
    status: "completed",
    summary: "",
    options: options.map(([name, cost], i) => ({
      id: `h${i}`,
      name,
      category: "hotel",
      description: "",
      estimated_cost: cost,
      attributes: {},
    })),
    evidence: [],
    warnings: [],
  };
}

export function plan(
  itinerary: ItineraryDay[],
  opts: { adults?: number; children?: number; worker_results?: WorkerResult[]; warnings?: string[] } = {},
): TravelPlanDraft {
  return {
    requirement: {
      destination: "成都",
      departure_date: "2026-12-01",
      days: itinerary.length,
      adults: opts.adults ?? 1,
      children: opts.children ?? 0,
      styles: [],
      food_preferences: [],
    },
    itinerary,
    worker_results: opts.worker_results ?? [],
    evidence: [],
    warnings: opts.warnings ?? [],
    status: "completed",
  };
}
