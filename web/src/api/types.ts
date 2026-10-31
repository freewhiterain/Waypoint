// 与后端 Pydantic 模型对应的最小类型。只声明前端用到的字段。

export interface User {
  id: string;
  username: string;
  email: string;
}

export interface Conversation {
  id: string;
  title: string;
  status: string;
  created_at: string;
  updated_at: string;
}

export interface HistoryMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  created_at: string;
  extra_info: Record<string, unknown> | null;
}

export type Period = "morning" | "lunch" | "afternoon" | "evening" | "night";

export interface TimeSlot {
  period: Period;
  title: string;
  description: string;
  location?: string | null;
  travel_minutes?: number | null;
  travel_mode?: "walk" | "drive" | null;
  opening_window?: string | null;
  estimated_cost?: number | null;
}

export interface ItineraryDay {
  day: number;
  date: string;
  slots: TimeSlot[];
  notes: string[];
}

export interface CandidateOption {
  id: string;
  name: string;
  category: string;
  description: string;
  estimated_cost?: number | null;
  attributes: Record<string, unknown>;
}

export interface Evidence {
  source: string;
  content: string;
  url?: string | null;
  is_mock?: boolean;
}

export type WorkerName = "attractions" | "weather" | "transport" | "hotel" | "food";

export interface WorkerResult {
  worker: WorkerName;
  status: "completed" | "partial" | "unavailable" | "failed" | string;
  summary: string;
  options: CandidateOption[];
  evidence: Evidence[];
  warnings: string[];
  is_mock?: boolean;
}

export interface TravelRequirement {
  destination: string;
  departure_date: string;
  days: number;
  origin?: string | null;
  adults: number;
  children: number;
  styles: string[];
  food_preferences: string[];
  must_visit?: string[];
  /** 整趟所有人的总预算（元），不含往返交通 */
  budget?: number | null;
}

export interface TravelPlanDraft {
  requirement: TravelRequirement;
  itinerary: ItineraryDay[];
  worker_results: WorkerResult[];
  evidence: Evidence[];
  warnings: string[];
  status: string;
}

export interface TripValues {
  destination: string;
  departure_date: string;
  days: string;
}

/** SSE 帧：后端 legacy_payload，常用字段被提升到顶层。 */
export interface StreamEvent {
  type: string;
  task_id?: string;
  sequence?: number;
  payload?: Record<string, any>;
  content?: string;
  message?: string;
  tool?: string;
  call_id?: string;
  arguments?: Record<string, any>;
  status?: string;
  result?: any;
  partial_values?: Record<string, any>;
}
