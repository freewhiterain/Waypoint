import { create } from "zustand";
import {
  api,
  saveSession,
  setAuthToken,
  setUnauthorizedHandler,
  streamPost,
  loadSession,
} from "@/api/client";
import type {
  Conversation,
  HistoryMessage,
  StreamEvent,
  TravelPlanDraft,
  TripValues,
  User,
  WorkerName,
} from "@/api/types";
import { todayIso } from "@/lib/format";

export interface Activity {
  id: string;
  tool: string;
  arguments: Record<string, any>;
  status: string;
  evidence?: number;
}

export type TripStatus = "pending" | "submitting" | "recommending" | "completed";

export interface TripCall {
  callId: string;
  values: TripValues;
  /** 对话里记下的、会带进这次规划的修改意见（不去的地点、轻松的天）。 */
  revisions: string[];
  /** 第一次出现时缺的槽位：卡片只问这些；为空表示三项都齐，直接显示确认。 */
  askFields: (keyof TripValues)[];
  status: TripStatus;
  recommendation: string;
  /** 推荐回答里提到的城市，卡片上给出"就去这里"的快捷选项。 */
  suggestions: string[];
  error: string;
}

export interface ResearchStep {
  worker: WorkerName;
  status: string;
  evidence?: number;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  streaming?: boolean;
  error?: boolean;
  activities: Activity[];
  trip?: TripCall;
  research?: Partial<Record<WorkerName, ResearchStep>>;
  plan?: TravelPlanDraft;
}

interface State {
  token: string;
  user: User | null;
  conversations: Conversation[];
  currentId: string | null;
  messages: ChatMessage[];
  loadingHistory: boolean;
  busy: boolean;
  activePlan: TravelPlanDraft | null;
  toast: { text: string; error: boolean } | null;

  login(username: string, password: string): Promise<void>;
  register(username: string, email: string, password: string): Promise<void>;
  logout(): void;
  loadConversations(): Promise<void>;
  selectConversation(id: string): Promise<void>;
  startNew(): void;
  renameConversation(id: string, title: string): Promise<void>;
  deleteConversation(id: string): Promise<void>;
  send(text: string): Promise<void>;
  /** 停止当前这轮问答的流式输出（规划不能中途停）。 */
  stop(): void;
  canStop: boolean;
  submitTrip(messageId: string, values: TripValues): Promise<void>;
  recommendDestination(messageId: string, values: TripValues): Promise<void>;
  openPlan(plan: TravelPlanDraft | null): void;
  notify(text: string, error?: boolean): void;
}

// 当前问答流的中止器；同一时间只有一轮问答在跑（busy 保证）。
let currentController: AbortController | null = null;

const uid = () => Math.random().toString(36).slice(2) + Date.now().toString(36);

export const TRIP_FIELDS: (keyof TripValues)[] = ["destination", "departure_date", "days"];

export function tripFieldError(field: keyof TripValues, values: TripValues): string {
  if (field === "destination" && !values.destination.trim()) return "请选择或输入目的地";
  if (field === "departure_date") {
    if (!values.departure_date) return "请选择出发日期";
    if (values.departure_date < todayIso()) return "出发日期不能早于今天";
  }
  if (field === "days") {
    const days = Number(values.days);
    if (!Number.isInteger(days) || days < 1 || days > 30) return "天数请填 1-30 的整数";
  }
  return "";
}

function tripFromCall(callId: string, args: Record<string, any> | undefined): TripCall {
  const initial = args?.initial_values || {};
  const values: TripValues = {
    destination: initial.destination || "",
    departure_date: initial.departure_date || "",
    days: initial.days == null ? "" : String(initial.days),
  };
  const context = args?.requirement_context || {};
  const revisions = [
    ...((context.avoid_places as string[] | undefined) || []).map((place) => `不去 ${place}`),
    ...((context.relaxed_days as number[] | undefined) || []).map((day) => `第 ${day} 天轻松些`),
  ];
  return {
    callId,
    values,
    revisions,
    askFields: TRIP_FIELDS.filter((field) => tripFieldError(field, values)),
    status: "pending",
    recommendation: "",
    suggestions: [],
    error: "",
  };
}

const REACT_TOOLS = new Set(["search_travel_knowledge", "research_topic", "estimate_travel_time", "estimate_trip_cost", "recommend_cities"]);

/** 把一条历史消息还原成界面消息；返回 null 表示这条不单独显示。 */
function fromHistory(message: HistoryMessage): ChatMessage | null {
  if (message.role === "system") return null;
  const extra = (message.extra_info || {}) as Record<string, any>;
  // 建会话时后端写入的主动邀请：新界面用空状态页代替，不在消息流里显示。
  if (extra.kind === "conversation_offer") return null;
  const chat: ChatMessage = {
    id: message.id,
    role: message.role,
    content: message.content,
    activities: [],
  };
  const toolCall = extra.tool_call;
  if (toolCall?.tool === "collect_trip_requirements") {
    chat.trip = tripFromCall(toolCall.call_id, toolCall.arguments);
    chat.content = "";
  } else if (toolCall?.tool) {
    chat.activities.push({
      id: toolCall.call_id || uid(),
      tool: toolCall.tool,
      arguments: toolCall.arguments || {},
      status: extra.tool_result?.status || "completed",
      evidence: extra.tool_result?.evidence_count,
    });
  }
  for (const step of extra.react_steps || []) {
    chat.activities.push({ id: uid(), tool: step.tool, arguments: step.arguments || {}, status: step.status });
  }
  if (extra.assistant_result?.itinerary) chat.plan = extra.assistant_result as TravelPlanDraft;
  return chat;
}

function rebuild(history: HistoryMessage[]): ChatMessage[] {
  const messages: ChatMessage[] = [];
  const trips = new Map<string, TripCall>();
  for (const item of history) {
    const chat = fromHistory(item);
    const extra = (item.extra_info || {}) as Record<string, any>;
    const result = extra.tool_result;
    if (result?.tool === "collect_trip_requirements") {
      const trip = trips.get(result.task_id || result.call_id) || [...trips.values()].pop();
      if (trip && result.status === "completed") {
        trip.status = "completed";
        if (result.result) trip.values = { ...trip.values, ...stringify(result.result) };
      } else if (trip && result.status === "awaiting_destination") {
        trip.values = { ...trip.values, ...stringify(result.partial_values || {}) };
        trip.recommendation = item.content;
        trip.suggestions = result.suggested_destinations || [];
        // 推荐内容已经显示在表单卡片里，不再作为单独一条消息重复显示。
        continue;
      }
    }
    if (chat?.trip) trips.set(chat.trip.callId, chat.trip);
    if (chat) messages.push(chat);
  }
  return messages;
}

function stringify(values: Record<string, any>): Partial<TripValues> {
  const out: Partial<TripValues> = {};
  for (const field of TRIP_FIELDS) if (values[field] != null) out[field] = String(values[field]);
  return out;
}

export const useChat = create<State>((set, get) => {
  const session = loadSession();

  const patchMessage = (id: string, patch: (message: ChatMessage) => ChatMessage) =>
    set((state) => ({ messages: state.messages.map((m) => (m.id === id ? patch(m) : m)) }));

  const patchTrip = (id: string, patch: Partial<TripCall>) =>
    patchMessage(id, (m) => (m.trip ? { ...m, trip: { ...m.trip, ...patch } } : m));

  const applyActivity = (id: string, event: StreamEvent) => {
    const tool = event.tool || "";
    if (event.type === "tool_call") {
      patchMessage(id, (m) => ({
        ...m,
        activities: [
          ...m.activities,
          { id: event.call_id || uid(), tool, arguments: event.arguments || {}, status: "running" },
        ],
      }));
    } else {
      patchMessage(id, (m) => ({
        ...m,
        activities: m.activities.map((a) =>
          a.id === event.call_id || (!event.call_id && a.tool === tool && a.status === "running")
            ? { ...a, status: event.status || "completed", evidence: event.result?.evidence_count }
            : a,
        ),
      }));
    }
  };

  const applyResearch = (id: string, event: StreamEvent) => {
    const payload = event.payload || {};
    const worker = payload.worker as WorkerName | undefined;
    if (!worker) return;
    patchMessage(id, (m) => {
      const research = { ...(m.research || {}) };
      const previous = research[worker] || { worker, status: "running" };
      if (event.type === "subagent_started") research[worker] = { ...previous, status: "running" };
      else if (event.type === "subagent_completed")
        research[worker] = {
          ...previous,
          status: payload.status || "completed",
          evidence: payload.evidence_count ?? previous.evidence,
        };
      else if (event.type === "evidence_collected")
        research[worker] = { ...previous, evidence: payload.evidence_count ?? previous.evidence };
      return { ...m, research };
    });
  };

  return {
    token: session.token,
    user: session.user,
    conversations: [],
    currentId: null,
    messages: [],
    loadingHistory: false,
    busy: false,
    canStop: false,
    activePlan: null,
    toast: null,

    async login(username, password) {
      const data = await api.login(username, password);
      setAuthToken(data.access_token);
      saveSession(data.access_token, data.user);
      set({ token: data.access_token, user: data.user });
      await get().loadConversations();
    },

    async register(username, email, password) {
      const data = await api.register(username, email, password);
      setAuthToken(data.access_token);
      saveSession(data.access_token, data.user);
      set({ token: data.access_token, user: data.user });
      await get().loadConversations();
    },

    logout() {
      setAuthToken("");
      saveSession("", null);
      set({ token: "", user: null, conversations: [], currentId: null, messages: [], activePlan: null });
    },

    async loadConversations() {
      const conversations = await api.listConversations();
      set({ conversations });
    },

    async selectConversation(id) {
      if (get().busy) return;
      set({ currentId: id, loadingHistory: true, messages: [], activePlan: null });
      try {
        const data = await api.history(id);
        if (get().currentId !== id) return;
        const messages = rebuild(data.messages);
        const lastPlan = [...messages].reverse().find((m) => m.plan)?.plan || null;
        set({ messages, activePlan: lastPlan });
      } catch (error) {
        get().notify((error as Error).message, true);
      } finally {
        set({ loadingHistory: false });
      }
    },

    startNew() {
      if (get().busy) return;
      set({ currentId: null, messages: [], activePlan: null });
    },

    async renameConversation(id, title) {
      const updated = await api.renameConversation(id, title);
      set((state) => ({
        conversations: state.conversations.map((c) => (c.id === id ? { ...c, title: updated.title } : c)),
      }));
    },

    async deleteConversation(id) {
      await api.deleteConversation(id);
      set((state) => ({
        conversations: state.conversations.filter((c) => c.id !== id),
        ...(state.currentId === id ? { currentId: null, messages: [], activePlan: null } : {}),
      }));
    },

    async send(text) {
      const content = text.trim();
      if (!content || get().busy) return;
      set({ busy: true });
      let conversationId = get().currentId;
      try {
        if (!conversationId) {
          // 标题由后端在收到第一条消息时生成。
          const created = await api.createConversation();
          conversationId = created.id;
          set((state) => ({ currentId: created.id, conversations: [created, ...state.conversations] }));
        }
      } catch (error) {
        set({ busy: false });
        get().notify((error as Error).message, true);
        return;
      }

      const assistantId = uid();
      set((state) => ({
        messages: [
          ...state.messages,
          { id: uid(), role: "user", content, activities: [] },
          { id: assistantId, role: "assistant", content: "", streaming: true, activities: [] },
        ],
      }));

      const controller = new AbortController();
      currentController = controller;
      set({ canStop: true });
      try {
        await streamPost(`/api/v1/chat/stream/${conversationId}`, { content }, (event) => {
          if (event.type === "tool_call" && event.tool === "collect_trip_requirements") {
            patchMessage(assistantId, (m) => ({ ...m, trip: tripFromCall(event.call_id!, event.arguments) }));
          } else if ((event.type === "tool_call" || event.type === "tool_result") && event.tool) {
            if (REACT_TOOLS.has(event.tool) || event.tool.startsWith("research_")) applyActivity(assistantId, event);
          } else if (event.type === "token") {
            patchMessage(assistantId, (m) => ({ ...m, content: m.content + (event.content || "") }));
          } else if (event.type === "token_reset") {
            // 模型先说了半句又转去调工具，或最终回答和流式内容不一致：撤回已显示的文字。
            patchMessage(assistantId, (m) => ({ ...m, content: "" }));
          } else if (event.type === "error") {
            patchMessage(assistantId, (m) => ({ ...m, error: true, content: event.message || "出错了，请稍后再试" }));
          }
        }, controller.signal);
      } catch (error) {
        if (controller.signal.aborted) {
          patchMessage(assistantId, (m) => ({ ...m, content: m.content ? `${m.content}\n\n_（已停止）_` : "_已停止_" }));
        } else {
          patchMessage(assistantId, (m) => ({ ...m, error: true, content: (error as Error).message }));
        }
      } finally {
        currentController = null;
        patchMessage(assistantId, (m) => ({ ...m, streaming: false }));
        set({ busy: false, canStop: false });
        // 会话标题由第一句话生成，列表顺序按更新时间刷新。
        get().loadConversations().catch(() => {});
      }
    },

    async submitTrip(messageId, values) {
      const message = get().messages.find((m) => m.id === messageId);
      const trip = message?.trip;
      if (!trip || get().busy) return;
      set({ busy: true });
      patchTrip(messageId, { status: "submitting", error: "", values });

      const planId = uid();
      set((state) => ({
        messages: [...state.messages, { id: planId, role: "assistant", content: "", streaming: true, activities: [], research: {} }],
      }));
      try {
        await streamPost(
          `/api/v1/chat/tools/${trip.callId}/result`,
          {
            tool: "collect_trip_requirements",
            status: "completed",
            result: {
              destination: values.destination.trim(),
              departure_date: values.departure_date,
              days: Number(values.days),
            },
          },
          (event) => {
            if (event.type.startsWith("subagent_") || event.type === "evidence_collected") {
              applyResearch(planId, event);
            } else if (event.type === "result") {
              const plan = (event.payload?.result || event.result) as TravelPlanDraft | undefined;
              if (plan?.itinerary) {
                patchMessage(planId, (m) => ({ ...m, plan }));
                set({ activePlan: plan });
              }
              patchTrip(messageId, { status: "completed" });
            } else if (event.type === "token") {
              patchMessage(planId, (m) => ({ ...m, content: m.content + (event.content || "") }));
            } else if (event.type === "error") {
              throw new Error(event.message || "规划失败，请稍后再试");
            }
          },
        );
      } catch (error) {
        patchTrip(messageId, { status: "pending", error: (error as Error).message });
        set((state) => ({ messages: state.messages.filter((m) => m.id !== planId) }));
      } finally {
        patchMessage(planId, (m) => ({ ...m, streaming: false }));
        set({ busy: false });
      }
    },

    async recommendDestination(messageId, values) {
      const trip = get().messages.find((m) => m.id === messageId)?.trip;
      if (!trip || get().busy) return;
      set({ busy: true });
      patchTrip(messageId, { status: "recommending", recommendation: "", suggestions: [], error: "" });
      const partial: Record<string, string> = {};
      for (const field of TRIP_FIELDS) if (values[field]?.trim() && field !== "destination") partial[field] = values[field];
      try {
        await streamPost(
          `/api/v1/chat/tools/${trip.callId}/result`,
          { tool: "collect_trip_requirements", status: "recommend_destination", partial_values: partial },
          (event) => {
            if (event.type === "token") {
              patchMessage(messageId, (m) =>
                m.trip ? { ...m, trip: { ...m.trip, recommendation: m.trip.recommendation + (event.content || "") } } : m,
              );
            } else if (event.type === "tool_result") {
              patchTrip(messageId, { suggestions: event.payload?.suggested_destinations || [] });
            } else if (event.type === "error") {
              throw new Error(event.message || "推荐失败");
            }
          },
        );
        patchTrip(messageId, { status: "pending" });
      } catch (error) {
        patchTrip(messageId, { status: "pending", error: (error as Error).message });
      } finally {
        set({ busy: false });
      }
    },

    stop() {
      currentController?.abort();
    },

    openPlan(plan) {
      set({ activePlan: plan });
    },

    notify(text, error = false) {
      set({ toast: { text, error } });
      window.setTimeout(() => {
        if (get().toast?.text === text) set({ toast: null });
      }, 3200);
    },
  };
});

setUnauthorizedHandler(() => {
  const state = useChat.getState();
  if (state.token) {
    state.logout();
    state.notify("登录已过期，请重新登录", true);
  }
});
