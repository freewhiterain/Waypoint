import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { StreamEvent } from "@/api/types";
import { localIso } from "@/lib/format";
import { day, plan, slot } from "@/test/fixtures";
import { tripFieldError, useChat } from "./chat";

/** 把事件编成 SSE 文本，并故意从帧中间切开，模拟网络分片。 */
function sse(events: StreamEvent[]): Response {
  const text = events.map((e) => `data: ${JSON.stringify(e)}\r\n\r\n`).join("");
  const bytes = new TextEncoder().encode(text);
  const cut = Math.floor(bytes.length / 2);
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      controller.enqueue(bytes.slice(0, cut));
      controller.enqueue(bytes.slice(cut));
      controller.close();
    },
  });
  return new Response(body, { status: 200, headers: { "Content-Type": "text/event-stream" } });
}

let streams: Record<string, StreamEvent[]>;
let posted: { url: string; body: any }[];

beforeEach(() => {
  streams = {};
  posted = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init?: RequestInit) => {
      if (init?.method === "POST") posted.push({ url, body: JSON.parse(String(init.body)) });
      const key = Object.keys(streams).find((prefix) => url.startsWith(prefix));
      if (key) return sse(streams[key]);
      return new Response("[]", { status: 200, headers: { "Content-Type": "application/json" } });
    }),
  );
  useChat.setState({ currentId: "c1", messages: [], busy: false, activePlan: null, toast: null });
});

afterEach(() => vi.unstubAllGlobals());

const lastMessage = () => useChat.getState().messages.at(-1)!;
const futureDate = () => {
  const d = new Date();
  d.setDate(d.getDate() + 10);
  return localIso(d);
};

const concretePlan = plan([day(1, [slot("morning", "武侯祠", 50)])]);
const freeOnlyPlan = plan([day(1, [slot("morning", "自由活动")])]);

async function sendWithTrip(args: Record<string, any>, extra: StreamEvent[] = []) {
  streams["/api/v1/chat/stream/"] = [
    { type: "token", content: "这次先规划成都。" },
    { type: "tool_call", tool: "collect_trip_requirements", call_id: "call-1", arguments: args },
    ...extra,
  ];
  await useChat.getState().send("想去成都");
  return lastMessage();
}

describe("send: chat stream", () => {
  it("appends user + assistant messages and accumulates tokens", async () => {
    streams["/api/v1/chat/stream/"] = [
      { type: "token", content: "你好" },
      { type: "token", content: "，成都" },
    ];
    await useChat.getState().send("  在吗  ");
    const { messages, busy } = useChat.getState();
    expect(messages.map((m) => [m.role, m.content])).toEqual([
      ["user", "在吗"],
      ["assistant", "你好，成都"],
    ]);
    expect(messages[1].streaming).toBe(false);
    expect(busy).toBe(false);
  });

  it("token_reset clears what was shown so far", async () => {
    streams["/api/v1/chat/stream/"] = [
      { type: "token", content: "我先查" },
      { type: "token_reset" },
      { type: "token", content: "查到了" },
    ];
    await useChat.getState().send("hi");
    expect(lastMessage().content).toBe("查到了");
  });

  it("an error event marks the message as an error", async () => {
    streams["/api/v1/chat/stream/"] = [{ type: "token", content: "半句" }, { type: "error", message: "模型超时" }];
    await useChat.getState().send("hi");
    expect(lastMessage()).toMatchObject({ error: true, content: "模型超时" });
  });

  it("an HTTP failure becomes an error message", async () => {
    vi.mocked(fetch).mockImplementationOnce(async () => new Response(JSON.stringify({ detail: "坏请求" }), { status: 400 }));
    await useChat.getState().send("hi");
    expect(lastMessage()).toMatchObject({ error: true, content: "坏请求" });
    expect(useChat.getState().busy).toBe(false);
  });

  it("tracks ReAct tool activities by call id and ignores unrelated tools", async () => {
    streams["/api/v1/chat/stream/"] = [
      { type: "tool_call", tool: "search_travel_knowledge", call_id: "a", arguments: { query: "宽窄巷子" } },
      { type: "tool_call", tool: "research_food", call_id: "b", arguments: {} },
      { type: "tool_call", tool: "something_internal", call_id: "x", arguments: {} },
      { type: "tool_result", tool: "search_travel_knowledge", call_id: "a", status: "completed", result: { evidence_count: 3 } },
      { type: "tool_result", tool: "research_food", status: "empty" },
    ];
    await useChat.getState().send("hi");
    expect(lastMessage().activities).toEqual([
      { id: "a", tool: "search_travel_knowledge", arguments: { query: "宽窄巷子" }, status: "completed", evidence: 3 },
      { id: "b", tool: "research_food", arguments: {}, status: "empty", evidence: undefined },
    ]);
  });

  it("does nothing for blank input or while busy", async () => {
    await useChat.getState().send("   ");
    useChat.setState({ busy: true });
    await useChat.getState().send("hi");
    expect(useChat.getState().messages).toEqual([]);
  });
});

describe("trip form", () => {
  it("builds the form from collect_trip_requirements, keeping the text above it", async () => {
    const message = await sendWithTrip({
      initial_values: { destination: "成都", days: 3 },
      requirement_context: { must_visit: ["武侯祠"], budget: 3000, avoid_places: ["锦里"], relaxed_days: [2] },
    });
    expect(message.content).toBe("这次先规划成都。");
    expect(message.trip).toMatchObject({
      callId: "call-1",
      values: { destination: "成都", departure_date: "", days: "3" },
      revisions: ["必去 武侯祠", "预算 ¥3,000", "不去 锦里", "第 2 天轻松些"],
      askFields: ["departure_date"],
      status: "pending",
    });
  });

  it("asks nothing when all three fields are already valid", async () => {
    const message = await sendWithTrip({ initial_values: { destination: "成都", departure_date: futureDate(), days: 2 } });
    expect(message.trip!.askFields).toEqual([]);
  });

  it("tripFieldError validates each field", () => {
    const values = { destination: " ", departure_date: "2000-01-01", days: "0" };
    expect(tripFieldError("destination", values)).toBeTruthy();
    expect(tripFieldError("departure_date", values)).toBe("出发日期不能早于今天");
    expect(tripFieldError("departure_date", { ...values, departure_date: "" })).toBe("请选择出发日期");
    expect(tripFieldError("days", values)).toBeTruthy();
    expect(tripFieldError("days", { ...values, days: "2.5" })).toBeTruthy();
    expect(tripFieldError("days", { ...values, days: "30" })).toBe("");
  });
});

describe("submitTrip: planning stream", () => {
  async function submit(events: StreamEvent[], notes = "") {
    const trip = await sendWithTrip({ initial_values: {} });
    streams["/api/v1/chat/tools/"] = events;
    await useChat.getState().submitTrip(trip.id, { destination: " 成都 ", departure_date: futureDate(), days: "2" }, notes);
    return trip.id;
  }

  it("posts the form values with trimmed notes", async () => {
    await submit([], "  不吃辣 ");
    const body = posted.find((p) => p.url.includes("/tools/call-1/result"))!.body;
    expect(body.result).toMatchObject({ destination: "成都", days: 2, notes: "不吃辣" });
  });

  it("tracks research progress per worker", async () => {
    await submit([
      { type: "subagent_started", payload: { worker: "food" } },
      { type: "subagent_started", payload: { worker: "hotel" } },
      { type: "evidence_collected", payload: { worker: "food", evidence_count: 4 } },
      { type: "subagent_completed", payload: { worker: "food", status: "completed" } },
      { type: "subagent_completed", payload: { worker: "hotel", status: "unavailable", evidence_count: 0 } },
      { type: "subagent_started", payload: {} },
    ]);
    expect(lastMessage().research).toEqual({
      food: { worker: "food", status: "completed", evidence: 4 },
      hotel: { worker: "hotel", status: "unavailable", evidence: 0 },
    });
  });

  it("a result with places attaches the plan, opens it and completes the form", async () => {
    const tripId = await submit([{ type: "token", content: "行程如下" }, { type: "result", payload: { result: concretePlan } }]);
    const state = useChat.getState();
    expect(lastMessage()).toMatchObject({ plan: concretePlan, content: "行程如下", streaming: false });
    expect(state.activePlan).toEqual(concretePlan);
    expect(state.messages.find((m) => m.id === tripId)!.trip!.status).toBe("completed");
    expect(state.busy).toBe(false);
  });

  it("a free-time-only result shows text, not a plan", async () => {
    await submit([{ type: "token", content: "这几天适合随意逛" }, { type: "result", result: freeOnlyPlan }]);
    expect(lastMessage().plan).toBeUndefined();
    expect(useChat.getState().activePlan).toBeNull();
  });

  it("an error event puts the form back with the error and drops the plan message", async () => {
    const tripId = await submit([{ type: "subagent_started", payload: { worker: "food" } }, { type: "error", message: "规划超时" }]);
    const state = useChat.getState();
    expect(state.messages.at(-1)!.id).toBe(tripId);
    expect(state.messages.find((m) => m.id === tripId)!.trip).toMatchObject({ status: "pending", error: "规划超时" });
    expect(state.busy).toBe(false);
  });
});

describe("recommendDestination", () => {
  it("streams the recommendation into the form and sets suggestions", async () => {
    const trip = await sendWithTrip({ initial_values: { days: 3 } });
    streams["/api/v1/chat/tools/"] = [
      { type: "token", content: "可以去" },
      { type: "token", content: "重庆" },
      { type: "tool_result", payload: { suggested_destinations: ["重庆", "西安"] } },
    ];
    await useChat.getState().recommendDestination(trip.id, { destination: "", departure_date: "", days: "3" });
    const body = posted.find((p) => p.body.status === "recommend_destination")!.body;
    expect(body.partial_values).toEqual({ days: "3" });
    expect(lastMessage().trip).toMatchObject({ recommendation: "可以去重庆", suggestions: ["重庆", "西安"], status: "pending", error: "" });
  });

  it("an error event is shown on the form", async () => {
    const trip = await sendWithTrip({ initial_values: {} });
    streams["/api/v1/chat/tools/"] = [{ type: "error", message: "推荐服务不可用" }];
    await useChat.getState().recommendDestination(trip.id, { destination: "", departure_date: "", days: "" });
    expect(lastMessage().trip).toMatchObject({ status: "pending", error: "推荐服务不可用" });
  });
});

describe("selectConversation: rebuilding history", () => {
  it("restores the trip form, its completion, activities and the last plan", async () => {
    const history = [
      { id: "m0", role: "assistant", content: "欢迎", created_at: "", extra_info: { kind: "conversation_offer" } },
      { id: "m1", role: "user", content: "想去成都", created_at: "", extra_info: null },
      {
        id: "m2",
        role: "assistant",
        content: "",
        created_at: "",
        extra_info: { tool_call: { tool: "collect_trip_requirements", call_id: "call-1", arguments: { initial_values: { destination: "成都" } } } },
      },
      {
        id: "m3",
        role: "assistant",
        content: "推荐重庆",
        created_at: "",
        extra_info: { tool_result: { tool: "collect_trip_requirements", call_id: "call-1", status: "awaiting_destination", partial_values: { days: 3 }, suggested_destinations: ["重庆"] } },
      },
      {
        id: "m4",
        role: "assistant",
        content: "行程如下",
        created_at: "",
        extra_info: {
          tool_result: { tool: "collect_trip_requirements", call_id: "call-1", status: "completed", result: { destination: "重庆", departure_date: "2026-12-01", days: 3, notes: "不吃辣" } },
          react_steps: [{ tool: "search_travel_knowledge", arguments: { query: "洪崖洞" }, status: "completed" }],
          assistant_result: concretePlan,
        },
      },
    ];
    vi.mocked(fetch).mockImplementationOnce(async () => new Response(JSON.stringify({ conversation: {}, messages: history }), { status: 200 }));
    await useChat.getState().selectConversation("c2");
    const { messages, activePlan } = useChat.getState();
    expect(messages.map((m) => m.id)).toEqual(["m1", "m2", "m4"]);
    expect(messages[1].trip).toMatchObject({
      status: "completed",
      values: { destination: "重庆", departure_date: "2026-12-01", days: "3" },
      notes: "不吃辣",
      recommendation: "推荐重庆",
      suggestions: ["重庆"],
    });
    expect(messages[2].activities.map((a) => a.tool)).toEqual(["search_travel_knowledge"]);
    expect(messages[2].plan).toEqual(concretePlan);
    expect(activePlan).toEqual(concretePlan);
  });
});
