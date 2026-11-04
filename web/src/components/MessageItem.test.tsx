import { afterEach, describe, expect, it } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import type { ChatMessage, TripCall } from "@/store/chat";
import { day, plan, slot } from "@/test/fixtures";
import { MessageItem } from "./MessageItem";

afterEach(cleanup);

const trip: TripCall = {
  callId: "c",
  values: { destination: "成都", departure_date: "", days: "3" },
  revisions: [],
  askFields: ["departure_date"],
  status: "pending",
  recommendation: "",
  suggestions: [],
  error: "",
};

const assistant = (patch: Partial<ChatMessage>): ChatMessage => ({ id: "m", role: "assistant", content: "", activities: [], ...patch });

describe("MessageItem", () => {
  it("shows the lead-in text above the trip form", () => {
    render(<MessageItem message={assistant({ content: "这次先规划成都。", trip })} interactive />);
    const text = screen.getByText("这次先规划成都。");
    const form = screen.getByText("补全旅行信息");
    expect(text.compareDocumentPosition(form) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it("shows the plan summary for a plan with concrete places", () => {
    const p = plan([day(1, [slot("morning", "武侯祠", 50), slot("afternoon", "锦里", 0)])], { adults: 2 });
    render(<MessageItem message={assistant({ content: "行程如下", plan: p })} interactive />);
    expect(screen.getByText("行程草稿")).toBeTruthy();
    expect(screen.getByText("武侯祠 → 锦里")).toBeTruthy();
    expect(screen.getByText(/约 ¥100/)).toBeTruthy();
    expect(screen.queryByText("行程如下")).toBeNull();
  });

  it("shows plain text (no summary) when there is no plan", () => {
    render(<MessageItem message={assistant({ content: "这几天适合随意逛" })} interactive />);
    expect(screen.getByText("这几天适合随意逛")).toBeTruthy();
    expect(screen.queryByText("行程草稿")).toBeNull();
  });

  it("renders user messages as plain bubbles", () => {
    render(<MessageItem message={{ id: "u", role: "user", content: "想去成都", activities: [] }} interactive />);
    expect(screen.getByText("想去成都")).toBeTruthy();
  });
});
