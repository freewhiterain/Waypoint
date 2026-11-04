import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { useChat, type TripCall } from "@/store/chat";
import { localIso } from "@/lib/format";
import { TripCard } from "./TripCard";

afterEach(cleanup);

const future = () => {
  const d = new Date();
  d.setDate(d.getDate() + 10);
  return localIso(d);
};

const trip = (patch: Partial<TripCall> = {}): TripCall => ({
  callId: "c",
  values: { destination: "成都", departure_date: future(), days: "3" },
  revisions: [],
  askFields: [],
  status: "pending",
  recommendation: "",
  suggestions: [],
  error: "",
  ...patch,
});

describe("TripCard", () => {
  it("submits the notes field with the form values", () => {
    const submitTrip = vi.fn();
    useChat.setState({ submitTrip, busy: false });
    const t = trip();
    render(<TripCard messageId="m1" trip={t} interactive />);
    fireEvent.change(screen.getByPlaceholderText(/洪崖洞必去/), { target: { value: "不吃辣" } });
    fireEvent.click(screen.getByText("开始规划"));
    expect(submitTrip).toHaveBeenCalledWith("m1", t.values, "不吃辣");
  });

  it("does not submit an invalid form and shows the errors", () => {
    const submitTrip = vi.fn();
    useChat.setState({ submitTrip, busy: false });
    render(<TripCard messageId="m1" trip={trip({ values: { destination: "", departure_date: "", days: "" }, askFields: ["destination", "departure_date", "days"] })} interactive />);
    fireEvent.click(screen.getByText("开始规划"));
    expect(submitTrip).not.toHaveBeenCalled();
    expect(screen.getByText("请选择或输入目的地")).toBeTruthy();
  });

  it("the collapsed summary shows the submitted notes", () => {
    render(<TripCard messageId="m1" trip={trip({ status: "completed", notes: "带老人别太累" })} interactive />);
    expect(screen.getByText(/已确认：成都/)).toBeTruthy();
    expect(screen.getByText("· 补充：带老人别太累")).toBeTruthy();
  });

  it("shows revisions carried into the plan", () => {
    render(<TripCard messageId="m1" trip={trip({ revisions: ["不去 锦里"] })} interactive />);
    expect(screen.getByText("不去 锦里")).toBeTruthy();
  });
});
