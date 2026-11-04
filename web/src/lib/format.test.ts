import { afterEach, describe, expect, it, vi } from "vitest";
import { formatDate, localIso, money, parseIso, quickDates, relativeTime, splitNotes, weatherIcon } from "./format";

afterEach(() => vi.useRealTimers());

describe("dates", () => {
  it("parseIso and localIso round-trip in local time", () => {
    const date = parseIso("2026-03-09");
    expect([date.getFullYear(), date.getMonth(), date.getDate()]).toEqual([2026, 2, 9]);
    expect(localIso(date)).toBe("2026-03-09");
  });

  it("formatDate shows month, day and weekday", () => {
    expect(formatDate("2026-11-04")).toBe("11月4日 周三");
    expect(formatDate("")).toBe("");
  });

  it("quickDates: this weekend, next weekend, first of next month", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date(2026, 11, 30, 15)); // 周三
    expect(quickDates().map((d) => d.value)).toEqual(["2027-01-02", "2027-01-09", "2027-01-01"]);
    vi.setSystemTime(new Date(2026, 10, 7, 9)); // 周六当天
    expect(quickDates()[0].value).toBe("2026-11-07");
  });

  it("relativeTime treats naive backend timestamps as UTC", () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-11-04T12:00:00Z"));
    expect(relativeTime("2026-11-04T11:59:30")).toBe("刚刚");
    expect(relativeTime("2026-11-04T11:55:00Z")).toBe("5 分钟前");
    expect(relativeTime("2026-11-04T09:00:00+00:00")).toBe("3 小时前");
    expect(relativeTime("2026-11-02T12:00:00")).toBe("2 天前");
    expect(relativeTime("2026-10-01T12:00:00")).toMatch(/^10\/1$/);
  });
});

describe("money", () => {
  it("formats prices", () => {
    expect(money(null)).toBeNull();
    expect(money(undefined)).toBeNull();
    expect(money(0)).toBe("免费");
    expect(money(49.6)).toBe("¥50");
  });
});

describe("splitNotes", () => {
  it("separates weather and hotel notes from the rest", () => {
    expect(splitNotes(["天气：晴 13~21℃", "住宿：桔子酒店。近地铁", "记得带伞"])).toEqual({
      weather: "晴 13~21℃",
      hotel: "桔子酒店。近地铁",
      other: ["记得带伞"],
    });
    expect(splitNotes([])).toEqual({ weather: undefined, hotel: undefined, other: [] });
  });
});

describe("weatherIcon", () => {
  it("picks the most severe matching icon", () => {
    expect(weatherIcon("雷阵雨")).toBe("⛈️");
    expect(weatherIcon("雨夹雪")).toBe("🌨️");
    expect(weatherIcon("小雨")).toBe("🌧️");
    expect(weatherIcon("多云")).toBe("⛅");
    expect(weatherIcon("晴")).toBe("☀️");
    expect(weatherIcon()).toBe("🌤️");
  });
});
