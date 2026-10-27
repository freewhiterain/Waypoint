import type { WorkerName } from "@/api/types";

export const WEEKDAYS = ["周日", "周一", "周二", "周三", "周四", "周五", "周六"];

export const WORKER_LABELS: Record<WorkerName, string> = {
  attractions: "景点",
  hotel: "住宿",
  food: "美食",
  weather: "天气",
  transport: "交通",
};

export const PERIOD_LABELS = { morning: "上午", lunch: "中午", afternoon: "下午", evening: "晚上" } as const;

export const POPULAR_DESTINATIONS = ["成都", "北京", "上海", "杭州", "南京", "重庆", "广州", "西安"];
export const DAY_OPTIONS = [2, 3, 4, 5];

export function localIso(date: Date): string {
  const offset = date.getTimezoneOffset() * 60000;
  return new Date(date.getTime() - offset).toISOString().slice(0, 10);
}

export function todayIso(): string {
  return localIso(new Date());
}

export function parseIso(iso: string): Date {
  const [y, m, d] = iso.split("-").map(Number);
  return new Date(y, m - 1, d);
}

export function formatDate(iso: string): string {
  if (!iso) return "";
  const date = parseIso(iso);
  return `${date.getMonth() + 1}月${date.getDate()}日 ${WEEKDAYS[date.getDay()]}`;
}

/** 常用出发日期按今天换算成具体日期：后端只接受 YYYY-MM-DD。 */
export function quickDates(): { label: string; value: string }[] {
  const now = new Date();
  const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
  const thisSaturday = new Date(today);
  thisSaturday.setDate(today.getDate() + ((6 - today.getDay() + 7) % 7));
  const nextSaturday = new Date(thisSaturday);
  nextSaturday.setDate(thisSaturday.getDate() + 7);
  const nextMonth = new Date(today.getFullYear(), today.getMonth() + 1, 1);
  return [
    { label: "本周末", value: localIso(thisSaturday) },
    { label: "下周末", value: localIso(nextSaturday) },
    { label: "下个月初", value: localIso(nextMonth) },
  ];
}

export function relativeTime(iso: string): string {
  const time = new Date(iso.endsWith("Z") || iso.includes("+") ? iso : `${iso}Z`).getTime();
  const diff = (Date.now() - time) / 1000;
  if (diff < 60) return "刚刚";
  if (diff < 3600) return `${Math.floor(diff / 60)} 分钟前`;
  if (diff < 86400) return `${Math.floor(diff / 3600)} 小时前`;
  if (diff < 86400 * 7) return `${Math.floor(diff / 86400)} 天前`;
  const date = new Date(time);
  return `${date.getMonth() + 1}/${date.getDate()}`;
}

export function money(value: number | null | undefined): string | null {
  if (value == null) return null;
  return value === 0 ? "免费" : `¥${Math.round(value)}`;
}

/** "天气：零星小雨 13~21℃；降水概率 17%" → 拆出天气备注和其余备注。 */
export function splitNotes(notes: string[]): { weather?: string; hotel?: string; other: string[] } {
  let weather: string | undefined;
  let hotel: string | undefined;
  const other: string[] = [];
  for (const note of notes) {
    if (note.startsWith("天气：")) weather = note.slice(3);
    else if (note.startsWith("住宿：")) hotel = note.slice(3);
    else other.push(note);
  }
  return { weather, hotel, other };
}

export function weatherIcon(text = ""): string {
  if (/雷/.test(text)) return "⛈️";
  if (/雪/.test(text)) return "🌨️";
  if (/雨/.test(text)) return "🌧️";
  if (/阴/.test(text)) return "☁️";
  if (/云/.test(text)) return "⛅";
  if (/晴/.test(text)) return "☀️";
  if (/雾|霾/.test(text)) return "🌫️";
  return "🌤️";
}
