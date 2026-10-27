import { lazy, Suspense, useState } from "react";
import clsx from "clsx";
import {
  BedDouble,
  Car,
  Clock,
  Copy,
  Database,
  Feather,
  ListOrdered,
  Map as MapIcon,
  RefreshCw,
  Ticket,
  Wallet,
  X,
} from "lucide-react";
import type { TimeSlot, TravelPlanDraft, WorkerName } from "@/api/types";
import { useChat } from "@/store/chat";
import { DAY_COLORS } from "@/lib/places";
import { estimateCost } from "@/lib/cost";
import { PERIOD_LABELS, WORKER_LABELS, formatDate, money, splitNotes, weatherIcon } from "@/lib/format";

const PlanMap = lazy(() => import("./PlanMap").then((m) => ({ default: m.PlanMap })));

type Tab = "timeline" | "map" | "sources";

const PERIOD_STYLES = {
  morning: "bg-amber-100 text-amber-700 dark:bg-amber-500/15 dark:text-amber-300",
  lunch: "bg-emerald-100 text-emerald-700 dark:bg-emerald-500/15 dark:text-emerald-300",
  afternoon: "bg-sky-100 text-sky-700 dark:bg-sky-500/15 dark:text-sky-300",
  evening: "bg-violet-100 text-violet-700 dark:bg-violet-500/15 dark:text-violet-300",
};

const isFree = (slot: TimeSlot) => slot.title.endsWith("自由活动") || slot.title.startsWith("自由安排");

function Slot({ slot, first, onReplace }: { slot: TimeSlot; first: boolean; onReplace?: () => void }) {
  const cost = money(slot.estimated_cost);
  return (
    <li className="group relative pl-6">
      <span className="absolute left-0 top-1.5 h-2.5 w-2.5 rounded-full border-2 border-white bg-ink-300 ring-2 ring-ink-100 dark:border-ink-900 dark:ring-ink-800" />
      {slot.travel_minutes != null && (
        <p className="mb-1.5 flex items-center gap-1 text-xs text-ink-400">
          <Car className="h-3.5 w-3.5" /> {first ? "从住处出发" : "路上"}约 {slot.travel_minutes} 分钟
        </p>
      )}
      <div className="flex items-start gap-2">
        <span className={clsx("mt-0.5 shrink-0 rounded-md px-1.5 py-0.5 text-[11px] font-medium", PERIOD_STYLES[slot.period])}>
          {PERIOD_LABELS[slot.period]}
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex items-start justify-between gap-2">
            <p className="font-medium leading-6">{slot.title}</p>
            {onReplace && !isFree(slot) && (
              <button
                onClick={onReplace}
                className="flex shrink-0 items-center gap-1 rounded-md px-1.5 py-0.5 text-xs text-ink-400 opacity-100 transition hover:bg-sun-50 hover:text-sun-700 sm:opacity-0 sm:group-hover:opacity-100 dark:hover:bg-sun-500/10"
                title="不想去这里，换一个重新规划"
              >
                <RefreshCw className="h-3 w-3" /> 换掉
              </button>
            )}
          </div>
          <p className="mt-0.5 text-sm leading-6 text-ink-500 dark:text-ink-300">{slot.description}</p>
          {(slot.opening_window || cost) && (
            <div className="mt-1.5 flex flex-wrap gap-3 text-xs text-ink-400">
              {slot.opening_window && (
                <span className="flex items-center gap-1">
                  <Clock className="h-3.5 w-3.5" /> {slot.opening_window}
                </span>
              )}
              {cost && (
                <span className="flex items-center gap-1">
                  <Ticket className="h-3.5 w-3.5" /> {cost}
                  {slot.period === "evening" || slot.period === "lunch" ? " / 人" : ""}
                </span>
              )}
            </div>
          )}
        </div>
      </div>
    </li>
  );
}

function CostCard({ plan }: { plan: TravelPlanDraft }) {
  const cost = estimateCost(plan);
  if (!cost.total) return null;
  const rows = [
    { label: "门票", value: cost.tickets },
    { label: "餐饮", value: cost.meals },
    { label: `住宿 · ${cost.nights} 晚`, value: cost.hotel },
  ];
  return (
    <div className="card p-4">
      <div className="flex items-baseline justify-between">
        <p className="flex items-center gap-1.5 text-sm font-medium">
          <Wallet className="h-4 w-4 text-sun-500" /> 费用估算
        </p>
        <p className="font-display text-xl font-semibold">¥{Math.round(cost.total).toLocaleString()}</p>
      </div>
      <div className="mt-3 grid grid-cols-3 gap-2 text-center">
        {rows.map((row) => (
          <div key={row.label} className="rounded-xl bg-ink-50 px-2 py-2 dark:bg-ink-800">
            <p className="text-[11px] text-ink-400">{row.label}</p>
            <p className="text-sm font-semibold">¥{Math.round(row.value).toLocaleString()}</p>
          </div>
        ))}
      </div>
      <p className="mt-2 text-[11px] leading-4 text-ink-400">
        按 {cost.people} 人计，住宿按一间房；不含城际交通和市内打车
        {cost.unknown ? `，另有 ${cost.unknown} 处没有价格数据` : ""}。价格来自数据快照，仅供参考。
      </p>
    </div>
  );
}

function Timeline({
  plan,
  onFocus,
  onRevise,
}: {
  plan: TravelPlanDraft;
  onFocus: (day: number) => void;
  onRevise?: (request: string) => void;
}) {
  const hotel = splitNotes(plan.itinerary[0]?.notes || []).hotel;
  return (
    <div className="space-y-4">
      <CostCard plan={plan} />
      {hotel && (
        <div className="flex gap-3 rounded-2xl bg-ink-50 p-4 dark:bg-ink-800">
          <BedDouble className="mt-0.5 h-5 w-5 shrink-0 text-ink-500" />
          <div className="text-sm">
            <p className="font-medium">{hotel.split("。")[0]}</p>
            <p className="mt-0.5 text-ink-500 dark:text-ink-300">{hotel.split("。").slice(1).join("。")}</p>
          </div>
        </div>
      )}
      {plan.itinerary.map((day) => {
        const { weather, other } = splitNotes(day.notes);
        const color = DAY_COLORS[(day.day - 1) % DAY_COLORS.length];
        return (
          <section key={day.day} className="card overflow-hidden">
            <button
              className="flex w-full items-center gap-3 border-b border-ink-100 px-4 py-3 text-left dark:border-ink-800"
              onClick={() => onFocus(day.day)}
              title="在地图上查看这一天"
            >
              <span className="flex h-9 w-9 items-center justify-center rounded-xl font-display font-semibold text-white" style={{ background: color }}>
                {day.day}
              </span>
              <div className="min-w-0 flex-1">
                <p className="font-semibold">第 {day.day} 天</p>
                <p className="text-xs text-ink-400">{formatDate(day.date)}</p>
              </div>
              {weather && (
                <span className="flex max-w-[50%] items-center gap-1.5 rounded-full bg-ink-50 px-2.5 py-1 text-xs text-ink-600 dark:bg-ink-800 dark:text-ink-200">
                  <span>{weatherIcon(weather)}</span>
                  <span className="truncate">{weather.split("；")[0]}</span>
                </span>
              )}
            </button>
            <ol className="space-y-5 border-l-0 px-4 py-4">
              {day.slots.map((slot, index) => (
                <Slot
                  key={`${slot.period}-${slot.title}`}
                  slot={slot}
                  first={index === 0}
                  onReplace={onRevise && (() => onRevise(`把${slot.title}换成别的`))}
                />
              ))}
            </ol>
            {onRevise &&
              day.slots.filter((slot) => (slot.period === "morning" || slot.period === "afternoon") && !isFree(slot)).length > 1 && (
                <div className="border-t border-ink-100 px-4 py-2 dark:border-ink-800">
                  <button
                    onClick={() => onRevise(`第${day.day}天轻松一点，少安排一个景点`)}
                    className="flex items-center gap-1 text-xs text-ink-400 hover:text-sun-600"
                  >
                    <Feather className="h-3.5 w-3.5" /> 这天轻松点
                  </button>
                </div>
              )}
            {other.length > 0 && <p className="border-t border-ink-100 px-4 py-2.5 text-xs text-ink-400 dark:border-ink-800">{other.join(" ")}</p>}
          </section>
        );
      })}
    </div>
  );
}

const STATUS_TEXT: Record<string, string> = {
  completed: "完成",
  partial: "部分可用",
  unavailable: "不可用",
  failed: "失败",
};

function Sources({ plan }: { plan: TravelPlanDraft }) {
  return (
    <div className="space-y-3">
      {plan.worker_results.map((result) => (
        <details key={result.worker} className="card group p-4">
          <summary className="flex cursor-pointer list-none items-center justify-between">
            <span className="font-medium">{WORKER_LABELS[result.worker as WorkerName] || result.worker}</span>
            <span className="flex items-center gap-2 text-xs text-ink-400">
              {result.options.length} 个候选 · {result.evidence.length} 条证据
              <span
                className={clsx(
                  "rounded-full px-2 py-0.5",
                  result.status === "completed" ? "bg-moss-500/10 text-moss-600" : "bg-amber-100 text-amber-700",
                )}
              >
                {STATUS_TEXT[result.status] || result.status}
              </span>
            </span>
          </summary>
          <p className="mt-3 text-sm text-ink-500 dark:text-ink-300">{result.summary}</p>
          <ul className="mt-3 space-y-2">
            {result.evidence.slice(0, 6).map((item, index) => (
              <li key={index} className="rounded-lg bg-ink-50 p-2.5 text-xs leading-5 text-ink-500 dark:bg-ink-800 dark:text-ink-300">
                <span className="font-medium text-ink-700 dark:text-ink-100">{item.source}</span>
                {item.is_mock && <span className="ml-1 text-amber-600">（模拟资料）</span>}
                <br />
                {item.content.slice(0, 160)}
              </li>
            ))}
          </ul>
        </details>
      ))}
      <p className="px-1 text-xs leading-5 text-ink-400">
        地点数据来自 ChinaTravel 数据集快照（CC BY 4.0），天气来自 Open-Meteo，路程来自高德与交通图谱。价格和营业时间以实际为准。
      </p>
    </div>
  );
}

function toMarkdown(plan: TravelPlanDraft): string {
  const lines = [`# ${plan.requirement.destination} ${plan.requirement.days} 日行程`, ""];
  for (const day of plan.itinerary) {
    lines.push(`## 第${day.day}天 · ${day.date}`);
    for (const note of day.notes) lines.push(`- ${note}`);
    for (const slot of day.slots) lines.push(`- **${PERIOD_LABELS[slot.period]}**：${slot.title}。${slot.description}`);
    lines.push("");
  }
  return lines.join("\n");
}

export function PlanPanel({ plan, onClose }: { plan: TravelPlanDraft; onClose: () => void }) {
  const { notify, send, busy } = useChat();
  // 一键修改：替用户发出修改请求，走对话理解 → 记下约束 → 确认后重新规划。
  const revise = (request: string) => {
    if (busy) return;
    send(request);
    if (window.innerWidth < 1280) onClose();
  };
  const [tab, setTab] = useState<Tab>("timeline");
  const [focusDay, setFocusDay] = useState<number | null>(null);
  const req = plan.requirement;

  const tabs: { id: Tab; label: string; icon: typeof MapIcon }[] = [
    { id: "timeline", label: "行程", icon: ListOrdered },
    { id: "map", label: "地图", icon: MapIcon },
    { id: "sources", label: "依据", icon: Database },
  ];

  return (
    <aside className="flex h-full w-full flex-col bg-[#fbfaf7] dark:bg-ink-900">
      <header className="flex items-start gap-3 border-b border-ink-100 px-5 py-4 dark:border-ink-800">
        <div className="min-w-0 flex-1">
          <p className="text-xs uppercase tracking-widest text-sun-600 dark:text-sun-400">行程草稿</p>
          <h2 className="mt-0.5 truncate font-display text-xl font-semibold">
            {req.destination} · {req.days} 天
          </h2>
          <p className="text-xs text-ink-400">
            {formatDate(req.departure_date)} 出发
            {req.styles.length ? ` · ${req.styles.join("、")}` : ""}
          </p>
        </div>
        <button
          className="btn-ghost h-9 w-9 p-0"
          title="复制为 Markdown"
          onClick={async () => {
            try {
              await navigator.clipboard.writeText(toMarkdown(plan));
              notify("已复制行程");
            } catch {
              notify("复制失败", true);
            }
          }}
        >
          <Copy className="h-4 w-4" />
        </button>
        <button className="btn-ghost h-9 w-9 p-0" onClick={onClose} aria-label="关闭行程">
          <X className="h-4 w-4" />
        </button>
      </header>

      <div className="flex gap-1 px-5 pt-3">
        {tabs.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            className={clsx(
              "flex items-center gap-1.5 rounded-lg px-3 py-1.5 text-sm font-medium transition",
              tab === id ? "bg-ink-900 text-white dark:bg-white dark:text-ink-900" : "text-ink-500 hover:bg-ink-100 dark:hover:bg-ink-800",
            )}
          >
            <Icon className="h-4 w-4" /> {label}
          </button>
        ))}
      </div>

      <div className="scrollbar-thin flex-1 overflow-y-auto p-5">
        {tab === "timeline" && (
          <Timeline
            plan={plan}
            onRevise={revise}
            onFocus={(day) => {
              setFocusDay(day);
              setTab("map");
            }}
          />
        )}
        {tab === "map" && (
          <div className="flex h-full min-h-[420px] flex-col gap-3">
            <div className="flex flex-wrap gap-1.5">
              <button className={clsx("chip py-1 text-xs", focusDay === null && "chip-active")} onClick={() => setFocusDay(null)}>
                全部
              </button>
              {plan.itinerary.map((day) => (
                <button
                  key={day.day}
                  className={clsx("chip py-1 text-xs", focusDay === day.day && "chip-active")}
                  onClick={() => setFocusDay(day.day)}
                >
                  <span className="h-2 w-2 rounded-full" style={{ background: DAY_COLORS[(day.day - 1) % DAY_COLORS.length] }} />第{day.day}天
                </button>
              ))}
            </div>
            <div className="min-h-0 flex-1">
              <Suspense fallback={<div className="skeleton h-full w-full rounded-2xl" />}>
                <PlanMap plan={plan} focusDay={focusDay} />
              </Suspense>
            </div>
          </div>
        )}
        {tab === "sources" && <Sources plan={plan} />}
      </div>
    </aside>
  );
}
