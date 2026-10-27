import clsx from "clsx";
import {
  AlertTriangle,
  BookOpen,
  Building2,
  CheckCircle2,
  CloudSun,
  Loader2,
  MapPinned,
  Navigation,
  Route,
  Search,
  UtensilsCrossed,
} from "lucide-react";
import type { WorkerName } from "@/api/types";
import { useChat, type Activity, type ChatMessage, type ResearchStep } from "@/store/chat";
import { WORKER_LABELS, formatDate } from "@/lib/format";
import { Markdown } from "./Markdown";
import { TripCard } from "./TripCard";
import { LogoMark } from "./Logo";

const WORKER_ICONS: Record<WorkerName, typeof MapPinned> = {
  attractions: MapPinned,
  hotel: Building2,
  food: UtensilsCrossed,
  weather: CloudSun,
  transport: Route,
};
const WORKER_ORDER: WorkerName[] = ["attractions", "hotel", "food", "weather", "transport"];

function activityLabel(activity: Activity): { icon: typeof Search; text: string } {
  const args = activity.arguments || {};
  if (activity.tool === "search_travel_knowledge") return { icon: BookOpen, text: `查资料：${args.query || ""}` };
  if (activity.tool === "estimate_travel_time") return { icon: Navigation, text: `估算路程：${args.destination || ""}` };
  const topic = (args.topic || activity.tool.replace("research_", "")) as WorkerName;
  return {
    icon: Search,
    text: `研究${args.destination ? ` ${args.destination} ` : ""}${WORKER_LABELS[topic] || topic}`,
  };
}

function Activities({ items }: { items: Activity[] }) {
  if (!items.length) return null;
  return (
    <div className="mb-3 flex flex-wrap gap-1.5">
      {items.map((activity) => {
        const { icon: Icon, text } = activityLabel(activity);
        const running = activity.status === "running";
        const empty = activity.status === "empty" || activity.status === "failed";
        return (
          <span
            key={activity.id}
            className={clsx(
              "inline-flex max-w-full items-center gap-1.5 rounded-full px-2.5 py-1 text-xs",
              running
                ? "bg-sun-50 text-sun-700 dark:bg-sun-500/10 dark:text-sun-300"
                : empty
                  ? "bg-ink-100 text-ink-400 line-through dark:bg-ink-800"
                  : "bg-ink-100 text-ink-600 dark:bg-ink-800 dark:text-ink-200",
            )}
          >
            {running ? <Loader2 className="h-3 w-3 animate-spin" /> : <Icon className="h-3 w-3" />}
            <span className="truncate">{text}</span>
          </span>
        );
      })}
    </div>
  );
}

function ResearchProgress({ steps, done }: { steps: Partial<Record<WorkerName, ResearchStep>>; done: boolean }) {
  // 五个子 Agent 都结束后还有选点、排程、检查和成文几步，单独给出提示，避免看起来像卡住了。
  const researched = WORKER_ORDER.every((worker) => steps[worker] && steps[worker]!.status !== "running");
  const label = done ? "研究完成" : researched ? "研究完成，正在选点、排日程…" : "子 Agent 正在并行研究…";
  return (
    <div className="card mb-3 max-w-xl p-4">
      <p className="mb-3 flex items-center gap-2 text-sm font-semibold">
        {done ? <CheckCircle2 className="h-4 w-4 text-moss-500" /> : <Loader2 className="h-4 w-4 animate-spin text-sun-500" />}
        {label}
      </p>
      <div className="grid grid-cols-5 gap-2">
        {WORKER_ORDER.map((worker) => {
          const step = steps[worker];
          const Icon = WORKER_ICONS[worker];
          const status = step?.status || (done ? "completed" : "waiting");
          const running = status === "running";
          const failed = status === "failed" || status === "unavailable";
          return (
            <div
              key={worker}
              className={clsx(
                "flex flex-col items-center gap-1.5 rounded-xl px-1 py-2.5 text-center transition",
                running && "bg-sun-50 dark:bg-sun-500/10",
                status === "waiting" && "opacity-40",
              )}
            >
              <span
                className={clsx(
                  "flex h-8 w-8 items-center justify-center rounded-full",
                  failed
                    ? "bg-red-100 text-red-500 dark:bg-red-500/10"
                    : running
                      ? "bg-sun-500 text-white"
                      : status === "waiting"
                        ? "bg-ink-100 text-ink-400 dark:bg-ink-800"
                        : "bg-moss-500 text-white",
                )}
              >
                {running ? <Loader2 className="h-4 w-4 animate-spin" /> : <Icon className="h-4 w-4" />}
              </span>
              <span className="text-xs font-medium">{WORKER_LABELS[worker]}</span>
              {step?.evidence != null && <span className="text-[10px] text-ink-400">{step.evidence} 条证据</span>}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function PlanSummary({ message }: { message: ChatMessage }) {
  const { openPlan, activePlan } = useChat();
  const plan = message.plan!;
  const req = plan.requirement;
  const stops = plan.itinerary.flatMap((day) => day.slots.filter((s) => s.period !== "evening").map((s) => s.title));
  const active = activePlan === plan;
  return (
    <div className="card max-w-xl overflow-hidden">
      <div className="relative bg-ink-950 px-5 py-5 text-white">
        <div className="pointer-events-none absolute -right-10 -top-10 h-40 w-40 rounded-full bg-sun-500/30 blur-2xl" />
        <p className="text-xs uppercase tracking-widest text-ink-300">行程草稿</p>
        <h3 className="mt-1 font-display text-2xl font-semibold">
          {req.destination} · {req.days} 天
        </h3>
        <p className="mt-1 text-sm text-ink-300">
          {formatDate(req.departure_date)} 出发 · {req.adults} 位成人{req.children ? ` · ${req.children} 位儿童` : ""}
        </p>
      </div>
      <div className="px-5 py-4">
        <p className="line-clamp-2 text-sm text-ink-500 dark:text-ink-300">{stops.slice(0, 8).join(" → ")}</p>
        {plan.warnings.some((w) => w.startsWith("planning_degraded")) && (
          <p className="mt-2 flex items-center gap-1.5 text-xs text-amber-600">
            <AlertTriangle className="h-3.5 w-3.5" /> 部分数据源不可用，结果已降级
          </p>
        )}
        <button className={clsx("mt-4 w-full", active ? "btn-ghost border border-ink-200 dark:border-ink-700" : "btn-primary")} onClick={() => openPlan(plan)}>
          {active ? "正在右侧查看" : "查看完整行程与地图"}
        </button>
      </div>
    </div>
  );
}

export function MessageItem({ message, interactive }: { message: ChatMessage; interactive: boolean }) {
  if (message.role === "user") {
    return (
      <div className="flex justify-end animate-fade-up">
        <div className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-br-md bg-ink-900 px-4 py-2.5 text-[15px] leading-7 text-white dark:bg-sun-500 dark:text-ink-950">
          {message.content}
        </div>
      </div>
    );
  }

  const thinking = message.streaming && !message.content && !message.trip && !message.activities.length && !message.research;
  const researching = message.research !== undefined && !message.plan;

  return (
    <div className="flex gap-3 animate-fade-up">
      <LogoMark className="mt-0.5 h-8 w-8 shrink-0" />
      <div className="min-w-0 flex-1">
        <Activities items={message.activities} />
        {researching && <ResearchProgress steps={message.research!} done={!message.streaming} />}
        {thinking && (
          <div className="flex items-center gap-1.5 py-2">
            {[0, 1, 2].map((i) => (
              <span key={i} className="h-2 w-2 animate-bounce rounded-full bg-sun-400" style={{ animationDelay: `${i * 0.15}s` }} />
            ))}
          </div>
        )}
        {message.trip && <TripCard messageId={message.id} trip={message.trip} interactive={interactive} />}
        {message.plan ? (
          <PlanSummary message={message} />
        ) : (
          message.content && (
            <div className={clsx(message.error && "rounded-xl bg-red-50 px-4 py-3 text-red-600 dark:bg-red-500/10 dark:text-red-300")}>
              <Markdown>{message.content}</Markdown>
            </div>
          )
        )}
      </div>
    </div>
  );
}
