import { useEffect, useState } from "react";
import clsx from "clsx";
import { CalendarDays, Check, Clock3, Loader2, MapPin, Pencil, Sparkles } from "lucide-react";
import type { TripValues } from "@/api/types";
import { TRIP_FIELDS, tripFieldError, useChat, type TripCall } from "@/store/chat";
import { DAY_OPTIONS, POPULAR_DESTINATIONS, formatDate, quickDates, todayIso } from "@/lib/format";
import { Markdown } from "./Markdown";

const FIELD_META: Record<keyof TripValues, { label: string; icon: typeof MapPin; question: string }> = {
  destination: { label: "目的地", icon: MapPin, question: "想去哪里？" },
  departure_date: { label: "出发", icon: CalendarDays, question: "哪天出发？" },
  days: { label: "天数", icon: Clock3, question: "玩几天？" },
};

/**
 * 规划表单：只问后端没能从对话里确定的槽位；三项都齐时直接给出确认。
 * 提交后卡片收起成一行摘要。
 */
export function TripCard({ messageId, trip, interactive }: { messageId: string; trip: TripCall; interactive: boolean }) {
  const { submitTrip, recommendDestination, busy } = useChat();
  const [values, setValues] = useState<TripValues>(trip.values);
  const [editing, setEditing] = useState(trip.askFields.length > 0);
  const [touched, setTouched] = useState(false);

  useEffect(() => setValues(trip.values), [trip.values]);

  const fields = editing ? (trip.askFields.length ? trip.askFields : TRIP_FIELDS) : [];
  const errors = Object.fromEntries(TRIP_FIELDS.map((f) => [f, tripFieldError(f, values)])) as Record<
    keyof TripValues,
    string
  >;
  const valid = TRIP_FIELDS.every((field) => !errors[field]);
  const summary = [values.destination, values.departure_date && formatDate(values.departure_date), values.days && `${values.days} 天`]
    .filter(Boolean)
    .join(" · ");

  if (trip.status === "completed" || !interactive) {
    return (
      <div className="inline-flex items-center gap-2 rounded-full border border-moss-500/30 bg-moss-500/10 px-3.5 py-1.5 text-sm text-moss-600 dark:text-moss-400">
        <Check className="h-4 w-4" />
        {trip.status === "completed" ? `已确认：${summary}` : `需求表单：${summary || "未填写"}`}
      </div>
    );
  }

  const submitting = trip.status === "submitting";
  const recommending = trip.status === "recommending";

  function submit() {
    setTouched(true);
    if (!valid) {
      setEditing(true);
      return;
    }
    submitTrip(messageId, values);
  }

  return (
    <div className="card w-full max-w-xl overflow-hidden animate-fade-up">
      <div className="flex items-center justify-between border-b border-ink-100 bg-gradient-to-r from-sun-50 to-transparent px-5 py-3.5 dark:border-ink-800 dark:from-sun-500/10">
        <div className="flex items-center gap-2 text-sm font-semibold">
          <Sparkles className="h-4 w-4 text-sun-500" />
          {editing ? "补全旅行信息" : "确认后开始规划"}
        </div>
        {!editing && (
          <button className="flex items-center gap-1 text-xs text-ink-500 hover:text-ink-900 dark:hover:text-white" onClick={() => setEditing(true)}>
            <Pencil className="h-3.5 w-3.5" /> 修改
          </button>
        )}
      </div>

      <div className="space-y-5 px-5 py-4">
        {!editing && (
          <div className="grid grid-cols-3 gap-2">
            {TRIP_FIELDS.map((field) => {
              const { label, icon: Icon } = FIELD_META[field];
              const value =
                field === "departure_date" ? formatDate(values[field]) : field === "days" ? `${values[field]} 天` : values[field];
              return (
                <div key={field} className="rounded-xl bg-ink-50 px-3 py-2.5 dark:bg-ink-800">
                  <p className="flex items-center gap-1 text-xs text-ink-400">
                    <Icon className="h-3.5 w-3.5" />
                    {label}
                  </p>
                  <p className="mt-1 truncate text-sm font-semibold">{value}</p>
                </div>
              );
            })}
          </div>
        )}

        {trip.revisions.length > 0 && (
          <div className="flex flex-wrap items-center gap-1.5">
            <span className="text-xs text-ink-400">会按你的修改重新规划：</span>
            {trip.revisions.map((item) => (
              <span key={item} className="rounded-full bg-sun-50 px-2.5 py-1 text-xs text-sun-700 dark:bg-sun-500/10 dark:text-sun-300">
                {item}
              </span>
            ))}
          </div>
        )}

        {fields.map((field) => {
          const { question } = FIELD_META[field];
          const error = touched ? errors[field] : "";
          return (
            <div key={field}>
              <p className="mb-2 text-sm font-medium">{question}</p>
              {field === "destination" && (
                <div className="flex flex-wrap gap-2">
                  {POPULAR_DESTINATIONS.map((city) => (
                    <button
                      key={city}
                      className={clsx("chip", values.destination === city && "chip-active")}
                      onClick={() => setValues({ ...values, destination: city })}
                    >
                      {city}
                    </button>
                  ))}
                  <input
                    className="input w-32 py-1.5"
                    placeholder="其他城市"
                    value={POPULAR_DESTINATIONS.includes(values.destination) ? "" : values.destination}
                    onChange={(e) => setValues({ ...values, destination: e.target.value })}
                  />
                  <button
                    className="chip border-dashed"
                    disabled={busy}
                    onClick={() => recommendDestination(messageId, values)}
                  >
                    {recommending ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Sparkles className="h-3.5 w-3.5" />}
                    还没想好，帮我推荐
                  </button>
                </div>
              )}
              {field === "departure_date" && (
                <div className="flex flex-wrap items-center gap-2">
                  {quickDates().map((option) => (
                    <button
                      key={option.label}
                      className={clsx("chip", values.departure_date === option.value && "chip-active")}
                      onClick={() => setValues({ ...values, departure_date: option.value })}
                    >
                      {option.label}
                      <span className="text-xs opacity-60">{formatDate(option.value)}</span>
                    </button>
                  ))}
                  <input
                    type="date"
                    className="input w-40 py-1.5"
                    min={todayIso()}
                    value={values.departure_date}
                    onChange={(e) => setValues({ ...values, departure_date: e.target.value })}
                  />
                </div>
              )}
              {field === "days" && (
                <div className="flex flex-wrap items-center gap-2">
                  {DAY_OPTIONS.map((days) => (
                    <button
                      key={days}
                      className={clsx("chip", values.days === String(days) && "chip-active")}
                      onClick={() => setValues({ ...values, days: String(days) })}
                    >
                      {days} 天
                    </button>
                  ))}
                  <input
                    type="number"
                    min={1}
                    max={30}
                    className="input w-24 py-1.5"
                    placeholder="自定义"
                    value={DAY_OPTIONS.map(String).includes(values.days) ? "" : values.days}
                    onChange={(e) => setValues({ ...values, days: e.target.value })}
                  />
                </div>
              )}
              {error && <p className="mt-1.5 text-xs text-red-500">{error}</p>}
            </div>
          );
        })}

        {trip.recommendation && (
          <div className="rounded-xl border border-dashed border-sun-300 bg-sun-50/60 p-4 dark:border-sun-500/40 dark:bg-sun-500/5">
            <Markdown>{trip.recommendation}</Markdown>
            {trip.suggestions.length > 0 && (
              <div className="mt-3 flex flex-wrap gap-2">
                {trip.suggestions.map((city) => (
                  <button
                    key={city}
                    className={clsx("chip", values.destination === city && "chip-active")}
                    onClick={() => setValues({ ...values, destination: city })}
                  >
                    <MapPin className="h-3.5 w-3.5" /> 就去{city}
                  </button>
                ))}
              </div>
            )}
          </div>
        )}

        {trip.error && <p className="text-sm text-red-500">{trip.error}</p>}

        <div className="flex items-center justify-between gap-3 pt-1">
          <p className="truncate text-xs text-ink-400">{summary || "选好后开始规划"}</p>
          <button className="btn-accent shrink-0" disabled={busy || (touched && !valid)} onClick={submit}>
            {submitting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
            {submitting ? "规划中…" : "开始规划"}
          </button>
        </div>
      </div>
    </div>
  );
}
