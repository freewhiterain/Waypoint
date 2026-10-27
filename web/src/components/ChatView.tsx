import { useEffect, useRef, useState } from "react";
import { ArrowUp, Loader2, Menu, PanelRightOpen, Square } from "lucide-react";
import { useChat } from "@/store/chat";
import { MessageItem } from "./MessageItem";

const SUGGESTIONS = [
  { emoji: "🐼", title: "成都 3 天美食之旅", text: "帮我规划成都三天的行程，下周六出发，喜欢美食" },
  { emoji: "🏯", title: "带孩子去北京", text: "下个月想带孩子去北京玩4天，帮我安排一下" },
  { emoji: "🌧️", title: "看看天气再决定", text: "杭州这几天天气怎么样，适合去西湖吗" },
  { emoji: "🧭", title: "还没想好去哪", text: "周末有两天假，推荐几个适合短途旅行的城市" },
];

function Composer() {
  const { send, busy, canStop, stop } = useChat();
  const [text, setText] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 160)}px`;
  }, [text]);

  function submit() {
    if (!text.trim() || busy) return;
    send(text);
    setText("");
  }

  return (
    <div className="mx-auto w-full max-w-3xl px-4 pb-5 sm:px-6">
      <div className="card flex items-end gap-2 p-2 focus-within:border-sun-300 focus-within:ring-4 focus-within:ring-sun-100 dark:focus-within:ring-sun-900/30">
        <textarea
          ref={ref}
          rows={1}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            // 中文输入法选词时按回车只是确认候选，不能当成发送。
            if (e.nativeEvent.isComposing || e.keyCode === 229) return;
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit();
            }
          }}
          placeholder="说说你想去哪、玩几天、喜欢什么…"
          className="max-h-40 min-h-[44px] flex-1 resize-none bg-transparent px-3 py-2.5 text-[15px] outline-none placeholder:text-ink-300 dark:placeholder:text-ink-500"
        />
        {canStop ? (
          <button className="btn-primary h-11 w-11 shrink-0 p-0" onClick={stop} aria-label="停止生成" title="停止生成">
            <Square className="h-4 w-4 fill-current" />
          </button>
        ) : (
          <button className="btn-accent h-11 w-11 shrink-0 p-0" disabled={busy || !text.trim()} onClick={submit} aria-label="发送">
            {busy ? <Loader2 className="h-5 w-5 animate-spin" /> : <ArrowUp className="h-5 w-5" />}
          </button>
        )}
      </div>
      <p className="mt-2 text-center text-xs text-ink-400">Enter 发送 · Shift+Enter 换行 · 行程为草稿，价格与营业时间以实际为准</p>
    </div>
  );
}

function EmptyState() {
  const { send, busy, user } = useChat();
  return (
    <div className="mx-auto flex w-full max-w-3xl flex-1 flex-col justify-center px-4 py-10 sm:px-6">
      <p className="text-sm font-medium text-sun-600 dark:text-sun-400">你好，{user?.username}</p>
      <h1 className="mt-2 font-display text-3xl font-semibold leading-tight sm:text-4xl">
        这次想去哪里？
        <br />
        <span className="text-ink-400">告诉我，剩下的交给 WayPoint。</span>
      </h1>
      <div className="mt-10 grid gap-3 sm:grid-cols-2">
        {SUGGESTIONS.map((item) => (
          <button
            key={item.title}
            disabled={busy}
            onClick={() => send(item.text)}
            className="card group p-4 text-left transition hover:-translate-y-0.5 hover:border-sun-300 hover:shadow-float"
          >
            <span className="text-2xl">{item.emoji}</span>
            <p className="mt-3 font-medium">{item.title}</p>
            <p className="mt-1 text-sm text-ink-400 group-hover:text-ink-500 dark:group-hover:text-ink-300">{item.text}</p>
          </button>
        ))}
      </div>
    </div>
  );
}

export function ChatView({ onOpenSidebar }: { onOpenSidebar: () => void }) {
  const { messages, loadingHistory, conversations, currentId, activePlan, openPlan } = useChat();
  const scroller = useRef<HTMLDivElement>(null);
  const title = conversations.find((c) => c.id === currentId)?.title;
  const lastPlan = [...messages].reverse().find((m) => m.plan)?.plan;
  const lastTripId = [...messages].reverse().find((m) => m.trip)?.id;

  useEffect(() => {
    const el = scroller.current;
    if (el) el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
  }, [messages]);

  return (
    <div className="flex h-full min-w-0 flex-1 flex-col">
      <header className="flex h-14 shrink-0 items-center gap-3 border-b border-ink-100/80 px-4 dark:border-ink-800/80">
        <button className="btn-ghost h-9 w-9 p-0 lg:hidden" onClick={onOpenSidebar} aria-label="打开会话列表">
          <Menu className="h-5 w-5" />
        </button>
        <h2 className="min-w-0 flex-1 truncate font-medium">{title || "新的旅程"}</h2>
        {lastPlan && !activePlan && (
          <button className="btn-ghost text-sm" onClick={() => openPlan(lastPlan)}>
            <PanelRightOpen className="h-4 w-4" /> 行程
          </button>
        )}
      </header>

      <div ref={scroller} className="scrollbar-thin flex flex-1 flex-col overflow-y-auto">
        {loadingHistory ? (
          <div className="mx-auto w-full max-w-3xl space-y-6 px-6 py-10">
            {[0, 1, 2].map((i) => (
              <div key={i} className="space-y-2">
                <div className="skeleton h-4 w-1/3" />
                <div className="skeleton h-4 w-2/3" />
              </div>
            ))}
          </div>
        ) : messages.length === 0 ? (
          <EmptyState />
        ) : (
          <div className="mx-auto w-full max-w-3xl space-y-6 px-4 py-8 sm:px-6">
            {messages.map((message) => (
              <MessageItem key={message.id} message={message} interactive={message.id === lastTripId} />
            ))}
          </div>
        )}
      </div>

      <Composer />
    </div>
  );
}
