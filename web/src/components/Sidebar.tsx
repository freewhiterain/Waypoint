import { useState } from "react";
import clsx from "clsx";
import { Check, LogOut, MessageSquare, Moon, Pencil, Plus, Sun, Trash2, X } from "lucide-react";
import { useChat } from "@/store/chat";
import { relativeTime } from "@/lib/format";
import { Logo } from "./Logo";

export function useTheme() {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains("dark"));
  const toggle = () => {
    const next = !dark;
    document.documentElement.classList.toggle("dark", next);
    try {
      localStorage.setItem("waypoint.theme", next ? "dark" : "light");
    } catch {
      /* 存不下也不影响本次切换 */
    }
    setDark(next);
  };
  return { dark, toggle };
}

export function Sidebar({ onNavigate }: { onNavigate?: () => void }) {
  const {
    user,
    conversations,
    currentId,
    busy,
    selectConversation,
    startNew,
    renameConversation,
    deleteConversation,
    logout,
    notify,
  } = useChat();
  const { dark, toggle } = useTheme();
  const [editing, setEditing] = useState<string | null>(null);
  const [draft, setDraft] = useState("");

  async function commitRename(id: string) {
    const title = draft.trim();
    setEditing(null);
    if (!title) return;
    try {
      await renameConversation(id, title);
    } catch (error) {
      notify((error as Error).message, true);
    }
  }

  async function remove(id: string) {
    if (!window.confirm("删除这个会话？")) return;
    try {
      await deleteConversation(id);
    } catch (error) {
      notify((error as Error).message, true);
    }
  }

  return (
    <aside className="flex h-full w-full flex-col bg-white/80 backdrop-blur dark:bg-ink-900/80">
      <div className="flex items-center justify-between px-5 pb-4 pt-5">
        <Logo />
        <button className="btn-ghost h-9 w-9 p-0" onClick={toggle} aria-label="切换深浅色">
          {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
        </button>
      </div>

      <div className="px-4">
        <button
          className="btn-primary w-full"
          disabled={busy}
          onClick={() => {
            startNew();
            onNavigate?.();
          }}
        >
          <Plus className="h-4 w-4" /> 新的旅程
        </button>
      </div>

      <p className="px-5 pb-2 pt-6 text-xs font-medium uppercase tracking-wider text-ink-400">最近的会话</p>
      <nav className="scrollbar-thin flex-1 space-y-0.5 overflow-y-auto px-3 pb-4">
        {conversations.length === 0 && (
          <p className="px-2 py-6 text-center text-sm text-ink-400">还没有会话，先说说想去哪吧。</p>
        )}
        {conversations.map((conversation) => {
          const active = conversation.id === currentId;
          return (
            <div
              key={conversation.id}
              className={clsx(
                "group flex items-center gap-2 rounded-xl px-2.5 py-2 text-sm transition",
                active ? "bg-ink-900 text-white dark:bg-white/10" : "hover:bg-ink-100 dark:hover:bg-ink-800",
              )}
            >
              <MessageSquare className={clsx("h-4 w-4 shrink-0", active ? "text-sun-400" : "text-ink-400")} />
              {editing === conversation.id ? (
                <>
                  <input
                    autoFocus
                    value={draft}
                    onChange={(e) => setDraft(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.nativeEvent.isComposing) return;
                      if (e.key === "Enter") commitRename(conversation.id);
                      if (e.key === "Escape") setEditing(null);
                    }}
                    className="min-w-0 flex-1 rounded bg-white px-1.5 py-0.5 text-ink-900 outline-none"
                  />
                  <button onClick={() => commitRename(conversation.id)} aria-label="保存">
                    <Check className="h-4 w-4" />
                  </button>
                  <button onClick={() => setEditing(null)} aria-label="取消">
                    <X className="h-4 w-4" />
                  </button>
                </>
              ) : (
                <>
                  <button
                    className="min-w-0 flex-1 text-left"
                    disabled={busy}
                    onClick={() => {
                      selectConversation(conversation.id);
                      onNavigate?.();
                    }}
                  >
                    <span className="block truncate font-medium">{conversation.title}</span>
                    <span className={clsx("block text-xs", active ? "text-ink-300" : "text-ink-400")}>
                      {relativeTime(conversation.updated_at)}
                    </span>
                  </button>
                  <div className="hidden shrink-0 items-center gap-1 group-hover:flex">
                    <button
                      onClick={() => {
                        setEditing(conversation.id);
                        setDraft(conversation.title);
                      }}
                      className="rounded p-1 hover:bg-black/10"
                      aria-label="重命名"
                    >
                      <Pencil className="h-3.5 w-3.5" />
                    </button>
                    <button onClick={() => remove(conversation.id)} className="rounded p-1 hover:bg-black/10" aria-label="删除">
                      <Trash2 className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </>
              )}
            </div>
          );
        })}
      </nav>

      <div className="flex items-center gap-3 border-t border-ink-100 px-4 py-4 dark:border-ink-800">
        <div className="flex h-9 w-9 items-center justify-center rounded-full bg-sun-500 font-semibold text-white">
          {user?.username?.[0]?.toUpperCase() || "?"}
        </div>
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-medium">{user?.username}</p>
          <p className="truncate text-xs text-ink-400">{user?.email}</p>
        </div>
        <button className="btn-ghost h-9 w-9 p-0" onClick={logout} aria-label="退出登录" title="退出登录">
          <LogOut className="h-4 w-4" />
        </button>
      </div>
    </aside>
  );
}
