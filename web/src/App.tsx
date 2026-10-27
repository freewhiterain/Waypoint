import { useEffect, useState } from "react";
import clsx from "clsx";
import { CheckCircle2, XCircle } from "lucide-react";
import { useChat } from "@/store/chat";
import { AuthScreen } from "@/components/AuthScreen";
import { Sidebar } from "@/components/Sidebar";
import { ChatView } from "@/components/ChatView";
import { PlanPanel } from "@/components/PlanPanel";

function Toast() {
  const toast = useChat((s) => s.toast);
  if (!toast) return null;
  const Icon = toast.error ? XCircle : CheckCircle2;
  return (
    <div className="pointer-events-none fixed inset-x-0 top-5 z-50 flex justify-center px-4">
      <div
        role="status"
        className={clsx(
          "flex items-center gap-2 rounded-full px-4 py-2 text-sm shadow-float animate-fade-up",
          toast.error ? "bg-red-600 text-white" : "bg-ink-900 text-white dark:bg-white dark:text-ink-900",
        )}
      >
        <Icon className="h-4 w-4" />
        {toast.text}
      </div>
    </div>
  );
}

export default function App() {
  const { token, activePlan, openPlan, loadConversations, notify } = useChat();
  const [sidebarOpen, setSidebarOpen] = useState(false);

  useEffect(() => {
    if (token) loadConversations().catch((error) => notify((error as Error).message, true));
  }, [token, loadConversations, notify]);

  if (!token) {
    return (
      <>
        <AuthScreen />
        <Toast />
      </>
    );
  }

  return (
    <div className="flex h-full overflow-hidden">
      {/* 会话列表：桌面常驻，移动端抽屉 */}
      <div className="hidden w-72 shrink-0 border-r border-ink-100 dark:border-ink-800 lg:block">
        <Sidebar />
      </div>
      {sidebarOpen && (
        <div className="fixed inset-0 z-40 lg:hidden">
          <div className="absolute inset-0 bg-ink-950/40" onClick={() => setSidebarOpen(false)} />
          <div className="absolute inset-y-0 left-0 w-72 shadow-float animate-fade-up">
            <Sidebar onNavigate={() => setSidebarOpen(false)} />
          </div>
        </div>
      )}

      <main className="flex min-w-0 flex-1">
        <ChatView onOpenSidebar={() => setSidebarOpen(true)} />
      </main>

      {/* 行程面板：宽屏在右侧，窄屏全屏覆盖 */}
      {activePlan && (
        <div
          className={clsx(
            "z-30 shrink-0 border-l border-ink-100 dark:border-ink-800",
            "fixed inset-0 xl:static xl:w-[460px] 2xl:w-[520px]",
          )}
        >
          <PlanPanel plan={activePlan} onClose={() => openPlan(null)} />
        </div>
      )}
      <Toast />
    </div>
  );
}
