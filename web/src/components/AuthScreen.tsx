import { useState, type FormEvent } from "react";
import { ArrowRight, Compass, Map, Sparkles } from "lucide-react";
import clsx from "clsx";
import { useChat } from "@/store/chat";
import { Logo, LogoMark } from "./Logo";

const FEATURES = [
  { icon: Sparkles, title: "一句话开始", text: "说出想去哪、玩几天，缺的信息只问一次" },
  { icon: Compass, title: "五个子 Agent 并行研究", text: "景点、住宿、美食、天气、交通同时查" },
  { icon: Map, title: "按真实路程排日程", text: "算法排顺序，开放时间和路上耗时都会检查" },
];

const ROUTE_POINTS: [number, number][] = [
  [40, 720],
  [260, 420],
  [380, 200],
  [560, 60],
];

export function AuthScreen() {
  const { login, register } = useChat();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [form, setForm] = useState({ username: "", email: "", password: "" });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError("");
    setLoading(true);
    try {
      if (mode === "login") await login(form.username.trim(), form.password);
      else await register(form.username.trim(), form.email.trim(), form.password);
    } catch (err) {
      setError((err as Error).message);
    } finally {
      setLoading(false);
    }
  }

  const field = (key: keyof typeof form) => ({
    value: form[key],
    onChange: (e: React.ChangeEvent<HTMLInputElement>) => setForm({ ...form, [key]: e.target.value }),
  });

  return (
    <div className="grid min-h-full lg:grid-cols-[1.1fr_1fr]">
      <section className="relative hidden overflow-hidden bg-ink-950 p-12 text-white lg:flex lg:flex-col">
        <div className="pointer-events-none absolute inset-0 opacity-70">
          <div className="absolute -left-20 top-24 h-80 w-80 rounded-full bg-sun-500/30 blur-3xl" />
          <div className="absolute bottom-0 right-0 h-96 w-96 rounded-full bg-moss-500/20 blur-3xl" />
          <svg className="absolute inset-0 h-full w-full" viewBox="0 0 600 800" preserveAspectRatio="none">
            <path
              d="M40 720 C 160 600, 120 480, 260 420 S 420 300, 380 200 S 480 80, 560 60"
              fill="none"
              stroke="rgba(255,255,255,0.25)"
              strokeWidth="2"
              strokeDasharray="6 10"
            />
            {ROUTE_POINTS.map(([x, y]) => (
              <circle key={`${x}-${y}`} cx={x} cy={y} r="6" fill="#fb6110" />
            ))}
          </svg>
        </div>
        <div className="relative flex items-center gap-2.5">
          <svg viewBox="0 0 64 64" className="h-8 w-8" aria-hidden="true">
            <rect width="64" height="64" rx="16" fill="#fff" />
            <path d="M32 12c-8.3 0-15 6.5-15 14.6C17 37.5 32 52 32 52s15-14.5 15-25.4C47 18.5 40.3 12 32 12z" fill="#fb6110" />
            <circle cx="32" cy="27" r="5.5" fill="#fff" />
          </svg>
          <span className="font-display text-xl font-semibold">
            Way<span className="text-sun-400">Point</span>
          </span>
        </div>
        <div className="relative mt-auto max-w-lg">
          <h1 className="font-display text-5xl font-semibold leading-tight">
            每一个落脚点，
            <br />
            都有依据。
          </h1>
          <p className="mt-5 text-lg text-ink-200">
            WayPoint 是一个基于 LangGraph 的多 Agent 旅行规划助手。真实地点数据、真实天气和路程，不编造班次和价格。
          </p>
          <ul className="mt-10 space-y-5">
            {FEATURES.map(({ icon: Icon, title, text }) => (
              <li key={title} className="flex gap-4">
                <span className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-white/10">
                  <Icon className="h-5 w-5 text-sun-300" />
                </span>
                <div>
                  <p className="font-medium">{title}</p>
                  <p className="text-sm text-ink-300">{text}</p>
                </div>
              </li>
            ))}
          </ul>
        </div>
      </section>

      <section className="flex items-center justify-center px-4 py-12 sm:px-8">
        <div className="w-full max-w-sm animate-fade-up">
          <Logo className="mb-10 lg:hidden" />
          <LogoMark className="mb-6 hidden h-10 w-10 lg:block" />
          <h2 className="font-display text-3xl font-semibold">{mode === "login" ? "欢迎回来" : "创建账号"}</h2>
          <p className="mt-2 text-sm text-ink-500 dark:text-ink-300">
            {mode === "login" ? "登录后继续规划你的下一段旅程。" : "注册后即可保存会话和行程。"}
          </p>

          <div className="mt-8 grid grid-cols-2 rounded-xl bg-ink-100 p-1 text-sm dark:bg-ink-800">
            {(["login", "register"] as const).map((value) => (
              <button
                key={value}
                type="button"
                onClick={() => {
                  setMode(value);
                  setError("");
                }}
                className={clsx(
                  "rounded-lg py-2 font-medium transition",
                  mode === value ? "bg-white shadow-sm dark:bg-ink-950" : "text-ink-500 dark:text-ink-300",
                )}
              >
                {value === "login" ? "登录" : "注册"}
              </button>
            ))}
          </div>

          <form onSubmit={submit} className="mt-6 space-y-4">
            <label className="block">
              <span className="mb-1.5 block text-sm font-medium">用户名</span>
              <input className="input" autoComplete="username" required {...field("username")} />
            </label>
            {mode === "register" && (
              <label className="block">
                <span className="mb-1.5 block text-sm font-medium">邮箱</span>
                <input className="input" type="email" autoComplete="email" required {...field("email")} />
              </label>
            )}
            <label className="block">
              <span className="mb-1.5 block text-sm font-medium">密码</span>
              <input
                className="input"
                type="password"
                minLength={6}
                autoComplete={mode === "login" ? "current-password" : "new-password"}
                required
                {...field("password")}
              />
            </label>
            {error && (
              <p role="alert" className="rounded-lg bg-red-50 px-3 py-2 text-sm text-red-600 dark:bg-red-500/10 dark:text-red-300">
                {error}
              </p>
            )}
            <button className="btn-accent w-full py-3" disabled={loading}>
              {loading ? "请稍候…" : mode === "login" ? "登录" : "注册并开始"}
              {!loading && <ArrowRight className="h-4 w-4" />}
            </button>
          </form>
        </div>
      </section>
    </div>
  );
}
