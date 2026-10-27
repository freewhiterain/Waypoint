import clsx from "clsx";

export function LogoMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 64 64" className={className} aria-hidden="true">
      <rect width="64" height="64" rx="16" className="fill-ink-900 dark:fill-white" />
      <path d="M32 12c-8.3 0-15 6.5-15 14.6C17 37.5 32 52 32 52s15-14.5 15-25.4C47 18.5 40.3 12 32 12z" fill="#fb6110" />
      <circle cx="32" cy="27" r="5.5" className="fill-ink-900 dark:fill-white" />
    </svg>
  );
}

export function Logo({ className, compact }: { className?: string; compact?: boolean }) {
  return (
    <div className={clsx("flex items-center gap-2.5", className)}>
      <LogoMark className="h-8 w-8 shrink-0" />
      {!compact && (
        <span className="font-display text-xl font-semibold tracking-tight">
          Way<span className="text-sun-500">Point</span>
        </span>
      )}
    </div>
  );
}
