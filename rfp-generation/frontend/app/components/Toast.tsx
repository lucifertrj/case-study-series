"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  ReactNode,
} from "react";

type ToastKind = "success" | "error" | "info";
interface ToastItem {
  id: number;
  kind: ToastKind;
  message: string;
}

interface ToastCtx {
  push: (kind: ToastKind, message: string) => void;
  success: (m: string) => void;
  error: (m: string) => void;
  info: (m: string) => void;
}

const Ctx = createContext<ToastCtx | null>(null);

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  const remove = useCallback((id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  }, []);

  const push = useCallback(
    (kind: ToastKind, message: string) => {
      const id = Date.now() + Math.random();
      setToasts((prev) => [...prev, { id, kind, message }]);
      setTimeout(() => remove(id), 4000);
    },
    [remove]
  );

  const value = useMemo<ToastCtx>(
    () => ({
      push,
      success: (m) => push("success", m),
      error: (m) => push("error", m),
      info: (m) => push("info", m),
    }),
    [push]
  );

  return (
    <Ctx.Provider value={value}>
      {children}
      <ToastContainer toasts={toasts} onDismiss={remove} />
    </Ctx.Provider>
  );
}

export function useToast(): ToastCtx {
  const c = useContext(Ctx);
  if (!c) throw new Error("useToast must be used inside <ToastProvider>");
  return c;
}

function ToastContainer({
  toasts,
  onDismiss,
}: {
  toasts: ToastItem[];
  onDismiss: (id: number) => void;
}) {
  return (
    <div className="fixed top-4 right-4 z-50 flex flex-col gap-2 max-w-sm">
      {toasts.map((t) => (
        <ToastRow key={t.id} toast={t} onDismiss={() => onDismiss(t.id)} />
      ))}
    </div>
  );
}

function ToastRow({
  toast,
  onDismiss,
}: {
  toast: ToastItem;
  onDismiss: () => void;
}) {
  const [leaving, setLeaving] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setLeaving(true), 3600);
    return () => clearTimeout(t);
  }, []);

  const colors: Record<ToastKind, string> = {
    success: "bg-emerald-600 text-white",
    error: "bg-rose-600 text-white",
    info: "bg-sky-600 text-white",
  };

  return (
    <div
      className={`shadow-lg rounded-md px-4 py-3 text-sm ${colors[toast.kind]} ${
        leaving ? "opacity-0 translate-y-1" : "opacity-100"
      } transition-all duration-300 flex items-start justify-between gap-3`}
    >
      <span className="flex-1">{toast.message}</span>
      <button
        onClick={onDismiss}
        className="text-white/70 hover:text-white leading-none"
        aria-label="Dismiss"
      >
        ×
      </button>
    </div>
  );
}
