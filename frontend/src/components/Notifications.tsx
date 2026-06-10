import { useEffect } from "react";
import { useDashboardStore } from "../store/dashboardStore";

// Status toasts (bottom-right): view saved (with its S3 path), view loaded, and
// errors. Each auto-dismisses after a few seconds; click to dismiss now.
const KIND_BG: Record<string, string> = {
  success: "bg-green-600",
  error: "bg-red-600",
  info: "bg-gray-800",
};

export function Notifications() {
  const notifications = useDashboardStore((s) => s.notifications);
  const dismiss = useDashboardStore((s) => s.dismissNotification);

  useEffect(() => {
    const timers = notifications.map((n) => setTimeout(() => dismiss(n.id), 6000));
    return () => timers.forEach(clearTimeout);
  }, [notifications, dismiss]);

  if (notifications.length === 0) return null;
  return (
    <div className="fixed bottom-4 right-4 z-50 flex flex-col gap-2 max-w-md">
      {notifications.map((n) => (
        <div
          key={n.id}
          onClick={() => dismiss(n.id)}
          className={`cursor-pointer rounded shadow-lg px-4 py-2 text-base text-white break-words ${KIND_BG[n.kind] ?? "bg-gray-800"}`}
          title="click to dismiss"
        >
          {n.message}
        </div>
      ))}
    </div>
  );
}
