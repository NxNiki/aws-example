import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { CHAT_MODELS, useDashboardStore } from "../../store/dashboardStore";
import type { ChatEvent } from "../../store/dashboardStore";

// Agent chat panel (Phase 3): docked to the right edge, streams the agent's
// tokens live, shows tool calls and dashboard actions inline, and renders
// clarify questions as quick-reply chips. The agent drives the dashboard via
// the action dispatcher — see store/actionDispatcher.ts.
function EventChip({ ev }: { ev: ChatEvent }) {
  const icon = ev.kind === "action" ? "⚡" : "🔧";
  const status = ev.kind === "action" ? ev.detail : ev.done ? "✓" : "…";
  const failed = ev.kind === "action" && ev.detail === "failed";
  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-sm mr-1 mb-1 ${
        failed ? "border-red-300 bg-red-50 text-red-700" : "border-gray-300 bg-gray-100 text-gray-700"
      }`}
      title={ev.detail}
    >
      {icon} {ev.name} <span className="text-gray-400">{status}</span>
    </span>
  );
}

export function ChatPanel() {
  const chat = useDashboardStore((s) => s.chat);
  const { toggleChat, setChatWidth, setChatModel, clearChat, stopChat, sendChat } = useDashboardStore();
  const [draft, setDraft] = useState("");
  const scrollRef = useRef<HTMLDivElement>(null);

  // Keep the newest message in view while streaming.
  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [chat.messages]);

  if (!chat.open) return null;

  const submit = () => {
    const text = draft.trim();
    if (!text || chat.streaming) return;
    setDraft("");
    void sendChat(text);
  };

  // Drag the left edge to resize; width is store state so the app layout's
  // content padding follows along (charts re-flow via their ResizeObserver).
  const startResize = (e: React.MouseEvent) => {
    e.preventDefault();
    const onMove = (ev: MouseEvent) => setChatWidth(window.innerWidth - ev.clientX);
    const onUp = () => {
      document.removeEventListener("mousemove", onMove);
      document.removeEventListener("mouseup", onUp);
      document.body.style.cursor = "";
      document.body.style.userSelect = "";
    };
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
  };

  return (
    <div
      className="fixed right-0 top-14 bottom-0 z-40 flex flex-col border-l bg-white shadow-lg"
      style={{ width: chat.width, maxWidth: "85vw" }}
    >
      <div
        className="absolute left-0 top-0 bottom-0 w-1.5 cursor-col-resize hover:bg-blue-300/60"
        onMouseDown={startResize}
        title="drag to resize"
      />
      <div className="flex items-center gap-2 border-b px-3 py-2">
        <span className="font-semibold text-gray-700">🤖 AI Assistant</span>
        <select
          className="ml-auto rounded border px-1 py-0.5 text-sm"
          value={chat.model}
          onChange={(e) => setChatModel(e.target.value)}
        >
          {CHAT_MODELS.map((m) => (
            <option key={m} value={m}>
              {m.split(":")[1]}
            </option>
          ))}
        </select>
        <button className="rounded border px-2 py-0.5 text-sm text-gray-600 hover:bg-gray-50" onClick={clearChat}>
          clear
        </button>
        <button className="rounded border px-2 py-0.5 text-sm text-gray-600 hover:bg-gray-50" onClick={toggleChat}>
          ✕
        </button>
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto px-3 py-2">
        {chat.messages.length === 0 && (
          <div className="mt-6 text-center text-sm text-gray-400">
            Ask about metrics, or tell me what to show — e.g.
            <br />
            <em>“show num_active_users in ss01 by date”</em>
          </div>
        )}
        {chat.messages.map((m, i) => (
          <div key={i} className={`mb-3 ${m.role === "user" ? "text-right" : ""}`}>
            {m.role === "user" ? (
              <div className="inline-block max-w-[90%] rounded-lg bg-blue-600 px-3 py-2 text-left text-white">
                {m.content}
              </div>
            ) : (
              <div className="max-w-[95%]">
                {m.events.length > 0 && (
                  <div className="mb-1">
                    {m.events.map((ev, j) => (
                      <EventChip key={j} ev={ev} />
                    ))}
                  </div>
                )}
                {m.content ? (
                  <div className="prose-sm rounded-lg bg-gray-100 px-3 py-2 [&_code]:text-[13px] [&_p]:my-1">
                    <ReactMarkdown>{m.content}</ReactMarkdown>
                  </div>
                ) : (
                  chat.streaming && i === chat.messages.length - 1 && (
                    <div className="rounded-lg bg-gray-100 px-3 py-2 text-gray-400">…</div>
                  )
                )}
              </div>
            )}
          </div>
        ))}

        {chat.pendingClarify && !chat.streaming && (
          <div className="mb-3">
            <div className="mb-1 flex flex-wrap gap-2">
              {chat.pendingClarify.options.map((opt) => (
                <button
                  key={opt}
                  className="rounded-full border border-blue-400 bg-blue-50 px-3 py-1 text-sm text-blue-700 hover:bg-blue-100"
                  onClick={() => void sendChat(opt)}
                >
                  {opt}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      <div className="border-t p-2">
        <div className="flex gap-2">
          <textarea
            className="max-h-28 min-h-[2.5rem] flex-1 resize-y rounded border px-2 py-1"
            placeholder="Ask the agent… (Enter to send, Shift+Enter for newline)"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              // Ignore Enter while an IME composition is active (Chinese/Japanese
              // input): that Enter commits the composition, not the message.
              // keyCode 229 covers a Safari quirk where compositionend fires
              // before the keydown.
              if (e.nativeEvent.isComposing || e.keyCode === 229) return;
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit();
              }
            }}
          />
          {chat.streaming ? (
            <button className="rounded bg-gray-500 px-3 py-1 text-white" onClick={stopChat}>
              stop
            </button>
          ) : (
            <button
              className="rounded bg-blue-600 px-3 py-1 text-white disabled:opacity-40"
              disabled={!draft.trim()}
              onClick={submit}
            >
              send
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
