// Streaming client for the agent chat (POST /api/agent/chat, SSE).
// EventSource can't POST, so this fetches and parses the text/event-stream from
// the response body. Caller cancels via the AbortSignal (stop button / unmount).

// The JSON request body POSTed to /api/agent/chat (mirrors the backend's
// AgentChatRequest): the new user message, prior turns, and a snapshot of the
// dashboard the agent can read/drive.
export interface AgentChatPayload {
  message: string;
  history: { role: string; content: string }[];
  model?: string | null;
  dashboard_config?: string | null;
  dashboard_state: Record<string, unknown>;
}

// Callbacks the caller supplies — one per SSE event type the agent emits.
// streamAgentChat invokes the matching one as each frame arrives.
export interface AgentStreamHandlers {
  onToken: (text: string) => void;
  onToolStart: (name: string, input: string) => void;
  onToolEnd: (name: string, output: string) => void;
  onAction: (action: Record<string, unknown>) => void;
  onClarify: (question: string, options: string[]) => void;
  onDone: (elapsedMs: number) => void;
  onError: (message: string) => void;
}

export async function streamAgentChat(
  payload: AgentChatPayload,
  on: AgentStreamHandlers,
  signal: AbortSignal,
): Promise<void> {
  const resp = await fetch("/api/agent/chat", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
  if (!resp.ok || !resp.body) {
    on.onError(`Agent request failed (HTTP ${resp.status})`);
    return;
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let event = "";
  let data = "";

  const dispatch = () => {
    if (!event) return;
    let parsed: Record<string, unknown> = {};
    try {
      parsed = data ? (JSON.parse(data) as Record<string, unknown>) : {};
    } catch {
      /* malformed frame — skip */
    }
    switch (event) {
      case "token":
        on.onToken(String(parsed.text ?? ""));
        break;
      case "tool_start":
        on.onToolStart(String(parsed.name ?? ""), String(parsed.input ?? ""));
        break;
      case "tool_end":
        on.onToolEnd(String(parsed.name ?? ""), String(parsed.output ?? ""));
        break;
      case "action":
        on.onAction(parsed);
        break;
      case "clarify":
        on.onClarify(String(parsed.question ?? ""), Array.isArray(parsed.options) ? parsed.options.map(String) : []);
        break;
      case "done":
        on.onDone(Number(parsed.elapsed_ms ?? 0));
        break;
      case "error":
        on.onError(String(parsed.message ?? "unknown agent error"));
        break;
    }
    event = "";
    data = "";
  };

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let nl: number;
    while ((nl = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, nl).replace(/\r$/, "");
      buffer = buffer.slice(nl + 1);
      if (line === "") dispatch(); // blank line terminates a frame
      else if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) data += line.slice(5).trim();
    }
  }
  dispatch(); // flush a trailing frame without a final blank line
}
