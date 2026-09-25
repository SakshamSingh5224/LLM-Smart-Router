// No build step, no framework: plain fetch() + ReadableStream SSE parsing
// (EventSource can't send a POST body, which /api/chat/stream needs).

const thread = document.getElementById("thread");
const form = document.getElementById("composer");
const input = document.getElementById("query");
const sendBtn = document.getElementById("send");
const modeSel = document.getElementById("mode");
const explainBox = document.getElementById("explain");
const healthDot = document.getElementById("health");

const API_BASE = ""; // same-origin: gateway serves this file

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
}

function addUserMessage(text) {
  const msg = el("div", "msg user");
  msg.appendChild(el("div", "bubble", text));
  thread.appendChild(msg);
  thread.scrollTop = thread.scrollHeight;
}

function addAssistantMessage() {
  const msg = el("div", "msg assistant");
  const meta = el("div", "meta");
  meta.appendChild(el("span", "tier-badge", "routing…"));
  const bubble = el("div", "bubble", "");
  msg.appendChild(meta);
  msg.appendChild(bubble);
  thread.appendChild(msg);
  thread.scrollTop = thread.scrollHeight;
  return { msg, meta, bubble };
}

function renderMeta(meta, decision, extra) {
  meta.innerHTML = "";
  const badge = el("span", `tier-badge ${decision.tier}`, decision.tier.toUpperCase());
  meta.appendChild(badge);
  meta.appendChild(el("span", "", `p(strong)=${decision.p_strong.toFixed(2)}`));
  meta.appendChild(el("span", "", `mode=${decision.mode}`));
  if (decision.source === "rules_fallback") {
    meta.appendChild(el("span", "", "⚠ fallback router"));
  }
  if (extra) meta.appendChild(el("span", "", extra));
  if (explainBox.checked && decision.reasoning) {
    const r = el("div", "reasoning", decision.reasoning);
    meta.parentElement.appendChild(r);
  }
}

async function health() {
  try {
    const r = await fetch(`${API_BASE}/health`);
    const j = await r.json();
    healthDot.className = `health ${j.status === "ok" ? "ok" : "degraded"}`;
    healthDot.title = JSON.stringify(j, null, 2);
  } catch {
    healthDot.className = "health down";
    healthDot.title = "gateway unreachable";
  }
}
health();
setInterval(health, 15000);

async function streamChat(query, mode, explain) {
  const { meta, bubble } = addAssistantMessage();
  const t0 = performance.now();
  let decision = null;

  let resp;
  try {
    resp = await fetch(`${API_BASE}/api/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, mode, explain }),
    });
  } catch (e) {
    bubble.classList.add("error-bubble");
    bubble.textContent = `Could not reach the gateway: ${e}`;
    return;
  }
  if (!resp.ok || !resp.body) {
    bubble.classList.add("error-bubble");
    bubble.textContent = `Gateway error: HTTP ${resp.status}`;
    return;
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });

    let idx;
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      const raw = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const lines = raw.split("\n");
      let event = "message", data = "";
      for (const line of lines) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data = line.slice(5).trim();
      }
      if (!data) continue;
      let payload;
      try {
        payload = JSON.parse(data);
      } catch {
        continue;
      }

      if (event === "meta") {
        decision = payload.decision;
        renderMeta(meta, decision);
      } else if (event === "delta") {
        bubble.textContent += payload.text;
        thread.scrollTop = thread.scrollHeight;
      } else if (event === "error") {
        bubble.classList.add("error-bubble");
        bubble.textContent += `\n[error: ${payload.message}]`;
      } else if (event === "done") {
        const ms = Math.round(performance.now() - t0);
        const cacheNote = payload.cache_hit ? " · from cache" : "";
        const costNote = payload.est_cost_usd > 0 ? ` · ~$${payload.est_cost_usd.toFixed(5)}` : "";
        renderMeta(meta, decision, `${ms} ms${cacheNote}${costNote}`);
      }
    }
  }
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const query = input.value.trim();
  if (!query) return;
  input.value = "";
  sendBtn.disabled = true;
  addUserMessage(query);
  try {
    await streamChat(query, modeSel.value, explainBox.checked);
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.requestSubmit();
  }
});
