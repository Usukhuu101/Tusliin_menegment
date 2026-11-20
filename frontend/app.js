// Mongolian Health Chatbot — minimal vanilla JS client for the FastAPI backend.
(() => {
  const $ = (id) => document.getElementById(id);
  const chat = $("chat"), form = $("form"), input = $("input"), send = $("send");
  const RISK = { LOW: "Бага эрсдэл", MODERATE: "Дунд эрсдэл", HIGH: "Өндөр эрсдэл", EMERGENCY: "ЯАРАЛТАЙ" };
  const KIND = { ANSWER: "Хариулт", FOLLOWUP: "Тодруулах асуулт", EMERGENCY: "Яаралтай", FALLBACK: "Мэдээлэл хангалтгүй",
                 GREETING: "", THANKS: "", OUT_OF_SCOPE: "Хамрах хүрээнээс гадуур", CLARIFY: "Тодруулах",
                 RECALL: "Санах ой", NOTED: "Тэмдэглэв" };
  const NO_RISK = ["GREETING", "THANKS", "OUT_OF_SCOPE", "RECALL", "NOTED"];
  const REASONS = [["incorrect", "Буруу мэдээлэл"], ["not_relevant", "Хамааралгүй"],
                   ["did_not_understand", "Асуултыг ойлгоогүй"], ["not_enough", "Мэдээлэл хангалтгүй"], ["other", "Бусад"]];

  const store = {
    get: (k) => { try { return localStorage.getItem(k); } catch { return null; } },
    set: (k, v) => { try { localStorage.setItem(k, v); } catch {} },
    del: (k) => { try { localStorage.removeItem(k); } catch {} },
  };
  let sessionId = store.get("hc_session");
  let showAgents = store.get("hc_agents") === "1";
  const STEP_ICON = { ok: "✅", alert: "🚨", skipped: "⏭️", fallback: "↩️" };
  const agentsBtn = $("agentsToggle");
  const paintAgentsBtn = () => agentsBtn.classList.toggle("on", showAgents);
  paintAgentsBtn();
  agentsBtn.addEventListener("click", () => {
    showAgents = !showAgents;
    store.set("hc_agents", showAgents ? "1" : "0");
    paintAgentsBtn();
    document.querySelectorAll("details.agents").forEach((d) => { d.open = showAgents; });
  });

  // ---------- tiny, safe markdown (escape first, then format) ----------
  const esc = (s) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  function inline(s) {
    return esc(s)
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|\s)_(.+?)_(?=\s|$|[.,])/g, "$1<em>$2</em>")
      .replace(/https?:\/\/[^\s<]+/g, (u) => {
        // keep "(…)" that belongs to the URL (WHO links), drop unbalanced trailing ")"
        let tail = "";
        const count = (s, c) => s.split(c).length - 1;
        while (/[.,;:]$/.test(u) || (u.endsWith(")") && count(u, "(") < count(u, ")"))) { tail = u.slice(-1) + tail; u = u.slice(0, -1); }
        return `<a href="${u}" target="_blank" rel="noopener">${u}</a>${tail}`;
      });
  }
  function md(text) {
    const out = []; let list = null;
    const close = () => { if (list) { out.push(`</${list}>`); list = null; } };
    for (const raw of text.split("\n")) {
      const line = raw.trimEnd();
      let m;
      if ((m = line.match(/^\s*[-*]\s+(.*)/))) { if (list !== "ul") { close(); out.push("<ul>"); list = "ul"; } out.push(`<li>${inline(m[1])}</li>`); }
      else if ((m = line.match(/^\s*\d+\.\s+(.*)/))) { if (list !== "ol") { close(); out.push("<ol>"); list = "ol"; } out.push(`<li>${inline(m[1])}</li>`); }
      else if (!line.trim()) { close(); }
      else { close(); out.push(`<p>${inline(line)}</p>`); }
    }
    close();
    return out.join("");
  }

  // ---------- rendering ----------
  function scroll() { chat.scrollTop = chat.scrollHeight; }
  function addUser(text) {
    $("welcome")?.remove();
    const el = document.createElement("div");
    el.className = "msg user";
    el.innerHTML = `<div class="bubble"></div>`;
    el.firstChild.textContent = text;
    chat.appendChild(el); scroll();
  }
  function addTyping() {
    const el = document.createElement("div");
    el.className = "msg bot typing";
    el.innerHTML = `<div class="bubble"><span class="dots">Бодож байна</span></div>`;
    chat.appendChild(el); scroll();
    return el;
  }
  function addBot(r) {
    const el = document.createElement("div");
    el.className = `msg bot ${r.message_type}`;
    const showRisk = !NO_RISK.includes(r.message_type);
    let html = `<div class="bubble">`;
    if (showRisk || KIND[r.message_type]) {
      html += `<div class="meta">${showRisk ? `<span class="badge ${r.risk_level}">${RISK[r.risk_level]}</span>` : ""}`
            + `<span class="kind">${KIND[r.message_type] || ""}</span></div>`;
    }
    if (r.message_type === "EMERGENCY") html += `<a class="call103" href="tel:103">📞 103 руу залгах</a>`;
    html += md(r.reply);
    if (r.content_notice) html += `<div class="notice">⚠️ ${esc(r.content_notice)}</div>`;
    if (r.sources?.length) {
      html += `<details class="sources"><summary>Эх сурвалж (${r.sources.length})</summary><ul>` +
        r.sources.map((s) => {
          const link = /^https?:/.test(s.url) ? `<a href="${esc(s.url)}" target="_blank" rel="noopener">${esc(s.source)}</a>` : esc(s.source);
          return `<li><strong>${esc(s.doc_id)}</strong> ${esc(s.title)} — ${link}<span class="tier">Tier ${s.tier}</span> <span class="tier">${esc(s.status)}</span></li>`;
        }).join("") + `</ul></details>`;
    }
    if (r.agents?.length) {
      const ran = r.agents.filter((a) => a.status !== "skipped");
      const total = ran.reduce((s, a) => s + a.ms, 0);
      const mode = r.mode?.startsWith("llm") ? "LLM" : "офлайн";
      html += `<details class="agents"${showAgents ? " open" : ""}><summary>🧠 Агентуудын ажиллагаа · ${ran.length} алхам · ${total.toFixed(1)} мс · ${mode}</summary><ol>` +
        r.agents.map((a) => `<li class="${a.status}"><span class="ico">${STEP_ICON[a.status] || "•"}</span>` +
          `<div><strong>${esc(a.title)}</strong>${a.status !== "skipped" ? ` <span class="ms">${a.ms} мс</span>` : " <span class=\"ms\">алгассан</span>"}` +
          `<div class="adetail">${esc(a.detail)}</div></div></li>`).join("") + `</ol></details>`;
    }
    if (!["GREETING", "THANKS"].includes(r.message_type)) {
      html += `<div class="feedback" data-turn="${r.turn_id}">Энэ хариулт тус болсон уу?
        <button type="button" data-v="1">👍 Тийм</button><button type="button" data-v="0">👎 Үгүй</button></div>`;
    }
    html += `</div>`;
    el.innerHTML = html;
    chat.appendChild(el); scroll();
    el.querySelector(".feedback")?.addEventListener("click", onFeedback);
  }

  // ---------- feedback ----------
  async function postFeedback(turnId, helpful, reason, comment) {
    await fetch("/api/feedback", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, turn_id: Number(turnId), helpful, reason, comment }) });
  }
  async function onFeedback(e) {
    const btn = e.target.closest("button[data-v]");
    if (!btn) return;
    const box = e.currentTarget, turn = box.dataset.turn;
    if (btn.dataset.v === "1") {
      await postFeedback(turn, true, null, null);
      box.textContent = "Баярлалаа! 🙏";
      return;
    }
    box.innerHTML = `Юу нь буруу байсан бэ?
      <select>${REASONS.map(([v, l]) => `<option value="${v}">${l}</option>`).join("")}</select>
      <input type="text" maxlength="500" placeholder="Нэмэлт тайлбар (заавал биш)">
      <button type="button" class="submit">Илгээх</button>`;
    box.querySelector(".submit").addEventListener("click", async () => {
      await postFeedback(turn, false, box.querySelector("select").value, box.querySelector("input").value || null);
      box.textContent = "Санал хүсэлтийг хүлээж авлаа. Хянаж сайжруулна. 🙏";
    }, { once: true });
  }

  // ---------- send ----------
  async function sendMessage(text) {
    text = text.trim();
    if (!text) return;
    addUser(text);
    input.value = ""; autosize();
    send.disabled = true;
    const typing = addTyping();
    try {
      const res = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, session_id: sessionId }) });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      sessionId = data.session_id; store.set("hc_session", sessionId);
      typing.remove(); addBot(data);
    } catch (err) {
      typing.remove();
      addBot({ message_type: "FALLBACK", risk_level: "LOW", turn_id: 0, sources: [],
        reply: "⚠️ Сервертэй холбогдож чадсангүй. Дахин оролдоно уу. **Яаралтай үед 103 дугаарт залгаарай.**" });
    } finally {
      send.disabled = false; input.focus();
    }
  }

  function autosize() { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 160) + "px"; }
  input.addEventListener("input", autosize);
  input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); form.requestSubmit(); } });
  form.addEventListener("submit", (e) => { e.preventDefault(); sendMessage(input.value); });
  $("chips")?.addEventListener("click", (e) => { if (e.target.tagName === "BUTTON") sendMessage(e.target.textContent); });
  $("newChat").addEventListener("click", () => { store.del("hc_session"); sessionId = null; location.reload(); });
  $("deleteData").addEventListener("click", async () => {
    if (!sessionId) { alert("Устгах яриа алга."); return; }
    if (!confirm("Энэ ярианы бүх мэдээллийг серверээс устгах уу?")) return;
    await fetch(`/api/session/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
    store.del("hc_session"); sessionId = null; location.reload();
  });

  // ---------- restore this chat after a page reload (the server remembers it) ----------
  async function restoreHistory() {
    if (!sessionId) return;
    try {
      const res = await fetch(`/api/session/${encodeURIComponent(sessionId)}/history`);
      if (!res.ok) return;
      const { turns } = await res.json();
      if (!turns?.length) return;
      for (const t of turns) {
        addUser(t.user);
        addBot({ reply: t.reply, message_type: t.message_type, risk_level: t.risk_level, turn_id: t.turn_id, sources: [] });
      }
      const note = document.createElement("div");
      note.className = "restored";
      note.textContent = "↩️ Өмнөх яриа сэргээгдлээ — таны хэлсэн мэдээллийг санаж байна. «＋ Шинэ яриа» дарвал шинээр эхэлнэ.";
      chat.appendChild(note); scroll();
    } catch {}
  }
  restoreHistory();

  fetch("/api/health").then((r) => r.json()).then((h) => {
    $("mode").textContent = h.mode.startsWith("llm") ? "🤖 LLM горим" : "📘 Офлайн горим (дүрэм + мэдлэгийн сан)";
  }).catch(() => { $("mode").textContent = "холболтгүй"; });

  if (store.get("hc_consent") !== "1") {
    const dlg = $("consent");
    dlg.showModal?.();
    dlg.addEventListener("close", () => store.set("hc_consent", "1"));
  }
})();
