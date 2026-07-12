"use strict";
/* Audicop frontend — vanilla JS, offline, no build. Talks to the FastAPI
   backend over JSON + SSE on the same origin (localhost). */

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const LABELS = {
  language: { auto: "Auto-detectar", es: "Español", en: "Inglés", pt: "Portugués", fr: "Francés", it: "Italiano", de: "Alemán" },
  task: { transcribe: "Transcribir (mismo idioma)", translate: "Traducir a inglés" },
  device: { cuda: "GPU (NVIDIA)", cpu: "CPU" },
  model: { tiny: "Tiny", base: "Base", small: "Small", medium: "Medium", "large-v3-turbo": "Large v3 Turbo (rápido)", "large-v3": "Large v3 (máxima calidad)" },
};

// App state (kept in memory only)
const state = {
  segments: [],
  meta: { language: "es", duration: 0, prob: 0 },
  baseFilename: "transcripcion",
  chosen: null, // { kind: "file"|"path", file?, path? }
  ai: { local: null },
  chatHistory: [],
  playerUrl: null,
  meetingId: null, // id in the local meetings library (SQLite, on-disk)
  lastAnswer: "", // last AI note, for copy/download/save
};

function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => (t.hidden = true), 2200);
}

function fillSelect(sel, values, labelMap, selected) {
  sel.innerHTML = "";
  for (const v of values) {
    const opt = document.createElement("option");
    opt.value = v;
    opt.textContent = labelMap && labelMap[v] ? labelMap[v] : v;
    if (v === selected) opt.selected = true;
    sel.appendChild(opt);
  }
}

function setStepEnabled(id, enabled) {
  $(id).classList.toggle("is-disabled", !enabled);
}

// ---------------------------------------------------------------------------
// Client-side formatting (mirrors backend services/formatting.py)
// ---------------------------------------------------------------------------
function fmtTs(seconds, withMillis, comma) {
  if (seconds < 0) seconds = 0;
  const totalMs = Math.round(seconds * 1000);
  const h = Math.floor(totalMs / 3600000);
  const m = Math.floor((totalMs % 3600000) / 60000);
  const s = Math.floor((totalMs % 60000) / 1000);
  const ms = totalMs % 1000;
  const pad = (n, w = 2) => String(n).padStart(w, "0");
  if (withMillis) {
    const sep = comma ? "," : ".";
    return `${pad(h)}:${pad(m)}:${pad(s)}${sep}${pad(ms, 3)}`;
  }
  return h > 0 ? `${pad(h)}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`;
}
const nonEmpty = () => state.segments.filter((s) => s.text.trim());
function toPlain() { return nonEmpty().map((s) => s.text.trim()).join("\n"); }
function toTimestamped() { return nonEmpty().map((s) => `[${fmtTs(s.start)}] ${s.text.trim()}`).join("\n"); }
function toSRT() {
  return nonEmpty().map((s, i) =>
    `${i + 1}\n${fmtTs(s.start, true, true)} --> ${fmtTs(s.end, true, true)}\n${s.text.trim()}`
  ).join("\n\n") + "\n";
}
function toVTT() {
  return "WEBVTT\n\n" + nonEmpty().map((s) =>
    `${fmtTs(s.start, true, false)} --> ${fmtTs(s.end, true, false)}\n${s.text.trim()}`
  ).join("\n\n") + "\n";
}

function download(text, ext, mime) {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${state.baseFilename}.${ext}`;
  a.click();
  URL.revokeObjectURL(url);
}

function fmtDuration(sec) {
  sec = Math.max(0, Math.round(sec));
  if (sec < 60) return `${sec} s`;
  const m = Math.floor(sec / 60);
  if (m < 60) return `${m} min`;
  const h = Math.floor(m / 60);
  return `${h} h ${m % 60} min`;
}

// ---------------------------------------------------------------------------
// Markdown + timestamp rendering (safe: escape first, then add our own tags)
// ---------------------------------------------------------------------------
function tsToSec(t) {
  const p = t.split(":").map(Number);
  return p.length === 3 ? p[0] * 3600 + p[1] * 60 + p[2] : p[0] * 60 + p[1];
}
function escapeHtml(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function escapeAttr(s) {
  return escapeHtml(s).replace(/"/g, "&quot;");
}
function fragmentAt(sec) {
  let best = null;
  for (const s of state.segments) {
    if (sec >= s.start && sec <= s.end) return s.text.trim();
    if (s.start <= sec) best = s;
  }
  return best ? best.text.trim() : "";
}
// Matches [MM:SS], [HH:MM:SS] and ranges like [00:58–01:28] (en dash, em dash
// or hyphen, with optional spaces). In a range each endpoint is its own
// clickable link — same behavior as YouTube/Gemini chapter ranges.
const TS_RE = /\[(\d{1,2}:\d{2}(?::\d{2})?)(?:\s*[–—-]\s*(\d{1,2}:\d{2}(?::\d{2})?))?\]/g;
function tsLink(t) {
  const sec = tsToSec(t);
  return `<span class="ts" data-sec="${sec}" title="${escapeAttr(fragmentAt(sec))}">${t}</span>`;
}
function inlineMd(s) {
  s = escapeHtml(s).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  return s.replace(TS_RE, (_m, t1, t2) =>
    t2 ? `[${tsLink(t1)}–${tsLink(t2)}]` : `[${tsLink(t1)}]`
  );
}
function renderMarkdown(text) {
  const out = [];
  for (const raw of text.split("\n")) {
    if (!raw.trim()) { out.push('<div class="md-gap"></div>'); continue; }
    const indent = raw.match(/^\s*/)[0].length;
    let line = raw.trim();
    let bullet = "";
    let m;
    if (/^[-*]\s+/.test(line)) { bullet = "•"; line = line.replace(/^[-*]\s+/, ""); }
    else if ((m = line.match(/^(\d+)\.\s+/))) { bullet = m[1] + "."; line = line.replace(/^\d+\.\s+/, ""); }
    const pad = Math.min(3, Math.floor(indent / 2)) * 16;
    const b = bullet ? `<span class="md-bullet">${bullet}</span> ` : "";
    out.push(`<div class="md-line" style="padding-left:${pad}px">${b}${inlineMd(line)}</div>`);
  }
  return out.join("");
}

// ---------------------------------------------------------------------------
// Tabs (generic)
// ---------------------------------------------------------------------------
function wireTabs(tabAttr, panelAttr) {
  document.querySelectorAll(`[${tabAttr}]`).forEach((tab) => {
    tab.addEventListener("click", () => {
      const group = tab.parentElement;
      group.querySelectorAll(`[${tabAttr}]`).forEach((t) => t.classList.remove("is-active"));
      tab.classList.add("is-active");
      const key = tab.getAttribute(tabAttr);
      const scope = group.parentElement;
      scope.querySelectorAll(`[${panelAttr}]`).forEach((p) => {
        p.classList.toggle("is-active", p.getAttribute(panelAttr) === key);
      });
    });
  });
}

// ---------------------------------------------------------------------------
// Bootstrap: load hardware + options
// ---------------------------------------------------------------------------
async function loadHardware() {
  let data;
  try {
    const r = await fetch("/api/hardware");
    data = await r.json();
  } catch (e) {
    $("status-banner").className = "banner banner--cpu";
    $("status-banner").innerHTML = '<span class="banner__icon">⚠️</span><span class="banner__text">No se pudo contactar al servidor.</span>';
    return;
  }
  const { hardware: hw, recommendation: rec, capacity: cap, options: opt, ai } = data;

  // Banner
  const banner = $("status-banner");
  const dev = rec.device === "cuda" && hw.gpu_name
    ? `GPU <strong>${hw.gpu_name.replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")}</strong>`
    : rec.device === "cuda" ? "<strong>GPU NVIDIA</strong>" : `<strong>CPU</strong> (${hw.cpu_cores_physical} núcleos)`;
  banner.className = "banner " + (rec.device === "cuda" ? "banner--ok" : "banner--cpu");
  banner.innerHTML = `<span class="banner__icon">✅</span><span class="banner__text">Listo. Transcribiré en ${dev} con el modelo <strong>${LABELS.model[rec.model_size] || rec.model_size}</strong> · ~<strong>${cap.minutes_per_hour} min</strong> por hora de audio · hasta <strong>${cap.max_duration_hours} h</strong>.</span>`;

  // Details
  $("hw-details").hidden = false;
  const vram = hw.gpu_vram_free_gb != null ? `${hw.gpu_vram_free_gb} / ${hw.gpu_vram_total_gb} GB libres` : "—";
  $("hw-details-body").innerHTML = `
    <p><strong>Tu equipo:</strong> ${hw.os_name} · CPU ${hw.cpu_cores_physical}/${hw.cpu_cores_logical} ·
    RAM ${hw.ram_available_gb}/${hw.ram_total_gb} GB libres ·
    ${hw.has_cuda ? `GPU ${hw.gpu_name} (${vram})` : "sin GPU CUDA"}</p>
    <p><strong>Modelo elegido:</strong> <code>${rec.model_size}</code> · <code>${rec.compute_type}</code> · <code>${rec.device}</code></p>
    <p class="muted">${rec.rationale}</p>`;
  $("upload-hint").textContent = `audio o vídeo · hasta ${Math.round(cap.max_upload_mb / 1000)} GB`;

  // Option selects
  fillSelect($("opt-language"), opt.languages, LABELS.language, "auto");
  fillSelect($("opt-task"), opt.tasks, LABELS.task, "transcribe");
  fillSelect($("opt-model"), opt.model_sizes, LABELS.model, rec.model_size);
  fillSelect($("opt-compute"), opt.compute_types, null, rec.compute_type);
  fillSelect($("opt-device"), ["cuda", "cpu"], LABELS.device, rec.device);

  // AI options — 100% local. Just show the recommended on-device model.
  state.ai = { local: ai.local };
  renderLocalNote();

  // Quick-action buttons (prompts come from the server's editable .md files)
  const qa = $("quick-actions");
  qa.innerHTML = "";
  for (const action of ai.quick_actions || []) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "quick-btn";
    btn.textContent = action.label;
    // Show the friendly label in the chat; the full prompt is internal.
    btn.addEventListener("click", () => sendChat(action.prompt, action.label));
    qa.appendChild(btn);
  }
}

function renderLocalNote() {
  const loc = (state.ai && state.ai.local) || {};
  if (loc.available) {
    const gb = Math.round((loc.download_size_mb || 0) / 100) / 10;
    $("ai-local-model").textContent = loc.cached
      ? loc.label
      : `${loc.label} (1.ª vez descarga ~${gb} GB)`;
  } else {
    $("ai-local-model").textContent =
      loc.rationale || "No hay un modelo local para este equipo; cierra apps y recarga.";
  }
}

// ---------------------------------------------------------------------------
// Theme (light ⟷ dark). CSS already honors prefers-color-scheme when no
// explicit choice exists, so first paint is correct; here we just reflect the
// effective theme and persist the user's toggle.
// ---------------------------------------------------------------------------
function currentTheme() {
  const saved = localStorage.getItem("audicop-theme");
  if (saved === "light" || saved === "dark") return saved;
  return matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}
function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const btn = $("theme-toggle");
  btn.textContent = theme === "dark" ? "☀️" : "🌙"; // icon = the mode you'd switch to
  btn.title = theme === "dark" ? "Cambiar a modo claro" : "Cambiar a modo oscuro";
}
function toggleTheme() {
  const next = currentTheme() === "dark" ? "light" : "dark";
  localStorage.setItem("audicop-theme", next);
  applyTheme(next);
}

// ---------------------------------------------------------------------------
// Views (new transcription ⟷ meeting) + local meetings library
// ---------------------------------------------------------------------------
function showView(name) {
  $("view-new").hidden = name !== "new";
  $("view-meeting").hidden = name !== "meeting";
}

function selectVtab(key) {
  document.querySelectorAll("[data-vtab]").forEach((b) =>
    b.classList.toggle("is-active", b.getAttribute("data-vtab") === key)
  );
  document.querySelectorAll("[data-vpanel]").forEach((p) =>
    p.classList.toggle("is-active", p.getAttribute("data-vpanel") === key)
  );
}

function fmtMeetingDate(iso) {
  const d = new Date(iso);
  const today = new Date();
  const yest = new Date(today);
  yest.setDate(today.getDate() - 1);
  const same = (a, b) => a.toDateString() === b.toDateString();
  if (same(d, today)) return "Hoy";
  if (same(d, yest)) return "Ayer";
  return d.toLocaleDateString("es", { day: "2-digit", month: "short", year: "numeric" });
}

async function loadMeetings(query) {
  let data;
  try {
    const r = await fetch(`/api/meetings?q=${encodeURIComponent(query || "")}`);
    data = await r.json();
  } catch (e) { return; }
  const list = $("meetings-list");
  list.innerHTML = "";
  const meetings = data.meetings || [];
  if (meetings.length === 0) {
    const p = document.createElement("p");
    p.className = "meetings__empty";
    p.textContent = query ? "Sin resultados." : "Tus transcripciones aparecerán aquí.";
    list.appendChild(p);
    return;
  }
  let lastGroup = "";
  for (const m of meetings) {
    const group = fmtMeetingDate(m.created_at);
    if (group !== lastGroup) {
      const g = document.createElement("div");
      g.className = "meetings__group";
      g.textContent = group;
      list.appendChild(g);
      lastGroup = group;
    }
    const item = document.createElement("div");
    item.className = "meeting-item" + (m.id === state.meetingId ? " is-active" : "");
    item.dataset.id = m.id;
    const body = document.createElement("div");
    body.className = "meeting-item__body";
    const t = document.createElement("span");
    t.className = "meeting-item__title";
    t.textContent = m.title;
    const meta = document.createElement("span");
    meta.className = "meeting-item__meta";
    const time = new Date(m.created_at).toLocaleTimeString("es", { hour: "2-digit", minute: "2-digit" });
    meta.textContent = `${time} · ${fmtDuration(m.duration)}${m.has_notes ? " · 📝" : ""}`;
    body.append(t, meta);
    const dots = document.createElement("button");
    dots.type = "button";
    dots.className = "meeting-item__dots";
    dots.textContent = "⋮";
    dots.title = "Opciones";
    dots.addEventListener("click", (e) => { e.stopPropagation(); showMeetingMenu(dots, m); });
    item.append(body, dots);
    item.addEventListener("click", () => openMeeting(m.id));
    list.appendChild(item);
  }
}

function resetChatPanel() {
  state.chatHistory = [];
  state.lastAnswer = "";
  $("chat-log").innerHTML = "";
}

function enterMeetingView(title) {
  $("meeting-title").value = title || state.baseFilename;
  setStepEnabled("step-result", true);
  setStepEnabled("step-ai", true);
  showView("meeting");
  selectVtab("notes");
  loadMeetings($("meeting-search").value.trim());
}

async function openMeeting(id) {
  let m;
  try {
    const r = await fetch(`/api/meetings/${id}`);
    if (!r.ok) { toast("No se pudo abrir la reunión."); return; }
    m = await r.json();
  } catch (e) { toast("No se pudo contactar al servidor."); return; }
  state.meetingId = m.id;
  state.segments = m.segments || [];
  state.meta.language = m.language;
  state.meta.duration = m.duration;
  state.meta.prob = 1;
  state.baseFilename = m.title.replace(/[^\wáéíóúñ -]/gi, "").trim() || "reunion";
  state.chosen = null;
  $("out-plain").textContent = toPlain();
  renderTsList();
  setupPlayer();
  if (m.has_audio) {
    // Reopened meetings play from the locally stored copy on the server.
    const p = $("player");
    p.src = `/api/meetings/${m.id}/audio`;
    p.hidden = false;
  }
  $("result-meta").textContent =
    `${fmtMeetingDate(m.created_at)} · idioma ${m.language} · ${fmtDuration(m.duration)}`;
  resetChatPanel();
  if (m.notes) {
    const bubble = addBubble("assistant", "");
    bubble.innerHTML = renderMarkdown(m.notes);
    addBubbleActions(bubble, m.notes);
    state.chatHistory.push({ role: "assistant", content: m.notes });
    state.lastAnswer = m.notes;
  }
  enterMeetingView(m.title);
}

// AI-generated meeting title (fire-and-forget; the filename title stays valid
// if the model isn't ready). Only when the local model is already downloaded —
// never trigger a GB download silently in the background.
async function requestAutotitle() {
  const loc = (state.ai && state.ai.local) || {};
  const id = state.meetingId;
  if (!id || !loc.available || !loc.cached) return;
  try {
    const r = await fetch(`/api/meetings/${id}/autotitle`, { method: "POST" });
    if (!r.ok) return;
    const { title } = await r.json();
    if (title && state.meetingId === id) $("meeting-title").value = title;
    loadMeetings($("meeting-search").value.trim());
  } catch (e) { /* keep the filename title */ }
}

// ---------------------------------------------------------------------------
// Sidebar ⋮ menu (rename / delete any meeting, Claude-style)
// ---------------------------------------------------------------------------
function closeCtxMenu() {
  const el = $("ctx-menu");
  if (el) el.remove();
}

function showMeetingMenu(anchor, meeting) {
  closeCtxMenu();
  const menu = document.createElement("div");
  menu.id = "ctx-menu";
  menu.className = "ctx-menu";
  const mk = (label, cls, fn) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = cls;
    b.textContent = label;
    b.addEventListener("click", (e) => { e.stopPropagation(); closeCtxMenu(); fn(); });
    return b;
  };
  menu.append(
    mk("✏️ Renombrar", "ctx-menu__item", () => renameMeetingById(meeting.id, meeting.title)),
    mk("🗑 Eliminar", "ctx-menu__item ctx-menu__item--danger", () => deleteMeetingById(meeting.id)),
  );
  const rect = anchor.getBoundingClientRect();
  menu.style.top = `${rect.bottom + 4}px`;
  menu.style.left = `${Math.max(8, rect.right - 150)}px`;
  document.body.appendChild(menu);
  setTimeout(() => document.addEventListener("click", closeCtxMenu, { once: true }), 0);
}

async function renameMeetingById(id, currentTitle) {
  const title = prompt("Nuevo título:", currentTitle);
  if (!title || !title.trim()) return;
  try {
    await fetch(`/api/meetings/${id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title: title.trim() }),
    });
  } catch (e) { toast("No se pudo renombrar."); return; }
  if (state.meetingId === id) $("meeting-title").value = title.trim();
  loadMeetings($("meeting-search").value.trim());
}

async function deleteMeetingById(id) {
  if (!confirm("¿Borrar esta reunión de tu equipo? Esta acción no se puede deshacer.")) return;
  try {
    await fetch(`/api/meetings/${id}`, { method: "DELETE" });
  } catch (e) { toast("No se pudo borrar."); return; }
  toast("Reunión borrada.");
  if (state.meetingId === id) {
    state.meetingId = null;
    showView("new");
  }
  loadMeetings($("meeting-search").value.trim());
}

// ---------------------------------------------------------------------------
// Chat bubbles: thinking indicator + per-response actions
// ---------------------------------------------------------------------------
function setThinking(bubble, label) {
  bubble.innerHTML = "";
  const wrap = document.createElement("span");
  wrap.className = "think";
  const star = document.createElement("span");
  star.className = "think__star";
  star.textContent = "✳";
  const txt = document.createElement("span");
  txt.textContent = label;
  wrap.append(star, txt);
  bubble.appendChild(wrap);
}

// Remove [MM:SS] / [MM:SS–MM:SS] marks for a clean copy of the note text.
function stripMarks(text) {
  return text
    .replace(/\[\d{1,2}:\d{2}(?::\d{2})?(?:\s*[–—-]\s*\d{1,2}:\d{2}(?::\d{2})?)?\]\s?/g, "")
    .replace(/ {2,}/g, " ")
    .trim();
}

function addBubbleActions(bubble, text) {
  const row = document.createElement("div");
  row.className = "bubble-actions";
  const mk = (label, value) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "bubble-actions__btn";
    b.textContent = label;
    b.addEventListener("click", () =>
      navigator.clipboard.writeText(value).then(() => toast("Copiado al portapapeles"))
    );
    return b;
  };
  row.append(mk("📋 Copiar", text), mk("Sin minutos", stripMarks(text)));
  bubble.appendChild(row);
}

async function renameCurrentMeeting() {
  const title = $("meeting-title").value.trim();
  if (!state.meetingId || !title) return;
  try {
    await fetch(`/api/meetings/${state.meetingId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    });
    loadMeetings($("meeting-search").value.trim());
  } catch (e) { /* offline rename is non-fatal */ }
}

async function deleteCurrentMeeting() {
  if (!state.meetingId) { showView("new"); return; }
  if (!confirm("¿Borrar esta reunión de tu equipo? Esta acción no se puede deshacer.")) return;
  try {
    await fetch(`/api/meetings/${state.meetingId}`, { method: "DELETE" });
  } catch (e) { toast("No se pudo borrar."); return; }
  state.meetingId = null;
  toast("Reunión borrada.");
  showView("new");
  loadMeetings($("meeting-search").value.trim());
}

async function autoSaveNotes() {
  if (!state.meetingId || !state.lastAnswer) return;
  try {
    await fetch(`/api/meetings/${state.meetingId}/notes`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ notes: state.lastAnswer }),
    });
    loadMeetings($("meeting-search").value.trim()); // refresh the 📝 badge
  } catch (e) { /* saving notes is best-effort */ }
}

// ---------------------------------------------------------------------------
// File selection
// ---------------------------------------------------------------------------
function chooseFile(file) {
  state.chosen = { kind: "file", file };
  state.baseFilename = file.name.replace(/\.[^.]+$/, "");
  // escapeHtml: filenames are OS-controlled input (Linux/macOS allow `<>`).
  showChosen(`📤 ${escapeHtml(file.name)} · ${(file.size / 1048576).toFixed(1)} MB`);
}
function choosePath(path) {
  const name = path.split(/[\\/]/).pop() || "archivo";
  state.chosen = { kind: "path", path };
  state.baseFilename = name.replace(/\.[^.]+$/, "");
  showChosen(`📁 ${escapeHtml(name)} <span class="muted">(${escapeHtml(path)})</span>`);
}
function showChosen(html) {
  const c = $("file-chosen");
  c.hidden = false;
  c.innerHTML = html;
  setStepEnabled("step-transcribe", true);
  $("btn-transcribe").disabled = false;
}

// ---------------------------------------------------------------------------
// Transcription
// ---------------------------------------------------------------------------
function startTranscription() {
  if (!state.chosen) return;
  $("transcribe-error").hidden = true;
  $("btn-transcribe").disabled = true;
  $("progress-wrap").hidden = false;
  setProgress(0, "Subiendo…");
  $("live-text").textContent = "";
  state.segments = [];

  const fd = new FormData();
  if (state.chosen.kind === "file") fd.append("file", state.chosen.file);
  else fd.append("path", state.chosen.path);
  fd.append("language", $("opt-language").value);
  fd.append("task", $("opt-task").value);
  fd.append("vad_filter", $("opt-vad").checked ? "true" : "false");
  fd.append("initial_prompt", $("opt-vocab").value.trim());
  fd.append("model_size", $("opt-model").value);
  fd.append("compute_type", $("opt-compute").value);
  fd.append("device", $("opt-device").value);

  const xhr = new XMLHttpRequest();
  xhr.open("POST", "/api/transcribe");
  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) setProgress((e.loaded / e.total) * 0.15, "Subiendo…");
  };
  xhr.onload = () => {
    if (xhr.status !== 200) {
      showTranscribeError(safeDetail(xhr.responseText) || "No se pudo iniciar la transcripción.");
      return;
    }
    const { job_id, model_cached } = JSON.parse(xhr.responseText);
    if (!model_cached) toast("📥 Descargando el modelo (solo la primera vez)…");
    streamEvents(job_id);
  };
  xhr.onerror = () => showTranscribeError("Error de red al subir el archivo.");
  xhr.send(fd);
}

function streamEvents(jobId) {
  const es = new EventSource(`/api/transcribe/${jobId}/events`);
  es.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    if (ev.type === "status") {
      setProgress(null, ev.label);
    } else if (ev.type === "meta") {
      state.meta.duration = ev.duration;
      if (ev.duration > 3600) {
        setProgress(null, `Audio de ${fmtDuration(ev.duration)} · estimado ~${fmtDuration(ev.estimated_seconds)}…`);
      }
    } else if (ev.type === "segment") {
      state.segments.push({ start: ev.start, end: ev.end, text: ev.text });
      const pctText = `${Math.round((ev.pct || 0) * 100)}%`;
      const eta = ev.eta != null ? ` · queda ~${fmtDuration(ev.eta)}` : "";
      setProgress(ev.pct, `Transcribiendo… ${pctText}${eta}`);
      const live = $("live-text");
      live.textContent = state.segments.slice(-25).map((s) => s.text.trim()).filter(Boolean).join(" ");
      live.scrollTop = live.scrollHeight;
    } else if (ev.type === "done") {
      state.meta.language = ev.language;
      state.meta.prob = ev.language_probability;
      state.meta.duration = ev.duration;
      state.meetingId = ev.meeting_id || null; // saved in the local library
      setProgress(1, `100% · completado en ${fmtDuration(ev.duration / 1)}`);
      es.close();
      renderResults();
      requestAutotitle(); // AI names the meeting (like chat apps do)
    } else if (ev.type === "error") {
      es.close();
      showTranscribeError(ev.message);
    }
  };
  es.onerror = () => {
    es.close();
    if (state.segments.length === 0) showTranscribeError("Se perdió la conexión con el servidor.");
  };
}

function setProgress(pct, label) {
  if (pct != null) $("progress-bar").style.width = `${Math.min(100, Math.max(0, pct * 100))}%`;
  if (label != null) $("progress-label").textContent = label;
}
function showTranscribeError(msg) {
  const e = $("transcribe-error");
  e.hidden = false;
  e.textContent = `❌ ${msg}`;
  $("btn-transcribe").disabled = false;
}
function safeDetail(text) {
  try { return JSON.parse(text).detail; } catch { return null; }
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------
function renderResults() {
  if (nonEmpty().length === 0) {
    showTranscribeError("No se detectó voz en el archivo.");
    return;
  }
  $("out-plain").textContent = toPlain();
  renderTsList();
  setupPlayer();
  $("result-meta").textContent =
    `Idioma: ${state.meta.language} (prob. ${state.meta.prob.toFixed(2)}) · duración ${fmtDuration(state.meta.duration)}`;
  $("btn-transcribe").disabled = false;
  $("progress-wrap").hidden = true;
  resetChatPanel();
  enterMeetingView(state.baseFilename);
}

function renderTsList() {
  const el = $("out-ts");
  el.innerHTML = "";
  nonEmpty().forEach((s) => {
    const line = document.createElement("div");
    line.className = "ts-line";
    line.dataset.start = s.start;
    const t = document.createElement("span");
    t.className = "ts-line__t";
    t.dataset.sec = Math.floor(s.start);
    t.textContent = `[${fmtTs(s.start)}]`;
    const x = document.createElement("span");
    x.className = "ts-line__x";
    x.textContent = s.text.trim();
    line.append(t, x);
    el.appendChild(line);
  });
}

function setupPlayer() {
  const p = $("player");
  if (state.playerUrl) { URL.revokeObjectURL(state.playerUrl); state.playerUrl = null; }
  if (state.chosen && state.chosen.kind === "file") {
    state.playerUrl = URL.createObjectURL(state.chosen.file);
    p.src = state.playerUrl;
    p.hidden = false;
  } else {
    p.removeAttribute("src");
    p.hidden = true;
  }
}

function activateResultTab(key) {
  document.querySelectorAll("[data-rtab]").forEach((t) =>
    t.classList.toggle("is-active", t.getAttribute("data-rtab") === key)
  );
  document.querySelectorAll("[data-rpanel]").forEach((p) =>
    p.classList.toggle("is-active", p.getAttribute("data-rpanel") === key)
  );
}

// Seek an <audio> element robustly. Two traps handled here:
// 1. AI answers may cite a timestamp at/after the real end of the audio
//    (models round up). Seeking there lands on the very end and playback
//    stops instantly ("the audio cuts off") — so clamp just before the end.
// 2. Blob audio can report duration Infinity until scanned; the scan-to-end
//    trick must be muted or the jump is audible, and we wait for `seeked`
//    (not `timeupdate`) so repeated clicks don't stack stale listeners.
function seekAudioTo(p, sec) {
  const apply = () => {
    try {
      const d = p.duration;
      const t = isFinite(d) && d > 0 ? Math.min(Math.max(0, sec), Math.max(0, d - 0.3)) : sec;
      p.currentTime = t;
      p.play().catch(() => {});
    } catch (e) { /* ignore */ }
  };
  const ready = () => {
    if (isFinite(p.duration) && p.duration > 0) {
      apply();
    } else {
      const wasMuted = p.muted;
      p.muted = true;
      const onSeeked = () => {
        p.removeEventListener("seeked", onSeeked);
        p.muted = wasMuted;
        apply();
      };
      p.addEventListener("seeked", onSeeked);
      try { p.currentTime = 1e7; } catch (e) { p.muted = wasMuted; }
    }
  };
  if (p.readyState >= 1) ready();
  else {
    p.addEventListener("loadedmetadata", ready, { once: true });
    try { p.load(); } catch (e) { /* ignore */ }
  }
}

function seekTo(sec) {
  const p = $("player");
  if (p && !p.hidden && p.getAttribute("src")) seekAudioTo(p, sec);
  selectVtab("transcript"); // reveal the transcript panel before scrolling
  activateResultTab("ts");
  let target = null;
  for (const line of document.querySelectorAll("#out-ts .ts-line")) {
    if (parseFloat(line.dataset.start) <= sec + 0.001) target = line;
    else break;
  }
  if (target) {
    target.scrollIntoView({ behavior: "smooth", block: "center" });
    target.classList.remove("flash");
    void target.offsetWidth; // restart the animation
    target.classList.add("flash");
  }
}

// ---------------------------------------------------------------------------
// AI chat (streaming via fetch ReadableStream)
// ---------------------------------------------------------------------------
function addBubble(role, text) {
  const div = document.createElement("div");
  div.className = `bubble bubble--${role}`;
  div.textContent = text;
  $("chat-log").appendChild(div);
  div.scrollIntoView({ behavior: "smooth", block: "nearest" });
  return div;
}

// `displayText` (optional) is what the user sees in their bubble — for quick
// actions it's the button label; the full engineered prompt stays internal.
async function sendChat(prompt, displayText) {
  if (!prompt.trim()) return;

  addBubble("user", displayText || prompt);
  state.chatHistory.push({ role: "user", content: prompt });
  // First use downloads the model (~GBs): without a heads-up the bubble sits
  // on the thinking indicator for minutes and looks frozen.
  const loc = (state.ai && state.ai.local) || {};
  const firstUse = loc.available && !loc.cached;
  const bubble = addBubble("assistant", "");
  setThinking(
    bubble,
    firstUse ? "Descargando el modelo de IA (solo la primera vez)…" : "Pensando…"
  );
  if (firstUse) loc.cached = true; // only announce once per session
  let answer = "";

  const body = {
    transcript_timestamped: toTimestamped(),
    language: state.meta.language,
    duration: state.meta.duration,
    meeting_id: state.meetingId || "", // lets the server reuse condensed notes
    history: state.chatHistory,
  };

  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop();
      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith("data:")) continue;
        const payload = JSON.parse(line.slice(5).trim());
        if (payload.delta) { answer += payload.delta; bubble.textContent = answer; }
        else if (payload.phase === "map") {
          const round = payload.round > 1 ? ` · pasada ${payload.round}` : "";
          setThinking(bubble, `Audio largo: analizando parte ${payload.current}/${payload.total}${round}…`);
        }
        else if (payload.phase === "combine") {
          setThinking(bubble, "Audio largo: uniendo las notas (última fase, se guarda para las próximas preguntas)…");
        }
        else if (payload.error) { bubble.textContent = `❌ ${payload.error}`; }
      }
    }
    if (answer) {
      bubble.innerHTML = renderMarkdown(answer); // render once complete (bold, bullets, [MM:SS])
      addBubbleActions(bubble, answer); // 📋 copiar / copiar sin minutos
      state.chatHistory.push({ role: "assistant", content: answer });
      state.lastAnswer = answer;
      autoSaveNotes(); // persist the note with the meeting (local library)
    } else if (!bubble.textContent.startsWith("❌")) {
      bubble.textContent = "(sin respuesta)";
    }
  } catch (e) {
    bubble.textContent = "❌ No se pudo contactar al servidor.";
  }
}

// ---------------------------------------------------------------------------
// Recording (voice + meeting). Captures locally on the server, then feeds the
// resulting WAV into the same transcription flow as an uploaded file.
// ---------------------------------------------------------------------------
const rec = { active: false, mode: null, timer: null, t0: 0, paused: false, accumMs: 0 };
let captureAvailable = true;
let meetingPoll = null;
let lastMeetingApp = null; // for one-shot "meeting detected" notifications

function fmtClock(sec) {
  sec = Math.max(0, Math.floor(sec));
  const pad = (n) => String(n).padStart(2, "0");
  return `${pad(Math.floor(sec / 60))}:${pad(sec % 60)}`;
}

function recEls(mode) {
  const p = mode === "voice" ? "voice" : "meeting";
  return {
    start: $(`${p}-start`), live: $(`${p}-live`), time: $(`${p}-time`),
    stop: $(`${p}-stop`), pause: $(`${p}-pause`),
  };
}

function setRecUI(mode, recording) {
  const el = recEls(mode);
  el.start.hidden = recording;
  el.live.hidden = !recording;
  if (!recording) el.live.classList.remove("is-paused");
}

function startTimer(mode) {
  rec.t0 = Date.now();
  rec.accumMs = 0;
  rec.paused = false;
  const el = recEls(mode);
  el.time.textContent = "00:00";
  el.pause.textContent = "⏸ Pausar";
  rec.timer = setInterval(() => {
    const ms = rec.accumMs + (rec.paused ? 0 : Date.now() - rec.t0);
    el.time.textContent = fmtClock(ms / 1000);
  }, 500);
}
function stopTimer() {
  if (rec.timer) { clearInterval(rec.timer); rec.timer = null; }
}

async function togglePause(mode) {
  if (!rec.active) return;
  const el = recEls(mode);
  const pausing = !rec.paused;
  try {
    const r = await fetch(`/api/record/${pausing ? "pause" : "resume"}`, { method: "POST" });
    if (!r.ok) { toast("No se pudo pausar/reanudar la grabación."); return; }
  } catch (e) {
    toast("No se pudo contactar al servidor.");
    return;
  }
  if (pausing) {
    rec.accumMs += Date.now() - rec.t0; // freeze elapsed; paused time isn't recorded
    rec.paused = true;
    el.pause.textContent = "▶ Reanudar";
    el.live.classList.add("is-paused");
  } else {
    rec.t0 = Date.now();
    rec.paused = false;
    el.pause.textContent = "⏸ Pausar";
    el.live.classList.remove("is-paused");
  }
}

// Live draft transcript (SSE) shown while recording.
let liveES = null;
function openLiveStream() {
  $("rec-live-text").textContent = "";
  $("rec-live-panel").hidden = false;
  liveES = new EventSource("/api/record/live");
  liveES.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    if (ev.done) { closeLiveStream(false); return; }
    if (!ev.text) return;
    const el = $("rec-live-text");
    el.textContent += (el.textContent ? "\n" : "") + `[${fmtTs(ev.start)}] ${ev.text}`;
    el.scrollTop = el.scrollHeight;
  };
  liveES.onerror = () => closeLiveStream(false);
}
function closeLiveStream(hidePanel) {
  if (liveES) { liveES.close(); liveES = null; }
  if (hidePanel) $("rec-live-panel").hidden = true;
}

async function startRecording(mode) {
  if (rec.active) return;
  const includeMic = mode === "meeting" ? $("rec-mic").checked : true;
  let data;
  try {
    const r = await fetch("/api/record/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode, include_mic: includeMic }),
    });
    if (!r.ok) { toast(safeDetail(await r.text()) || "No se pudo iniciar la grabación."); return; }
    data = await r.json();
    rec.active = true;
    rec.mode = mode;
    setRecUI(mode, true);
    startTimer(mode);
    if (data.live) openLiveStream();
  } catch (e) {
    toast("No se pudo contactar al servidor para grabar.");
  }
}

async function stopRecording(mode) {
  if (!rec.active) return;
  stopTimer();
  let data;
  try {
    const r = await fetch("/api/record/stop", { method: "POST" });
    if (!r.ok) {
      toast(safeDetail(await r.text()) || "No se pudo detener la grabación.");
      rec.active = false; setRecUI(mode, false); closeLiveStream(true);
      return;
    }
    data = await r.json();
  } catch (e) {
    toast("No se pudo contactar al servidor.");
    rec.active = false; setRecUI(mode, false); closeLiveStream(true);
    return;
  }
  rec.active = false;
  setRecUI(mode, false);
  closeLiveStream(true); // the definitive transcription replaces the draft

  // Hand the recorded WAV to the existing transcription flow and auto-start.
  const label = mode === "meeting" ? "reunión" : "nota de voz";
  state.chosen = { kind: "path", path: data.path };
  state.baseFilename = mode === "meeting" ? "reunion" : "grabacion";
  showChosen(`🎙️ Grabación lista (${label}) · ${fmtDuration(data.duration || 0)}`);
  startTranscription();
}

async function pollMeeting() {
  try {
    const r = await fetch("/api/record/meeting");
    const d = await r.json();
    captureAvailable = !!d.capture_available;
    recEls("voice").start.disabled = !captureAvailable;
    const el = $("meeting-detect");
    if (!captureAvailable) {
      el.className = "meeting-detect";
      el.textContent = "⚠️ Este equipo no tiene captura de audio disponible.";
    } else if (d.detected) {
      el.className = "meeting-detect is-on";
      el.innerHTML = `🟢 Detecté <strong>${d.app}</strong> — listo para grabar.`;
    } else {
      el.className = "meeting-detect";
      el.textContent = "No detecté una reunión activa. Puedes grabar igualmente.";
    }
    updateMeetingStart();
    // Notion-style alert: fire once when a meeting first appears.
    const app = captureAvailable && d.detected ? d.app : null;
    if (app && app !== lastMeetingApp) maybeNotifyMeeting(app);
    lastMeetingApp = app;
  } catch (e) {
    /* ignore polling errors */
  }
}
function updateMeetingStart() {
  $("meeting-start").disabled = !captureAvailable || !$("rec-consent").checked || rec.active;
}
function startMeetingPoll() { if (!meetingPoll) meetingPoll = setInterval(pollMeeting, 4000); }

// Ask once (on a user gesture) so we can alert about a meeting in the background.
function requestMeetingNotifications() {
  if ("Notification" in window && Notification.permission === "default") {
    Notification.requestPermission().catch(() => {});
  }
}
function maybeNotifyMeeting(app) {
  if (rec.active) return; // already recording — no need to nag
  if (!("Notification" in window) || Notification.permission !== "granted") return;
  const n = new Notification("🎙️ Audicop — reunión detectada", {
    body: `Detecté ${app}. Abre Audicop para grabarla.`,
    tag: "audicop-meeting",
  });
  n.onclick = () => {
    window.focus();
    const tab = document.querySelector('[data-tab="rec-meeting"]');
    if (tab) tab.click();
    n.close();
  };
}

// ---------------------------------------------------------------------------
// Wire up
// ---------------------------------------------------------------------------
function init() {
  applyTheme(currentTheme());
  $("theme-toggle").addEventListener("click", toggleTheme);
  wireTabs("data-tab", "data-panel");
  wireTabs("data-rtab", "data-rpanel");
  loadHardware();
  loadMeetings(); // sidebar library — past work survives reloads and restarts

  // Sidebar: new transcription + search
  $("btn-new").addEventListener("click", () => {
    state.meetingId = null;
    showView("new");
    loadMeetings($("meeting-search").value.trim());
  });
  let searchT;
  $("meeting-search").addEventListener("input", (e) => {
    clearTimeout(searchT);
    searchT = setTimeout(() => loadMeetings(e.target.value.trim()), 250);
  });

  // Meeting header: rename + delete; Notas ⟷ Transcripción toggle
  $("meeting-title").addEventListener("change", renameCurrentMeeting);
  $("btn-del-meeting").addEventListener("click", deleteCurrentMeeting);
  document.querySelectorAll("[data-vtab]").forEach((b) =>
    b.addEventListener("click", () => selectVtab(b.getAttribute("data-vtab")))
  );

  // Note actions: copy / download the last AI note
  $("note-copy").addEventListener("click", () => {
    if (!state.lastAnswer) { toast("Genera primero una nota (Resumen, Acta…)."); return; }
    navigator.clipboard.writeText(state.lastAnswer).then(() => toast("Nota copiada"));
  });
  $("note-download").addEventListener("click", () => {
    if (!state.lastAnswer) { toast("Genera primero una nota (Resumen, Acta…)."); return; }
    const title = $("meeting-title").value.trim() || state.baseFilename;
    download(`# ${title}\n\n${state.lastAnswer}\n`, "md", "text/markdown");
  });

  // file input + drag/drop
  const dz = $("dropzone");
  $("file-input").addEventListener("change", (e) => { if (e.target.files[0]) chooseFile(e.target.files[0]); });
  ["dragover", "dragenter"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("is-drag"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("is-drag"); }));
  dz.addEventListener("drop", (e) => { if (e.dataTransfer.files[0]) chooseFile(e.dataTransfer.files[0]); });

  $("path-input").addEventListener("change", (e) => { if (e.target.value.trim()) choosePath(e.target.value.trim()); });

  // recording: voice + meeting
  $("voice-start").addEventListener("click", () => startRecording("voice"));
  $("voice-stop").addEventListener("click", () => stopRecording("voice"));
  $("voice-pause").addEventListener("click", () => togglePause("voice"));
  $("meeting-start").addEventListener("click", () => startRecording("meeting"));
  $("meeting-stop").addEventListener("click", () => stopRecording("meeting"));
  $("meeting-pause").addEventListener("click", () => togglePause("meeting"));
  $("rec-consent").addEventListener("change", updateMeetingStart);
  // Ask for notification permission when the user opens the meeting tab.
  document.querySelectorAll("[data-tab]").forEach((tab) =>
    tab.addEventListener("click", () => {
      if (tab.getAttribute("data-tab") === "rec-meeting") { requestMeetingNotifications(); pollMeeting(); }
    })
  );
  // Poll continuously so we can alert about a meeting even from another tab.
  pollMeeting();
  startMeetingPoll();

  $("btn-transcribe").addEventListener("click", startTranscription);

  // copy buttons
  document.querySelectorAll("[data-copy]").forEach((b) =>
    b.addEventListener("click", () => {
      const txt = b.dataset.copy === "ts" ? toTimestamped() : toPlain();
      navigator.clipboard.writeText(txt).then(() => toast("Copiado al portapapeles"));
    })
  );
  // download buttons
  document.querySelectorAll("[data-dl]").forEach((b) =>
    b.addEventListener("click", () => {
      const k = b.dataset.dl;
      if (k === "txt") download(toPlain(), "txt", "text/plain");
      else if (k === "srt") download(toSRT(), "srt", "text/plain");
      else if (k === "vtt") download(toVTT(), "vtt", "text/vtt");
    })
  );

  // Clic en cualquier [MM:SS] (chat o transcripción) → saltar a ese momento
  document.addEventListener("click", (e) => {
    const t = e.target.closest(".ts, .ts-line__t");
    if (t && t.dataset.sec != null) seekTo(parseFloat(t.dataset.sec));
  });

  // AI
  $("chat-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const v = $("chat-input").value;
    $("chat-input").value = "";
    sendChat(v);
  });
}

document.addEventListener("DOMContentLoaded", init);
