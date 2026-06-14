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
  model: { tiny: "Tiny", base: "Base", small: "Small", medium: "Medium", "large-v3": "Large v3" },
};

// App state (kept in memory only)
const state = {
  segments: [],
  meta: { language: "es", duration: 0, prob: 0 },
  baseFilename: "transcripcion",
  chosen: null, // { kind: "file"|"path", file?, path? }
  ai: { providers: {}, models: {}, keyHelp: {} },
  chatHistory: [],
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

  // AI options
  state.ai = { providers: ai.provider_labels, models: ai.models_by_provider, keyHelp: ai.api_key_help };
  fillSelect($("ai-provider"), Object.keys(ai.provider_labels), ai.provider_labels, ai.default_provider);
  refreshAiModels();

  // Quick-action buttons (prompts come from the server's editable .md files)
  const qa = $("quick-actions");
  qa.innerHTML = "";
  for (const action of ai.quick_actions || []) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "quick-btn";
    btn.textContent = action.label;
    btn.addEventListener("click", () => sendChat(action.prompt));
    qa.appendChild(btn);
  }
}

function refreshAiModels() {
  const prov = $("ai-provider").value;
  fillSelect($("ai-model"), state.ai.models[prov] || [], null);
  $("ai-key-help").textContent = state.ai.keyHelp[prov] || "";
}

// ---------------------------------------------------------------------------
// File selection
// ---------------------------------------------------------------------------
function chooseFile(file) {
  state.chosen = { kind: "file", file };
  state.baseFilename = file.name.replace(/\.[^.]+$/, "");
  showChosen(`📤 ${file.name} · ${(file.size / 1048576).toFixed(1)} MB`);
}
function choosePath(path) {
  const name = path.split(/[\\/]/).pop() || "archivo";
  state.chosen = { kind: "path", path };
  state.baseFilename = name.replace(/\.[^.]+$/, "");
  showChosen(`📁 ${name} <span class="muted">(${path})</span>`);
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
      setProgress(1, `100% · completado en ${fmtDuration(ev.duration / 1)}`);
      es.close();
      renderResults();
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
  $("out-ts").textContent = toTimestamped();
  $("result-meta").textContent =
    `Idioma: ${state.meta.language} (prob. ${state.meta.prob.toFixed(2)}) · duración ${fmtDuration(state.meta.duration)}`;
  setStepEnabled("step-result", true);
  setStepEnabled("step-ai", true);
  $("btn-transcribe").disabled = false;
  $("step-result").scrollIntoView({ behavior: "smooth", block: "start" });
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

async function sendChat(prompt) {
  const key = $("ai-key").value.trim();
  if (!key) { toast("Pega tu API key arriba para usar el chat."); return; }
  if (!prompt.trim()) return;

  addBubble("user", prompt);
  state.chatHistory.push({ role: "user", content: prompt });
  const bubble = addBubble("assistant", "…");
  let answer = "";

  const body = {
    provider: $("ai-provider").value,
    model: $("ai-model").value,
    api_key: key,
    transcript_timestamped: toTimestamped(),
    language: state.meta.language,
    duration: state.meta.duration,
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
        else if (payload.error) { bubble.textContent = `❌ ${payload.error}`; }
      }
    }
    if (answer) state.chatHistory.push({ role: "assistant", content: answer });
    else if (bubble.textContent === "…") bubble.textContent = "(sin respuesta)";
  } catch (e) {
    bubble.textContent = "❌ No se pudo contactar al servidor.";
  }
}

// ---------------------------------------------------------------------------
// Wire up
// ---------------------------------------------------------------------------
function init() {
  wireTabs("data-tab", "data-panel");
  wireTabs("data-rtab", "data-rpanel");
  loadHardware();

  // file input + drag/drop
  const dz = $("dropzone");
  $("file-input").addEventListener("change", (e) => { if (e.target.files[0]) chooseFile(e.target.files[0]); });
  ["dragover", "dragenter"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("is-drag"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("is-drag"); }));
  dz.addEventListener("drop", (e) => { if (e.dataTransfer.files[0]) chooseFile(e.dataTransfer.files[0]); });

  $("path-input").addEventListener("change", (e) => { if (e.target.value.trim()) choosePath(e.target.value.trim()); });

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

  // AI
  $("ai-provider").addEventListener("change", refreshAiModels);
  $("chat-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const v = $("chat-input").value;
    $("chat-input").value = "";
    sendChat(v);
  });
}

document.addEventListener("DOMContentLoaded", init);
