/* Reachy Wheels D-pad.
 *
 * Hold-to-drive: while a button (or key) is held we re-send the command every
 * 300 ms; the chassis-side deadman (2 s) stops the wheels if we vanish, and
 * releasing sends an explicit /api/stop. Never queue commands — one in flight
 * at a time, latest wins.
 */

const $ = (sel) => document.querySelector(sel);

const REPEAT_MS = 300;
let speed = 0.8;
let held = null;          // command currently held, or null
let repeatTimer = null;
let inflight = false;     // drop overlapping sends instead of queueing

async function api(path, body) {
  const opts = body
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : (path === "/api/stop" ? { method: "POST" } : {});
  const resp = await fetch(path, opts);
  return resp.json().catch(() => ({}));
}

async function sendCmd(command) {
  if (inflight) return;
  inflight = true;
  try {
    const out = await api("/api/cmd", { command, speed });
    if (out && out.ok === false) flashError(out.error);
  } catch (e) {
    flashError(String(e));
  } finally {
    inflight = false;
  }
}

async function sendStop() {
  clearInterval(repeatTimer);
  repeatTimer = null;
  held = null;
  document.querySelectorAll(".drive.held").forEach((b) => b.classList.remove("held"));
  try { await api("/api/stop"); } catch (e) { /* deadman covers us */ }
}

function beginHold(command, btn) {
  if (held === command) return;
  held = command;
  if (btn) btn.classList.add("held");
  sendCmd(command);
  clearInterval(repeatTimer);
  repeatTimer = setInterval(() => { if (held) sendCmd(held); }, REPEAT_MS);
}

/* ---- pad buttons ------------------------------------------------------ */

document.querySelectorAll(".drive").forEach((btn) => {
  const cmd = btn.dataset.cmd;
  btn.addEventListener("pointerdown", (e) => { e.preventDefault(); beginHold(cmd, btn); });
  ["pointerup", "pointercancel", "pointerleave"].forEach((ev) =>
    btn.addEventListener(ev, () => { if (held === cmd) sendStop(); }));
  btn.addEventListener("contextmenu", (e) => e.preventDefault());
});

$("#stop").addEventListener("pointerdown", (e) => { e.preventDefault(); sendStop(); });

/* ---- keyboard --------------------------------------------------------- */

const KEYMAP = {
  w: "forward", arrowup: "forward",
  s: "reverse", arrowdown: "reverse",
  a: "strafe_left", arrowleft: "strafe_left",
  d: "strafe_right", arrowright: "strafe_right",
  q: "rotate_ccw", e: "rotate_cw",
};

function typing(ev) {
  const el = ev.target;
  return el && (el.tagName === "INPUT" || el.tagName === "TEXTAREA" ||
                el.tagName === "SELECT" || el.isContentEditable);
}

window.addEventListener("keydown", (ev) => {
  if (typing(ev)) return;     // "d" in the follow box is a letter, not a strafe
  const key = ev.key.toLowerCase();
  if (key === " ") { ev.preventDefault(); sendStop(); return; }
  const cmd = KEYMAP[key];
  if (!cmd || ev.repeat) return;
  ev.preventDefault();
  beginHold(cmd, document.querySelector(`.drive[data-cmd="${cmd}"]`));
});

window.addEventListener("keyup", (ev) => {
  if (typing(ev)) return;
  const cmd = KEYMAP[ev.key.toLowerCase()];
  if (cmd && held === cmd) sendStop();
});

window.addEventListener("blur", () => { if (held) sendStop(); });

/* ---- speed ------------------------------------------------------------ */

const speedInput = $("#speed");
speedInput.addEventListener("input", () => {
  speed = parseFloat(speedInput.value);
  $("#speed-val").textContent = speed.toFixed(2);
});

/* ---- status polling --------------------------------------------------- */

let lastError = 0;
function flashError(msg) {
  const now = Date.now();
  if (now - lastError < 2000) return; // don't spam while holding
  lastError = now;
  const pill = $("#status");
  pill.className = "pill offline";
  pill.textContent = msg || "command failed";
}

const WHEEL_LABELS = { front_left: "FL", front_right: "FR", rear_left: "RL", rear_right: "RR" };

async function poll() {
  try {
    const st = await api("/api/status");
    const pill = $("#status");
    if (st.connected) {
      pill.className = "pill online";
      pill.textContent = `${st.host} · ${st.state.last_command}`;
      const wheels = st.state.wheels || {};
      $("#wheels").innerHTML = Object.entries(WHEEL_LABELS)
        .map(([k, label]) => `<div>${label} <b>${(wheels[k] ?? 0).toFixed(2)}</b></div>`)
        .join("");
      $("#deadman").textContent = st.state.stops_in != null
        ? `deadman stops in ${st.state.stops_in.toFixed(1)} s` : "";
      labLive(wheels);
    } else {
      pill.className = "pill offline";
      pill.textContent = `${st.host} unreachable`;
      $("#deadman").textContent = "";
    }
  } catch (e) { /* app itself unreachable; keep last pill */ }
}
setInterval(poll, 2000);

/* ---- voice ------------------------------------------------------------ */

const WHO_LABEL = { user: "you", reachy: "reachy", tool: "⚙", system: "!" };

async function pollVoice() {
  try {
    const st = await api("/api/voice/status");
    const pill = $("#voice-status");
    if (st.connected) {
      pill.className = "pill online";
      pill.textContent = st.detail || "listening";
    } else {
      pill.className = "pill offline";
      pill.textContent = st.detail || "off";
    }
    $("#voice-key-row").hidden = st.key_present || !st.voice_enabled;

    const box = $("#transcript");
    const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 30;
    box.innerHTML = (st.transcript || [])
      .map((l) => `<div class="line-${l.who}"><span class="who">${WHO_LABEL[l.who] || l.who}</span>${escapeHtml(l.text)}</div>`)
      .join("");
    if (atBottom) box.scrollTop = box.scrollHeight;
  } catch (e) { /* app unreachable; keep last state */ }
}
setInterval(pollVoice, 2000);

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

$("#save-key").addEventListener("click", async () => {
  const key = $("#gemini-key").value.trim();
  if (!key) return;
  await api("/api/voice/config", { gemini_api_key: key });
  $("#gemini-key").value = "";
  pollVoice();
});

/* ---- mount detection --------------------------------------------------- */

/* The meter spans the two measured baselines with a margin, so the needle's
 * position is meaningful relative to THIS robot's calibration rather than
 * some absolute dBm scale nobody can read at a glance. */
const MOUNT_LABEL = {
  on_wheels: ["on the wheels", "online"],
  off_wheels: ["off the wheels", "offline"],
  no_signal: ["no beacon heard", "offline"],
  uncalibrated: ["needs calibrating", "offline"],
  unknown: ["deciding…", "offline"],
  unavailable: ["not running", "offline"],
};

function mountScale(st) {
  const lo = Math.min(st.off_dbm ?? -75, st.smoothed_dbm ?? -75) - 4;
  const hi = Math.max(st.on_dbm ?? -30, st.smoothed_dbm ?? -30) + 4;
  return { lo, hi, pct: (v) => `${Math.max(0, Math.min(100, ((v - lo) / (hi - lo)) * 100))}%` };
}

async function pollMount() {
  let st;
  try { st = await api("/api/mount/status"); } catch (e) { return; }

  const state = st.available === false ? "unavailable" : (st.state || "unknown");
  const [text, cls] = MOUNT_LABEL[state] || MOUNT_LABEL.unknown;
  const pill = $("#mount-status");
  pill.className = `pill ${state === "on_wheels" ? "online" : "offline"}`;
  pill.textContent = st.smoothed_dbm != null
    ? `${text} · ${st.smoothed_dbm.toFixed(0)} dBm` : text;

  $("#mount-figure").dataset.state = state;
  $("#mount-gap-label").textContent =
    state === "on_wheels" ? "mounted" :
    state === "off_wheels" ? "away from the base" :
    state === "no_signal" ? "beacon silent" : "";

  const bits = [];
  if (st.calibrated) {
    const s = mountScale(st);
    $("#mount-band-off").style.left = "0%";
    $("#mount-band-off").style.width = s.pct(st.threshold_dbm);
    $("#mount-band-on").style.left = s.pct(st.threshold_dbm);
    $("#mount-band-on").style.right = "0";
    $("#mount-thresh").style.left = s.pct(st.threshold_dbm);
    if (st.smoothed_dbm != null) {
      $("#mount-needle").style.left = s.pct(st.smoothed_dbm);
      $("#mount-needle").style.display = "block";
    } else {
      $("#mount-needle").style.display = "none";
    }
    bits.push(`on <b>${st.on_dbm}</b> · off <b>${st.off_dbm}</b> · threshold <b>${st.threshold_dbm.toFixed(1)}</b> dBm`);
    if (st.confidence != null) bits.push(`confidence <b>${(st.confidence * 100).toFixed(0)}%</b>`);
  } else {
    $("#mount-needle").style.display = "none";
    bits.push("not calibrated yet — record both states below");
    if (st.smoothed_dbm != null) bits.push(`hearing <b>${st.smoothed_dbm.toFixed(1)}</b> dBm right now`);
  }
  if (st.freq_mhz) bits.push(`${st.freq_mhz} MHz`);
  if (st.misses > 0) bits.push(`${st.misses} scans missed the beacon`);
  $("#mount-detail").innerHTML = bits.join(" · ");
}
setInterval(pollMount, 3000);

async function calibrateMount(which, button) {
  const buttons = document.querySelectorAll(".mount-cal button");
  buttons.forEach((b) => (b.disabled = true));
  const original = button.textContent;
  button.textContent = "listening…";
  try {
    const out = await api("/api/mount/calibrate", { state: which, samples: 8 });
    button.textContent = out.ok
      ? `${out.median_dbm} dBm ±${out.spread_db}`
      : (out.error || "failed").slice(0, 40);
  } catch (e) {
    button.textContent = "failed";
  } finally {
    buttons.forEach((b) => (b.disabled = false));
    setTimeout(() => { button.textContent = original; }, 4000);
    pollMount();
  }
}

$("#cal-on").addEventListener("click", (e) => calibrateMount("on", e.target));
$("#cal-off").addEventListener("click", (e) => calibrateMount("off", e.target));

/* ---- visual following -------------------------------------------------- */

/* The server cancels any follow on /api/stop, so the STOP button and the
 * space bar already end one — we only have to refresh the panel. */

let trackActive = false;
let previewTimer = null;

async function startFollow(target) {
  const value = (target || $("#track-target").value || "").trim();
  if (!value) return;
  $("#track-target").value = value;
  const out = await api("/api/track/start", { target: value });
  if (out && out.status !== "ok") setTrackDetail(out.error || "could not start");
  pollTrack();
}

async function stopFollow() {
  try { await api("/api/track/stop", {}); } catch (e) { /* deadman covers us */ }
  pollTrack();
}

function setTrackDetail(html) { $("#track-detail").innerHTML = html; }

/* The endpoint 404s until there is a frame; hide the box rather than
 * showing a broken image, and keep the final frame after a session ends so
 * you can see what the robot was actually looking at when it gave up. */
function refreshPreview() {
  const img = $("#track-preview");
  img.onerror = () => { img.hidden = true; };
  img.onload = () => { img.hidden = false; };
  img.src = `/api/track/preview?t=${Date.now()}`;
}

function trackSummary(st) {
  const bits = [];
  if (st.track_id != null && st.track_id >= 0) bits.push(`track <b>#${st.track_id}</b>`);
  if (st.distance_m != null) bits.push(`<b>${st.distance_m.toFixed(1)}</b> m`);
  if (st.bearing_deg != null) bits.push(`bearing <b>${st.bearing_deg.toFixed(0)}°</b>`);
  if (st.body_yaw != null) bits.push(`body <b>${st.body_yaw.toFixed(0)}°</b>`);
  if (st.head_yaw != null) bits.push(`head <b>${st.head_yaw.toFixed(0)}°</b>`);
  if (st.fps) bits.push(`<b>${st.fps.toFixed(1)}</b> fps`);
  if (st.seen != null) bits.push(`${st.seen} seen`);
  const line = bits.join(" · ");
  const detail = st.detail ? escapeHtml(st.detail) : "";
  return [line, detail].filter(Boolean).join("<br>");
}

async function pollTrack() {
  try {
    const st = await api("/api/track/status");
    trackActive = !!st.active;
    const pill = $("#track-status");
    pill.className = `pill ${trackActive ? "online" : "offline"}`;
    pill.textContent = trackActive
      ? `${st.phase} · ${st.target || ""}`.trim()
      : (st.available === false ? "unavailable" : (st.detail || "idle"));
    setTrackDetail(trackActive || st.detail ? trackSummary(st) : "");
    if (trackActive && !previewTimer) {
      previewTimer = setInterval(refreshPreview, 400);
      refreshPreview();
    } else if (!trackActive && previewTimer) {
      clearInterval(previewTimer);
      previewTimer = null;
      refreshPreview();   // keep the last thing it saw on screen
    }
  } catch (e) { /* app unreachable; keep last state */ }
}
setInterval(pollTrack, 1500);

$("#track-go").addEventListener("click", () => startFollow());
$("#track-halt").addEventListener("click", stopFollow);
$("#track-target").addEventListener("keydown", (ev) => {
  if (ev.key === "Enter") { ev.preventDefault(); startFollow(); }
});
document.querySelectorAll(".chip").forEach((chip) =>
  chip.addEventListener("click", () => startFollow(chip.dataset.target)));

/* ---- wheel lab --------------------------------------------------------- */

/* Hand-driving the four corners. The chassis rotates badly in place and
 * `move(omega)` cannot express an asymmetric fix, so this panel sends an
 * explicit per-wheel mix and lets a human judge the result. Same hold
 * discipline as the pad: repeat while held, explicit stop on release, and
 * the board's 2 s deadman behind both. */

const LAB = {
  wheels: ["front_left", "front_right", "rear_left", "rear_right"],
  labels: WHEEL_LABELS,
  mix: { front_left: 0, front_right: 0, rear_left: 0, rear_right: 0 },
  // What a wheel goes back to when it is switched on again, so toggling a
  // corner off to test without it does not lose the value you dialled in.
  lastNonZero: { front_left: 1, front_right: -1, rear_left: 1, rear_right: -1 },
  trim: {},
  invert: {},
  speed: 0.8,
  duration: 1.0,
  holding: false,
  timer: null,
};

async function labSend(duration) {
  if (inflight) return;
  inflight = true;
  try {
    const out = await api("/api/wheels/mix", {
      speeds: LAB.mix, speed: LAB.speed, duration,
    });
    if (out && out.ok === false) flashError(out.error);
  } catch (e) {
    flashError(String(e));
  } finally {
    inflight = false;
  }
}

function labStop() {
  LAB.holding = false;
  clearInterval(LAB.timer);
  LAB.timer = null;
  $("#lab-hold").classList.remove("held");
  sendStop();
}

function labBeginHold() {
  if (LAB.holding) return;
  LAB.holding = true;
  $("#lab-hold").classList.add("held");
  labSend(null);              // no duration: the board's deadman is the limit
  clearInterval(LAB.timer);
  LAB.timer = setInterval(() => { if (LAB.holding) labSend(null); }, REPEAT_MS);
}

function labMixLine() {
  return LAB.wheels
    .map((n) => `${LAB.labels[n]} ${LAB.mix[n] >= 0 ? "+" : ""}${LAB.mix[n].toFixed(2)}`)
    .join(" · ") + ` @ speed ${LAB.speed.toFixed(2)}`;
}

function renderLabValues() {
  LAB.wheels.forEach((name) => {
    const card = document.querySelector(`.wheel-card[data-wheel="${name}"]`);
    if (!card) return;
    const value = LAB.mix[name];
    card.querySelector(".wc-mix").value = value;
    card.querySelector(".wc-value").textContent =
      `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
    card.classList.toggle("off", value === 0);
    card.querySelector(".wc-power").textContent = value === 0 ? "off" : "on";
  });
  $("#lab-mix-line").textContent = labMixLine();
}

function setWheel(name, value) {
  LAB.mix[name] = Math.max(-1, Math.min(1, value));
  if (value !== 0) LAB.lastNonZero[name] = LAB.mix[name];
  renderLabValues();
  if (LAB.holding) labSend(null);   // live: retune a corner mid-spin
}

function renderWheelGrid() {
  $("#wheel-grid").innerHTML = LAB.wheels.map((name) => `
    <div class="wheel-card" data-wheel="${name}">
      <div class="wc-head">
        <span class="wc-label">${LAB.labels[name]}</span>
        <button class="wc-power">on</button>
      </div>
      <input class="wc-mix" type="range" min="-1" max="1" step="0.05" value="0">
      <div class="wc-value">+0.00</div>
      <div class="wc-live">—</div>
    </div>`).join("");

  document.querySelectorAll(".wheel-card").forEach((card) => {
    const name = card.dataset.wheel;
    card.querySelector(".wc-mix").addEventListener("input", (ev) =>
      setWheel(name, parseFloat(ev.target.value)));
    card.querySelector(".wc-power").addEventListener("click", () =>
      setWheel(name, LAB.mix[name] === 0 ? LAB.lastNonZero[name] : 0));
  });
}

function renderTrimGrid() {
  const [lo, hi] = LAB.trimRange || [0.3, 1.5];
  $("#trim-grid").innerHTML = LAB.wheels.map((name) => {
    const trim = LAB.trim[name] ?? 1.0;
    return `
    <div class="trim-card" data-wheel="${name}">
      <span class="wc-label">${LAB.labels[name]}</span>
      <input class="tc-trim" type="range" min="${lo}" max="${hi}" step="0.05" value="${trim}">
      <b class="tc-val">${trim.toFixed(2)}</b>
      <button class="tc-save">save</button>
      <button class="tc-flip">flip</button>
    </div>`;
  }).join("");

  document.querySelectorAll(".trim-card").forEach((card) => {
    const name = card.dataset.wheel;
    const slider = card.querySelector(".tc-trim");
    slider.addEventListener("input", () => {
      card.querySelector(".tc-val").textContent = parseFloat(slider.value).toFixed(2);
    });
    // Trim is pushed on release, not on every drag frame: the board serves
    // one connection at a time and a slider emits dozens of events.
    ["change", "pointerup"].forEach((ev) => slider.addEventListener(ev, () =>
      saveTrim(name, { trim: parseFloat(slider.value) })));
    card.querySelector(".tc-save").addEventListener("click", () =>
      saveTrim(name, { trim: parseFloat(slider.value) }));
    card.querySelector(".tc-flip").addEventListener("click", () =>
      saveTrim(name, { invert: !(LAB.invert[name] ?? true) }));
  });
}

async function saveTrim(name, patch) {
  const out = await api("/api/wheels/tune", { wheel: name, ...patch });
  if (!out || out.ok === false) { flashError(out && out.error); return; }
  if (out.trim != null) LAB.trim[name] = out.trim;
  if (out.invert != null) LAB.invert[name] = out.invert;
  if (out.pins_snippet) $("#pins-snippet").textContent = out.pins_snippet;
}

function renderPresets() {
  $("#lab-presets").innerHTML = (LAB.presets || [])
    .map((p, i) => `<button class="chip" data-preset="${i}">${escapeHtml(p.label)}</button>`)
    .join("");
  document.querySelectorAll("[data-preset]").forEach((btn) =>
    btn.addEventListener("click", () => {
      const preset = LAB.presets[parseInt(btn.dataset.preset, 10)];
      LAB.wheels.forEach((name) => setWheel(name, preset.mix[name] ?? 0));
      $("#lab-note").textContent = preset.note || "";
      document.querySelectorAll("[data-preset]").forEach((b) =>
        b.classList.toggle("on", b === btn));
    }));
}

/* Board readback, so a wheel that is commanded but not turning (dead zone,
 * a loose lead) is visible rather than inferred. */
function labLive(wheels) {
  LAB.wheels.forEach((name) => {
    const el = document.querySelector(`.wheel-card[data-wheel="${name}"] .wc-live`);
    if (el) el.textContent = `board ${(wheels[name] ?? 0).toFixed(2)}`;
  });
}

$("#lab-hold").addEventListener("pointerdown", (e) => { e.preventDefault(); labBeginHold(); });
["pointerup", "pointercancel", "pointerleave"].forEach((ev) =>
  $("#lab-hold").addEventListener(ev, () => { if (LAB.holding) labStop(); }));
$("#lab-pulse").addEventListener("click", () => labSend(LAB.duration));
$("#lab-stop").addEventListener("click", labStop);
$("#lab-mirror").addEventListener("click", () =>
  LAB.wheels.forEach((name) => setWheel(name, -LAB.mix[name])));
$("#lab-zero").addEventListener("click", () =>
  LAB.wheels.forEach((name) => setWheel(name, 0)));
$("#lab-copy").addEventListener("click", async () => {
  const text = labMixLine();
  try { await navigator.clipboard.writeText(text); $("#lab-copy").textContent = "copied"; }
  catch (e) { $("#lab-copy").textContent = "select it"; }
  setTimeout(() => { $("#lab-copy").textContent = "copy"; }, 1500);
});

$("#lab-speed").addEventListener("input", (ev) => {
  LAB.speed = parseFloat(ev.target.value);
  $("#lab-speed-val").textContent = LAB.speed.toFixed(2);
  $("#lab-mix-line").textContent = labMixLine();
  if (LAB.holding) labSend(null);
});
$("#lab-duration").addEventListener("input", (ev) => {
  LAB.duration = parseFloat(ev.target.value);
  $("#lab-duration-val").textContent = LAB.duration.toFixed(1);
});

$("#trim-apply").addEventListener("click", async (ev) => {
  const btn = ev.target;
  const original = btn.textContent;
  btn.disabled = true;
  btn.textContent = "pushing…";
  const out = await api("/api/wheels/tune/apply", {});
  btn.textContent = out.ok ? `applied ${(out.applied || []).length}` : "failed";
  if (out.pins_snippet) $("#pins-snippet").textContent = out.pins_snippet;
  btn.disabled = false;
  setTimeout(() => { btn.textContent = original; }, 2500);
});

async function loadLab() {
  let lab;
  try { lab = await api("/api/wheels/lab"); } catch (e) { return; }
  LAB.wheels = lab.wheels || LAB.wheels;
  LAB.labels = lab.labels || LAB.labels;
  LAB.presets = lab.presets || [];
  LAB.trim = lab.trim || {};
  LAB.invert = lab.invert || {};
  LAB.trimRange = lab.trim_range;
  LAB.speed = lab.speed ?? LAB.speed;
  LAB.duration = lab.duration ?? LAB.duration;
  // The board's own view of trim wins when it has one: the app remembers
  // what it pushed, the board is what is actually driving the motors.
  if (lab.board_tuning) {
    Object.entries(lab.board_tuning).forEach(([name, t]) => {
      LAB.trim[name] = t.trim;
      LAB.invert[name] = t.invert;
    });
  }
  renderWheelGrid();
  renderPresets();
  renderTrimGrid();
  $("#pins-snippet").textContent = lab.pins_snippet || "";
  $("#lab-speed").value = LAB.speed;
  $("#lab-speed-val").textContent = LAB.speed.toFixed(2);
  $("#lab-duration").value = LAB.duration;
  $("#lab-duration-val").textContent = LAB.duration.toFixed(1);
  LAB.wheels.forEach((name) => {
    const value = (lab.mix || {})[name] ?? 0;
    LAB.mix[name] = value;
    if (value !== 0) LAB.lastNonZero[name] = value;
  });
  renderLabValues();
  // First run has no saved mix, and a panel where HOLD does nothing reads as
  // broken. Start from the built-in rotate — the mix being compared against.
  if (LAB.presets.length && LAB.wheels.every((n) => LAB.mix[n] === 0)) {
    document.querySelector('[data-preset="0"]').click();
  }
}

/* ---- settings --------------------------------------------------------- */

$("#save-host").addEventListener("click", async () => {
  const host = $("#host").value.trim();
  if (!host) return;
  await api("/api/config", { host });
  poll();
});

$("#save-detector").addEventListener("click", async () => {
  await api("/api/track/config", { track_detector: $("#track-detector").value });
  loadVocabulary();
  pollTrack();
});

$("#save-remote-url").addEventListener("click", async () => {
  await api("/api/track/config", { track_remote_url: $("#track-remote-url").value.trim() });
  pollTrack();
});

async function loadVocabulary() {
  try {
    const vocab = await api("/api/track/vocabulary");
    $("#track-classes").innerHTML = (vocab.classes || [])
      .map((c) => `<option value="${escapeHtml(c)}">`).join("");
    $("#track-target").placeholder = vocab.open_vocabulary
      ? "anything you can name" : "person";
  } catch (e) {}
}

(async () => {
  try {
    const cfg = await api("/api/config");
    $("#host").value = cfg.host || "";
    $("#track-detector").value = cfg.track_detector || "onnx";
    $("#track-remote-url").value = cfg.track_remote_url || "";
    if (cfg.default_speed) {
      speed = cfg.default_speed;
      speedInput.value = speed;
      $("#speed-val").textContent = speed.toFixed(2);
    }
  } catch (e) {}
  poll();
  pollVoice();
  loadLab();
  loadVocabulary();
  pollTrack();
  pollMount();
})();

// Sensor readiness is independent of chassis connectivity.
async function pollSensors() {
  try {
    const response = await fetch('/api/sensors/status');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const state = await response.json();
    const depth = state.depth;
    $('#sensor-state').textContent = depth.connected
      ? `L515 live · ${depth.point_count} points · ${depth.published_fps} fps · ${depth.last_frame_age_ms} ms old`
      : `L515 unavailable: ${depth.error}`;
    if (!$('#depth-url').value && depth.url) $('#depth-url').value = depth.url;
    // No source configured yet (DEPTH_SERVER_URL unset): leave the link inert.
    if (depth.url) $('#depth-viewer').href = depth.url;
    else $('#depth-viewer').removeAttribute('href');
  } catch (error) {
    $('#sensor-state').textContent = `Sensor status unavailable: ${error}`;
  } finally {
    setTimeout(pollSensors, 3000);
  }
}
$('#depth-save').addEventListener('click', async () => {
  try {
    const result = await api('/api/sensors/config', {depth_url: $('#depth-url').value});
    if (!result.ok) throw new Error(result.error || 'Could not save source');
    $('#sensor-state').textContent = 'Source saved. Checking connection…';
  } catch (error) { $('#sensor-state').textContent = String(error); }
});
pollSensors();
