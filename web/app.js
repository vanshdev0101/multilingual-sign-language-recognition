import { HandLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const SLOTS = ["Left", "Right"];
const DIM = 126;
const USER_KEY = "signreader.samples.v1";

const $ = (id) => document.getElementById(id);
const els = {
  video: $("video"), canvas: $("overlay"), gate: $("gate"), gateText: $("gateText"),
  start: $("start"), status: $("status"), statusText: $("statusText"),
  letter: $("letter"), conf: $("conf"), confText: $("confText"), live: $("live"),
  buffer: $("buffer"), back: $("back"), clear: $("clear"),
  fps: $("fps"), hands: $("hands"), dist: $("dist"),
  sens: $("sens"), sensText: $("sensText"), hold: $("hold"), holdText: $("holdText"),
};

let base = { X: null, y: [], labels: [] };   // shipped model
let user = { X: [], y: [] };                 // samples taught in this browser
let maxDist = 1.2, landmarker = null, running = false;

/* ---------- features: must match the Python pipeline exactly ---------- */
function features(hands, w, h) {
  const v = new Float32Array(DIM);
  const aspect = w / h;
  for (const { label, pts } of hands) {
    const slot = SLOTS.indexOf(label);
    if (slot < 0) continue;
    const off = slot * 63;
    const wx = pts[0].x * aspect, wy = pts[0].y, wz = pts[0].z * aspect;
    let max = 0;
    const tmp = new Float32Array(63);
    for (let i = 0; i < 21; i++) {
      const x = pts[i].x * aspect - wx, y = pts[i].y - wy, z = pts[i].z * aspect - wz;
      tmp[i * 3] = x; tmp[i * 3 + 1] = y; tmp[i * 3 + 2] = z;
      const n = Math.hypot(x, y, z);
      if (n > max) max = n;
    }
    const s = max > 1e-6 ? 1 / max : 1;
    for (let i = 0; i < 63; i++) v[off + i] = tmp[i] * s;
  }
  return v;
}

// same sign with the other hand: swap slots, negate x
function mirror(v) {
  const m = new Float32Array(DIM);
  for (let slot = 0; slot < 2; slot++) {
    const from = slot * 63, to = (1 - slot) * 63;
    for (let i = 0; i < 21; i++) {
      m[to + i * 3]     = -v[from + i * 3];
      m[to + i * 3 + 1] =  v[from + i * 3 + 1];
      m[to + i * 3 + 2] =  v[from + i * 3 + 2];
    }
  }
  return m;
}

// tolerate a few degrees of wrist rotation, like the training augmentation
function rotate(v, deg) {
  if (!deg) return v;
  const t = deg * Math.PI / 180, c = Math.cos(t), s = Math.sin(t);
  const r = new Float32Array(DIM);
  for (let slot = 0; slot < 2; slot++) {
    const o = slot * 63;
    let occupied = false;
    for (let i = 0; i < 63; i++) if (v[o + i] !== 0) { occupied = true; break; }
    if (!occupied) continue;
    for (let i = 0; i < 21; i++) {
      const x = v[o + i * 3], y = v[o + i * 3 + 1];
      r[o + i * 3] = x * c - y * s;
      r[o + i * 3 + 1] = x * s + y * c;
      r[o + i * 3 + 2] = v[o + i * 3 + 2];
    }
  }
  return r;
}

/* ---------- nearest-neighbour vote over shipped + user samples ---------- */
function d2(a, bArr, off) {
  let s = 0;
  for (let i = 0; i < DIM; i++) { const d = a[i] - bArr[off + i]; s += d * d; }
  return s;
}

function classify(vec) {
  const variants = [];
  for (const deg of [-10, 0, 10]) {
    const r = rotate(vec, deg);
    variants.push(r, mirror(r));
  }
  const found = [];   // {dist, label}
  const scan = (arr, labelsOf, n, bias) => {
    for (let j = 0; j < n; j++) {
      let best = Infinity;
      for (const q of variants) {
        const d = d2(q, arr, j * DIM);
        if (d < best) best = d;
      }
      found.push({ d: Math.sqrt(best) * bias, label: labelsOf[j] });
    }
  };
  // user samples win ties: they match this person's hand
  scan(base.X, base.y, base.y.length, 1.0);
  if (user.y.length) scan(user.flat, user.y, user.y.length, 0.85);

  found.sort((a, b) => a.d - b.d);
  const k = found.slice(0, 5);
  if (!k.length) return null;
  const votes = new Map();
  for (const n of k) {
    const w = 1 / Math.max(n.d, 1e-6);
    votes.set(n.label, (votes.get(n.label) || 0) + w);
  }
  let label = null, top = 0, total = 0;
  for (const [l, w] of votes) { total += w; if (w > top) { top = w; label = l; } }
  return { label, conf: top / total, dist: k[0].d };
}

/* ---------- user-taught samples, kept in this browser ---------- */
function loadUser() {
  try {
    const raw = localStorage.getItem(USER_KEY);
    if (!raw) return;
    const o = JSON.parse(raw);
    user.y = o.y || [];
    user.flat = new Float32Array(o.x || []);
  } catch { user.y = []; user.flat = new Float32Array(0); }
  if (!user.flat) user.flat = new Float32Array(0);
}
function saveUser() {
  try {
    localStorage.setItem(USER_KEY, JSON.stringify({
      y: user.y, x: Array.from(user.flat, (n) => Math.round(n * 1000) / 1000),
    }));
    return true;
  } catch { return false; }
}
function addSamples(label, vecs) {
  const merged = new Float32Array(user.flat.length + vecs.length * DIM);
  merged.set(user.flat);
  vecs.forEach((v, i) => merged.set(v, user.flat.length + i * DIM));
  user.flat = merged;
  vecs.forEach(() => user.y.push(label));
}

/* ---------- camera loop ---------- */
const ctx = els.canvas.getContext("2d");
const CONNECTIONS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];

function draw(hands, w, h) {
  els.canvas.width = w; els.canvas.height = h;
  ctx.clearRect(0, 0, w, h);
  ctx.lineWidth = Math.max(2, w / 320);
  for (const { pts } of hands) {
    ctx.strokeStyle = "#22C55E";
    for (const [a, b] of CONNECTIONS) {
      ctx.beginPath();
      ctx.moveTo(pts[a].x * w, pts[a].y * h);
      ctx.lineTo(pts[b].x * w, pts[b].y * h);
      ctx.stroke();
    }
    ctx.fillStyle = "#F8FAFC";
    for (const p of pts) {
      ctx.beginPath(); ctx.arc(p.x * w, p.y * h, ctx.lineWidth * 1.1, 0, 6.284); ctx.fill();
    }
  }
}

let recent = [], lastCommitted = null, frames = 0, fpsAt = performance.now();
let capture = null;   // active teaching session

function setStatus(text, cls) {
  els.statusText.textContent = text;
  els.status.className = "pill" + (cls ? " " + cls : "");
}

function loop() {
  if (!running) return;
  const v = els.video;
  if (v.readyState >= 2) {
    const res = landmarker.detectForVideo(v, performance.now());
    const hands = (res.landmarks || []).map((pts, i) => ({
      pts,
      label: res.handedness?.[i]?.[0]?.categoryName ?? "Right",
    }));
    // the video is mirrored for the user, so mirror the landmarks to match
    const shown = hands.map(({ pts, label }) => ({
      label: label === "Left" ? "Right" : "Left",
      pts: pts.map((p) => ({ x: 1 - p.x, y: p.y, z: p.z })),
    }));
    draw(shown, v.videoWidth, v.videoHeight);
    els.hands.textContent = hands.length;

    let result = null;
    if (hands.length) {
      const vec = features(shown, v.videoWidth, v.videoHeight);
      if (capture) capture.vecs.push(vec);
      result = classify(vec);
    }
    render(result);

    frames++;
    const now = performance.now();
    if (now - fpsAt > 500) {
      els.fps.textContent = Math.round((frames * 1000) / (now - fpsAt));
      frames = 0; fpsAt = now;
    }
  }
  requestAnimationFrame(loop);
}

function render(result) {
  const hold = +els.hold.value;
  let label = null;
  if (result) {
    els.dist.textContent = result.dist.toFixed(2);
    const known = result.dist <= maxDist;
    label = known ? result.label : null;
    els.letter.textContent = known ? result.label : "?";
    els.letter.className = "letter " + (known ? "ok" : "no");
    els.conf.style.width = Math.round(result.conf * 100) + "%";
    els.confText.textContent = known
      ? `${Math.round(result.conf * 100)}% match`
      : "too far from any known sign";
  } else {
    els.dist.textContent = "—";
    els.letter.textContent = "—"; els.letter.className = "letter no";
    els.conf.style.width = "0%"; els.confText.textContent = "waiting for a hand";
  }

  recent.push(label);
  if (recent.length > hold) recent.shift();
  const stable = recent.length === hold && recent.every((x) => x && x === recent[0]) ? recent[0] : null;
  if (stable && stable !== lastCommitted) {
    els.buffer.textContent = (els.buffer.textContent + stable).slice(-40);
    els.live.textContent = `Added ${stable}`;
    lastCommitted = stable;
  }
  if (!stable) lastCommitted = null;
}

/* ---------- start-up ---------- */
async function init() {
  try {
    const [meta, buf, vision] = await Promise.all([
      fetch("model.json").then((r) => r.json()),
      fetch("model.bin").then((r) => r.arrayBuffer()),
      FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"),
    ]);
    base.X = new Float32Array(buf);
    base.labels = meta.labels;
    base.y = meta.y.map((i) => meta.labels[i]);
    maxDist = meta.maxDist;
    els.sens.value = maxDist;
    updateSens();

    landmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO", numHands: 2,
      minHandDetectionConfidence: 0.4, minHandPresenceConfidence: 0.4, minTrackingConfidence: 0.4,
    });
    loadUser();
    refreshTaught();
    setStatus("Ready — start the camera");
    els.start.disabled = false;
  } catch (e) {
    setStatus("Could not load", "err");
    els.gateText.textContent = "Failed to load the model: " + e.message;
    console.error(e);
  }
}

els.start.addEventListener("click", async () => {
  els.start.disabled = true;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 960, height: 720 } });
    els.video.srcObject = stream;
    await els.video.play();
    els.gate.style.display = "none";
    running = true;
    setStatus("Live", "live");
    loop();
  } catch (e) {
    els.start.disabled = false;
    setStatus("Camera blocked", "err");
    els.gateText.textContent = "Camera permission was refused. Allow it in your browser, then try again.";
  }
});

els.clear.addEventListener("click", () => { els.buffer.textContent = ""; lastCommitted = null; });
els.back.addEventListener("click", () => {
  els.buffer.textContent = els.buffer.textContent.slice(0, -1); lastCommitted = null;
});

function updateSens() {
  maxDist = +els.sens.value;
  els.sensText.textContent = `Accepting matches closer than ${maxDist.toFixed(2)}. Lower = fewer wrong guesses, more "?".`;
}
els.sens.addEventListener("input", updateSens);
function updateHold() {
  els.holdText.textContent = `A sign must be held for ${els.hold.value} frames (about ${(els.hold.value / 30).toFixed(1)}s) before it is added.`;
}
els.hold.addEventListener("input", updateHold);
updateHold();

init();

/* ---------- teaching: record your own hand for a sign ---------- */
const ALL = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("").concat("123456789".split(""));
const TEACH_FRAMES = 30;

function refreshTaught() {
  const sel = $("teachLabel");
  if (!sel.options.length) {
    for (const l of ALL) {
      const o = document.createElement("option");
      o.value = l; o.textContent = l;
      sel.appendChild(o);
    }
    sel.value = "A";
  }
  const counts = {};
  for (const l of user.y) counts[l] = (counts[l] || 0) + 1;
  const taught = Object.keys(counts).sort();
  $("teachText").textContent = taught.length
    ? `You have taught ${taught.length} sign${taught.length > 1 ? "s" : ""}: ${taught.map((l) => `${l} (${counts[l]})`).join(", ")}`
    : "You have not taught any signs yet. The shipped model is being used.";
}

$("teach").addEventListener("click", async () => {
  if (!running) { $("teachText").textContent = "Start the camera first."; return; }
  if (capture) return;
  const label = $("teachLabel").value;
  const btn = $("teach");
  btn.disabled = true;

  for (let n = 3; n > 0; n--) {
    $("teachText").textContent = `Get into "${label}" — recording in ${n}…`;
    await new Promise((r) => setTimeout(r, 1000));
  }
  capture = { vecs: [] };
  while (capture.vecs.length < TEACH_FRAMES) {
    $("teachText").textContent = `Hold "${label}" — ${capture.vecs.length}/${TEACH_FRAMES}. Move your hand slightly.`;
    await new Promise((r) => setTimeout(r, 100));
  }
  const vecs = capture.vecs.slice(0, TEACH_FRAMES);
  capture = null;

  addSamples(label, vecs);
  const saved = saveUser();
  refreshTaught();
  if (!saved) $("teachText").textContent += "  (Too much data to save — older signs may be lost on reload.)";
  btn.disabled = false;
});

$("forget").addEventListener("click", () => {
  if (!user.y.length) return;
  user.y = []; user.flat = new Float32Array(0);
  localStorage.removeItem(USER_KEY);
  refreshTaught();
});
