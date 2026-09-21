"""
Local High-Speed Viewpoint Annotation Server for Real Cattle Images.

Features:
- Zero external dependencies (uses standard library http.server).
- Blazing-fast keyboard-first workflow (press 1-6 to save and advance instantly).
- Autosaves immediately to CSV on every label.
- Resumes automatically at the first unreviewed sample.
- Zero metadata display (no dataset name, cow ID, camera ID, BCS, behavior, or hints) to prevent anchoring.
- Serves images via opaque /api/image?id=vp1k_XXXX endpoint so client never inspects dataset paths.
- Preloads upcoming images for 0ms transition latency.

Usage:
    python scripts/viewpoint_annotator.py [--port 8088] [--manifest artifacts/perception_audit/viewpoint_1000_annotation_manifest.csv]
"""

import argparse
import csv
import json
import mimetypes
import os
import sys
import threading
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_MANIFEST = os.path.join(REPO_ROOT, "artifacts", "perception_audit", "viewpoint_1000_annotation_manifest.csv")

VALID_VIEWPOINTS = ["rear", "rear-oblique", "side", "front-oblique", "front", "unknown / ambiguous"]
OCCLUSION_CYCLE = ["none", "partial", "severe"]
BODY_CUTOFF_CYCLE = ["none", "partial", "severe"]
MULTIPLE_COWS_VALUES = ["no", "yes"]

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Real-Cattle Viewpoint Annotator</title>
<style>
  :root {
    --bg-dark: #0f172a;
    --card-bg: #1e293b;
    --border-color: #334155;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --accent: #3b82f6;
    --accent-hover: #2563eb;
    --badge-none: #475569;
    --badge-partial: #d97706;
    --badge-severe: #dc2626;
    --badge-yes: #7c3aed;
    --badge-active: #10b981;
  }

  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
    user-select: none;
  }

  body {
    background-color: var(--bg-dark);
    color: var(--text-primary);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    height: 100vh;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  /* Header */
  header {
    background: var(--card-bg);
    border-bottom: 1px solid var(--border-color);
    padding: 10px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-shrink: 0;
  }

  .title-group {
    display: flex;
    align-items: center;
    gap: 16px;
  }

  h1 {
    font-size: 1.1rem;
    font-weight: 700;
    color: #e2e8f0;
    letter-spacing: 0.5px;
  }

  .sample-badge {
    background: #0ea5e9;
    color: #032030;
    font-weight: 800;
    font-family: monospace;
    font-size: 1rem;
    padding: 3px 10px;
    border-radius: 6px;
  }

  .progress-group {
    display: flex;
    align-items: center;
    gap: 16px;
  }

  .progress-bar-container {
    width: 200px;
    height: 10px;
    background: #334155;
    border-radius: 5px;
    overflow: hidden;
  }

  .progress-bar {
    height: 100%;
    background: #10b981;
    width: 0%;
    transition: width 0.2s ease;
  }

  .progress-text {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text-secondary);
    font-variant-numeric: tabular-nums;
  }

  /* Main Workspace */
  main {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 12px 24px;
    min-height: 0;
    position: relative;
  }

  .image-stage {
    flex: 1;
    width: 100%;
    max-width: 1200px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #020617;
    border: 2px solid var(--border-color);
    border-radius: 12px;
    overflow: hidden;
    position: relative;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
  }

  #main-image {
    max-width: 100%;
    max-height: 100%;
    object-fit: contain;
    display: block;
  }

  /* Status Toast */
  .toast {
    position: absolute;
    top: 20px;
    right: 20px;
    background: #10b981;
    color: #ffffff;
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 0.85rem;
    font-weight: 700;
    opacity: 0;
    transform: translateY(-8px);
    transition: all 0.2s ease;
    pointer-events: none;
  }
  .toast.show {
    opacity: 1;
    transform: translateY(0);
  }

  /* Metadata Controls Row */
  .metadata-row {
    display: flex;
    gap: 14px;
    margin-top: 10px;
    margin-bottom: 8px;
    flex-shrink: 0;
  }

  .meta-btn {
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    padding: 6px 14px;
    color: var(--text-secondary);
    font-size: 0.85rem;
    font-weight: 600;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 8px;
    transition: all 0.15s ease;
  }

  .meta-btn:hover {
    border-color: #64748b;
    color: var(--text-primary);
  }

  .key-pill {
    background: #334155;
    color: #f1f5f9;
    font-family: monospace;
    font-size: 0.75rem;
    font-weight: 800;
    padding: 2px 6px;
    border-radius: 4px;
    border: 1px solid #475569;
  }

  .tag-val {
    font-weight: 800;
    text-transform: uppercase;
    font-size: 0.8rem;
    padding: 2px 6px;
    border-radius: 4px;
  }
  .tag-none { background: #334155; color: #cbd5e1; }
  .tag-partial { background: #b45309; color: #fef3c7; }
  .tag-severe { background: #b91c1c; color: #fee2e2; }
  .tag-no { background: #334155; color: #cbd5e1; }
  .tag-yes { background: #6d28d9; color: #ede9fe; }

  /* Viewpoint Action Buttons */
  .viewpoint-row {
    display: flex;
    gap: 10px;
    margin-bottom: 10px;
    flex-shrink: 0;
    max-width: 1200px;
    width: 100%;
    justify-content: center;
  }

  .vp-btn {
    flex: 1;
    max-width: 180px;
    background: var(--card-bg);
    border: 2px solid var(--border-color);
    border-radius: 10px;
    padding: 10px 8px;
    color: var(--text-primary);
    font-size: 0.9rem;
    font-weight: 700;
    cursor: pointer;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 4px;
    transition: all 0.12s ease;
  }

  .vp-btn .vp-key {
    background: #3b82f6;
    color: #ffffff;
    font-family: monospace;
    font-size: 0.85rem;
    font-weight: 900;
    padding: 2px 8px;
    border-radius: 4px;
  }

  .vp-btn:hover {
    border-color: #3b82f6;
    background: #1e3a8a;
    transform: translateY(-2px);
  }

  .vp-btn.selected {
    border-color: #10b981;
    background: #064e3b;
    box-shadow: 0 0 12px rgba(16, 185, 129, 0.4);
  }
  .vp-btn.selected .vp-key {
    background: #10b981;
  }

  /* Navigation and Footer */
  footer {
    background: var(--card-bg);
    border-top: 1px solid var(--border-color);
    padding: 8px 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-shrink: 0;
    font-size: 0.85rem;
    color: var(--text-secondary);
  }

  .nav-group {
    display: flex;
    align-items: center;
    gap: 10px;
  }

  .nav-btn {
    background: #334155;
    border: 1px solid #475569;
    border-radius: 6px;
    color: #f1f5f9;
    padding: 4px 10px;
    font-size: 0.8rem;
    font-weight: 600;
    cursor: pointer;
  }
  .nav-btn:hover { background: #475569; }

  .jump-input {
    width: 60px;
    background: #0f172a;
    border: 1px solid #475569;
    border-radius: 4px;
    color: #fff;
    padding: 3px 6px;
    text-align: center;
    font-size: 0.85rem;
  }

  .shortcuts-help {
    display: flex;
    gap: 16px;
    align-items: center;
  }
  .shortcut-item {
    display: flex;
    align-items: center;
    gap: 4px;
  }
</style>
</head>
<body>

<header>
  <div class="title-group">
    <h1>CATTLE VIEWPOINT ANNOTATOR</h1>
    <span class="sample-badge" id="sample-id-display">vp1k_????</span>
    <span id="review-status-badge" style="font-size: 0.75rem; padding: 2px 8px; border-radius: 4px; background: #475569; color: #fff; font-weight: 700;">UNREVIEWED</span>
  </div>

  <div class="progress-group">
    <div class="progress-bar-container">
      <div class="progress-bar" id="progress-bar"></div>
    </div>
    <span class="progress-text" id="progress-text">0 / 1000 (0.0%)</span>
  </div>
</header>

<main>
  <div class="image-stage">
    <img id="main-image" src="" alt="Cattle Sample" />
    <div class="toast" id="toast">Saved!</div>
  </div>

  <!-- Metadata controls -->
  <div class="metadata-row">
    <button class="meta-btn" id="btn-cycle-occlusion" onclick="cycleOcclusion()">
      <span class="key-pill">O</span>
      <span>Occlusion:</span>
      <span class="tag-val tag-none" id="val-occlusion">NONE</span>
    </button>

    <button class="meta-btn" id="btn-cycle-cutoff" onclick="cycleCutoff()">
      <span class="key-pill">C</span>
      <span>Body Cutoff:</span>
      <span class="tag-val tag-none" id="val-cutoff">NONE</span>
    </button>

    <button class="meta-btn" id="btn-toggle-multi" onclick="toggleMulti()">
      <span class="key-pill">M</span>
      <span>Multiple Cows:</span>
      <span class="tag-val tag-no" id="val-multi">NO</span>
    </button>
  </div>

  <!-- Viewpoint Selection Buttons -->
  <div class="viewpoint-row">
    <button class="vp-btn" id="vp-btn-1" onclick="selectViewpoint('rear')">
      <span class="vp-key">1</span>
      <span>Rear</span>
    </button>
    <button class="vp-btn" id="vp-btn-2" onclick="selectViewpoint('rear-oblique')">
      <span class="vp-key">2</span>
      <span>Rear-Oblique</span>
    </button>
    <button class="vp-btn" id="vp-btn-3" onclick="selectViewpoint('side')">
      <span class="vp-key">3</span>
      <span>Side</span>
    </button>
    <button class="vp-btn" id="vp-btn-4" onclick="selectViewpoint('front-oblique')">
      <span class="vp-key">4</span>
      <span>Front-Oblique</span>
    </button>
    <button class="vp-btn" id="vp-btn-5" onclick="selectViewpoint('front')">
      <span class="vp-key">5</span>
      <span>Front</span>
    </button>
    <button class="vp-btn" id="vp-btn-6" onclick="selectViewpoint('unknown / ambiguous')">
      <span class="vp-key">6</span>
      <span>Unknown / Ambiguous</span>
    </button>
  </div>
</main>

<footer>
  <div class="shortcuts-help">
    <div class="shortcut-item"><span class="key-pill">1-6</span> Viewpoint & Advance</div>
    <div class="shortcut-item"><span class="key-pill">O</span> Cycle Occlusion</div>
    <div class="shortcut-item"><span class="key-pill">C</span> Cycle Cutoff</div>
    <div class="shortcut-item"><span class="key-pill">M</span> Toggle Multi-Cow</div>
    <div class="shortcut-item"><span class="key-pill">← / Backspace</span> Prev</div>
    <div class="shortcut-item"><span class="key-pill">→</span> Next</div>
  </div>

  <div class="nav-group">
    <button class="nav-btn" onclick="prevSample()">← Prev</button>
    <button class="nav-btn" onclick="nextSample()">Next →</button>
    <span>Jump to:</span>
    <input type="number" id="jump-index" class="jump-input" min="1" max="1000" onkeydown="if(event.key==='Enter') jumpToSample()" />
  </div>
</footer>

<script>
let currentIndex = 0;
let totalSamples = 1000;
let currentSample = null;

const OCCLUSION_CYCLE = ["none", "partial", "severe"];
const CUTOFF_CYCLE = ["none", "partial", "severe"];

const VP_MAP = {
  "1": "rear",
  "2": "rear-oblique",
  "3": "side",
  "4": "front-oblique",
  "5": "front",
  "6": "unknown / ambiguous"
};

const VP_BUTTONS = {
  "rear": "vp-btn-1",
  "rear-oblique": "vp-btn-2",
  "side": "vp-btn-3",
  "front-oblique": "vp-btn-4",
  "front": "vp-btn-5",
  "unknown / ambiguous": "vp-btn-6"
};

async function init() {
  const res = await fetch('/api/status');
  const status = await res.json();
  totalSamples = status.total;
  currentIndex = status.first_unreviewed_index >= 0 ? status.first_unreviewed_index : 0;
  loadSample(currentIndex);
}

async function loadSample(index) {
  if (index < 0 || index >= totalSamples) return;
  currentIndex = index;

  const res = await fetch(`/api/sample?index=${index}`);
  const sample = await res.json();
  currentSample = sample;

  // Update UI Elements
  document.getElementById('sample-id-display').textContent = sample.sample_id;
  document.getElementById('main-image').src = `/api/image?id=${sample.sample_id}`;
  document.getElementById('jump-index').value = index + 1;

  // Review Status
  const statusBadge = document.getElementById('review-status-badge');
  if (sample.review_status === 'human_verified') {
    statusBadge.textContent = 'REVIEWED';
    statusBadge.style.background = '#10b981';
  } else {
    statusBadge.textContent = 'PENDING';
    statusBadge.style.background = '#d97706';
  }

  // Metadata Display
  updateMetaDisplay('val-occlusion', sample.occlusion);
  updateMetaDisplay('val-cutoff', sample.body_cutoff);
  updateMetaDisplay('val-multi', sample.multiple_cows);

  // Viewpoint Buttons highlight
  for (const [vp, btnId] of Object.entries(VP_BUTTONS)) {
    const btn = document.getElementById(btnId);
    if (sample.viewpoint === vp) {
      btn.classList.add('selected');
    } else {
      btn.classList.remove('selected');
    }
  }

  // Preload next 2 images
  for (let offset = 1; offset <= 2; offset++) {
    const nextIdx = index + offset;
    if (nextIdx < totalSamples) {
      fetch(`/api/sample?index=${nextIdx}`).then(r => r.json()).then(nextS => {
        const preloadImg = new Image();
        preloadImg.src = `/api/image?id=${nextS.sample_id}`;
      });
    }
  }

  updateProgress();
}

function updateMetaDisplay(elementId, val) {
  const el = document.getElementById(elementId);
  el.textContent = val.toUpperCase();
  el.className = `tag-val tag-${val}`;
}

async function updateProgress() {
  const res = await fetch('/api/status');
  const status = await res.json();
  const pct = ((status.reviewed_count / status.total) * 100).toFixed(1);
  document.getElementById('progress-bar').style.width = `${pct}%`;
  document.getElementById('progress-text').textContent = `${status.reviewed_count} / ${status.total} (${pct}%)`;
}

function cycleOcclusion() {
  if (!currentSample) return;
  let idx = OCCLUSION_CYCLE.indexOf(currentSample.occlusion);
  idx = (idx + 1) % OCCLUSION_CYCLE.length;
  currentSample.occlusion = OCCLUSION_CYCLE[idx];
  updateMetaDisplay('val-occlusion', currentSample.occlusion);
}

function cycleCutoff() {
  if (!currentSample) return;
  let idx = CUTOFF_CYCLE.indexOf(currentSample.body_cutoff);
  idx = (idx + 1) % CUTOFF_CYCLE.length;
  currentSample.body_cutoff = CUTOFF_CYCLE[idx];
  updateMetaDisplay('val-cutoff', currentSample.body_cutoff);
}

function toggleMulti() {
  if (!currentSample) return;
  currentSample.multiple_cows = currentSample.multiple_cows === 'yes' ? 'no' : 'yes';
  updateMetaDisplay('val-multi', currentSample.multiple_cows);
}

async function selectViewpoint(vp) {
  if (!currentSample) return;
  currentSample.viewpoint = vp;

  // Immediately send save request
  const payload = {
    index: currentIndex,
    sample_id: currentSample.sample_id,
    viewpoint: vp,
    occlusion: currentSample.occlusion,
    body_cutoff: currentSample.body_cutoff,
    multiple_cows: currentSample.multiple_cows
  };

  showToast();

  try {
    const res = await fetch('/api/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const result = await res.json();
    if (result.success) {
      if (currentIndex + 1 < totalSamples) {
        loadSample(currentIndex + 1);
      } else {
        alert("All 1,000 samples have been reviewed! Amazing work! 🎉");
      }
    }
  } catch (err) {
    console.error("Save failed:", err);
    alert("Error saving annotation! Check server logs.");
  }
}

function showToast() {
  const toast = document.getElementById('toast');
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 400);
}

function prevSample() {
  if (currentIndex > 0) loadSample(currentIndex - 1);
}

function nextSample() {
  if (currentIndex + 1 < totalSamples) loadSample(currentIndex + 1);
}

function jumpToSample() {
  const val = parseInt(document.getElementById('jump-index').value, 10);
  if (!isNaN(val) && val >= 1 && val <= totalSamples) {
    loadSample(val - 1);
  }
}

// Global Keyboard Handler
window.addEventListener('keydown', (e) => {
  // If user is focused on jump input, do not intercept
  if (e.target.tagName === 'INPUT') return;

  const key = e.key.toLowerCase();

  // 1-6: Select Viewpoint and advance
  if (key in VP_MAP) {
    e.preventDefault();
    selectViewpoint(VP_MAP[key]);
  } else if (key === 'o') {
    e.preventDefault();
    cycleOcclusion();
  } else if (key === 'c') {
    e.preventDefault();
    cycleCutoff();
  } else if (key === 'm') {
    e.preventDefault();
    toggleMulti();
  } else if (key === 'backspace' || key === 'arrowleft') {
    e.preventDefault();
    prevSample();
  } else if (key === 'arrowright') {
    e.preventDefault();
    nextSample();
  }
});

init();
</script>
</body>
</html>
"""


class ManifestManager:
    """Thread-safe CSV manifest manager with immediate atomic writes."""
    def __init__(self, manifest_path):
        self.manifest_path = manifest_path
        self.lock = threading.Lock()
        self.rows = []
        self.sample_id_to_idx = {}
        self.load()

    def load(self):
        with self.lock:
            self.rows = []
            self.sample_id_to_idx = {}
            with open(self.manifest_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                self.fieldnames = reader.fieldnames
                for i, row in enumerate(reader):
                    self.rows.append(row)
                    self.sample_id_to_idx[row["sample_id"]] = i

    def get_status(self):
        with self.lock:
            total = len(self.rows)
            reviewed = sum(1 for r in self.rows if r.get("review_status") == "human_verified" or (r.get("viewpoint") != "UNREVIEWED" and r.get("viewpoint") != ""))
            first_unreviewed = -1
            for i, r in enumerate(self.rows):
                if r.get("review_status") != "human_verified" and (r.get("viewpoint") == "UNREVIEWED" or not r.get("viewpoint")):
                    first_unreviewed = i
                    break
            return {
                "total": total,
                "reviewed_count": reviewed,
                "first_unreviewed_index": first_unreviewed,
            }

    def get_sample(self, index):
        with self.lock:
            if index < 0 or index >= len(self.rows):
                return None
            row = self.rows[index]
            # Omit sensitive provenance from client payload to prevent anchoring
            return {
                "index": index,
                "sample_id": row["sample_id"],
                "viewpoint": row.get("viewpoint", "UNREVIEWED"),
                "occlusion": row.get("occlusion", "none") or "none",
                "body_cutoff": row.get("body_cutoff", "none") or "none",
                "multiple_cows": row.get("multiple_cows", "no") or "no",
                "review_status": row.get("review_status", "pending_human_review") or "pending_human_review",
            }

    def get_image_path(self, sample_id):
        with self.lock:
            idx = self.sample_id_to_idx.get(sample_id)
            if idx is None:
                return None
            rel_p = self.rows[idx]["source_image_path"]
            return os.path.abspath(os.path.join(REPO_ROOT, rel_p)) if not os.path.isabs(rel_p) else os.path.abspath(rel_p)

    def save_annotation(self, index, sample_id, viewpoint, occlusion, body_cutoff, multiple_cows):
        with self.lock:
            if index < 0 or index >= len(self.rows):
                return False
            row = self.rows[index]
            if row["sample_id"] != sample_id:
                return False

            row["viewpoint"] = viewpoint
            row["occlusion"] = occlusion
            row["body_cutoff"] = body_cutoff
            row["multiple_cows"] = multiple_cows
            row["review_status"] = "human_verified"

            # Immediate atomic write to manifest
            tmp_path = self.manifest_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
                writer.writerows(self.rows)
            os.replace(tmp_path, self.manifest_path)
            return True


class AnnotationServerHandler(BaseHTTPRequestHandler):
    manager: ManifestManager = None

    def log_message(self, format, *args):
        # Silence routine static asset logs for quiet terminal operation
        pass

    def send_json_response(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif path == "/api/status":
            status = self.manager.get_status()
            self.send_json_response(status)

        elif path == "/api/sample":
            try:
                idx = int(query.get("index", [0])[0])
            except (ValueError, IndexError):
                idx = 0
            sample = self.manager.get_sample(idx)
            if sample is None:
                self.send_json_response({"error": "Index out of range"}, status=404)
            else:
                self.send_json_response(sample)

        elif path == "/api/image":
            sample_id = query.get("id", [""])[0]
            img_path = self.manager.get_image_path(sample_id)
            if not img_path or not os.path.exists(img_path):
                self.send_response(404)
                self.end_headers()
                return

            mime_type, _ = mimetypes.guess_type(img_path)
            if not mime_type:
                mime_type = "image/jpeg"

            try:
                with open(img_path, "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", mime_type)
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Cache-Control", "public, max-age=86400")
                self.end_headers()
                self.wfile.write(data)
            except Exception as e:
                self.send_response(500)
                self.end_headers()

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/save":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode("utf-8"))
                idx = int(data["index"])
                sample_id = str(data["sample_id"])
                viewpoint = str(data["viewpoint"])
                occlusion = str(data.get("occlusion", "none"))
                body_cutoff = str(data.get("body_cutoff", "none"))
                multiple_cows = str(data.get("multiple_cows", "no"))

                if viewpoint not in VALID_VIEWPOINTS:
                    self.send_json_response({"error": "Invalid viewpoint"}, status=400)
                    return

                success = self.manager.save_annotation(
                    idx, sample_id, viewpoint, occlusion, body_cutoff, multiple_cows
                )
                if success:
                    self.send_json_response({"success": True, "saved_index": idx, "next_index": idx + 1})
                else:
                    self.send_json_response({"error": "Save failed"}, status=500)
            except Exception as e:
                self.send_json_response({"error": str(e)}, status=400)
        else:
            self.send_response(404)
            self.end_headers()


def run_server(port=8088, manifest_path=DEFAULT_MANIFEST):
    if not os.path.exists(manifest_path):
        print(f"Error: Manifest not found at {manifest_path}")
        sys.exit(1)

    manager = ManifestManager(manifest_path)
    AnnotationServerHandler.manager = manager

    server = HTTPServer(("127.0.0.1", port), AnnotationServerHandler)
    status = manager.get_status()
    print("=" * 65)
    print(" REAL-CATTLE VIEWPOINT ANNOTATION SERVER")
    print("=" * 65)
    print(f" Port:             {port}")
    print(f" Manifest:         {manifest_path}")
    print(f" Total Samples:    {status['total']}")
    print(f" Reviewed:         {status['reviewed_count']} / {status['total']}")
    print(f" Starting Index:   {status['first_unreviewed_index']}")
    print("-" * 65)
    print(f" Access URL:       http://localhost:{port}")
    print("=" * 65)
    print(" Controls:")
    print("   1-6: Set Viewpoint & Advance immediately")
    print("     1 = rear")
    print("     2 = rear-oblique")
    print("     3 = side")
    print("     4 = front-oblique")
    print("     5 = front")
    print("     6 = unknown / ambiguous")
    print("   O:   Cycle Occlusion (none -> partial -> severe)")
    print("   C:   Cycle Body Cutoff (none -> partial -> severe)")
    print("   M:   Toggle Multiple Cows (no <-> yes)")
    print("   ← / Backspace: Previous sample / correct previous label")
    print("   →:   Next sample")
    print("=" * 65)
    print(" Press Ctrl+C to stop the server.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nAnnotation server stopped cleanly.")
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Cattle Viewpoint Annotation Server")
    parser.add_argument("--port", type=int, default=8088, help="Port to bind server (default: 8088)")
    parser.add_argument("--manifest", type=str, default=DEFAULT_MANIFEST, help="Path to annotation CSV manifest")
    args = parser.parse_args()

    run_server(port=args.port, manifest_path=args.manifest)
