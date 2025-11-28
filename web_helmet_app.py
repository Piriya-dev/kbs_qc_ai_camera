# web_helmet_yolo_app.py
# Confidential – Internal Use Only
#
# Flask web app for real-time helmet detection using YOLO PPE model (hardhat/no_hardhat).
#
# Assumed classes from PPE model (dataset2, Roboflow):
#   names: ['gloves', 'hardhat', 'no_gloves', 'no_hardhat',
#           'no_safety_shoes', 'no_safety_vest', 'person',
#           'safety_shoes', 'safety_vest']
#
# Logic:
#   - Run YOLO on each frame.
#   - If any "no_hardhat" (no helmet) detection => FAIL (red) for 3 seconds.
#   - Else if any "hardhat" detection => PASS (green) for 3 seconds.
#
# UI:
#   - Video stream (MJPEG) in <img>.
#   - Detection ON/OFF toggle.
#   - Detection duration dropdown (1, 30, 60 s) with auto-stop.
#   - PASS / FAIL big text OUTSIDE video, centered below.
#   - Status label overlaid in top-left of video.
#   - Test Snapshot button (manual snapshot).
#
# NOTE: No Keras, no Teachable Machine here – YOLO best.pt only.

import os
import time
import cv2
import numpy as np
from ultralytics import YOLO
from flask import (
    Flask,
    Response,
    render_template_string,
    jsonify,
    request,
    make_response,
)

# ───────────────────────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# RTSP stream (sub-stream 102 recommended for speed)
RTSP_URL = "rtsp://admin:KBSit%402468@192.168.4.190:554/ISAPI/Streaming/channels/102"

# YOLO PPE model path (trained on dataset2 with hardhat/no_hardhat)
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

# YOLO settings
YOLO_CONF = 0.50
YOLO_IMGSZ = 600
RUN_EVERY_N = 2  # run YOLO every N frames for speed

# Display size
MAX_WIDTH = 1048

# Detection session duration
ALLOWED_DURATIONS = [1, 30, 60, 90, 120]
DEFAULT_DURATION = 60

# PASS / FAIL display window (seconds)
PASS_SEC = 3.0
FAIL_SEC = 3.0

# RTSP stability
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "1"

# ───────────────────────────────────────────────────────────────
# SNAPSHOT CONFIG
# ───────────────────────────────────────────────────────────────

SNAPSHOT_DIR = os.path.join(BASE_DIR, "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
print(f"[INFO] Snapshot directory: {SNAPSHOT_DIR}")

MIN_SNAPSHOT_INTERVAL = 1.0  # seconds between auto snapshots

# ───────────────────────────────────────────────────────────────
# LOAD YOLO MODEL
# ───────────────────────────────────────────────────────────────

if not os.path.isfile(YOLO_MODEL_PATH):
    raise FileNotFoundError(f"YOLO model not found: {YOLO_MODEL_PATH}")

print(f"[INFO] Loading YOLO PPE model from: {YOLO_MODEL_PATH}")
helmet_model = YOLO(YOLO_MODEL_PATH)
CLASS_NAMES = helmet_model.names
print("[INFO] Model classes:")
for cid, cname in CLASS_NAMES.items():
    print(f"  id={cid}: {cname}")

# ───────────────────────────────────────────────────────────────
# GLOBAL STATE
# ───────────────────────────────────────────────────────────────

state = {
    "detect_enabled": True,
    "duration_sec": DEFAULT_DURATION,
    "detect_started_at": None,
}

# PASS / FAIL timers
last_pass_ts = None
last_fail_ts = None

# Snapshot state
last_snapshot_ts = 0.0          # auto snapshot timer
last_frame_for_snapshot = None  # last frame for manual snapshot

# Status for web polling
# pass_fail: "pass", "fail", "none"
last_status = {
    "pass_fail": "none",
    "text": "NO DETECTION",
}

# ───────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ───────────────────────────────────────────────────────────────

def resize_for_display(frame, max_width=MAX_WIDTH):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / float(w)
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)


def maybe_auto_stop_detection():
    if not state["detect_enabled"]:
        return
    duration = state["duration_sec"]
    start_ts = state["detect_started_at"]
    if duration is None or start_ts is None:
        return
    elapsed = time.time() - start_ts
    if elapsed > duration:
        state["detect_enabled"] = False
        state["detect_started_at"] = None
        print(f"[INFO] Auto-stopped detection after {duration} seconds")


def compute_remaining_seconds():
    if not state["detect_enabled"]:
        return None
    duration = state["duration_sec"]
    start_ts = state["detect_started_at"]
    if duration is None or start_ts is None:
        return None
    elapsed = time.time() - start_ts
    remaining = int(duration - elapsed)
    return max(0, remaining)


def is_helmet_label(label: str) -> bool:
    """
    Helmet (PASS) for dataset2:
      - 'hardhat'
      - or any label containing 'hardhat'/'helmet' without 'no'
    """
    l = label.lower()
    if "no_hardhat" in l:
        return False
    return ("hardhat" in l or "helmet" in l) and ("no" not in l)


def is_no_helmet_label(label: str) -> bool:
    """
    No helmet (FAIL) for dataset2:
      - 'no_hardhat'
      - or label containing both 'no' and 'hardhat'/'helmet'
    """
    l = label.lower()
    if "no_hardhat" in l:
        return True
    return ("no" in l) and ("hardhat" in l or "helmet" in l)


def save_snapshot(image: np.ndarray, prefix: str = "snap") -> str:
    """Save a snapshot image to SNAPSHOT_DIR and return filepath."""
    ts_str = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{ts_str}.jpg"
    filepath = os.path.join(SNAPSHOT_DIR, filename)
    ok = cv2.imwrite(filepath, image)
    if ok:
        print(f"[SNAPSHOT] Saved snapshot: {filepath}")
    else:
        print(f"[SNAPSHOT] ERROR: Failed to save snapshot: {filepath}")
    return filepath

# ───────────────────────────────────────────────────────────────
# FLASK APP + HTML
# ───────────────────────────────────────────────────────────────

app = Flask(__name__)

HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>KBS Helmet Detector Camera</title>
  <style>
    body { background:#111; color:#eee; font-family:Arial,sans-serif; text-align:center; }
    h1 { margin-top:20px; }
    .container { display:flex; flex-direction:column; align-items:center; margin-top:20px; gap:10px; }
    .video-frame { border:3px solid #0f0; border-radius:8px; overflow:hidden; max-width:90vw; }
    img.stream { max-width:100%; height:auto; display:block; }
    .controls { margin-top:10px; display:flex; gap:10px; align-items:center; justify-content:center; flex-wrap:wrap; }
    button {
      padding:8px 16px;
      border-radius:6px;
      border:none;
      cursor:pointer;
      font-size:14px;
      font-weight:bold;
    }
    #toggleBtn.on  { background:#c00; color:#fff; }
    #toggleBtn.off { background:#0a0; color:#fff; }
    #snapshotBtn { background:#0066cc; color:#fff; }
    select {
      padding:6px 10px;
      border-radius:4px;
      border:1px solid #555;
      background:#222;
      color:#eee;
    }
    label { font-size:14px; }
    .hint { margin-top:10px; color:#aaa; font-size:13px; }
    .pass-fail-banner {
      margin-top:15px;
      font-size:40px;
      font-weight:bold;
      text-align:center;
      min-height:40px;
    }
    .pass-fail-pass { color:#00ff00; }
    .pass-fail-fail { color:#ff0000; }
  </style>
</head>
<body>
  <h1>KBS Safety Helmet Detector</h1>
  <div class="container">
    <div class="video-frame">
      <img class="stream" id="streamImg" src="{{ url_for('video_feed') }}" alt="Video stream">
    </div>

    <div class="controls">
      <button id="toggleBtn" class="on" onclick="toggleDetection()">Turn OFF detection</button>
      <label for="durationSelect">Detection duration:</label>
      <select id="durationSelect" onchange="changeDuration()">
        <option value="1">1s</option>
        <option value="30">30s</option>
        <option value="60">60s</option>
      </select>
      <button id="snapshotBtn" onclick="manualSnapshot()">Test Snapshot</button>
    </div>

    <!-- PASS / FAIL OUTSIDE VIDEO -->
    <div class="pass-fail-banner">
      <span id="passFailText"></span>
    </div>

    <div class="hint">
      PASS (green) when helmet is detected, FAIL (red) when NO HELMET is detected.<br>
      PASS/FAIL stays for 3 seconds after the last detection.<br>
      Click "Test Snapshot" to save current frame to the snapshots folder.
    </div>
  </div>

  <script>
    let detectionOn = true;
    let currentDuration = 60;

    function updateToggleButton(on) {
      const btn = document.getElementById('toggleBtn');
      detectionOn = on;
      if (on) {
        btn.textContent = 'Turn OFF detection';
        btn.classList.remove('off');
        btn.classList.add('on');
      } else {
        btn.textContent = 'Turn ON detection';
        btn.classList.remove('on');
        btn.classList.add('off');
      }
    }

    function updateDurationSelect(sec) {
      currentDuration = sec;
      const sel = document.getElementById('durationSelect');
      if (![1, 30, 60].includes(sec)) sec = 60;
      sel.value = String(sec);
    }

    function toggleDetection() {
      fetch('/toggle_detection', { method: 'POST' })
        .then(resp => resp.json())
        .then(data => {
          updateToggleButton(data.detect_enabled);
          if (data.detect_enabled && data.duration_sec) {
            updateDurationSelect(data.duration_sec);
          }
        })
        .catch(err => console.error('Toggle error:', err));
    }

    function changeDuration() {
      const sel = document.getElementById('durationSelect');
      const sec = parseInt(sel.value, 10);
      fetch('/set_duration', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ duration_sec: sec })
      })
      .then(resp => resp.json())
      .then(data => {
        updateDurationSelect(data.duration_sec);
      })
      .catch(err => console.error('Set duration error:', err));
    }

    function updatePassFailBanner(passFail) {
      const textEl = document.getElementById('passFailText');
      textEl.classList.remove('pass-fail-pass', 'pass-fail-fail');

      if (passFail === 'pass') {
        textEl.textContent = 'PASS';
        textEl.classList.add('pass-fail-pass');
      } else if (passFail === 'fail') {
        textEl.textContent = 'FAIL';
        textEl.classList.add('pass-fail-fail');
      } else {
        textEl.textContent = '';
      }
    }

    function pollStatus() {
      fetch('/latest_status')
        .then(resp => resp.json())
        .then(data => {
          updatePassFailBanner(data.pass_fail);
        })
        .catch(err => console.error('Status poll error:', err));
    }

    function syncStates() {
      fetch('/detection_state')
        .then(resp => resp.json())
        .then(data => {
          updateToggleButton(data.detect_enabled);
          updateDurationSelect(data.duration_sec);
        })
        .catch(err => console.error('State fetch error:', err));
    }

    function manualSnapshot() {
      fetch('/manual_snapshot', { method: 'POST' })
        .then(resp => resp.json())
        .then(data => {
          if (data.ok) {
            alert('Snapshot saved: ' + data.file);
          } else {
            alert('Snapshot error: ' + (data.error || 'Unknown error'));
          }
        })
        .catch(err => {
          console.error('Manual snapshot error:', err);
          alert('Request error when saving snapshot');
        });
    }

    syncStates();
    setInterval(pollStatus, 500);
  </script>
</body>
</html>
"""

# ───────────────────────────────────────────────────────────────
# ROUTES
# ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    html = render_template_string(HTML_PAGE)
    resp = make_response(html)
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route("/detection_state")
def detection_state():
    return jsonify({
        "detect_enabled": state["detect_enabled"],
        "duration_sec": state["duration_sec"],
    })


@app.route("/latest_status")
def latest_status_endpoint():
    # Only pass PASS/FAIL info (text already implied by color)
    return jsonify(last_status)


@app.route("/toggle_detection", methods=["POST"])
def toggle_detection():
    global last_pass_ts, last_fail_ts
    new_state = not state["detect_enabled"]
    state["detect_enabled"] = new_state
    state["detect_started_at"] = time.time() if new_state else None

    if not new_state:
        last_pass_ts = None
        last_fail_ts = None
        last_status["pass_fail"] = "none"
        last_status["text"] = "DETECTION OFF"

    print(f"[INFO] Detection toggled -> {'ON' if new_state else 'OFF'}")
    return jsonify({
        "detect_enabled": state["detect_enabled"],
        "duration_sec": state["duration_sec"],
    })


@app.route("/set_duration", methods=["POST"])
def set_duration():
    data = request.get_json(silent=True) or {}
    try:
        dur = int(data.get("duration_sec", DEFAULT_DURATION))
    except (TypeError, ValueError):
        dur = DEFAULT_DURATION
    if dur not in ALLOWED_DURATIONS:
        dur = DEFAULT_DURATION
    state["duration_sec"] = dur
    if state["detect_enabled"]:
        state["detect_started_at"] = time.time()
        print(f"[INFO] Detection duration changed to {dur}s (timer restarted)")
    return jsonify({"duration_sec": state["duration_sec"]})


@app.route("/manual_snapshot", methods=["POST"])
def manual_snapshot():
    """Manually save a snapshot of the latest frame."""
    global last_frame_for_snapshot
    if last_frame_for_snapshot is None:
        return jsonify({"ok": False, "error": "No frame available yet"}), 500
    filepath = save_snapshot(last_frame_for_snapshot, prefix="manual")
    return jsonify({"ok": True, "file": os.path.basename(filepath)})

# ───────────────────────────────────────────────────────────────
# FRAME GENERATOR
# ───────────────────────────────────────────────────────────────

def generate_frames():
    global last_pass_ts, last_fail_ts, last_status
    global last_snapshot_ts, last_frame_for_snapshot

    print(f"[INFO] Opening RTSP: {RTSP_URL}")
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print("[WARN] FFMPEG backend failed, trying default backend...")
        cap.release()
        cap = cv2.VideoCapture(RTSP_URL)

    if not cap.isOpened():
        print("[ERROR] Cannot open RTSP stream.")
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "ERROR: Cannot open RTSP stream",
                    (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255), 2)
        ret, buffer = cv2.imencode(".jpg", blank)
        frame = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to read frame, retry...")
            time.sleep(0.05)
            continue

        frame_idx += 1
        draw = frame.copy()
        h, w = frame.shape[:2]

        maybe_auto_stop_detection()
        detect_enabled = state["detect_enabled"]
        remaining_sec = compute_remaining_seconds()

        any_helmet = False
        any_no_helmet = False

        if detect_enabled and (frame_idx % RUN_EVERY_N == 0):
            results = helmet_model(frame, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, verbose=False)[0]

            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))

                label = CLASS_NAMES.get(cls_id, str(cls_id))

                # Debug confidence values
                print(f"[DETECT] {label} conf={conf:.2f}")

                if is_no_helmet_label(label):
                    any_no_helmet = True
                    color = (0, 0, 255)
                elif is_helmet_label(label):
                    any_helmet = True
                    color = (0, 255, 0)
                else:
                    # other class, draw blue
                    color = (255, 0, 0)

                cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
                cv2.putText(draw, f"{label} {conf:.2f}",
                            (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # ───────────────────────────────────
        # Decide PASS / FAIL for this moment
        # ───────────────────────────────────
        now = time.time()
        status_text = "NO DETECTION"
        status_color = (128, 128, 128)

        if any_no_helmet:
            status_text = "NO HELMET (ALERT)"
            status_color = (0, 0, 255)
            last_fail_ts = now

            # Auto snapshot on NO-HELMET
            if now - last_snapshot_ts > MIN_SNAPSHOT_INTERVAL:
                save_snapshot(draw, prefix="no_helmet")
                last_snapshot_ts = now

        elif any_helmet:
            status_text = "HELMET DETECTED"
            status_color = (0, 255, 0)
            last_pass_ts = now

        # PASS/FAIL state for banner
        pass_fail_state = "none"
        if last_pass_ts is not None and now - last_pass_ts <= PASS_SEC:
            pass_fail_state = "pass"
        if last_fail_ts is not None and now - last_fail_ts <= FAIL_SEC:
            # FAIL overrides PASS
            pass_fail_state = "fail"

        last_status["pass_fail"] = pass_fail_state
        last_status["text"] = status_text

        # ───────────────────────────────────
        # Overlays in video
        # ───────────────────────────────────
        frame_to_show = resize_for_display(draw, MAX_WIDTH)
        h_show, w_show = frame_to_show.shape[:2]

        # Keep last frame for manual snapshot (with overlays)
        last_frame_for_snapshot = frame_to_show.copy()

        # Status (top-left)
        cv2.putText(frame_to_show, status_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    status_color, 2)

        # Detection ON/OFF (top-right)
        det_text = f"Detection: {'ON' if detect_enabled else 'OFF'}"
        (tw, th), _ = cv2.getTextSize(det_text,
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(frame_to_show, det_text,
                    (w_show - tw - 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2)

        # Remaining time (bottom-left small)
        if detect_enabled and remaining_sec is not None:
            remain_text = f"Time left: {remaining_sec}s"
        else:
            remain_text = "Time left: -"
        cv2.putText(frame_to_show, remain_text,
                    (10, h_show - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (200, 200, 200), 2)

        # JPEG encode
        ret2, buffer = cv2.imencode(".jpg", frame_to_show)
        if not ret2:
            continue
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


@app.route("/video_feed")
def video_feed():
    resp = Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


if __name__ == "__main__":
    # Use 5050 (not 5000) to avoid conflict with AirPlay/AirTunes on macOS
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
