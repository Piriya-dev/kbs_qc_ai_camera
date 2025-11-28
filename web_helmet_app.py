# web_helmet_app.py
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
#   - Run YOLO on each frame (every RUN_EVERY_N frames) when detection is ON.
#   - If any "no_hardhat" (no helmet) detection => FAIL (red) for FAIL_SEC seconds.
#   - Else if any "hardhat" detection => PASS (green) for PASS_SEC seconds.
#   - If YOLO sees person but no helmet/no_helmet boxes => infer NO HELMET (FAIL).
#
# Snapshots:
#   - Auto snapshot saved when there is at least ONE NO_HELMET (explicit or inferred),
#     respecting snapshot_interval_sec (10/30/60).
#   - Manual snapshot button saves current frame immediately.
#
# Telegram sending:
#   - send_mode = "auto":
#       * On NO_HELMET auto snapshot, send to Telegram immediately (real-time alert).
#   - send_mode = "manual":
#       * Auto snapshots are saved to disk only (no Telegram).
#       * Manual "Snapshot" button always sends to Telegram instantly.
#
# Counters:
#   - Per-poll counts (helmet_count, no_helmet_count) shown with %.
#   - Session totals (total_helmet_count, total_no_helmet_count) accumulated
#     from server start until process stops (not reset on toggle OFF/ON).
#
# UI:
#   - Video stream (MJPEG) in <img>.
#   - Detection ON/OFF toggle (no duration).
#   - Manual snapshot button with camera icon next to ON/OFF.
#   - Auto snapshot interval (10/30/60s).
#   - Send mode select (Auto / Manual).
#   - PASS / FAIL big text OUTSIDE video, centered below, with counts + %.
#   - Session total line below, with cumulative counts.
#   - Status label overlaid in top-left of video.

import os
import time
import cv2
import numpy as np
import requests
from ultralytics import YOLO
from flask import (
    Flask,
    Response,
    render_template_string,
    jsonify,
    request,
    make_response,
)

# ───────────────── CONFIG ─────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CAMERA_NAME = "KBS-HelmetCam-01"

# RTSP stream (Hikvision)
RTSP_URL = "rtsp://admin:KBSit%402468@192.168.5.61:554/ISAPI/Streaming/channels/102"

# YOLO PPE model path (trained on dataset2 with hardhat/no_hardhat)
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

# YOLO settings
YOLO_CONF = 0.50       # base YOLO confidence threshold
BOX_CONF_FILTER = 0.65 # extra filter for drawing/classifying boxes
YOLO_IMGSZ = 840
RUN_EVERY_N = 2        # run YOLO every N frames for speed

# Display size
MAX_WIDTH = 1048

# PASS / FAIL display window (seconds)
PASS_SEC = 2.0
FAIL_SEC = 2.0

# RTSP stability
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "1"

# ───────────── SNAPSHOT CONFIG ─────────────

SNAPSHOT_DIR = os.path.join(BASE_DIR, "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
print(f"[INFO] Snapshot directory: {SNAPSHOT_DIR}")

SNAPSHOT_INTERVAL_OPTIONS = [10, 30, 60]
DEFAULT_SNAPSHOT_INTERVAL = 30  # seconds between auto snapshots

# ───────────── TELEGRAM CONFIG ─────────────

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "YOUR_BOT_TOKEN_HERE")
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

CHAT_IDS = [
    6953358112,     # you (DM)
    -1003103459072  # supergroup
]

def send_telegram_photo(image_path: str, caption: str = ""):
    """Send a photo file to all CHAT_IDS via Telegram Bot API."""
    if not TELEGRAM_BOT_TOKEN or "YOUR_BOT_TOKEN_HERE" in TELEGRAM_BOT_TOKEN:
        print("[TELEGRAM] Bot token not set. Skipping sendPhoto.")
        return

    if not os.path.isfile(image_path):
        print(f"[TELEGRAM] File does not exist: {image_path}")
        return

    for chat_id in CHAT_IDS:
        try:
            with open(image_path, "rb") as f:
                files = {"photo": f}
                data = {"chat_id": chat_id, "caption": caption}
                url = f"{TELEGRAM_API_URL}/sendPhoto"
                resp = requests.post(url, data=data, files=files, timeout=15)
            if resp.status_code == 200:
                print(f"[TELEGRAM] sendPhoto OK -> chat_id={chat_id}")
            else:
                print(
                    f"[TELEGRAM] sendPhoto FAILED -> chat_id={chat_id}, "
                    f"status={resp.status_code}, body={resp.text}"
                )
        except Exception as e:
            print(f"[TELEGRAM] Error sending photo to chat_id={chat_id}: {e}")

# ───────────── LOAD YOLO MODEL ─────────────

if not os.path.isfile(YOLO_MODEL_PATH):
    raise FileNotFoundError(f"YOLO model not found: {YOLO_MODEL_PATH}")

print(f"[INFO] Loading YOLO PPE model from: {YOLO_MODEL_PATH}")
helmet_model = YOLO(YOLO_MODEL_PATH)
CLASS_NAMES = helmet_model.names
print("[INFO] Model classes:")
for cid, cname in CLASS_NAMES.items():
    print(f"  id={cid}: {cname}")

# Build explicit ID sets
HELMET_IDS = set()
NO_HELMET_IDS = set()
PERSON_IDS = set()

for cid, cname in CLASS_NAMES.items():
    name = cname.lower()
    if "person" in name:
        PERSON_IDS.add(cid)
    if "no_hardhat" in name or ("no" in name and "helmet" in name):
        NO_HELMET_IDS.add(cid)
    elif "hardhat" in name or "helmet" in name:
        HELMET_IDS.add(cid)

print(f"[INFO] HELMET_IDS: {HELMET_IDS}")
print(f"[INFO] NO_HELMET_IDS: {NO_HELMET_IDS}")
print(f"[INFO] PERSON_IDS: {PERSON_IDS}")

# ───────────── GLOBAL STATE ─────────────

state = {
    "detect_enabled": True,
    "snapshot_interval_sec": DEFAULT_SNAPSHOT_INTERVAL,
}

# send_mode: "auto" (send on detection) / "manual" (only manual snapshots send)
send_config = {
    "mode": "auto",
}

# PASS / FAIL timers
last_pass_ts = None
last_fail_ts = None

# Snapshot state
last_snapshot_ts = 0.0
last_frame_for_snapshot = None

# For smoothed counts
last_helmet_count = 0
last_no_helmet_count = 0

# Session total counters
total_helmet_count = 0
total_no_helmet_count = 0

# Status for web polling
last_status = {
    "pass_fail": "none",
    "text": "NO DETECTION",
    "helmet_count": 0,
    "no_helmet_count": 0,
    "total_helmet_count": 0,
    "total_no_helmet_count": 0,
}

# ───────────── HELPERS ─────────────

def resize_for_display(frame, max_width=MAX_WIDTH):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / float(w)
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)


def save_snapshot(image: np.ndarray, prefix: str = "snap", send_to_telegram: bool = False) -> str:
    """Save a snapshot image to SNAPSHOT_DIR and (optionally) send to Telegram."""
    try:
        os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    except Exception as e:
        print(f"[SNAPSHOT] ERROR: could not create directory {SNAPSHOT_DIR}: {e}")

    ts_str = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{ts_str}.jpg"
    filepath = os.path.join(SNAPSHOT_DIR, filename)

    print(f"[SNAPSHOT] Trying to write file: {filepath}")
    ok = cv2.imwrite(filepath, image)
    if ok:
        print(f"[SNAPSHOT] Saved snapshot: {filepath}")
        if send_to_telegram:
            human_time = time.strftime('%Y-%m-%d %H:%M:%S')
            caption = f"{CAMERA_NAME} | {prefix.upper()} | {human_time}"
            send_telegram_photo(filepath, caption=caption)
    else:
        print(f"[SNAPSHOT] ERROR: Failed to save snapshot with cv2.imwrite: {filepath}")
    return filepath

# ───────────── FLASK APP + HTML ─────────────

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
    #snapshotBtn { background:#0066cc; color:#fff; display:flex; align-items:center; gap:6px; }
    #snapshotBtn span.icon { font-size:16px; }
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
      min-height:60px;
    }
    .pass-fail-pass { color:#00ff00; }
    .pass-fail-fail { color:#ff0000; }
    .detail-line {
      display:block;
      font-size:20px;
      color:#ddd;
      margin-top:4px;
    }
    .total-line {
      display:block;
      font-size:16px;
      color:#bbb;
      margin-top:2px;
    }
    .debug-box {
      margin-top:8px;
      font-size:12px;
      color:#888;
    }
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

      <button id="snapshotBtn" onclick="manualSnapshot()">
        <span class="icon">📸</span>
        <span>Snapshot</span>
      </button>

      <label for="snapIntervalSelect">Auto snapshot every:</label>
      <select id="snapIntervalSelect" onchange="changeSnapshotInterval()">
        <option value="10">10s</option>
        <option value="30" selected>30s</option>
        <option value="60">60s</option>
      </select>

      <label for="sendModeSelect">Send mode:</label>
      <select id="sendModeSelect" onchange="changeSendMode()">
        <option value="auto">Auto (alert)</option>
        <option value="manual">Manual only</option>
      </select>
    </div>

    <div class="pass-fail-banner" id="passFailText"></div>
    <div class="debug-box" id="debugStatus"></div>

    <div class="hint">
      PASS (green) when helmet is detected, FAIL (red) when NO HELMET is detected.<br>
      Auto snapshot runs ONLY on NO HELMET (red box or inferred), with the selected interval.<br>
      <b>Auto mode:</b> each auto snapshot (NO HELMET) is also sent to Telegram in real-time.<br>
      <b>Manual mode:</b> auto snapshots stay on disk only; use 📸 Snapshot to send instantly.<br>
      Session totals accumulate from server start until you restart the app.
    </div>
  </div>

  <script>
    let detectionOn = true;
    let snapshotIntervalSec = 30;
    let sendMode = "auto";

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

    function updateSnapshotSelect(sec) {
      snapshotIntervalSec = sec;
      const sel = document.getElementById('snapIntervalSelect');
      if (![10, 30, 60].includes(sec)) sec = 30;
      sel.value = String(sec);
    }

    function updateSendMode(mode) {
      sendMode = mode;
      document.getElementById('sendModeSelect').value = mode;
    }

    function toggleDetection() {
      fetch('/toggle_detection', { method: 'POST' })
        .then(resp => resp.json())
        .then(data => {
          updateToggleButton(data.detect_enabled);
          updateSnapshotSelect(data.snapshot_interval_sec);
          updateSendMode(data.send_mode);
        })
        .catch(err => console.error('Toggle error:', err));
    }

    function changeSnapshotInterval() {
      const sel = document.getElementById('snapIntervalSelect');
      const sec = parseInt(sel.value, 10);
      fetch('/set_snapshot_interval', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ snapshot_interval_sec: sec })
      })
      .then(resp => resp.json())
      .then(data => {
        updateSnapshotSelect(data.snapshot_interval_sec);
      })
      .catch(err => console.error('Set snapshot interval error:', err));
    }

    function changeSendMode() {
      const sel = document.getElementById('sendModeSelect');
      const mode = sel.value;
      fetch('/set_send_mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: mode })
      })
      .then(resp => resp.json())
      .then(data => {
        updateSendMode(data.mode);
      })
      .catch(err => console.error('Set send mode error:', err));
    }

    function updatePassFailBanner(statusData) {
      const passFail = statusData.pass_fail;
      const helmetCount = statusData.helmet_count ?? 0;
      const noHelmetCount = statusData.no_helmet_count ?? 0;
      const totalHelmet = statusData.total_helmet_count ?? 0;
      const totalNoHelmet = statusData.total_no_helmet_count ?? 0;

      const bannerEl = document.getElementById('passFailText');
      const debugEl = document.getElementById('debugStatus');

      debugEl.textContent =
        'DEBUG => pass_fail=' + passFail +
        ', helmet_count=' + helmetCount +
        ', no_helmet_count=' + noHelmetCount +
        ', total_helmet=' + totalHelmet +
        ', total_no_helmet=' + totalNoHelmet +
        ', auto_snapshot_interval=' + snapshotIntervalSec + 's' +
        ', send_mode=' + sendMode;

      bannerEl.classList.remove('pass-fail-pass', 'pass-fail-fail');

      let titleText = '';
      if (passFail === 'pass') {
        titleText = 'PASS';
        bannerEl.classList.add('pass-fail-pass');
      } else if (passFail === 'fail') {
        titleText = 'FAIL';
        bannerEl.classList.add('pass-fail-fail');
      } else {
        titleText = 'NO DETECTION';
      }

      const totalDetected = helmetCount + noHelmetCount;
      let helmetPct = 0;
      let noHelmetPct = 0;
      if (totalDetected > 0) {
        helmetPct = Math.round((helmetCount / totalDetected) * 100);
        noHelmetPct = Math.round((noHelmetCount / totalDetected) * 100);
      }

      const detailHtml =
        '<span class="detail-line">' +
        'Frame: Helmet ' + helmetCount + ' (' + helmetPct + '%) | ' +
        'No helmet ' + noHelmetCount + ' (' + noHelmetPct + '%)' +
        '</span>' +
        '<span class="total-line">' +
        'Session total – Helmet: ' + totalHelmet +
        ' | No helmet: ' + totalNoHelmet +
        '</span>';

      bannerEl.innerHTML = titleText + detailHtml;
    }

    function pollStatus() {
      fetch('/latest_status')
        .then(resp => resp.json())
        .then(data => {
          updatePassFailBanner(data);
        })
        .catch(err => console.error('Status poll error:', err));
    }

    function syncStates() {
      fetch('/detection_state')
        .then(resp => resp.json())
        .then(data => {
          updateToggleButton(data.detect_enabled);
          updateSnapshotSelect(data.snapshot_interval_sec);
          updateSendMode(data.send_mode);
        })
        .catch(err => console.error('State fetch error:', err));
    }

    function manualSnapshot() {
      fetch('/manual_snapshot', { method: 'POST' })
        .then(resp => resp.json())
        .then(data => {
          if (data.ok) {
            alert('Snapshot saved to:\\n' + data.file + '\\n(and sent to Telegram)');
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

# ───────────── ROUTES ─────────────

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
        "snapshot_interval_sec": state["snapshot_interval_sec"],
        "send_mode": send_config["mode"],
    })


@app.route("/latest_status")
def latest_status_endpoint():
    return jsonify(last_status)


@app.route("/toggle_detection", methods=["POST"])
def toggle_detection():
    global last_pass_ts, last_fail_ts
    new_state = not state["detect_enabled"]
    state["detect_enabled"] = new_state

    if not new_state:
        # clear per-frame status (but keep totals)
        last_pass_ts = None
        last_fail_ts = None
        last_status["pass_fail"] = "none"
        last_status["text"] = "DETECTION OFF"
        last_status["helmet_count"] = 0
        last_status["no_helmet_count"] = 0

    print(f"[INFO] Detection toggled -> {'ON' if new_state else 'OFF'}")
    return jsonify({
        "detect_enabled": state["detect_enabled"],
        "snapshot_interval_sec": state["snapshot_interval_sec"],
        "send_mode": send_config["mode"],
    })


@app.route("/set_snapshot_interval", methods=["POST"])
def set_snapshot_interval():
    data = request.get_json(silent=True) or {}
    try:
        sec = int(data.get("snapshot_interval_sec", DEFAULT_SNAPSHOT_INTERVAL))
    except (TypeError, ValueError):
        sec = DEFAULT_SNAPSHOT_INTERVAL

    if sec not in SNAPSHOT_INTERVAL_OPTIONS:
        sec = DEFAULT_SNAPSHOT_INTERVAL

    state["snapshot_interval_sec"] = sec
    print(f"[INFO] Auto snapshot interval set to {sec} seconds")
    return jsonify({"snapshot_interval_sec": state["snapshot_interval_sec"]})


@app.route("/set_send_mode", methods=["POST"])
def set_send_mode():
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "auto")
    if mode not in ("auto", "manual"):
        mode = "auto"
    send_config["mode"] = mode
    print(f"[INFO] Send mode set to {mode}")
    return jsonify({"mode": send_config["mode"]})


@app.route("/manual_snapshot", methods=["POST"])
def manual_snapshot():
    global last_frame_for_snapshot
    if last_frame_for_snapshot is None:
        print("[SNAPSHOT] ERROR: manual snapshot requested but no frame yet")
        return jsonify({"ok": False, "error": "No frame available yet"}), 500

    filepath = save_snapshot(last_frame_for_snapshot, prefix="manual", send_to_telegram=True)
    return jsonify({"ok": True, "file": filepath})

# ───────────── FRAME GENERATOR ─────────────

def generate_frames():
    global last_pass_ts, last_fail_ts, last_status
    global last_snapshot_ts, last_frame_for_snapshot
    global last_helmet_count, last_no_helmet_count
    global total_helmet_count, total_no_helmet_count

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

        detect_enabled = state["detect_enabled"]
        snapshot_interval_sec = state["snapshot_interval_sec"]

        frame_helmet_count = 0
        frame_no_helmet_count = 0
        frame_person_count = 0
        detection_ran = False

        if detect_enabled and (frame_idx % RUN_EVERY_N == 0):
            detection_ran = True
            results = helmet_model(frame, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, verbose=False)[0]

            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < BOX_CONF_FILTER:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))

                label = CLASS_NAMES.get(cls_id, str(cls_id))
                print(f"[DETECT] id={cls_id} name={label} conf={conf:.2f}")

                if cls_id in NO_HELMET_IDS:
                    frame_no_helmet_count += 1
                    color = (0, 0, 255)     # red
                elif cls_id in HELMET_IDS:
                    frame_helmet_count += 1
                    color = (0, 255, 0)     # green
                elif cls_id in PERSON_IDS:
                    frame_person_count += 1
                    color = (255, 255, 0)   # yellow for person
                else:
                    color = (255, 0, 0)     # blue for other

                cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
                cv2.putText(draw, f"{label} {conf:.2f}",
                            (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # logic for this detection cycle
        any_no_helmet_box = detection_ran and frame_no_helmet_count > 0
        any_helmet_box = detection_ran and frame_helmet_count > 0
        any_person_box = detection_ran and frame_person_count > 0

        inferred_no_helmet = False
        if detection_ran:
            if any_person_box and not any_helmet_box and not any_no_helmet_box:
                inferred_no_helmet = True

        if detection_ran:
            if inferred_no_helmet and frame_no_helmet_count == 0:
                frame_no_helmet_count = 1

            helmet_count = frame_helmet_count
            no_helmet_count = frame_no_helmet_count

            last_helmet_count = helmet_count
            last_no_helmet_count = no_helmet_count

            total_helmet_count += frame_helmet_count
            total_no_helmet_count += frame_no_helmet_count
        else:
            helmet_count = last_helmet_count
            no_helmet_count = last_no_helmet_count

        # PASS / FAIL decision
        now = time.time()
        status_text = "NO DETECTION"
        status_color = (128, 128, 128)

        if detect_enabled:
            if any_no_helmet_box or inferred_no_helmet:
                status_text = "NO HELMET (ALERT)"
                status_color = (0, 0, 255)
                last_fail_ts = now

                if now - last_snapshot_ts > snapshot_interval_sec:
                    send_to_tg = (send_config["mode"] == "auto")
                    print(
                        f"[AUTO SNAPSHOT] NO HELMET at {time.strftime('%H:%M:%S')}, "
                        f"interval={snapshot_interval_sec}s, send_to_tg={send_to_tg}"
                    )
                    save_snapshot(draw, prefix="no_helmet", send_to_telegram=send_to_tg)
                    last_snapshot_ts = now

            elif any_helmet_box:
                status_text = "HELMET DETECTED"
                status_color = (0, 255, 0)
                last_pass_ts = now
        else:
            status_text = "DETECTION OFF"
            status_color = (128, 128, 128)

        pass_fail_state = "none"
        if detect_enabled and last_pass_ts is not None and now - last_pass_ts <= PASS_SEC:
            pass_fail_state = "pass"
        if detect_enabled and last_fail_ts is not None and now - last_fail_ts <= FAIL_SEC:
            pass_fail_state = "fail"

        last_status["pass_fail"] = pass_fail_state
        last_status["text"] = status_text
        last_status["helmet_count"] = helmet_count
        last_status["no_helmet_count"] = no_helmet_count
        last_status["total_helmet_count"] = total_helmet_count
        last_status["total_no_helmet_count"] = total_no_helmet_count

        frame_to_show = resize_for_display(draw, MAX_WIDTH)
        h_show, w_show = frame_to_show.shape[:2]

        last_frame_for_snapshot = frame_to_show.copy()

        cv2.putText(frame_to_show, status_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    status_color, 2)

        det_text = f"Detection: {'ON' if detect_enabled else 'OFF'}"
        (tw, th), _ = cv2.getTextSize(det_text,
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(frame_to_show, det_text,
                    (w_show - tw - 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2)

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
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
