# web_helmet_app.py
# Confidential – Internal Use Only
#
# Flask web app for real-time helmet detection using YOLO PPE model (hardhat/no_hardhat).
#
# Logic:
#   - Run YOLO on each frame (every RUN_EVERY_N frames) when detection is ON.
#   - If any "no_hardhat" (no helmet) detection (inside ROI) => FAIL (red) for 2 seconds.
#   - Else if any "hardhat" detection (inside ROI) => PASS (green) for 2 seconds.
#
# Snapshots:
#   - Auto snapshot saved when there is at least ONE "no_helmet" (red box INSIDE ROI),
#     respecting snapshot_interval_sec.
#   - Manual snapshot button saves current frame immediately and sends to Telegram.
#
# Telegram sending:
#   - send_mode = "auto":
#       * On NO_HELMET auto snapshot, send to Telegram immediately
#         (real-time alert), rate-limited by snapshot_interval_sec.
#   - send_mode = "manual":
#       * Auto snapshots are saved to disk only (no Telegram).
#       * Manual "Snapshot" button always sends to Telegram instantly.
#
# Multi-camera:
#   - CAMERAS dict defines cam1, cam2, ... (RTSP URL + name).
#   - Dropdown on UI selects active camera; all controls operate on selected cam.
#
# ROI (Area trigger):
#   - ROI_RECT per camera (x1, y1, x2, y2) in ORIGINAL frame coordinates.
#   - Only detections whose center is inside ROI are counted for:
#       * PASS / FAIL status
#       * auto-snapshot trigger
#   - Boxes outside ROI are still drawn but counted only visually.
#
# UI:
#   - Camera dropdown (cam1, cam2,...).
#   - One video stream <img> (switches by selected camera).
#   - Detection ON/OFF toggle (per cam).
#   - Manual snapshot button with camera icon (per cam).
#   - Auto snapshot interval select (per cam).
#   - Send mode select (Auto / Manual) (per cam).
#   - PASS / FAIL big text OUTSIDE video, centered below, with counts + %.
#   - Status label overlaid in top-left of video.
#   - ROI rectangle overlaid on video.

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

# ───────────────────────────────────────────────────────────────
# CONFIG
# ───────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Multi-camera configuration
CAMERAS = {
    "cam1": {
        "name": "KBS-HelmetCam-01",
        "rtsp": "rtsp://admin:KBSit%402468@192.168.4.190:554/ISAPI/Streaming/channels/101",
    },
    "cam2": {
        "name": "KBS-HelmetCam-02",
        "rtsp": "rtsp://admin:KBSit%402468@192.168.5.61:554/ISAPI/Streaming/channels/102",
    },
}
CAMERA_IDS = list(CAMERAS.keys())

# YOLO PPE model path (trained on dataset2 with hardhat/no_hardhat)
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "best.pt")

# YOLO settings
YOLO_CONF = 0.50
YOLO_IMGSZ = 608
RUN_EVERY_N = 2  # run YOLO every N frames for speed

# Display size
MAX_WIDTH = 1048

# PASS / FAIL display window (seconds)
PASS_SEC = 2.0
FAIL_SEC = 2.0

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

# Auto snapshot interval options (seconds)
SNAPSHOT_INTERVAL_OPTIONS = [3, 5, 10, 15, 30, 60]
DEFAULT_SNAPSHOT_INTERVAL = 10

# ───────────────────────────────────────────────────────────────
# TELEGRAM CONFIG
# ───────────────────────────────────────────────────────────────

TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN",
    "8559813204:AAH0wnA83cvbjOr99fbCr_isV5tXepjJxkg"
)
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

CHAT_IDS = [
    6953358112,     # you (DM)
    -1003103459072  # supergroup
]


def send_telegram_photo(image_path, caption=""):
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
# GLOBAL STATE (PER CAMERA)
# ───────────────────────────────────────────────────────────────

# detection + snapshot config per camera
state = {
    cam_id: {
        "detect_enabled": True,
        "snapshot_interval_sec": DEFAULT_SNAPSHOT_INTERVAL,
    }
    for cam_id in CAMERA_IDS
}

# send_mode: "auto" / "manual" per camera
send_config = {
    cam_id: {"mode": "auto"}
    for cam_id in CAMERA_IDS
}

# PASS / FAIL timers per camera
last_pass_ts = {cam_id: None for cam_id in CAMERA_IDS}
last_fail_ts = {cam_id: None for cam_id in CAMERA_IDS}

# Snapshot state per camera
last_snapshot_ts = {cam_id: 0.0 for cam_id in CAMERA_IDS}
last_frame_for_snapshot = {cam_id: None for cam_id in CAMERA_IDS}

# For smoothed counts per camera
last_helmet_count = {cam_id: 0 for cam_id in CAMERA_IDS}
last_no_helmet_count = {cam_id: 0 for cam_id in CAMERA_IDS}

# Status for web polling per camera
# pass_fail: "pass", "fail", "none"
last_status = {
    cam_id: {
        "pass_fail": "none",
        "text": "NO DETECTION",
        "helmet_count": 0,
        "no_helmet_count": 0,
    }
    for cam_id in CAMERA_IDS
}

# ───────────────────────────────────────────────────────────────
# ROI CONFIG (AREA TRIGGER)
# ───────────────────────────────────────────────────────────────
# ROI rectangles PER CAMERA in ORIGINAL frame pixels (x1, y1, x2, y2).
# IMPORTANT:
#   - Start with full-frame, then narrow down after you know your resolution.
#   - Example below is full frame (0..big number) so behavior is same as no ROI.
#   - Tune numbers by printing frame width/height once or using OpenCV window.

ROI_RECT = {
    "cam1": {"x1": 0, "y1": 0, "x2": 9999, "y2": 9999},  # TODO: set to real area
    "cam2": {"x1": 0, "y1": 0, "x2": 9999, "y2": 9999},  # TODO: set to real area
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


def is_helmet_label(label):
    """
    Helmet (PASS) for dataset2:
      - 'hardhat'
      - or any label containing 'hardhat'/'helmet' without 'no'
    """
    l = label.lower()
    if "no_hardhat" in l:
        return False
    return ("hardhat" in l or "helmet" in l) and ("no" not in l)


def is_no_helmet_label(label):
    """
    No helmet (FAIL) for dataset2:
      - 'no_hardhat'
      - or label containing both 'no' and 'hardhat'/'helmet'
    """
    l = label.lower()
    if "no_hardhat" in l:
        return True
    return ("no" in l) and ("hardhat" in l or "helmet" in l)


def save_snapshot(image, prefix="snap", send_to_telegram=False, camera_name="Camera"):
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
            caption = f"{camera_name} | {prefix.upper()} | {human_time}"
            send_telegram_photo(filepath, caption=caption)
    else:
        print(f"[SNAPSHOT] ERROR: Failed to save snapshot with cv2.imwrite: {filepath}")
    return filepath


def get_cam_from_request(default="cam1"):
    """Get camera id from ?cam= or JSON body; fallback to default."""
    cam_id = request.args.get("cam")
    if not cam_id and request.is_json:
        data = request.get_json(silent=True) or {}
        cam_id = data.get("cam")
    if cam_id not in CAMERAS:
        cam_id = default
    return cam_id


def point_in_roi(cam_id, x, y):
    """Check if a point (x,y) in original frame coords lies inside ROI for that camera."""
    cfg = ROI_RECT.get(cam_id)
    if not cfg:
        return True  # no ROI config -> treat as inside
    x1, y1, x2, y2 = cfg["x1"], cfg["y1"], cfg["x2"], cfg["y2"]
    return x1 <= x <= x2 and y1 <= y <= y2

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
    .top-controls {
      margin-top:10px;
      display:flex;
      gap:10px;
      align-items:center;
      justify-content:center;
      flex-wrap:wrap;
    }
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
    .hint { margin-top:10px; color:#aaa; font-size:13px; max-width:1100px; }
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
    .debug-box {
      margin-top:8px;
      font-size:12px;
      color:#888;
    }
  </style>
</head>
<body>
  <h1>KBS Safety Helmet Detector</h1>

  <div class="top-controls">
    <label for="camSelect">Camera:</label>
    <select id="camSelect" onchange="onCameraChange()">
      {% for cam_id, cfg in cameras.items() %}
      <option value="{{ cam_id }}">{{ cam_id }} – {{ cfg['name'] }}</option>
      {% endfor %}
    </select>
  </div>

  <div class="container">
    <div class="video-frame">
      <img class="stream" id="streamImg" src="{{ url_for('video_feed') }}?cam=cam1" alt="Video stream">
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
      Only detections INSIDE the configured ROI area will change PASS/FAIL and trigger auto snapshots.<br>
      Auto snapshot runs ONLY on NO HELMET (red box in ROI), with the selected interval.<br>
      <b>Auto mode:</b> each auto snapshot (NO HELMET) is also sent to Telegram in real-time.<br>
      <b>Manual mode:</b> auto snapshots stay on disk only; use 📸 Snapshot to send instantly.
    </div>
  </div>

  <script>
    const CAM_IDS = {{ cameras.keys()|list|tojson }};
    let currentCam = 'cam1';

    const camState = {};
    CAM_IDS.forEach(camId => {
      camState[camId] = {
        snapshotIntervalSec: 30,
        sendMode: 'auto',
        detectEnabled: true,
      };
    });

    function getSelectedCam() {
      return currentCam;
    }

    function onCameraChange() {
      const sel = document.getElementById('camSelect');
      currentCam = sel.value || 'cam1';
      // switch video feed
      const img = document.getElementById('streamImg');
      img.src = '/video_feed?cam=' + currentCam + '&_t=' + Date.now();

      // sync button/select states
      syncStateForCam(currentCam);
    }

    function updateToggleButton(on) {
      const btn = document.getElementById('toggleBtn');
      camState[currentCam].detectEnabled = on;
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
      const sel = document.getElementById('snapIntervalSelect');
      if (![10, 30, 60].includes(sec)) sec = 30;
      sel.value = String(sec);
      camState[currentCam].snapshotIntervalSec = sec;
    }

    function updateSendMode(mode) {
      const sel = document.getElementById('sendModeSelect');
      if (!['auto', 'manual'].includes(mode)) mode = 'auto';
      sel.value = mode;
      camState[currentCam].sendMode = mode;
    }

    function toggleDetection() {
      const camId = getSelectedCam();
      fetch('/toggle_detection?cam=' + camId, { method: 'POST' })
        .then(resp => resp.json())
        .then(data => {
          if (camId === currentCam) {
            updateToggleButton(data.detect_enabled);
            updateSnapshotSelect(data.snapshot_interval_sec);
            updateSendMode(data.send_mode);
          }
          camState[camId].detectEnabled = data.detect_enabled;
          camState[camId].snapshotIntervalSec = data.snapshot_interval_sec;
          camState[camId].sendMode = data.send_mode;
        })
        .catch(err => console.error('Toggle error:', err));
    }

    function changeSnapshotInterval() {
      const camId = getSelectedCam();
      const sel = document.getElementById('snapIntervalSelect');
      const sec = parseInt(sel.value, 10);
      fetch('/set_snapshot_interval?cam=' + camId, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ snapshot_interval_sec: sec })
      })
      .then(resp => resp.json())
      .then(data => {
        if (camId === currentCam) {
          updateSnapshotSelect(data.snapshot_interval_sec);
        }
        camState[camId].snapshotIntervalSec = data.snapshot_interval_sec;
      })
      .catch(err => console.error('Set snapshot interval error:', err));
    }

    function changeSendMode() {
      const camId = getSelectedCam();
      const sel = document.getElementById('sendModeSelect');
      const mode = sel.value;
      fetch('/set_send_mode?cam=' + camId, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: mode })
      })
      .then(resp => resp.json())
      .then(data => {
        if (camId === currentCam) {
          updateSendMode(data.mode);
        }
        camState[camId].sendMode = data.mode;
      })
      .catch(err => console.error('Set send mode error:', err));
    }

    function updatePassFailBanner(statusData) {
      const passFail = statusData.pass_fail;
      const helmetCount = statusData.helmet_count ?? 0;
      const noHelmetCount = statusData.no_helmet_count ?? 0;

      const bannerEl = document.getElementById('passFailText');
      const debugEl = document.getElementById('debugStatus');

      const camId = getSelectedCam();
      debugEl.textContent =
        'DEBUG => cam=' + camId +
        ', pass_fail=' + passFail +
        ', helmet_count=' + helmetCount +
        ', no_helmet_count=' + noHelmetCount +
        ', auto_snapshot_interval=' + camState[camId].snapshotIntervalSec + 's' +
        ', send_mode=' + camState[camId].sendMode;

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
        'Helmet (in ROI): ' + helmetCount + ' (' + helmetPct + '%) | ' +
        'No helmet (in ROI): ' + noHelmetCount + ' (' + noHelmetPct + '%)' +
        '</span>';

      bannerEl.innerHTML = titleText + detailHtml;
    }

    function pollStatus() {
      const camId = getSelectedCam();
      fetch('/latest_status?cam=' + camId)
        .then(resp => resp.json())
        .then(data => {
          updatePassFailBanner(data);
        })
        .catch(err => console.error('Status poll error:', err));
    }

    function syncStateForCam(camId) {
      fetch('/detection_state?cam=' + camId)
        .then(resp => resp.json())
        .then(data => {
          camState[camId].detectEnabled = data.detect_enabled;
          camState[camId].snapshotIntervalSec = data.snapshot_interval_sec;
          camState[camId].sendMode = data.send_mode;

          if (camId === currentCam) {
            updateToggleButton(data.detect_enabled);
            updateSnapshotSelect(data.snapshot_interval_sec);
            updateSendMode(data.send_mode);
          }
        })
        .catch(err => console.error('State fetch error:', err));
    }

    function manualSnapshot() {
      const camId = getSelectedCam();
      fetch('/manual_snapshot?cam=' + camId, { method: 'POST' })
        .then(resp => resp.json())
        .then(data => {
          if (data.ok) {
            alert('[' + camId + '] Snapshot saved to:\\n' + data.file + '\\n(and sent to Telegram)');
          } else {
            alert('[' + camId + '] Snapshot error: ' + (data.error || 'Unknown error'));
          }
        })
        .catch(err => {
          console.error('Manual snapshot error (' + camId + '):', err);
          alert('[' + camId + '] Request error when saving snapshot');
        });
    }

    function init() {
      // initial state for cam1
      syncStateForCam('cam1');
      // also pre-fetch cam2 state in background
      syncStateForCam('cam2');
      setInterval(pollStatus, 500);
    }

    init();
  </script>
</body>
</html>
"""

# ───────────────────────────────────────────────────────────────
# ROUTES
# ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    html = render_template_string(HTML_PAGE, cameras=CAMERAS)
    resp = make_response(html)
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route("/detection_state")
def detection_state():
    cam_id = get_cam_from_request()
    return jsonify({
        "detect_enabled": state[cam_id]["detect_enabled"],
        "snapshot_interval_sec": state[cam_id]["snapshot_interval_sec"],
        "send_mode": send_config[cam_id]["mode"],
    })


@app.route("/latest_status")
def latest_status_endpoint():
    cam_id = get_cam_from_request()
    return jsonify(last_status[cam_id])


@app.route("/toggle_detection", methods=["POST"])
def toggle_detection():
    cam_id = get_cam_from_request()
    cam_state = state[cam_id]

    new_state = not cam_state["detect_enabled"]
    cam_state["detect_enabled"] = new_state

    if not new_state:
        # When turning OFF, clear timers & status
        last_pass_ts[cam_id] = None
        last_fail_ts[cam_id] = None
        last_status[cam_id]["pass_fail"] = "none"
        last_status[cam_id]["text"] = "DETECTION OFF"
        last_status[cam_id]["helmet_count"] = 0
        last_status[cam_id]["no_helmet_count"] = 0

    print(f"[INFO] [{cam_id}] Detection toggled -> {'ON' if new_state else 'OFF'}")
    return jsonify({
        "detect_enabled": cam_state["detect_enabled"],
        "snapshot_interval_sec": cam_state["snapshot_interval_sec"],
        "send_mode": send_config[cam_id]["mode"],
    })


@app.route("/set_snapshot_interval", methods=["POST"])
def set_snapshot_interval():
    """Set auto snapshot interval (in seconds) from the UI dropdown."""
    cam_id = get_cam_from_request()
    data = request.get_json(silent=True) or {}
    try:
        sec = int(data.get("snapshot_interval_sec", DEFAULT_SNAPSHOT_INTERVAL))
    except (TypeError, ValueError):
        sec = DEFAULT_SNAPSHOT_INTERVAL

    if sec not in SNAPSHOT_INTERVAL_OPTIONS:
        sec = DEFAULT_SNAPSHOT_INTERVAL

    state[cam_id]["snapshot_interval_sec"] = sec
    print(f"[INFO] [{cam_id}] Auto snapshot interval set to {sec} seconds")
    return jsonify({"snapshot_interval_sec": state[cam_id]["snapshot_interval_sec"]})


@app.route("/set_send_mode", methods=["POST"])
def set_send_mode():
    """Toggle auto/manual sending to Telegram."""
    cam_id = get_cam_from_request()
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "auto")
    if mode not in ("auto", "manual"):
        mode = "auto"
    send_config[cam_id]["mode"] = mode
    print(f"[INFO] [{cam_id}] Send mode set to {mode}")
    return jsonify({"mode": send_config[cam_id]["mode"]})


@app.route("/manual_snapshot", methods=["POST"])
def manual_snapshot():
    """Manually save a snapshot of the latest frame and send to Telegram."""
    cam_id = get_cam_from_request()
    frame = last_frame_for_snapshot[cam_id]
    if frame is None:
        print(f"[SNAPSHOT] ERROR: manual snapshot requested but no frame yet ({cam_id})")
        return jsonify({"ok": False, "error": "No frame available yet"}), 500

    filepath = save_snapshot(
        frame,
        prefix=f"{cam_id}_manual",
        send_to_telegram=True,
        camera_name=CAMERAS[cam_id]["name"],
    )
    return jsonify({"ok": True, "file": filepath})

# ───────────────────────────────────────────────────────────────
# FRAME GENERATOR (PER CAMERA with ROI)
# ───────────────────────────────────────────────────────────────

def generate_frames(cam_id):
    cam_cfg = CAMERAS[cam_id]
    rtsp_url = cam_cfg["rtsp"]
    cam_name = cam_cfg["name"]

    print(f"[INFO] [{cam_id}] Opening RTSP: {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print(f"[WARN] [{cam_id}] FFMPEG backend failed, trying default backend...")
        cap.release()
        cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print(f"[ERROR] [{cam_id}] Cannot open RTSP stream.")
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, f"ERROR: Cannot open RTSP stream ({cam_id})",
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
            print(f"[WARN] [{cam_id}] Failed to read frame, retry...")
            time.sleep(0.05)
            continue

        frame_idx += 1
        draw = frame.copy()
        h, w = frame.shape[:2]

        detect_enabled = state[cam_id]["detect_enabled"]
        snapshot_interval_sec = state[cam_id]["snapshot_interval_sec"]

        # counts for THIS detection run (inside ROI only)
        frame_helmet_count = 0
        frame_no_helmet_count = 0
        detection_ran = False

        if detect_enabled and (frame_idx % RUN_EVERY_N == 0):
            detection_ran = True
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

                # center of the bbox (for ROI)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                in_roi = point_in_roi(cam_id, cx, cy)

                # Debug confidence values
                print(f"[DETECT] [{cam_id}] {label} conf={conf:.2f}, in_roi={in_roi}")

                color = (255, 0, 0)  # default for other classes
                if is_no_helmet_label(label):
                    if in_roi:
                        frame_no_helmet_count += 1
                        color = (0, 0, 255)
                    else:
                        color = (0, 0, 150)
                elif is_helmet_label(label):
                    if in_roi:
                        frame_helmet_count += 1
                        color = (0, 255, 0)
                    else:
                        color = (0, 150, 0)

                # draw bbox
                cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
                cv2.putText(draw, f"{label} {conf:.2f}",
                            (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                # draw center point
                cv2.circle(draw, (cx, cy), 3, color, -1)

        # logic for this frame (based ONLY on detections that were inside ROI)
        any_no_helmet = detection_ran and frame_no_helmet_count > 0
        any_helmet = detection_ran and frame_helmet_count > 0

        # decide smoothed counts to send to UI
        if detection_ran:
            helmet_count = frame_helmet_count
            no_helmet_count = frame_no_helmet_count
            last_helmet_count[cam_id] = helmet_count
            last_no_helmet_count[cam_id] = no_helmet_count
        else:
            helmet_count = last_helmet_count[cam_id]
            no_helmet_count = last_no_helmet_count[cam_id]

        # ───────────────────────────────────
        # Decide PASS / FAIL for this moment
        # ───────────────────────────────────
        now = time.time()
        status_text = "NO DETECTION"
        status_color = (128, 128, 128)

        if detect_enabled:
            if any_no_helmet:
                status_text = "NO HELMET (ALERT) [ROI]"
                status_color = (0, 0, 255)
                last_fail_ts[cam_id] = now

                # Auto snapshot on NO-HELMET in ROI with interval
                if now - last_snapshot_ts[cam_id] > snapshot_interval_sec:
                    send_to_tg = (send_config[cam_id]["mode"] == "auto")
                    print(
                        f"[AUTO SNAPSHOT] [{cam_id}] NO HELMET in ROI at "
                        f"{time.strftime('%H:%M:%S')}, interval={snapshot_interval_sec}s, "
                        f"send_to_tg={send_to_tg}"
                    )
                    save_snapshot(
                        draw,
                        prefix=f"{cam_id}_no_helmet",
                        send_to_telegram=send_to_tg,
                        camera_name=cam_name,
                    )
                    last_snapshot_ts[cam_id] = now

            elif any_helmet:
                status_text = "HELMET DETECTED [ROI]"
                status_color = (0, 255, 0)
                last_pass_ts[cam_id] = now
        else:
            status_text = "DETECTION OFF"
            status_color = (128, 128, 128)

        # PASS/FAIL state for info line
        pass_fail_state = "none"
        if detect_enabled and last_pass_ts[cam_id] is not None and now - last_pass_ts[cam_id] <= PASS_SEC:
            pass_fail_state = "pass"
        if detect_enabled and last_fail_ts[cam_id] is not None and now - last_fail_ts[cam_id] <= FAIL_SEC:
            # FAIL overrides PASS
            pass_fail_state = "fail"

        # Update shared status for /latest_status
        last_status[cam_id]["pass_fail"] = pass_fail_state
        last_status[cam_id]["text"] = status_text
        last_status[cam_id]["helmet_count"] = helmet_count
        last_status[cam_id]["no_helmet_count"] = no_helmet_count

        # ───────────────────────────────────
        # Overlays in video
        # ───────────────────────────────────
        frame_to_show = resize_for_display(draw, MAX_WIDTH)
        h_show, w_show = frame_to_show.shape[:2]

        # Keep last frame for manual snapshot (with overlays)
        last_frame_for_snapshot[cam_id] = frame_to_show.copy()

        # draw ROI rectangle (scaled to display size)
        roi_cfg = ROI_RECT.get(cam_id)
        if roi_cfg:
            rx1, ry1, rx2, ry2 = roi_cfg["x1"], roi_cfg["y1"], roi_cfg["x2"], roi_cfg["y2"]
            # clamp to original frame
            rx1 = max(0, min(rx1, w - 1))
            ry1 = max(0, min(ry1, h - 1))
            rx2 = max(0, min(rx2, w))
            ry2 = max(0, min(ry2, h))

            scale_x = w_show / float(w)
            scale_y = h_show / float(h)
            rx1s = int(rx1 * scale_x)
            ry1s = int(ry1 * scale_y)
            rx2s = int(rx2 * scale_x)
            ry2s = int(ry2 * scale_y)

            cv2.rectangle(frame_to_show, (rx1s, ry1s), (rx2s, ry2s), (0, 255, 255), 2)
            cv2.putText(frame_to_show, "ROI",
                        (rx1s + 5, ry1s + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)

        # Status (top-left)
        cv2.putText(frame_to_show, status_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    status_color, 2)

        # Detection ON/OFF (top-right)
        det_text = f"{cam_name} | Detection: {'ON' if detect_enabled else 'OFF'}"
        (tw, th), _ = cv2.getTextSize(det_text,
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(frame_to_show, det_text,
                    (w_show - tw - 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2)

        # JPEG encode
        ret2, buffer = cv2.imencode(".jpg", frame_to_show)
        if not ret2:
            continue
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


@app.route("/video_feed")
def video_feed():
    cam_id = get_cam_from_request()
    resp = Response(
        generate_frames(cam_id),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


if __name__ == "__main__":
    # Use 5050 or any free port you like
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
