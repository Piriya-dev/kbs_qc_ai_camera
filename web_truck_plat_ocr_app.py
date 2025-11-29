# web_truck_plate_ocr_app.py
# Confidential – Internal Use Only
#
# Real-time truck + Thai plate detection & OCR:
#   - YOLO11n (COCO) for truck
#   - YOLO11 (thai_plate.pt) for license plates (class: license_plate)
#   - EasyOCR (th+en) for reading plate text
#   - Multi-camera (cam1, cam2, cam3)
#   - Accumulated truck events
#   - Auto snapshots with timestamp + plate text, filename includes plate
#   - Manual snapshot + Telegram send

import os
import time
import re
import cv2
import numpy as np
import requests
import easyocr
from ultralytics import YOLO
from flask import (
    Flask,
    Response,
    render_template_string,
    jsonify,
    request,
    make_response,
)

# ─────────────────────────────────────────────
# BASIC CONFIG
# ─────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CAMERAS = {
    "cam1": {
        "name": "KBS-IT Cam-01",
        "rtsp": "rtsp://admin:KBSit%402468@192.168.4.190:554/ISAPI/Streaming/channels/101",
    },
    "cam2": {
        "name": "KBS-Gate-02",
        "rtsp": "rtsp://admin:KBSit%402468@192.168.5.61:554/ISAPI/Streaming/channels/101",
    },
    "cam3": {
        "name": "KBS-Truck Tracking-03",
        "rtsp": "rtsp://admin:KBSit%402468@192.168.5.46:554/ISAPI/Streaming/channels/101",
    },
}
CAMERA_IDS = list(CAMERAS.keys())

TRUCK_MODEL_PATH = os.path.join(BASE_DIR, "yolo11n.pt")    # COCO, has 'truck'
PLATE_MODEL_PATH = os.path.join(BASE_DIR, "thai_plate.pt")  # your trained plate model

YOLO_CONF_TRUCK = 0.45
YOLO_CONF_PLATE = 0.40
YOLO_IMGSZ = 608
RUN_EVERY_N = 2
MAX_WIDTH = 1048

# RTSP stability
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_FFMPEG"] = "1"

# ─────────────────────────────────────────────
# SNAPSHOT / TELEGRAM
# ─────────────────────────────────────────────

SNAPSHOT_DIR = os.path.join(BASE_DIR, "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
print(f"[INFO] Snapshot directory: {SNAPSHOT_DIR}")

SNAPSHOT_INTERVAL_OPTIONS = [3, 5, 10, 15, 30, 60]
DEFAULT_SNAPSHOT_INTERVAL = 10

TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN",
    "8559813204:AAH0wnA83cvbjOr99fbCr_isV5tXepjJxkg"  # <-- replace if needed
)
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
CHAT_IDS = [
    6953358112,     # your DM
    -1003103459072  # your group
]

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────

if not os.path.isfile(TRUCK_MODEL_PATH):
    raise FileNotFoundError(f"TRUCK YOLO model not found: {TRUCK_MODEL_PATH}")
if not os.path.isfile(PLATE_MODEL_PATH):
    raise FileNotFoundError(f"PLATE YOLO model not found: {PLATE_MODEL_PATH}")

print(f"[INFO] Loading TRUCK model from: {TRUCK_MODEL_PATH}")
truck_model = YOLO(TRUCK_MODEL_PATH)
TRUCK_CLASS_NAMES = truck_model.names

print(f"[INFO] Loading PLATE model from: {PLATE_MODEL_PATH}")
plate_model = YOLO(PLATE_MODEL_PATH)
PLATE_CLASS_NAMES = plate_model.names

print("[INFO] TRUCK model classes:")
for cid, cname in TRUCK_CLASS_NAMES.items():
    print(f"  {cid}: {cname}")
print("[INFO] PLATE model classes:")
for cid, cname in PLATE_CLASS_NAMES.items():
    print(f"  {cid}: {cname}")

# ─────────────────────────────────────────────
# OCR
# ─────────────────────────────────────────────

print("[INFO] Loading EasyOCR (th+en)...")
ocr_reader = easyocr.Reader(["th", "en"], gpu=False)
print("[INFO] EasyOCR ready.")

# ─────────────────────────────────────────────
# STATE PER CAMERA
# ─────────────────────────────────────────────

state = {
    cam_id: {
        "detect_enabled": True,
        "snapshot_interval_sec": DEFAULT_SNAPSHOT_INTERVAL,
    }
    for cam_id in CAMERA_IDS
}

send_config = {
    cam_id: {"mode": "auto"}   # "auto" or "manual"
    for cam_id in CAMERA_IDS
}

last_snapshot_ts = {cam_id: 0.0 for cam_id in CAMERA_IDS}
last_frame_for_snapshot = {cam_id: None for cam_id in CAMERA_IDS}

last_truck_count = {cam_id: 0 for cam_id in CAMERA_IDS}
total_truck_events = {cam_id: 0 for cam_id in CAMERA_IDS}
prev_any_truck = {cam_id: False for cam_id in CAMERA_IDS}
last_plate_text = {cam_id: "" for cam_id in CAMERA_IDS}

last_status = {
    cam_id: {
        "text": "NO DETECTION",
        "truck_count": 0,
        "total_truck_events": 0,
        "last_plate_text": "",
    }
    for cam_id in CAMERA_IDS
}

# ─────────────────────────────────────────────
# ROI
# ─────────────────────────────────────────────

ROI_RECT = {
    "cam1": {"x1": 0, "y1": 0, "x2": 9999, "y2": 9999},
    "cam2": {"x1": 0, "y1": 0, "x2": 9999, "y2": 9999},
    "cam3": {"x1": 0, "y1": 0, "x2": 9999, "y2": 9999},
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

PLATE_KEYWORDS = ["license_plate"]  # from data.yaml: ['license_plate']

def send_telegram_photo(image_path, caption=""):
    if not TELEGRAM_BOT_TOKEN or "YOUR_BOT_TOKEN_HERE" in TELEGRAM_BOT_TOKEN:
        print("[TELEGRAM] Bot token not set, skip.")
        return
    if not os.path.isfile(image_path):
        print(f"[TELEGRAM] File not found: {image_path}")
        return

    for chat_id in CHAT_IDS:
        try:
            with open(image_path, "rb") as f:
                files = {"photo": f}
                data = {"chat_id": chat_id, "caption": caption}
                resp = requests.post(
                    f"{TELEGRAM_API_URL}/sendPhoto",
                    data=data,
                    files=files,
                    timeout=15
                )
            if resp.status_code == 200:
                print(f"[TELEGRAM] OK -> {chat_id}")
            else:
                print(f"[TELEGRAM] FAIL -> {chat_id}, code={resp.status_code}, body={resp.text}")
        except Exception as e:
            print(f"[TELEGRAM] Error sending to {chat_id}: {e}")


def resize_for_display(frame, max_width=MAX_WIDTH):
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / float(w)
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)


def is_truck_label(label: str) -> bool:
    return "truck" in label.lower()


def is_plate_label(label: str) -> bool:
    l = label.lower()
    return any(k in l for k in PLATE_KEYWORDS)


def save_snapshot(image, prefix="snap", send_to_telegram=False, camera_name="Camera"):
    ts_str = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{ts_str}.jpg"
    filepath = os.path.join(SNAPSHOT_DIR, filename)

    ok = cv2.imwrite(filepath, image)
    if ok:
        print(f"[SNAPSHOT] Saved: {filepath}")
        if send_to_telegram:
            human_time = time.strftime('%Y-%m-%d %H:%M:%S')
            caption = f"{camera_name} | {prefix.upper()} | {human_time}"
            send_telegram_photo(filepath, caption=caption)
    else:
        print(f"[SNAPSHOT] ERROR writing: {filepath}")
    return filepath


def get_cam_from_request(default="cam1"):
    cam_id = request.args.get("cam")
    if not cam_id and request.is_json:
        data = request.get_json(silent=True) or {}
        cam_id = data.get("cam")
    if cam_id not in CAMERAS:
        cam_id = default
    return cam_id


def point_in_roi(cam_id, x, y):
    cfg = ROI_RECT.get(cam_id)
    if not cfg:
        return True
    x1, y1, x2, y2 = cfg["x1"], cfg["y1"], cfg["x2"], cfg["y2"]
    return x1 <= x <= x2 and y1 <= y <= y2


def sanitize_plate_text(text: str) -> str:
    text = text.strip().replace(" ", "")
    if not text:
        return ""
    pattern = r"[^0-9A-Za-zก-ฮ๐-๙]"
    cleaned = re.sub(pattern, "", text)
    return cleaned


def ocr_plate_image(plate_bgr: np.ndarray) -> str:
    if plate_bgr is None or plate_bgr.size == 0:
        return ""
    plate_rgb = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2RGB)
    try:
        results = ocr_reader.readtext(plate_rgb, detail=0)
    except Exception as e:
        print(f"[OCR] Error: {e}")
        return ""
    if not results:
        return ""
    joined = "".join(results)
    cleaned = sanitize_plate_text(joined)
    print(f"[OCR] raw={results}, joined='{joined}', cleaned='{cleaned}'")
    return cleaned

# ─────────────────────────────────────────────
# FLASK APP + HTML
# ─────────────────────────────────────────────

app = Flask(__name__)

HTML_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>KBS Truck Counter + Thai Plate OCR</title>
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
    .container {
      display:flex;
      flex-direction:column;
      align-items:center;
      margin-top:20px;
      gap:10px;
    }
    .video-frame {
      border:3px solid #0f0;
      border-radius:8px;
      overflow:hidden;
      max-width:90vw;
    }
    img.stream { max-width:100%; height:auto; display:block; }
    .controls {
      margin-top:10px;
      display:flex;
      gap:10px;
      align-items:center;
      justify-content:center;
      flex-wrap:wrap;
    }
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
    .info-banner {
      margin-top:15px;
      font-size:22px;
      font-weight:bold;
      text-align:center;
      min-height:60px;
    }
    .info-line {
      display:block;
      font-size:18px;
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
  <h1>KBS Truck Counter + Thai Plate OCR</h1>

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

    <div class="info-banner" id="infoBanner"></div>
    <div class="debug-box" id="debugStatus"></div>
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
      const img = document.getElementById('streamImg');
      img.src = '/video_feed?cam=' + currentCam + '&_t=' + Date.now();
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

    function updateInfoBanner(statusData) {
      const truckCount = statusData.truck_count ?? 0;
      const totalEvents = statusData.total_truck_events ?? 0;
      const text = statusData.text || '';
      const plate = statusData.last_plate_text || '';

      const bannerEl = document.getElementById('infoBanner');
      const debugEl = document.getElementById('debugStatus');

      const camId = getSelectedCam();
      debugEl.textContent =
        'DEBUG => cam=' + camId +
        ', text=' + text +
        ', truck_count=' + truckCount +
        ', total_truck_events=' + totalEvents +
        ', last_plate_text=' + plate +
        ', auto_snapshot_interval=' + camState[camId].snapshotIntervalSec + 's' +
        ', send_mode=' + camState[camId].sendMode;

      const html =
        '<span class="info-line">' + text + '</span>' +
        '<span class="info-line">Current trucks in ROI: ' + truckCount + '</span>' +
        '<span class="info-line">Accumulated truck events: ' + totalEvents + '</span>' +
        '<span class="info-line">Last plate OCR: ' + plate + '</span>';

      bannerEl.innerHTML = html;
    }

    function pollStatus() {
      const camId = getSelectedCam();
      fetch('/latest_status?cam=' + camId)
        .then(resp => resp.json())
        .then(data => {
          updateInfoBanner(data);
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
      syncStateForCam('cam1');
      syncStateForCam('cam2');
      syncStateForCam('cam3');
      setInterval(pollStatus, 700);
    }

    init();
  </script>
</body>
</html>
"""

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

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
        last_status[cam_id]["text"] = "DETECTION OFF"
        last_status[cam_id]["truck_count"] = 0

    print(f"[INFO] [{cam_id}] Detection -> {'ON' if new_state else 'OFF'}")
    return jsonify({
        "detect_enabled": cam_state["detect_enabled"],
        "snapshot_interval_sec": cam_state["snapshot_interval_sec"],
        "send_mode": send_config[cam_id]["mode"],
    })


@app.route("/set_snapshot_interval", methods=["POST"])
def set_snapshot_interval():
    cam_id = get_cam_from_request()
    data = request.get_json(silent=True) or {}
    try:
        sec = int(data.get("snapshot_interval_sec", DEFAULT_SNAPSHOT_INTERVAL))
    except (TypeError, ValueError):
        sec = DEFAULT_SNAPSHOT_INTERVAL

    if sec not in SNAPSHOT_INTERVAL_OPTIONS:
        sec = DEFAULT_SNAPSHOT_INTERVAL

    state[cam_id]["snapshot_interval_sec"] = sec
    print(f"[INFO] [{cam_id}] Snapshot interval = {sec}s")
    return jsonify({"snapshot_interval_sec": sec})


@app.route("/set_send_mode", methods=["POST"])
def set_send_mode():
    cam_id = get_cam_from_request()
    data = request.get_json(silent=True) or {}
    mode = data.get("mode", "auto")
    if mode not in ("auto", "manual"):
        mode = "auto"
    send_config[cam_id]["mode"] = mode
    print(f"[INFO] [{cam_id}] Send mode = {mode}")
    return jsonify({"mode": mode})


@app.route("/manual_snapshot", methods=["POST"])
def manual_snapshot():
    cam_id = get_cam_from_request()
    frame = last_frame_for_snapshot[cam_id]
    if frame is None:
        return jsonify({"ok": False, "error": "No frame available yet"}), 500

    filepath = save_snapshot(
        frame,
        prefix=f"{cam_id}_manual",
        send_to_telegram=True,
        camera_name=CAMERAS[cam_id]["name"],
    )
    return jsonify({"ok": True, "file": filepath})

# ─────────────────────────────────────────────
# FRAME GENERATOR
# ─────────────────────────────────────────────

def generate_frames(cam_id):
    cam_cfg = CAMERAS[cam_id]
    rtsp_url = cam_cfg["rtsp"]
    cam_name = cam_cfg["name"]

    print(f"[INFO] [{cam_id}] Opening RTSP: {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

    if not cap.isOpened():
        print(f"[WARN] [{cam_id}] FFMPEG failed, try default backend...")
        cap.release()
        cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print(f"[ERROR] [{cam_id}] Cannot open RTSP stream.")
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, f"ERROR: Cannot open RTSP ({cam_id})",
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

        frame_truck_count = 0
        detection_ran = False
        trucks = []
        plates = []

        if detect_enabled and (frame_idx % RUN_EVERY_N == 0):
            detection_ran = True

            # TRUCK DETECTION
            t_results = truck_model(
                frame, imgsz=YOLO_IMGSZ, conf=YOLO_CONF_TRUCK, verbose=False
            )[0]
            for box in t_results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))
                label = TRUCK_CLASS_NAMES.get(cls_id, str(cls_id))

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                in_roi = point_in_roi(cam_id, cx, cy)

                if is_truck_label(label):
                    if in_roi:
                        frame_truck_count += 1
                    trucks.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "cx": cx, "cy": cy, "in_roi": in_roi,
                        "label": label, "conf": conf
                    })

            # PLATE DETECTION
            p_results = plate_model(
                frame, imgsz=YOLO_IMGSZ, conf=YOLO_CONF_PLATE, verbose=False
            )[0]
            for box in p_results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))
                label = PLATE_CLASS_NAMES.get(cls_id, str(cls_id))
                if not is_plate_label(label):
                    continue

                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                in_roi = point_in_roi(cam_id, cx, cy)

                plates.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "cx": cx, "cy": cy, "in_roi": in_roi,
                    "label": label, "conf": conf
                })

        any_truck = detection_ran and frame_truck_count > 0

        if detection_ran:
            truck_count = frame_truck_count
            last_truck_count[cam_id] = truck_count
        else:
            truck_count = last_truck_count[cam_id]

        # accumulated truck events
        if detect_enabled:
            if any_truck and not prev_any_truck[cam_id]:
                total_truck_events[cam_id] += 1
                print(f"[COUNT] [{cam_id}] New truck event -> {total_truck_events[cam_id]}")
            prev_any_truck[cam_id] = any_truck
        else:
            prev_any_truck[cam_id] = False

        # ASSOCIATE PLATES WITH TRUCKS + OCR + AUTO SNAPSHOT
        if detect_enabled and detection_ran and trucks and plates:
            for p in plates:
                plate_cx, plate_cy = p["cx"], p["cy"]
                for t in trucks:
                    if t["x1"] <= plate_cx <= t["x2"] and t["y1"] <= plate_cy <= t["y2"]:
                        if t["in_roi"] and p["in_roi"]:
                            y1p, y2p = p["y1"], p["y2"]
                            x1p, x2p = p["x1"], p["x2"]
                            plate_crop = frame[y1p:y2p, x1p:x2p]
                            plate_text = ocr_plate_image(plate_crop) or "unknown"
                            last_plate_text[cam_id] = plate_text

                            now = time.time()
                            if now - last_snapshot_ts[cam_id] > snapshot_interval_sec:
                                clean_plate = sanitize_plate_text(plate_text) or "unknown"
                                prefix = f"{cam_id}_truck_{clean_plate}"
                                send_to_tg = (send_config[cam_id]["mode"] == "auto")

                                print(
                                    f"[AUTO SNAPSHOT] [{cam_id}] TRUCK+PLATE, plate={plate_text}, "
                                    f"interval={snapshot_interval_sec}s, send_to_tg={send_to_tg}"
                                )

                                snapshot_image = draw.copy()
                                ts_text = time.strftime("%Y-%m-%d %H:%M:%S")
                                cv2.putText(snapshot_image, ts_text,
                                            (10, h - 25),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                            (0, 255, 255), 2)
                                cv2.putText(snapshot_image, f"Plate: {plate_text}",
                                            (10, h - 60),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                            (0, 255, 0), 2)

                                save_snapshot(
                                    snapshot_image,
                                    prefix=prefix,
                                    send_to_telegram=send_to_tg,
                                    camera_name=cam_name,
                                )
                                last_snapshot_ts[cam_id] = now
                        break  # one truck per plate is enough

        # DRAW BOXES
        for t in trucks:
            color = (0, 255, 255) if t["in_roi"] else (0, 150, 150)
            x1, y1, x2, y2 = t["x1"], t["y1"], t["x2"], t["y2"]
            cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
            cv2.putText(draw, f"{t['label']} {t['conf']:.2f}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        for p in plates:
            color = (0, 255, 0) if p["in_roi"] else (0, 150, 0)
            x1, y1, x2, y2 = p["x1"], p["y1"], p["x2"], p["y2"]
            cv2.rectangle(draw, (x1, y1), (x2, y2), color, 2)
            cv2.putText(draw, f"{p['label']} {p['conf']:.2f}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # STATUS TEXT
        if not detect_enabled:
            status_text = "DETECTION OFF"
        elif truck_count > 0:
            status_text = "TRUCK DETECTED"
        else:
            status_text = "NO TRUCK"

        last_status[cam_id]["text"] = status_text
        last_status[cam_id]["truck_count"] = truck_count
        last_status[cam_id]["total_truck_events"] = total_truck_events[cam_id]
        last_status[cam_id]["last_plate_text"] = last_plate_text[cam_id]

        # FINAL OVERLAYS
        frame_to_show = resize_for_display(draw, MAX_WIDTH)
        h_show, w_show = frame_to_show.shape[:2]
        last_frame_for_snapshot[cam_id] = frame_to_show.copy()

        roi_cfg = ROI_RECT.get(cam_id)
        if roi_cfg:
            rx1, ry1, rx2, ry2 = roi_cfg["x1"], roi_cfg["y1"], roi_cfg["x2"], roi_cfg["y2"]
            rx1 = max(0, min(rx1, w - 1))
            ry1 = max(0, min(ry1, h - 1))
            rx2 = max(0, min(rx2, w))
            ry2 = max(0, min(ry2, h))
            scale_x = w_show / float(w)
            scale_y = h_show / float(h)
            rx1s = int(rx1 * scale_x); ry1s = int(ry1 * scale_y)
            rx2s = int(rx2 * scale_x); ry2s = int(ry2 * scale_y)
            cv2.rectangle(frame_to_show, (rx1s, ry1s), (rx2s, ry2s), (0, 255, 255), 2)
            cv2.putText(frame_to_show, "ROI",
                        (rx1s + 5, ry1s + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)

        cv2.putText(frame_to_show, status_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 255), 2)

        det_text = f"{cam_name} | Detection: {'ON' if detect_enabled else 'OFF'}"
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
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
