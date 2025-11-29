# web_plate_only_app.py
# Confidential – Internal Use Only
#
# Thai License Plate Detection (YOLO11) + OCR (EasyOCR)
# With far-distance enhancements + CENTER ROI:
#   - Upscale frame before YOLO detection (small-object boost)
#   - imgsz=1280, conf=0.10
#   - Only plates whose center is in a CENTER ROI are used
#   - Shows ROI box on video
#   - Shows plate text on video and web
#   - Snapshot OK + BROKEN
#   - Telegram alert on BROKEN
#   - Debug logs

import os
import time
import re
import cv2
import numpy as np
import requests
import easyocr
from ultralytics import YOLO
from flask import Flask, Response, render_template_string, jsonify, request, make_response

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CAMERAS = {
    "cam1": {
        "name": "KBS-Gate-01",
        #"rtsp": "rtsp://admin:KBSit%402468@192.168.5.46:554/ISAPI/Streaming/channels/102",
        "rtsp": "rtsp://admin:KBSit%402468@192.168.4.190:554/ISAPI/Streaming/channels/101" #--> IT Camera
    },
}

CAMERA_IDS = list(CAMERAS.keys())

PLATE_MODEL_PATH = os.path.join(BASE_DIR, "thai_plate.pt")

YOLO_CONF_PLATE = 0.10           # lowered threshold
YOLO_IMGSZ = 840               # bigger input → detect small plates
RUN_EVERY_N = 2

UPSCALE_FACTOR = 1.8             # Upscale trick for far small plates

MAX_WIDTH = 840

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

# Center ROI (as ratios of width/height)
CENTER_ROI = {
    "x1_ratio": 0.25,  # 25% from left
    "x2_ratio": 0.75,  # 75% from left  -> middle 50%
    "y1_ratio": 0.30,  # 30% from top
    "y2_ratio": 0.80,  # 80% from top   -> lower-middle area
}

# Telegram
BROKEN_ALERT_MIN_INTERVAL_SEC = 5
TELEGRAM_BOT_TOKEN = "8559813204:AAH0wnA83cvbjOr99fbCr_isV5tXepjJxkg"
TELEGRAM_API_URL = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"
CHAT_IDS = [6953358112, -1003103459072]

# Snapshots
SNAPSHOT_DIR = os.path.join(BASE_DIR, "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# MODEL LOAD
# ─────────────────────────────────────────────
if not os.path.isfile(PLATE_MODEL_PATH):
    raise FileNotFoundError("thai_plate.pt NOT FOUND. Put model in project folder.")

print("[INFO] Loading Plate Model...")
plate_model = YOLO(PLATE_MODEL_PATH)
PLATE_CLASS_NAMES = plate_model.names
print(" → Plate model classes:", PLATE_CLASS_NAMES)

print("[INFO] Loading EasyOCR...")
ocr_reader = easyocr.Reader(["th", "en"], gpu=False)

# ─────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────
state = {cam_id: {"detect_enabled": True} for cam_id in CAMERA_IDS}
last_frame = {cam_id: None for cam_id in CAMERA_IDS}
good_plate_count = {cam_id: 0 for cam_id in CAMERA_IDS}
broken_plate_count = {cam_id: 0 for cam_id in CAMERA_IDS}
last_plate_text = {cam_id: "" for cam_id in CAMERA_IDS}
last_broken_alert_ts = {cam_id: 0 for cam_id in CAMERA_IDS}

last_status = {
    cam_id: {
        "text": "NO DETECTION",
        "good_plate_count": 0,
        "broken_plate_count": 0,
        "last_plate_text": "",
    }
    for cam_id in CAMERA_IDS
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def send_telegram_photo(path, caption=""):
    if not os.path.isfile(path):
        print("[TEL] Missing photo:", path)
        return
    for chat in CHAT_IDS:
        try:
            with open(path, "rb") as f:
                requests.post(
                    f"{TELEGRAM_API_URL}/sendPhoto",
                    data={"chat_id": chat, "caption": caption},
                    files={"photo": f},
                    timeout=10
                )
            print(f"[TEL] Sent to {chat}")
        except Exception as e:
            print("[TEL] Error:", e)

def sanitize_plate(text):
    if not text: 
        return ""
    return re.sub(r"[^0-9A-Za-zก-ฮ๐-๙]", "", text)

def ocr_plate(crop):
    if crop is None or crop.size == 0:
        print("[OCR] Empty crop")
        return ""
    try:
        crop = cv2.resize(crop, None, fx=2.0, fy=2.0)
        results = ocr_reader.readtext(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), detail=0)
    except Exception as e:
        print("[OCR] Error:", e)
        return ""
    if not results:
        print("[OCR] No text")
        return ""
    raw = "".join(results)
    cleaned = sanitize_plate(raw)
    print(f"[OCR] raw={results} cleaned={cleaned}")
    return cleaned

def save_snapshot(img, prefix, send_tg=False, cam_name=""):
    ts = time.strftime("%Y%m%d_%H%M%S")
    fname = f"{prefix}_{ts}.jpg"
    path = os.path.join(SNAPSHOT_DIR, fname)
    cv2.imwrite(path, img)
    print("[SNAP] Saved:", path)
    if send_tg:
        send_telegram_photo(path, caption=f"{cam_name} | {prefix}")
    return path

def upscale_frame(frame):
    return cv2.resize(frame, None, fx=UPSCALE_FACTOR, fy=UPSCALE_FACTOR)

def get_center_roi_rect(w, h):
    """Return ROI rectangle (x1,y1,x2,y2) in pixels for the center area."""
    x1 = int(CENTER_ROI["x1_ratio"] * w)
    x2 = int(CENTER_ROI["x2_ratio"] * w)
    y1 = int(CENTER_ROI["y1_ratio"] * h)
    y2 = int(CENTER_ROI["y2_ratio"] * h)
    return x1, y1, x2, y2

def box_center_in_roi(x1, y1, x2, y2, roi_rect):
    """Check if bbox center is inside ROI rectangle."""
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    rx1, ry1, rx2, ry2 = roi_rect
    return (rx1 <= cx <= rx2) and (ry1 <= cy <= ry2)

# ─────────────────────────────────────────────
# WEB UI
# ─────────────────────────────────────────────
app = Flask(__name__)
HTML = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>KBS Plate OCR</title>
<style>
body { background:#111; color:#eee; font-family:Arial; text-align:center; }
video, img { max-width:90vw; border:3px solid #0f0; border-radius:6px; }
.info { margin-top:15px; font-size:20px; }
</style>
</head>
<body>
<h1>KBS Thai Plate OCR (Center ROI)</h1>

<select id="camSel" onchange="chgCam()">
{% for cam, cfg in cams.items() %}
<option value="{{cam}}">{{cam}} - {{cfg['name']}}</option>
{% endfor %}
</select>

<div>
<img id="streamImg" src="/video_feed?cam=cam1">
</div>

<div class="info" id="infoText"></div>

<script>
let curCam="cam1";
function chgCam(){
  curCam=document.getElementById("camSel").value;
  document.getElementById("streamImg").src="/video_feed?cam="+curCam+"&t="+Date.now();
}
function poll(){
 fetch("/latest_status?cam="+curCam)
 .then(r=>r.json())
 .then(s=>{
   document.getElementById("infoText").innerHTML=
     "Status: "+s.text+"<br>"+
     "Good: "+s.good_plate_count+" | Broken: "+s.broken_plate_count+"<br>"+
     "Last Plate: "+s.last_plate_text;
 });
}
setInterval(poll,700);
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML, cams=CAMERAS)

@app.route("/latest_status")
def latest_status_view():
    return jsonify(last_status[get_cam()])

@app.route("/toggle_detection", methods=["POST"])
def toggle_detection():
    cam = get_cam()
    state[cam]["detect_enabled"] = not state[cam]["detect_enabled"]
    return jsonify({"ok": True})

def get_cam():
    cam = request.args.get("cam", "cam1")
    return cam if cam in CAMERAS else "cam1"

# ─────────────────────────────────────────────
# FRAME GENERATOR
# ─────────────────────────────────────────────
def generate_frames(cam_id):
    rtsp = CAMERAS[cam_id]["rtsp"]
    cam_name = CAMERAS[cam_id]["name"]
    print("[INFO] Open RTSP:", rtsp)

    cap = cv2.VideoCapture(rtsp, cv2.CAP_FFMPEG)

    frame_i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame_i += 1
        draw = frame.copy()
        h, w = frame.shape[:2]

        # Center ROI rectangle in original frame
        roi_x1, roi_y1, roi_x2, roi_y2 = get_center_roi_rect(w, h)

        plates = []

        if state[cam_id]["detect_enabled"] and (frame_i % RUN_EVERY_N == 0):

            # 🔥 UPSCALE FOR FAR SMALL PLATES
            big = upscale_frame(frame)
            bh, bw = big.shape[:2]

            # YOLO DETECT ON UPSCALED VERSION
            results = plate_model(big, imgsz=YOLO_IMGSZ, conf=YOLO_CONF_PLATE, verbose=False)[0]

            print(f"[DEBUG] YOLO plate box count (all) = {len(results.boxes)}")

            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                x1b, y1b, x2b, y2b = map(int, box.xyxy[0])
                label = PLATE_CLASS_NAMES.get(cls)

                # Only license_plate class
                if label != "license_plate":
                    continue

                # 🔥 Rescale box positions back to original frame
                x1 = int(x1b / UPSCALE_FACTOR)
                y1 = int(y1b / UPSCALE_FACTOR)
                x2 = int(x2b / UPSCALE_FACTOR)
                y2 = int(y2b / UPSCALE_FACTOR)

                x1 = max(0, min(x1, w-1))
                y1 = max(0, min(y1, h-1))
                x2 = max(0, min(x2, w))
                y2 = max(0, min(y2, h))

                # Keep only if bbox center inside CENTER ROI
                in_center = box_center_in_roi(x1, y1, x2, y2, (roi_x1, roi_y1, roi_x2, roi_y2))
                print(f"[DEBUG] plate box label={label}, conf={conf:.2f}, bbox=({x1},{y1},{x2},{y2}), in_center={in_center}")

                if not in_center:
                    continue

                plates.append((x1, y1, x2, y2, conf))

        # PROCESS PLATES INSIDE CENTER ROI
        status_text = "NO DETECTION"

        if plates:
            # choose largest plate box
            plates.sort(key=lambda p: (p[2]-p[0])*(p[3]-p[1]), reverse=True)
            x1, y1, x2, y2, conf = plates[0]

            # draw plate box
            cv2.rectangle(draw, (x1,y1), (x2,y2), (0,255,0), 2)
            plate_crop = frame[y1:y2, x1:x2]

            text = ocr_plate(plate_crop)

            if text and len(text) >= 3:
                # GOOD
                good_plate_count[cam_id] += 1
                last_plate_text[cam_id] = text
                status_text = f"PLATE OK: {text}"

                cv2.putText(draw, f"Plate: {text}", (x1, max(y1-20,20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

                save_snapshot(draw, f"{cam_id}_plate_{sanitize_plate(text)}")

            else:
                # BROKEN
                broken_plate_count[cam_id] += 1
                last_plate_text[cam_id] = ""
                status_text = "BROKEN PLATE (OCR FAIL)"

                cv2.putText(draw, "BROKEN PLATE", (x1, max(y1-20,20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

                now = time.time()
                if now - last_broken_alert_ts[cam_id] > BROKEN_ALERT_MIN_INTERVAL_SEC:
                    last_broken_alert_ts[cam_id] = now
                    fp = save_snapshot(draw, f"{cam_id}_broken_plate", send_tg=True, cam_name=cam_name)
                    print("[ALERT] BROKEN SENT:", fp)

        # update status
        last_status[cam_id] = {
            "text": status_text,
            "good_plate_count": good_plate_count[cam_id],
            "broken_plate_count": broken_plate_count[cam_id],
            "last_plate_text": last_plate_text[cam_id],
        }

        # draw CENTER ROI rectangle on video
        cv2.rectangle(draw, (roi_x1, roi_y1), (roi_x2, roi_y2), (0,255,255), 2)
        cv2.putText(draw, "CENTER ROI", (roi_x1+5, roi_y1+25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        # stream frame
        ok, buf = cv2.imencode(".jpg", draw)
        if not ok: 
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")


@app.route("/video_feed")
def video_feed():
    cam_id = get_cam()
    return Response(generate_frames(cam_id),
        mimetype="multipart/x-mixed-replace; boundary=frame")


# ─────────────────────────────────────────────
# START SERVER
# ─────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
