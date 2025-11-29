# image_detect_test.py
# Confidential – Internal Use Only

from ultralytics import YOLO
import cv2
import numpy as np
import os

# 1. Absolute path to your image
IMAGE_PATH = "/Users/piriya/kbs_qc_ai_camera/training/p6.png"

print("Current working dir:", os.getcwd())
print("Image exists? ", os.path.isfile(IMAGE_PATH))

img = cv2.imread(IMAGE_PATH)
print("cv2.imread type:", type(img))

if img is None:
    raise RuntimeError("cv2.imread failed – check IMAGE_PATH")

# double-check type is numpy.ndarray
print("isinstance(img, np.ndarray)?", isinstance(img, np.ndarray))

# 2. Load YOLO11n
model = YOLO("model/yolo11n.pt")  # auto-downloads if missing


# 3. Run detection on the numpy image instead of filename
results = model(img, conf=0.25, imgsz=640, classes=[0]) # predict on the image array

res = results[0]
print("Detections:", len(res.boxes))

for box in res.boxes:
    cls_id = int(box.cls[0]) # class id
    conf = float(box.conf[0]) # confidence
    xyxy = box.xyxy[0].tolist() # bounding box
    label = model.names[cls_id] # class name
    print(f"{label}: {conf:.2f} -> {xyxy}") # bounding box details

annotated = res.plot()
out_path = "/Users/piriya/kbs_qc_ai_camera/training/picture_detected.jpg"
cv2.imwrite(out_path, annotated)
print("Saved:", out_path)
