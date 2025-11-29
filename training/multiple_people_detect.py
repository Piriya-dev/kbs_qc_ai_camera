from ultralytics import YOLO
import cv2
import numpy as np
import os

IMAGE_PATH = "/Users/piriya/kbs_qc_ai_camera/training/p6.png"

img = cv2.imread(IMAGE_PATH)
model = YOLO("model/yolo11n.pt")

results = model(
    img,
    conf=0.10,
    imgsz=1280,
    iou=0.45,
    classes=[0]   # detect person only
)

res = results[0]
print("Number of people detected:", len(res.boxes))

for box in res.boxes:
    cls_id = int(box.cls[0])
    conf = float(box.conf[0])
    xyxy = box.xyxy[0].tolist()
    print(f"person: {conf:.2f} -> {xyxy}")

annotated = res.plot()
cv2.imshow("Annotated Image", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows() 
cv2.imwrite("multiple_people_detected.jpg", annotated)
print("Detection results saved to multiple_people_detected.jpg")