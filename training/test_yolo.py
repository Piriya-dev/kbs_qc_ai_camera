import cv2
from ultralytics import YOLO

def detect_from_image(image_path):
    # Load the YOLO model
    model=YOLO('model/yolo11n.pt')
    print(model.names)

detect_from_image('p1.JPG')