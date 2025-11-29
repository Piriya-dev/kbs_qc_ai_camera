from ultralytics import YOLO
import cv2

def detect_from_image(image_path):
    # Load the YOLO model
    model=YOLO("yolo11n.pt")
    print("Model loaded successfully.",model)
    
    
detect_from_image("test_images/p2.png")