import cv2
import platform

CAM_INDEX = 0
IS_MAC = platform.system() == "Darwin"

if IS_MAC:
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_AVFOUNDATION)
else:
    cap = cv2.VideoCapture(CAM_INDEX)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    cv2.imshow('test webcam - press q to quit', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
