from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # load a pretrained model (YOLOv11)

#results = model.predict(source="https://ultralytics.com/images/bus.jpg")  # predict on an image URL
results = model("dataset_raw_frames/frame_000830.jpg") # predict on an image URL
results[0].show()  # display results
results.print()  # print results to console