from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # load a pretrained model (YOLOv11)

#Train the model
train_results = model.train(data="dataset1/data.yaml", 
    epochs=100, imgsz=640, 
    device="cpu", 
   workers=0, batch=16
)
metric = model.val()
print(metric)