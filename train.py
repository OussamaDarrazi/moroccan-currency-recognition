from ultralytics import YOLO

model = YOLO('yolo11n.pt') 

model.train(data="ds.yaml", epochs=300, imgsz=640, batch=8, device="cpu", workers=0, patience=50)