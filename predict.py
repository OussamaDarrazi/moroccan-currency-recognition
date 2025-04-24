from ultralytics import YOLO
import os


model = YOLO("moroccan_currency_detector_yolo11.pt")


results = model.predict(
    source="to_predict",      
    save=True,                
    project="predicted",      
    name="results",           
    exist_ok=True             
)
