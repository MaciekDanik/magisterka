from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt') # Startujemy od zera (czyste wagi COCO)

print("Rozpoczynamy trening Eksperymentu B (Generative Negative Samples)...")

results = model.train(
    data='generative_data.yaml', 
    epochs=300,        
    patience=50,       
    imgsz=640,
    batch=16, 
    
    project='magisterka/eksperymenty_ssl',
    name='yolo_generative', # Zapis wyników pod nową nazwą
    save=True,
    save_period=20, # Zapisuje wagi co 20 epok
    
    # Identyczna augmentacja co w SSL, aby metodologicznie eksperyment był spójny
    mixup=0.2,       
    mosaic=1.0,      
    hsv_h=0.05,      
    hsv_s=0.7,       
    hsv_v=0.4,       
    degrees=25.0,    
    translate=0.2,   
    scale=0.6,       
    
    device='cpu' # (lub 'cpu' jeśli uczysz na procesorze)          
)

print("Trening Eksperymentu B zakończony!")