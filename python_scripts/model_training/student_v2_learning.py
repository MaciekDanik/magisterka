from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt') # Znowu ładujemy czysty model

print("Rozpoczynamy trening modelu Student_v2 (Iteracja 2)...")

results = model.train(
    data='student_data.yaml',
    epochs=300,        
    patience=50,       
    imgsz=640,
    batch=16,

    # Opcje zapisywania
    save=True,
    save_period=20, # Zapisuje wagi co 20 epok
    
    # Te same augmentacje!
    mixup=0.2,       
    mosaic=1.0,      
    hsv_h=0.05,      
    hsv_s=0.7,       
    hsv_v=0.4,       
    degrees=25.0,    
    translate=0.2,   
    scale=0.6,       
    
    name='student_v2', # <--- ZMIANA NAZWY
    device='cpu'  
)