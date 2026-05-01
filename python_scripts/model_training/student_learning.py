from ultralytics import YOLO

# Ważne: Zaczynamy od czystego modelu pretrenowanego na COCO, 
# NIE od wag Nauczyciela! Uczeń musi przejść własną ścieżkę.
model = YOLO('yolov8n-seg.pt') 

try:
    results = model.train(
        data='student_data.yaml',
        epochs=300,
        patience=50,
        imgsz=640,
        batch=16,
        
        # Opcje zapisywania
        save=True,
        save_period=5, # Zapisuje wagi co 5 epok
        
        # --- HARD AUGMENTATION (Dodawanie Hałasu) ---
        mixup=0.2,       
        mosaic=1.0,      
        hsv_h=0.05,      
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=25.0,    
        translate=0.2,   
        scale=0.6,       
        # --------------------------------------------
        
        name='student_v1',
        device='cpu'
    )
except KeyboardInterrupt:
    print("\n--- Uczenie przerwane ręcznie! ---")
    print("Biblioteka Ultralytics automatycznie zachowuje pliki 'best.pt' oraz 'last.pt' (jeśli zapisała się choć jedna epoka).")
    print("Sprawdź folder: runs/segment/student_v1/weights/")