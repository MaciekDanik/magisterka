from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt') 

print("Rozpoczynam TEST na sucho (1 epoka)...")

# Trening na zaledwie 1 epoce, żeby sprawdzić czy wszystko działa
results = model.train(
    data='generative_data.yaml', 
    epochs=1,          # <--- TYLKO 1 EPOKA
    imgsz=640,
    batch=16,          # Upewnij się, że RAM daje radę
    
    project='magisterka/testy',
    name='dry_run_generative', 
    
    # Wyłączamy augmentacje na ten test
    mixup=0.0, 
    mosaic=1.0, 
    device='cpu'       # Upewnij się, że 'cpu' jest tu wpisane, skoro tak trenujesz
)

print("Test na sucho zakończony sukcesem! Ścieżki są poprawne.")