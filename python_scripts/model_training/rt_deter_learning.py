from ultralytics import RTDETR

model = RTDETR('rtdetr-l.pt') 

print("Rozpoczynamy trening modelu bazowego RT-DETR...")

results = model.train(
    data='student_data.yaml', # Użyj odpowiedniego pliku yaml
    epochs=300,
    patience=50,
    imgsz=640,
    batch=16,
    
    project='magisterka/eksperymenty_rtdetr',
    name='rtdetr_baseline',
    
    # Używamy standardowych augmentacji (nie tych agresywnych SSL, bo to na razie baseline)
    mixup=0.0,
    mosaic=1.0,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=15.0,
    
    device='cpu' #<-- Ustaw na 'cuda' jeśli masz GPU i chcesz przyspieszyć trening!
)

print("Trening zakończony!")