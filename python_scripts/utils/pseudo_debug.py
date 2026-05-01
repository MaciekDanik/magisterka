import os
import cv2 as cv
from ultralytics import YOLO

# Ścieżki
MODEL_PATH = 'S:/MyFiles/Studia/Magisterskie/Sem3/magisterka/models/teacher.pt' 
IMG_DIR = 'S:/MyFiles/Studia/Magisterskie/Sem3/magisterka/datasets/pseudo/images'
DEBUG_DIR = 'S:/MyFiles/Studia/Magisterskie/Sem3/magisterka/datasets/pseudo/debug'

os.makedirs(DEBUG_DIR, exist_ok=True)
model = YOLO(MODEL_PATH)

print("Generowanie obrazów debugujących...")

for img_name in os.listdir(IMG_DIR):
    if not img_name.lower().endswith(('.jpg', '.png')):
        continue
        
    img_path = os.path.join(IMG_DIR, img_name)
    
    # Puszczamy model z bardzo niskim progiem (0.2), żeby zobaczyć WSZYSTKO co podejrzewa
    results = model.predict(source=img_path, conf=0.2, verbose=False)[0]
    
    # YOLO ma wbudowaną świetną funkcję do rysowania wyników na zdjęciu
    annotated_frame = results.plot() 
    
    # Zapisujemy pokolorowany obrazek
    debug_path = os.path.join(DEBUG_DIR, img_name)
    cv.imwrite(debug_path, annotated_frame)

print(f"Zakończono! Sprawdź folder: {DEBUG_DIR}")