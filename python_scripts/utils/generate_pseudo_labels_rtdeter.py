import os
from ultralytics import YOLO

# --- USTAWIENIA ---
CONF_THRESHOLD = 0.45     # Ufamy predykcjom na min. 45% (łapie większość prawdziwych worków)
MAX_AREA_THRESHOLD = 0.35 # Zabijamy gigantyczne halucynacje nieba i wody (>35% kafelka)

# MODEL_PATH = 'S:/MyFiles/Studia/Magisterskie/Sem3/magisterka/models/teacher.pt' 
MODEL_PATH = 'S:/MyFiles/Studia/Magisterskie/Sem3/magisterka/models/student_v1.pt' # Używamy modelu Studenta, bo jest bardziej konserwatywny i mniej "halucynuje" niż Nauczyciel.
IMG_DIR = 'datasets/pseudo/images'
LBL_DIR = 'datasets/pseudo/labels'

os.makedirs(LBL_DIR, exist_ok=True)
model = YOLO(MODEL_PATH)

stats = {'total': 0, 'saved_images': 0, 'empty_backgrounds': 0}
image_files = [f for f in os.listdir(IMG_DIR) if f.lower().endswith(('.jpg', '.png'))]

for img_name in image_files:
    stats['total'] += 1
    img_path = os.path.join(IMG_DIR, img_name)
    txt_path = os.path.join(LBL_DIR, os.path.splitext(img_name)[0] + '.txt')
    
    # Próg 0.1, żeby w ogóle dostać z modelu wszystkie propozycje
    results = model.predict(source=img_path, conf=0.10, verbose=False)[0]
    
    valid_polygons = []
    
    if results.boxes is not None and len(results.boxes) > 0:
        for box, conf, cls in zip(results.boxes.xywhn, results.boxes.conf, results.boxes.cls):
            # box.xywhn to format: [środek_x, środek_y, szerokość, wysokość] (znormalizowane 0-1)
            w, h = box[2].item(), box[3].item()
            area = w * h
            
            if float(conf) >= CONF_THRESHOLD and area <= MAX_AREA_THRESHOLD:
                # Zapis formatu detekcji YOLO: class x_center y_center width height
                line = f"{int(cls)} {box[0].item()} {box[1].item()} {w} {h}"
                valid_polygons.append(line)

    # Decyzja:
    if len(valid_polygons) > 0:
        # Znalazł worki spełniające warunki
        with open(txt_path, 'w') as f:
            f.write("\n".join(valid_polygons))
        stats['saved_images'] += 1
    else:
        # Wszystko odrzuciliśmy (np. były tam tylko małe, niepewne chmury albo nic). 
        # Zapisujemy PUSTY PLIK, ucząc sieć, że tu NIC NIE MA.
        open(txt_path, 'w').close()
        stats['empty_backgrounds'] += 1

print("\n--- PODSUMOWANIE ---")
print(f"Przetworzono obrazów: {stats['total']}")
print(f"Zapisano pseudo-etykiety (z workami): {stats['saved_images']}")
print(f"Zapisano czyste tła (puste pliki): {stats['empty_backgrounds']}")
print("--------------------\n")