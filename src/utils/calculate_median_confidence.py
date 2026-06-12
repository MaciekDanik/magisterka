import os
import numpy as np
from pathlib import Path

# Podaj nazwy plików (lub base name z/bez .txt), dla których chcesz policzyć medianę
TARGET_FILES = [
    "sacred_1_R001_C001.txt",
    "sacred_1_R001_C003.txt",
    "sacred_3.txt", 
    # dodaj kolejne nazwy plików...
]

# Upewniamy się, że każdy wpis na liście ma rozszerzenie .txt
TARGET_FILES = [f if f.endswith(".txt") else f"{f}.txt" for f in TARGET_FILES]

def calculate_median_confidence(base_results_path="results/test_sacred_crops"):
    base_dir = Path(base_results_path)
    
    if not base_dir.exists():
        print(f"Katalog bazowy {base_dir} nie istnieje.")
        return

    models = [d.name for d in base_dir.iterdir() if d.is_dir()]
    
    print(f"Wybrane pliki do sprawdzenia: {TARGET_FILES}\n")
    print("-" * 80)
    
    for file_name in TARGET_FILES:
        print(f"Wyniki dla pliku: {file_name}")
        for model in sorted(models):
            model_dir = base_dir / model
            methods = [d.name for d in model_dir.iterdir() if d.is_dir()]
            
            for method in sorted(methods):
                labels_dir = model_dir / method / "labels"
                file_path = labels_dir / file_name
                confidences = []
                
                if file_path.exists():
                    with open(file_path, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            # Sprawdzamy czy format to YOLO + confidence (min. 6 kolumn: class x y w h conf)
                            if len(parts) >= 6:
                                try:
                                    conf = float(parts[5])
                                    confidences.append(conf)
                                except ValueError:
                                    continue
                
                if confidences:
                    median_conf = np.median(confidences)
                    mean_conf = np.mean(confidences)
                    print(f"  [{model: <10}] {method: <15} -> Mediana: {median_conf:.4f} | Średnia: {mean_conf:.4f} | Łącznie detekcji: {len(confidences)}")
                else:
                    print(f"  [{model: <10}] {method: <15} -> Brak detekcji (lub brak pliku).")
        print("-" * 80)

if __name__ == "__main__":
    calculate_median_confidence()
