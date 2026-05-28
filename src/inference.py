import os
from ultralytics import YOLO

# 1. Słownik ze ścieżkami do wytrenowanych modeli
modele = {
    'yolov8': {
        'baseline': 'models/training_runs/yolov8_Baseline/yolov8_baseline-3/weights/best.pt',
        'ssl_pseudo': 'models/training_runs/yolov8_SSL_Pseudo/yolov8_student_iter_2-2/weights/best.pt',
        'ssl_generative': 'models/training_runs/yolov8_SSL_Generative/yolov8_generative-2/weights/best.pt'
    },
    'yolo26': {
        'baseline': 'models/training_runs/yolo26_Baseline/yolo26_baseline-2/weights/best.pt',
        'ssl_pseudo': 'models/training_runs/yolo26_SSL_Pseudo/yolo26_student_iter_2-2/weights/best.pt',
        'ssl_generative': 'models/training_runs/yolo26_SSL_Generative/yolo26_generative-2/weights/best.pt'
    },
    'rtdetr': {
        'baseline': 'models/training_runs/rtdetr_Baseline/rtdetr_baseline-2/weights/best.pt',
        'ssl_pseudo': 'models/training_runs/rtdetr_SSL_Pseudo/rtdetr_student_iter_2-2/weights/best.pt',
        'ssl_generative': 'models/training_runs/rtdetr_SSL_Generative/rtdetr_generative-2/weights/best.pt'
    }
}

# 2. Ścieżka do zdjęć testowych
zdjecia_testowe = 'datasets/manual/valid/images' 
# zdjecia_testowe = 'datasets/test_sacred/images'  # Zmieniamy na nową ścieżkę do zdjęć testowych
folder_zapisow = os.path.abspath('results/validation')  # Zmieniamy na nową ścieżkę do folderu wyników

print(f"Rozpoczynam inferencję modeli na zdjęciach z: {zdjecia_testowe}")

# 3. Uruchomienie predykcji dla każdego modelu w pętli
for model_name, metody in modele.items():
    for nazwa_metody, sciezka_modelu in metody.items():
        print(f"\n=========================================")
        print(f"Testowanie modelu: {model_name} | Metoda: {nazwa_metody}")
        print(f"Ścieżka do wag: {sciezka_modelu}")
        print(f"=========================================\n")
        
        # Oczekiwany rezultat: folder_zapisow/model_name/nazwa_metody
        project_path = os.path.join(folder_zapisow, model_name)
        
        try:
            model = YOLO(sciezka_modelu)
            results = model.predict(
                source=zdjecia_testowe,
                save=True,               # Zapisuje obrazki
                save_txt=True,           # Zapisuje bounding boxy
                save_conf=True,          # Dopisuje wywnioskowaną pewność do plików .txt
                project=project_path,    # Główny folder wyjściowy to np. results/test_sacred/yolov8
                name=nazwa_metody,       # Zapis w podfolderze metody (np. baseline)
                exist_ok=True,            # Pozwala nadpisać folder
                # visualize=True            # Wyświetla wyniki
                save_crop=True             # Zapisuje wycięte obiekty (crop) w folderze 'crops' wewnątrz folderu metody
            )
            print(f"Gotowe dla {model_name}/{nazwa_metody}! Obrazki i pliki .txt zapisano w folderze {project_path}/{nazwa_metody}")
        except Exception as e:
            print(f"Wystąpił błąd dla {model_name}/{nazwa_metody}: {e}")

print("\nWszystkie testy zostały zakończone!\n")