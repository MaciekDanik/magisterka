import os
import requests
import io
import time
from PIL import Image

# --- KONFIGURACJA ---
HF_TOKEN = ""
API_URL = "https://api-inference.huggingface.co/models/black-forest-labs/FLUX.1-schnell" 
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

IMG_DIR = "datasets/generative_negatives/images"
LBL_DIR = "datasets/generative_negatives/labels"
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)

IMAGES_PER_PROMPT = 15  # Dla 4 promptów da nam to 60 czystych negatywów

# Prompty celujące w "piętę achillesową" Twojego bazowego modelu YOLO
PROMPTS = [
    # 1. Błękitne niebo z koronami drzew (Zdjęcia 4 i 5 z inżynierki)
    "Aerial drone top down view of a dense deciduous forest without leaves, bare branches over green grass, visible bright blue sky between branches, highly detailed",
    
    # 2. Czyste błękitne niebo
    "Aerial drone photography looking straight up at solid bright blue sky with very light uniform clouds, nothing else",
    
    # 3. Woda (często mylona z folią)
    "Aerial top down view of a calm blue river surface reflecting the sky, rippling water texture",
    
    # 4. Błotnista ziemia
    "Drone shot looking straight down at messy muddy ground with patches of brown dry grass and small puddles"
]

def query(payload, retries=3):
    """Odpytuje API Hugging Face z automatycznym ponawianiem w razie przeciążenia serwera."""
    for attempt in range(retries):
        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            return response.content
        elif response.status_code == 503: # Model się ładuje (Cold Start)
            print(f"Serwer się ładuje, czekam 15 sekund... (Próba {attempt+1}/{retries})")
            time.sleep(15)
        else:
            print(f"Błąd API: {response.status_code} - {response.text}")
            time.sleep(5)
    return None

print(f"Rozpoczynamy generowanie {len(PROMPTS) * IMAGES_PER_PROMPT} syntetycznych negatywów...")

image_counter = 1

for prompt_idx, prompt in enumerate(PROMPTS):
    print(f"\n--- Generowanie promptu {prompt_idx+1}/{len(PROMPTS)} ---")
    print(f"Prompt: '{prompt}'")
    
    for i in range(IMAGES_PER_PROMPT):
        # Seed gwarantuje unikalność każdego generowanego zdjęcia
        payload = {
            "inputs": prompt,
            "parameters": {
                "width": 640, 
                "height": 640,
                "seed": i * 1000 + prompt_idx # Zmieniaj ziarno dla różnorodności
            } 
        }
        
        image_bytes = query(payload)
        
        if image_bytes:
            try:
                # Otwieranie obrazu
                image = Image.open(io.BytesIO(image_bytes))
                filename = f"gen_neg_P{prompt_idx+1}_{image_counter:03d}"
                
                # ZAPIS JPG (Obraz)
                img_path = os.path.join(IMG_DIR, f"{filename}.jpg")
                image.save(img_path)
                
                # ZAPIS TXT (Negative Sample - PUSTY PLIK!)
                lbl_path = os.path.join(LBL_DIR, f"{filename}.txt")
                open(lbl_path, 'w').close()
                
                print(f"  [{i+1}/{IMAGES_PER_PROMPT}] Zapisano {filename}.jpg")
                image_counter += 1
                
                # Lekkie opóźnienie, żeby nie zablokowali nam darmowego API
                time.sleep(1.5) 
                
            except Exception as e:
                 print(f"  Błąd parsowania obrazu: {e}")
        else:
            print("  Nie udało się pobrać obrazu po kilku próbach.")

print("\nGenerowanie zakończone! Sprawdź folder datasets/generative_negatives/images.")