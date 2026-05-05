import os
import torch
from diffusers import StableDiffusionPipeline
import time

# --- KONFIGURACJA ---
IMG_DIR = "datasets/generative_negatives/images"
LBL_DIR = "datasets/generative_negatives/labels"
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)

IMAGES_PER_PROMPT = 15

# Prompty dobrane tak, by przypominały typowe tło oryginalnego zbioru danych (z drona oraz z poziomu ziemi)
PROMPTS = [
    "Ground level photograph of an empty grassy field with some dry patches of dirt, eye level perspective, natural daylight, photorealistic",
    "Aerial top-down drone photography of an empty dirt road passing through a dense forest, daylight, empty scene",
    "Eye-level shot of a messy construction site or dirt ground with gravel, scattered rocks, and mud, high resolution",
    "Ground level view of an asphalt road edge or sidewalk with some dirt and green weeds, sunny day, highly detailed, no trash", #? not great results, maybe useful?
    "Ground level photo of a anti flood barrier or levee with grass and dirt, taken from the side, natural lighting, photorealistic"
]

print("Ładowanie modelu lokalnego (to może chwilę potrwać za pierwszym razem, bo pobierze wagi)...")
model_id = "runwayml/stable-diffusion-v1-5"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# Inicjalizacja pipeline'u
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=dtype)
pipe = pipe.to(device)

print(f"Rozpoczynamy generowanie {len(PROMPTS) * IMAGES_PER_PROMPT} syntetycznych negatywów lokalnie na urządzeniu: {device}...")

image_counter = 1

for prompt_idx, prompt in enumerate(PROMPTS):
    print(f"\n--- Generowanie promptu {prompt_idx+1}/{len(PROMPTS)} ---")
    print(f"Prompt: '{prompt}'")
    
    for i in range(IMAGES_PER_PROMPT):
        seed = i * 1000 + prompt_idx
        generator = torch.Generator(device).manual_seed(seed)
        
        try:
            result = pipe(
                prompt,
                num_inference_steps=25,
                generator=generator,
                height=640,
                width=640
            )
            image = result.images[0]
            
            filename = f"gen_neg_P{prompt_idx+1}_{image_counter:03d}"
            
            img_path = os.path.join(IMG_DIR, f"{filename}.jpg")
            image.save(img_path)
            
            lbl_path = os.path.join(LBL_DIR, f"{filename}.txt")
            open(lbl_path, 'w').close()
            
            print(f"  [{i+1}/{IMAGES_PER_PROMPT}] Zapisano {filename}.jpg")
            image_counter += 1
            
        except Exception as e:
            print(f"  Błąd generowania obrazu: {e}")

print("\nGenerowanie zakończone! Sprawdź folder datasets/generative_negatives/images.")