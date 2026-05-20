# Gotowy, oficjalny obraz twórców YOLO - ma CUDA, PyTorcha i ultralytics
FROM ultralytics/ultralytics:latest

# Ustawiamy folder roboczy
WORKDIR /workspace

# Kopiujemy Twój plik z wymaganiami
COPY requirements.txt .

# Instalujemy ewentualne dodatkowe biblioteki (jeśli masz tam coś poza ultralytics)
RUN pip install --no-cache-dir -r requirements.txt