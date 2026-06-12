# Raport Statystyczny: Redukcja Fałszywych Detekcji
## Wstęp i metodologia
Celem niniejszej analizy jest weryfikacja, czy zastosowane metody poprawy jakości widzenia (SSL Pseudo oraz SSL Generative) prowadzą do istotnej statystycznie redukcji 'halucynacji' modelu (fałszywych detekcji - FP).
Jako zmienną zależną przyjęto liczbę fałszywych detekcji (FP) na pojedynczym obrazie testowym. Ze względu na nieparametryczny charakter rozkładu danych oraz próbę powiązaną (ten sam zbiór testowy dla modeli Baseline i wariantów), zastosowano test znakowanych rang Wilcoxona (test jednostronny: alternative='less').

**Liczba obrazów walidacyjnych (próba N):** 43
## Tabela wyników: Redukcja fałszywych detekcji (FP)

| Architektura | Porównanie | Śr. FP (Baseline) | Śr. FP (Metoda) | p-value | Istotność (p<0.05 i redukcja) |
|--------------|------------|-------------------|-----------------|---------|-------------------------------|
| YOLOV8 | Baseline vs ssl_pseudo | 13.37 | 11.88 | 0.0030 | TAK |
| YOLOV8 | Baseline vs ssl_generative | 13.37 | 11.93 | 0.0012 | TAK |
| YOLO26 | Baseline vs ssl_pseudo | 10.00 | 10.88 | 0.9652 | NIE |
| YOLO26 | Baseline vs ssl_generative | 10.00 | 10.91 | 0.9626 | NIE |
| RTDETR | Baseline vs ssl_pseudo | 15.51 | 14.14 | 0.0103 | TAK |
| RTDETR | Baseline vs ssl_generative | 15.51 | 20.23 | 0.9997 | NIE |
