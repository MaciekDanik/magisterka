# Raport Statystyczny z Oceny Detekcji (Test Wilcoxona)
## Wstęp i metodologia (Background)
Celem niniejszej ewaluacji statystycznej jest rzetelne określenie, czy alternatywne metody uczenia (wykorzystanie pseudoznakowania oraz generatywnych masek tła) wprowadzają istotną statystycznie poprawę w zdolności modelu do poprawnej detekcji.
Podstawowym problemem w standardowej analizie agregowanej (np. globalnym mAP) jest to, że eksperymenty pojedyncze (single-seed runs) oferują tylko jeden zagregowany wynik. Aby zyskać odpowiednią moc statystyczną, wyliczono metrykę precyzji i czułości (F1-Score na podstawie IoU > 0.5) osobno dla **każdego wycinka ewaluacyjnego (każdego obrazu)**. Dzięki temu dla pojedynczego modelu uzyskujemy wektor $N$ obserwacji, gdzie $N$ to liczba kadrów testowych.
Ze względu na niespełnienie założeń o rozkładzie normalnym wyników oraz na fakt, iż porównujemy odpowiedzi różnych modeli dla tego samego zbioru ilustracji (próby połączone), zastosowano nieparametryczny **test znakowanych rang Wilcoxona**.

* Wartość **p-value < 0.05** przyjmuje się jako próg istotności statystycznej odrzucający hipotezę zerową (H0 mówiącą o braku różnic między modelem Baseline a wariantami SSL).

## Wyniki operacyjne
Niniejszy raport zawiera wyniki z testowania różnic jakości predykcji poszczególnych modeli. Pary powiązane wynikają z faktu wykorzystania tych samych obrazów testowych.

**Liczba obrazów walidacyjnych (próba N):** 43
**Zastosowana metryka (zmienna zależna):** F1-Score (IoU > 0.5)

## Tabela wyników

| Architektura | Porównanie | Śr. F1 (Baseline) | Śr. F1 (Metoda) | p-value | Istotność stat. (p<0.05) |
|--------------|------------|-------------------|-----------------|---------|--------------------------|
| YOLOV8 | Baseline vs ssl_pseudo | 0.0014 | 0.0000 | 0.1797 | - |
| YOLOV8 | Baseline vs ssl_generative | 0.0014 | 0.0019 | 0.1797 | - |
| YOLO26 | Baseline vs ssl_pseudo | 0.0014 | 0.0015 | 0.3173 | - |
| YOLO26 | Baseline vs ssl_generative | 0.0014 | 0.0000 | 0.3173 | - |
| RTDETR | Baseline vs ssl_pseudo | 0.0003 | 0.0020 | 0.1797 | - |
| RTDETR | Baseline vs ssl_generative | 0.0003 | 0.0024 | 0.2850 | - |
