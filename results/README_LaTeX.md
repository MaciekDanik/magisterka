# Skrypty i wykresy do pracy dyplomowej (LaTeX)

Katalog `plots/` zawiera wygenerowane wykresy ewaluacyjne gotowe do wstawienia do dokumentu LaTeX.
Poniżej znajdują się przygotowane definicje środowisk `\begin{figure}`, połączone ze zwięzłym opisem naukowym oraz zadeklarowanymi etykietami umożliwiającymi prawidłowe odwoływanie w tekście.

---

### 1. Wykres z metrykami (Trade-off Precyzji i Czułości)

**Pliki:** `wykres_1_metryki_YOLOv8.png`, `wykres_1_metryki_YOLO26.png`, `wykres_1_metryki_RT-DETR-L.png`
**Skrypt:** `plot_1_metrics.py`

**Opis w LaTeX:**
Skrypt w swojej nowej iteracji generuje macierz plików, umożliwiając stworzenie sub-figur dla poszczególnych architektur, zestawiających trójki decyzyjne (`mAP@0.5`, `Precision`, `Recall`).

```latex
\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Pics/wykres_1_metryki_YOLOv8.png}
        \caption{Zestawienie metryk dla YOLOv8.}
        \label{fig:metrics_yolov8}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Pics/wykres_1_metryki_YOLO26.png}
        \caption{Zestawienie metryk dla YOLO11 (YOLO26).}
        \label{fig:metrics_yolo26}
    \end{subfigure}
    
    \vspace{0.5cm} % Przerwa pomiędzy rzędami
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Pics/wykres_1_metryki_RT-DETR-L.png}
        \caption{Zestawienie metryk dla RT-DETR-L.}
        \label{fig:metrics_rtdetr}
    \end{subfigure}
    \caption{Porównanie wartości metryk ewaluacyjnych mAP@0.5, precyzji oraz czułości w zależności od wariantu treningowego w rozbiciu na testowane architektury sieci.}
    \label{fig:metrics_tradeoff_all}
\end{figure}
```

---

### 2. Wykres funkcji starty i zjawiska przeuczania

**Pliki:** `wykres_2_loss_YOLOv8.png`, `wykres_2_loss_YOLO26.png`, `wykres_2_loss_RT-DETR-L.png`
**Skrypt:** `plot_2_loss.py`

**Opis w LaTeX:**
Wykresy liniowe prezentujące postęp funkcji straty na zbiorze walidacyjnym (Validation Loss) względem przeprowadzonych epok. Pozwalają uzmysłowić wpływ zwiększenia zbioru danych (bądź dodania tła) na stabilność treningu we wszystkich architekturach.

```latex
\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Pics/wykres_2_loss_YOLOv8.png}
        \caption{Validation Loss - YOLOv8.}
        \label{fig:loss_yolov8}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Pics/wykres_2_loss_YOLO26.png}
        \caption{Validation Loss - YOLO11 (YOLO26).}
        \label{fig:loss_yolo26}
    \end{subfigure}
    
    \vspace{0.5cm}
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Pics/wykres_2_loss_RT-DETR-L.png}
        \caption{Validation Loss - RT-DETR-L.}
        \label{fig:loss_rtdetr}
    \end{subfigure}
    \caption{Krzywe funkcji straty dla zbioru walidacyjnego. Obejmują one porównanie standardowego uczonego modelu oraz modeli douczanych z wykorzystaniem metod SSL.}
    \label{fig:val_loss_all}
\end{figure}
```

---

### 3. Podatność architektur na metody Data Augmentation/SSL

**Plik:** `wykres_3_architektury.png`
**Skrypt:** `plot_3_architecture.py`

**Opis w LaTeX:**
Wykres słupkowy ilustrujący jak bardzo zastosowane metody wzbogacania danych (generatywne tła i testowe pseudo-etykiety) wpływają na przyrost mAP@50 względem bazowego modelu. Ograniczenie osi od -5 do 10 p.p. uwypukla istotne spadki u modeli transformatorowych (RT-DETR) podczas użycia syntetycznych teł oraz ogromne wzrosty po nałożeniu pseudo-etykiet.

```latex
\begin{figure}[htbp]
    \centering
    \includegraphics[width=0.85\textwidth]{Pics/wykres_3_architektury.png}
    \caption{Podatność badanych architektur sieci neuronowych na metody generatywne oraz Pseudo-etykietowanie. Wartości nad i pod słupkami opisują bezwzględną zmianę mAP@50 wyrażoną w punktach procentowych względem modelu bazowego (Baseline).}
    \label{fig:architecture_delta}
\end{figure}
```

---

### 4. Krzywa konwergencji (Learning Curve)

**Pliki:** `wykres_4_konwergencja_YOLOv8.png`, `wykres_4_konwergencja_YOLO26.png`, `wykres_4_konwergencja_RT-DETR-L.png`
**Skrypt:** `plot_4_convergence.py`

**Opis w LaTeX:**
Wykresy prezentujące tempo uczenia modeli (wzrost `mAP@50`) na przestrzeni minionych epok, razem z zastosowanym wygładzaniem (średnia ruchoma). Doskonałe do wykazania m.in. jak wariant 'SSL Pseudo' przyspiesza konwergencję badanych architektur.

```latex
\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Pics/wykres_4_konwergencja_YOLOv8.png}
        \caption{Zbieżność mAP@50 dla YOLOv8.}
        \label{fig:conv_yolov8}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Pics/wykres_4_konwergencja_YOLO26.png}
        \caption{Zbieżność mAP@50 dla YOLO11 (YOLO26).}
        \label{fig:conv_yolo26}
    \end{subfigure}
    
    \vspace{0.5cm}
    \begin{subfigure}[b]{0.48\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Pics/wykres_4_konwergencja_RT-DETR-L.png}
        \caption{Zbieżność mAP@50 dla RT-DETR-L.}
        \label{fig:conv_rtdetr}
    \end{subfigure}
    \caption{Krzywe zbieżności uczenia oceniane na zbiorze walidacyjnym za pomocą metryki mAP@50 w rozbiciu na architektury. Półprzezroczyste sygnatury reprezentują surowe odczyty, wyraźna linia nanosi 5-epokowe wygładzenie okna ruchomego.}
    \label{fig:map_convergence_all}
\end{figure}
```

---

### 5. Wielowymiarowy Profil Wydajności - Wykres Radarowy

**Pliki:** `wykres_5_radar_YOLOv8.png`, `wykres_5_radar_YOLO26.png`, `wykres_5_radar_RT-DETR-L.png`
**Skrypt:** `plot_5_radar.py`

**Opis w LaTeX:**
Trzyczęściowy Spider/Radar Plot łączący po 5 najważniejszych wskaźników sieci. Kolor niebieski obrazuje profil bazowy (Baseline), kolor zielony prezentuje ogromny obszar ekspansji ukształtowany po dodaniu SSL Pseudo, zaś kolor pomarańczowy uwydatnia mankamenty podejścia generatywnego (Generative Backgrounds).

```latex
\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Pics/wykres_5_radar_YOLOv8.png}
        \caption{YOLOv8}
        \label{fig:radar_yolov8}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Pics/wykres_5_radar_YOLO26.png}
        \caption{YOLO11 (YOLO26)}
        \label{fig:radar_yolo26}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{Pics/wykres_5_radar_RT-DETR-L.png}
        \caption{RT-DETR-L}
        \label{fig:radar_rtdetr}
    \end{subfigure}
    \caption{Wielowymiarowe wykresy radarowe uwypuklające zmianę pięciu czołowych parametrów jakości po dodaniu do środowiska metod SSL. Widoczne jest wyraźne powiększenie otoczki skuteczności (kolor zielony) dla procedur wspomaganych Pseudo-etykietami w stosunku do pierwotnych wariantów badawczych (kolor niebieski) oraz teł generatywnych (kolor pomarańczowy).}
    \label{fig:radar_chart_metrics_all}
\end{figure}
```
```
