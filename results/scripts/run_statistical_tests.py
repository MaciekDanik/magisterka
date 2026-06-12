import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# Ustawienia ścieżek
GT_DIR = "datasets/manual/valid/labels"
PREDS_BASE_DIR = "results/validation"
OUT_REPORT = "results/statistical_report.md"
OUT_CSV = "results/statistical_results.csv"
IOU_THRESHOLD = 0.5

def xywh2xyxy(x, y, w, h):
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return x1, y1, x2, y2

def calculate_iou(box1, box2):
    """
    Oblicza Intersection over Union (IoU) dla dwóch b-boxów.
    Format boxów: [x1, y1, x2, y2]
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0

    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area

def read_boxes(filepath):
    """Odczytuje boxy z pliku txt YOLO do formatu xyxy."""
    boxes = []
    if not os.path.exists(filepath):
        return boxes
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:  # class, x, y, w, h, [conf]
                _, x, y, w, h = map(float, parts[:5])
                boxes.append(xywh2xyxy(x, y, w, h))
    return boxes

def evaluate_image(gt_boxes, pred_boxes, iou_thresh=0.5):
    """Zwraca (TP, FP, FN) dla jednego obrazu."""
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 0, 0, 0
    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)

    matched_gt = set()
    matched_pred = set()
    
    tp = 0
    # Obliczamy macierz IoU
    for p_idx, p_box in enumerate(pred_boxes):
        best_iou = 0
        best_gt_idx = -1
        for gt_idx, gt_box in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            iou = calculate_iou(p_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx
                
        if best_iou >= iou_thresh:
            tp += 1
            matched_gt.add(best_gt_idx)
            matched_pred.add(p_idx)
            
    fp = len(pred_boxes) - len(matched_pred)
    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn

def calculate_metrics(_tp, _fp, _fn):
    precision = _tp / (_tp + _fp) if (_tp + _fp) > 0 else 0
    recall = _tp / (_tp + _fn) if (_tp + _fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

def run_pipeline():
    # Znajdź wszystkie pliki z etykietami dla testowych/validacyjnych obrazów
    gt_files = glob.glob(os.path.join(GT_DIR, "*.txt"))
    image_names = [os.path.basename(f) for f in gt_files]
    
    models = ['yolov8', 'yolo26', 'rtdetr']
    methods = ['baseline', 'ssl_pseudo', 'ssl_generative']
    
    # Przechowywanie wyników F1-score: results_dict[model][method] = list_of_f1_scores
    results_dict = {m: {meth: [] for meth in methods} for m in models}
    
    print("Przetwarzanie obrazków i wyliczanie metryk (F1-score) per obraz...")
    
    for img_name in image_names:
        gt_path = os.path.join(GT_DIR, img_name)
        gt_boxes = read_boxes(gt_path)
        
        for model in models:
            for method in methods:
                # Domyślny folder to np. validation/yolov8/baseline/labels
                pred_path = os.path.join(PREDS_BASE_DIR, model, method, "labels", img_name)
                pred_boxes = read_boxes(pred_path)
                
                tp, fp, fn = evaluate_image(gt_boxes, pred_boxes, IOU_THRESHOLD)
                _, _, f1 = calculate_metrics(tp, fp, fn)
                
                results_dict[model][method].append(f1)

    print("Obliczenia zakończone. Generowanie raportu statystycznego...")
    
    # Przeprowadzanie testów Wilcoxona (Baseline vs Metoda)
    report_lines = []
    report_lines.append("# Raport Statystyczny z Oceny Detekcji (Test Wilcoxona)")
    
    report_lines.append("## Wstęp i metodologia (Background)")
    report_lines.append("Celem niniejszej ewaluacji statystycznej jest rzetelne określenie, czy alternatywne "
                        "metody uczenia (wykorzystanie pseudoznakowania oraz generatywnych masek tła) "
                        "wprowadzają istotną statystycznie poprawę w zdolności modelu do poprawnej detekcji.")
    report_lines.append("Podstawowym problemem w standardowej analizie agregowanej (np. globalnym mAP) jest to, "
                        "że eksperymenty pojedyncze (single-seed runs) oferują tylko jeden zagregowany wynik. "
                        "Aby zyskać odpowiednią moc statystyczną, wyliczono metrykę precyzji i czułości "
                        "(F1-Score na podstawie IoU > 0.5) osobno dla **każdego wycinka ewaluacyjnego (każdego obrazu)**. "
                        "Dzięki temu dla pojedynczego modelu uzyskujemy wektor $N$ obserwacji, gdzie $N$ to "
                        "liczba kadrów testowych.")
    report_lines.append("Ze względu na niespełnienie założeń o rozkładzie normalnym wyników oraz na fakt, "
                        "iż porównujemy odpowiedzi różnych modeli dla tego samego zbioru ilustracji (próby połączone), "
                        "zastosowano nieparametryczny **test znakowanych rang Wilcoxona**.")
    report_lines.append("\n* Wartość **p-value < 0.05** przyjmuje się jako próg istotności statystycznej "
                        "odrzucający hipotezę zerową (H0 mówiącą o braku różnic między modelem Baseline a wariantami SSL).")

    report_lines.append("\n## Wyniki operacyjne")
    report_lines.append("Niniejszy raport zawiera wyniki z testowania różnic jakości predykcji poszczególnych "
                        "modeli. Pary powiązane wynikają z faktu wykorzystania tych samych obrazów testowych.\n")
    
    report_lines.append(f"**Liczba obrazów walidacyjnych (próba N):** {len(image_names)}")
    report_lines.append(f"**Zastosowana metryka (zmienna zależna):** F1-Score (IoU > {IOU_THRESHOLD})\n")
    
    report_lines.append("## Tabela wyników\n")
    report_lines.append("| Architektura | Porównanie | Śr. F1 (Baseline) | Śr. F1 (Metoda) | p-value | Istotność stat. (p<0.05) |")
    report_lines.append("|--------------|------------|-------------------|-----------------|---------|--------------------------|")
    
    csv_data = []

    for model in models:
        base_f1 = np.array(results_dict[model]['baseline'])
        mean_base = base_f1.mean()
        
        for variant in ['ssl_pseudo', 'ssl_generative']:
            var_f1 = np.array(results_dict[model][variant])
            mean_var = var_f1.mean()
            
            # Wymagane są różnice między pomiarami. Jeśli są absolutnie równe na każdym zdjęciu,
            # Wilcoxon może rzucić wyjątkiem. Zabezpieczamy się na to.
            diffs = var_f1 - base_f1
            if np.all(diffs == 0):
                p_value = 1.0  # W pełni identyczne dystrybucje
            else:
                stat, p_value = wilcoxon(base_f1, var_f1)
                
            is_sig = "+" if p_value < 0.05 else "-"
            
            # Dodaj do markdown
            report_lines.append(f"| {model.upper()} | Baseline vs {variant} | {mean_base:.4f} | {mean_var:.4f} | {p_value:.4f} | {is_sig} |")
            
            # Dodaj do CSV
            csv_data.append({
                "Model": model.upper(),
                "Comparison": f"Baseline vs {variant}",
                "Mean_F1_Baseline": round(mean_base, 4),
                "Mean_F1_Variant": round(mean_var, 4),
                "p_value": round(p_value, 4),
                "Significant_0.05": bool(p_value < 0.05)
            })

    # Zapis Markdown
    with open(OUT_REPORT, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines) + "\n")
        
    # Zapis CSV dla surowych danych statystycznych
    pd.DataFrame(csv_data).to_csv(OUT_CSV, index=False)

    print(f"Raport statystyczny zapisano w statystycznym ujęciu markdown w locji: {OUT_REPORT}")
    print(f"Dane tabelaryczne zapisano do: {OUT_CSV}")

if __name__ == "__main__":
    run_pipeline()