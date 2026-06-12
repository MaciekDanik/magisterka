import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

# Ustawienia ścieżek
GT_DIR = "datasets/manual/valid/labels"
PREDS_BASE_DIR = "results/validation"
OUT_REPORT = "results/statistical_report_FP.md"
OUT_CSV = "results/statistical_results_FP.csv"
IOU_THRESHOLD = 0.5

def xywh2xyxy(x, y, w, h):
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return x1, y1, x2, y2

def calculate_iou(box1, box2):
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
    boxes = []
    if not os.path.exists(filepath):
        return boxes
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                _, x, y, w, h = map(float, parts[:5])
                boxes.append(xywh2xyxy(x, y, w, h))
    return boxes

def evaluate_image(gt_boxes, pred_boxes, iou_thresh=0.5):
    """Zwraca (TP, FP, FN) dla jednego obrazu."""
    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)

    matched_gt = set()
    matched_pred = set()
    
    tp = 0
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

def run_pipeline():
    gt_files = glob.glob(os.path.join(GT_DIR, "*.txt"))
    image_names = [os.path.basename(f) for f in gt_files]
    
    models = ['yolov8', 'yolo26', 'rtdetr']
    methods = ['baseline', 'ssl_pseudo', 'ssl_generative']
    
    # Przechowujemy liczbę FP (False Positives) per obraz
    results_dict = {m: {meth: [] for meth in methods} for m in models}
    
    print("Przetwarzanie danych: Analiza liczby fałszywych detekcji (FP)...")
    
    for img_name in image_names:
        gt_path = os.path.join(GT_DIR, img_name)
        gt_boxes = read_boxes(gt_path)
        
        for model in models:
            for method in methods:
                pred_path = os.path.join(PREDS_BASE_DIR, model, method, "labels", img_name)
                pred_boxes = read_boxes(pred_path)
                
                _, fp, _ = evaluate_image(gt_boxes, pred_boxes, IOU_THRESHOLD)
                results_dict[model][method].append(fp)

    print("Generowanie raportu statystycznego...")
    
    report_lines = [
        "# Raport Statystyczny: Redukcja Fałszywych Detekcji",
        "## Wstęp i metodologia",
        "Celem niniejszej analizy jest weryfikacja, czy zastosowane metody poprawy jakości widzenia "
        "(SSL Pseudo oraz SSL Generative) prowadzą do istotnej statystycznie redukcji 'halucynacji' "
        "modelu (fałszywych detekcji - FP).",
        "Jako zmienną zależną przyjęto liczbę fałszywych detekcji (FP) na pojedynczym obrazie testowym. "
        "Ze względu na nieparametryczny charakter rozkładu danych oraz próbę powiązaną (ten sam zbiór "
        "testowy dla modeli Baseline i wariantów), zastosowano test znakowanych rang Wilcoxona (test jednostronny: alternative='less').",
        f"\n**Liczba obrazów walidacyjnych (próba N):** {len(image_names)}",
        "## Tabela wyników: Redukcja fałszywych detekcji (FP)\n",
        "| Architektura | Porównanie | Śr. FP (Baseline) | Śr. FP (Metoda) | p-value | Istotność (p<0.05 i redukcja) |",
        "|--------------|------------|-------------------|-----------------|---------|-------------------------------|"
    ]
    
    csv_data = []

    for model in models:
        base_fp = np.array(results_dict[model]['baseline'])
        mean_base = base_fp.mean()
        
        for variant in ['ssl_pseudo', 'ssl_generative']:
            var_fp = np.array(results_dict[model][variant])
            mean_var = var_fp.mean()
            
            # Test jednostronny: czy metoda ma MNIEJ fałszywych detekcji niż baseline?
            stat, p_value = wilcoxon(var_fp, base_fp, alternative='less')
                
            # Istotne, gdy p < 0.05 ORAZ średnia FP metody jest niższa od baseline
            is_sig = "TAK" if (p_value < 0.05 and mean_var < mean_base) else "NIE"
            
            report_lines.append(f"| {model.upper()} | Baseline vs {variant} | {mean_base:.2f} | {mean_var:.2f} | {p_value:.4f} | {is_sig} |")
            
            csv_data.append({
                "Model": model.upper(),
                "Comparison": f"Baseline vs {variant}",
                "Mean_FP_Baseline": round(mean_base, 4),
                "Mean_FP_Variant": round(mean_var, 4),
                "p_value": round(p_value, 4),
                "Significant_Reduction": is_sig
            })

    with open(OUT_REPORT, 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines) + "\n")
        
    pd.DataFrame(csv_data).to_csv(OUT_CSV, index=False)
    print(f"Raport zapisano do: {OUT_REPORT}")

if __name__ == "__main__":
    run_pipeline()