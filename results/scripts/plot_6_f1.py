import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
out_dir = os.path.join("results", "plots")
os.makedirs(out_dir, exist_ok=True)

def get_f1_curve(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    epochs = df['epoch'].values
    
    if 'metrics/precision(B)' in df.columns and 'metrics/recall(B)' in df.columns:
        p = df['metrics/precision(B)'].values
        r = df['metrics/recall(B)'].values
        # Obliczenie F1 score: 2 * (P * R) / (P + R)
        # Zabezpieczenie przed dzieleniem przez zero
        f1 = np.divide(2 * (p * r), p + r, out=np.zeros_like(p), where=(p+r)!=0)
    else:
        f1 = np.zeros(len(epochs))
        
    return epochs, f1

def find_latest_run(base_dir):
    search_path = os.path.join(base_dir, "*", "results.csv")
    csv_files = glob.glob(search_path)
    if not csv_files:
        return None
    # Celowo wybieramy najdłuższy pełny przebieg, pomijając krótkie uruchomienia kontrolne.
    return max(csv_files, key=os.path.getsize)

def main():
    base_path = os.path.join("models", "training_runs")
    
    architectures = {
        'YOLOv8': 'yolov8',
        'YOLO26': 'yolo26',
        'RT-DETR-L': 'rtdetr'
    }
    
    for arch_display, arch_prefix in architectures.items():
        path_baseline = find_latest_run(os.path.join(base_path, f"{arch_prefix}_Baseline"))
        path_generative = find_latest_run(os.path.join(base_path, f"{arch_prefix}_SSL_Generative"))
        path_pseudo = find_latest_run(os.path.join(base_path, f"{arch_prefix}_SSL_Pseudo"))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        if path_baseline and os.path.exists(path_baseline) and path_generative and os.path.exists(path_generative) and path_pseudo and os.path.exists(path_pseudo):
            epochs_b, f1_b = get_f1_curve(path_baseline)
            epochs_g, f1_g = get_f1_curve(path_generative)
            epochs_p, f1_p = get_f1_curve(path_pseudo)
            
            max_epoch_b = epochs_b[np.argmax(f1_b)]
            max_epoch_p = epochs_p[np.argmax(f1_p)]
            max_epoch_g = epochs_g[np.argmax(f1_g)]
            
            sns.lineplot(x=epochs_b, y=f1_b, label="Model odniesienia", linewidth=2.5, color='darkblue', ax=ax)
            sns.lineplot(x=epochs_p, y=f1_p, label="Metoda A (pseudoetykiety)", linewidth=2.5, color='#2ca02c', ax=ax)
            sns.lineplot(x=epochs_g, y=f1_g, label="Metoda B (syntetyczne tła)", linewidth=2.5, color='darkorange', ax=ax)
            
            ax.axvline(x=max_epoch_b, color='darkblue', linestyle='--', alpha=0.7, label=f"Max. model odniesienia (epoka {int(max_epoch_b)})")
            ax.axvline(x=max_epoch_p, color='#2ca02c', linestyle='--', alpha=0.7, label=f"Max. Metoda A (Epoka {int(max_epoch_p)})")
            ax.axvline(x=max_epoch_g, color='darkorange', linestyle='--', alpha=0.7, label=f"Max. Metoda B (Epoka {int(max_epoch_g)})")
        else:
            print(f"Brak pełnych danych dla {arch_display}. Trwa pomijanie...")
            continue
            
        ax.set_xlabel("Epoka treningu", fontsize=12)
        ax.set_ylabel("F1 Score", fontsize=12)
        ax.set_title(f"Rozwój wyniku F1 na przestrzeni epok - {arch_display}", fontsize=14)
        ax.legend()
        
        plt.tight_layout()
        save_path = os.path.join(out_dir, f"wykres_6_f1_{arch_display}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Zapisano wykres F1: {save_path}")

if __name__ == "__main__":
    main()
