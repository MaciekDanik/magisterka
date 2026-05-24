import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

sns.set_theme(style="whitegrid")
out_dir = os.path.join("results", "plots")
os.makedirs(out_dir, exist_ok=True)

def get_best_map(csv_path):
    if not csv_path or not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    return df['metrics/mAP50(B)'].max()

def find_latest_run(base_dir):
    search_path = os.path.join(base_dir, "*", "results.csv")
    csv_files = glob.glob(search_path)
    if not csv_files:
        return None
    # Zabezpieczenie przed przerwamnymi przebiegami - wybieramy ten co ma najwięcej danych
    return max(csv_files, key=os.path.getsize)

def main():
    base_path = os.path.join("models", "training_runs")
    
    architectures = {
        'YOLOv8n': 'yolov8',
        'YOLOv26n': 'yolo26',
        'RT-DETR-L': 'rtdetr'
    }
    
    deltas = []
    
    for arch_display, arch_prefix in architectures.items():
        base_p = find_latest_run(os.path.join(base_path, f"{arch_prefix}_Baseline"))
        ssl_gen_p = find_latest_run(os.path.join(base_path, f"{arch_prefix}_SSL_Generative"))
        ssl_pse_p = find_latest_run(os.path.join(base_path, f"{arch_prefix}_SSL_Pseudo"))
            
        map_base = get_best_map(base_p)
        map_gen = get_best_map(ssl_gen_p)
        map_pse = get_best_map(ssl_pse_p)
        
        # Generative Delta
        if map_base is not None and map_gen is not None:
            deltas.append({'Architektura': arch_display, 'Metoda': 'Syntetyczne tła (Generative)', 'Delta': map_gen - map_base})
        else:
            deltas.append({'Architektura': arch_display, 'Metoda': 'Syntetyczne tła (Generative)', 'Delta': 0})
            
        # Pseudo Delta
        if map_base is not None and map_pse is not None:
            deltas.append({'Architektura': arch_display, 'Metoda': 'Pseudo-etykiety', 'Delta': map_pse - map_base})
        else:
            deltas.append({'Architektura': arch_display, 'Metoda': 'Pseudo-etykiety', 'Delta': 0})
        
    df = pd.DataFrame(deltas)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    barplot = sns.barplot(
        data=df, 
        x='Architektura', 
        y='Delta', 
        hue='Metoda',
        palette='viridis',
        ax=ax
    )
    
    for p in barplot.patches:
        val = p.get_height()
        sign = "+" if val > 0 else ""
        y_offset = 4 if val > 0 else -12
        va_align = 'bottom' if val > 0 else 'top'
        
        if not pd.isna(val) and val != 0:
            ax.annotate(f"{sign}{val * 100:.1f}%",
                        (p.get_x() + p.get_width() / 2., val),
                        ha='center', va=va_align,
                        fontsize=11, fontweight='bold', color='black',
                        xytext=(0, y_offset), textcoords='offset points')

    ax.set_ylim(-0.05, 0.10)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    
    ax.axhline(0, color='black', linewidth=1)
    
    ax.set_ylabel("Zmiana mAP@50 względem modelu bazowego (p.p.)", fontsize=12)
    ax.set_xlabel("Architektura modeli", fontsize=12)
    ax.set_title("Podatność architektur na metody Data Augmentation/SSL", fontsize=14)
    
    plt.tight_layout()
    save_path = os.path.join(out_dir, "wykres_3_architektury.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Zapisano wykres: {save_path}")

if __name__ == "__main__":
    main()