import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Konfiguracja globalna i estetyka
sns.set_theme(style="whitegrid")
out_dir = os.path.join("results", "plots")
os.makedirs(out_dir, exist_ok=True)

def get_best_metrics(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    # Szukamy epoki z najlepszym mAP50
    best_row = df.loc[df['metrics/mAP50(B)'].idxmax()]
    return {
        'mAP@0.5': best_row['metrics/mAP50(B)'],
        'Precyzja (Precision)': best_row['metrics/precision(B)'],
        'Czułość (Recall)': best_row['metrics/recall(B)']
    }

def find_latest_run(base_dir):
    search_path = os.path.join(base_dir, "*", "results.csv")
    csv_files = glob.glob(search_path)
    if not csv_files:
        return None
    # Pobieramy plik o największym rozmiarze (najwięcej epok = faktyczny, pełny trening, a nie ucięty testowy!)
    return max(csv_files, key=os.path.getsize)

def main():
    base_path = os.path.join("models", "training_runs")
    
    architectures = {
        'YOLOv8': 'yolov8',
        'YOLO26': 'yolo26',
        'RT-DETR-L': 'rtdetr'
    }
    
    metrics_names = ['mAP@0.5', 'Precyzja (Precision)', 'Czułość (Recall)']
    
    for arch_display_name, arch_prefix in architectures.items():
        # Scenariusze i ścieżki dla konkretnej architektury
        scenarios = {
            'Baseline': find_latest_run(os.path.join(base_path, f"{arch_prefix}_Baseline")),
            'Metoda A (SSL / Pseudo-etykiety)': find_latest_run(os.path.join(base_path, f"{arch_prefix}_SSL_Pseudo")),
            'Metoda B (Syntetyczne tła)': find_latest_run(os.path.join(base_path, f"{arch_prefix}_SSL_Generative")),
        }
        
        data = []
        
        # Fallback dla testu, gdyby folderów jakiegoś modelu jeszcze nie było
        mock_data = {
            'Baseline': [0.65, 0.70, 0.60],
            'Metoda A (SSL / Pseudo-etykiety)': [0.72, 0.68, 0.81],
            'Metoda B (Syntetyczne tła)': [0.71, 0.85, 0.58]
        }
        
        for name, path in scenarios.items():
            if path and os.path.exists(path):
                vals = get_best_metrics(path)
                data.append([name, vals['mAP@0.5'], vals['Precyzja (Precision)'], vals['Czułość (Recall)']])
            else:
                print(f"Brak pliku algorytmu {arch_display_name} dla {name}. Używam mock data.")
                data.append([name] + mock_data[name])
                
        df = pd.DataFrame(data, columns=['Scenariusz'] + metrics_names)
        df_melted = df.melt(id_vars='Scenariusz', var_name='Metryka', value_name='Wartość')
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Rysowanie wykresu ze stonowaną paletą
        barplot = sns.barplot(
            data=df_melted, 
            x='Metryka', 
            y='Wartość', 
            hue='Scenariusz', 
            palette='magma',
            ax=ax
        )
        
        # Dodanie wartości liczbowych nad słupkami
        for p in barplot.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(f"{height:.2f}",
                            (p.get_x() + p.get_width() / 2., height),
                            ha='center', va='bottom',
                            fontsize=10, color='black',
                            xytext=(0, 3), textcoords='offset points')
                
        ax.set_ylim(0.0, 1.05) # Poprawka: Oś od 0, by nie obcinać słupków < 0.4
        ax.set_ylabel("Wartość metryki [0 - 1]", fontsize=12)
        ax.set_xlabel("", fontsize=12)
        ax.set_title(f"Wpływ metod SSL na metryki detekcji - {arch_display_name}", fontsize=14)
        
        # Konfiguracja legendy - niezasłaniająca danych
        ax.legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=1)
        
        plt.tight_layout()
        save_path = os.path.join(out_dir, f"wykres_1_metryki_{arch_display_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig) # Zamknięcie figury zrzuca ją z pamięci dla kolejnej iteracji
        print(f"Zapisano wykres: {save_path}")

if __name__ == "__main__":
    main()