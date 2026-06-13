import pandas as pd
from pathlib import Path

def generate_comparison_table(base_dir="models/training_runs"):
    base_path = Path(base_dir)
    
    architectures = {
        'YOLOv8': 'yolov8',
        'YOLO26': 'yolo26',
        'RT-DETR-L': 'rtdetr'
    }
    categories = {
        'Model odniesienia': 'Baseline',
        'Metoda A (SSL / Pseudo)': 'SSL_Pseudo',
        'Metoda B (Syntetyczne tła)': 'SSL_Generative'
    }
    
    results_files = []
    for arch_display, arch_prefix in architectures.items():
        for cat_display, cat_suffix in categories.items():
            run_dir = base_path / f"{arch_prefix}_{cat_suffix}"
            csv_files = list(run_dir.glob("*/results.csv"))
            if csv_files:
                # Celowo wybieramy najdłuższy pełny przebieg, pomijając krótkie uruchomienia kontrolne.
                best_file = max(csv_files, key=lambda p: p.stat().st_size)
                results_files.append((best_file, arch_display, cat_display))
    
    data = []
    
    for file_path, arch_display, cat_display in results_files:
        try:
            df = pd.read_csv(file_path)
            # Normalize column names by stripping whitespace
            df.columns = df.columns.str.strip()
            
            # Check for required columns
            p_col = 'metrics/precision(B)'
            r_col = 'metrics/recall(B)'
            map50_col = 'metrics/mAP50(B)'
            map95_col = 'metrics/mAP50-95(B)'
            
            if not all(col in df.columns for col in [p_col, r_col, map50_col, map95_col]):
                print(f"Skipping {file_path} - missing metric columns.")
                continue
            
            # Calculate F1 score
            df['F1_score'] = 2 * (df[p_col] * df[r_col]) / (df[p_col] + df[r_col] + 1e-16)
            
            # Calculate total validation loss to find the epoch with minimum loss
            # For YOLO: val/box_loss, val/cls_loss, val/dfl_loss
            # For RT-DETR: val/giou_loss, val/cls_loss, val/l1_loss
            val_loss_cols = [col for col in df.columns if col.startswith('val/') and 'loss' in col]
            if val_loss_cols:
                df['total_val_loss'] = df[val_loss_cols].sum(axis=1)
                # Ignore validation losses of 0.0 which are usually from skipped validation runs
                valid_losses = df[df['total_val_loss'] > 0]['total_val_loss']
                min_loss_epoch_idx = valid_losses.idxmin() if not valid_losses.empty else df['total_val_loss'].idxmin()
            else:
                df['total_val_loss'] = 0
                min_loss_epoch_idx = 0
                
            min_loss_epoch = int(df.iloc[min_loss_epoch_idx].get('epoch', min_loss_epoch_idx + 1))
            min_loss_val = df.iloc[min_loss_epoch_idx]['total_val_loss']
            
            # Find the best epoch based on mAP50-95
            best_epoch_idx = df[map95_col].idxmax()
            best_row = df.iloc[best_epoch_idx]
            
            # Extract names for identification
            run_name = file_path.parent.name
            
            data.append({
                'Architecture': arch_display,
                'Category': cat_display,
                'Run Name': run_name,
                'Best Epoch': int(best_row.get('epoch', best_epoch_idx + 1)),
                'Min Loss Epoch': min_loss_epoch,
                'Min Val Loss': min_loss_val,
                'Precision': best_row[p_col],
                'Recall': best_row[r_col],
                'F1 Score': best_row['F1_score'],
                'mAP50': best_row[map50_col],
                'mAP50-95': best_row[map95_col]
            })
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
    if not data:
        print("No valid results found. Ensure the directory path is correct.")
        return
        
    summary_df = pd.DataFrame(data)
    
    # Sort logically
    summary_df = summary_df.sort_values(by=['Architecture', 'Category'])
    
    # Round numerical metrics to 4 decimal places for readability
    metrics_cols = ['Min Val Loss', 'Precision', 'Recall', 'F1 Score', 'mAP50', 'mAP50-95']
    summary_df[metrics_cols] = summary_df[metrics_cols].round(4)
    
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    csv_path = output_dir / "comparison_table.csv"
    
    summary_df.to_csv(csv_path, index=False)
    
    print(f"Data successfully generated for {len(summary_df)} runs!")
    print(f"CSV saved to: {csv_path}")
    print("\nPreview of top 10 models (by mAP50-95):")
    print(summary_df.head(10).to_string(index=False))

if __name__ == "__main__":
    generate_comparison_table()
