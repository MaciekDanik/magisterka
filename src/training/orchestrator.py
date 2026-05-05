import argparse
import traceback
from training_methods import run_baseline, run_ssl_pseudo_labels, run_ssl_generative

def main():
    parser = argparse.ArgumentParser(description="Master Training Orchestrator: YOLO & RT-DETR")
    parser.add_argument('--epochs', type=int, default=None, help='Override default epochs')
    parser.add_argument('--device', type=str, default=None, help='Override compute device (e.g., 0, cpu)')
    parser.add_argument('--batch-size', type=int, default=None, help='Override batch size')
    parser.add_argument('--iterations', type=int, default=2, help='Number of pseudo-label SSL iterations')
    
    args = parser.parse_args()
    
    # Bundle CLI overrides
    overrides = {}
    if args.epochs is not None: overrides['epochs'] = args.epochs
    if args.device is not None: overrides['device'] = args.device
    if args.batch_size is not None: overrides['batch'] = args.batch_size

    print("=========================================================")
    print("   MASTER TRAINING ORCHESTRATOR: YOLO & RT-DETR")
    print(f"   OVERRIDES: {overrides}")
    print("=========================================================\n")
    
    # ====== YOLO PIPELINE ======
    try:
        print("[PIPELINE] 1. YOLO Baseline")
        yolo_base_weights = run_baseline(model_type='yolo', data_yaml='configs/baseline_data.yaml', overrides=overrides)
    except Exception as e:
        print("\n[ERROR] Failed during YOLO Baseline:")
        traceback.print_exc()
        
    try:
        print("[PIPELINE] 2. YOLO SSL Pseudo Labels")
        yolo_pseudo_weights = run_ssl_pseudo_labels(model_type='yolo', iterations=args.iterations, data_yaml='configs/student_data.yaml', overrides=overrides)
    except Exception as e:
        print("\n[ERROR] Failed during YOLO SSL Pseudo Labels:")
        traceback.print_exc()
        
    try:
        print("[PIPELINE] 3. YOLO SSL Generative Negative Samples")
        yolo_gen_weights = run_ssl_generative(model_type='yolo', data_yaml='configs/generative_data.yaml', overrides=overrides)
    except Exception as e:
        print("\n[ERROR] Failed during YOLO SSL Generative Negative:")
        traceback.print_exc()

    # ====== RT-DETR PIPELINE ======
    try:
        print("[PIPELINE] 4. RT-DETR Baseline")
        rtdetr_base_weights = run_baseline(model_type='rtdetr', data_yaml='configs/baseline_data.yaml', overrides=overrides)
    except Exception as e:
        print("\n[ERROR] Failed during RT-DETR Baseline:")
        traceback.print_exc()
        
    try:
        print("[PIPELINE] 5. RT-DETR SSL Pseudo Labels")
        rtdetr_pseudo_weights = run_ssl_pseudo_labels(model_type='rtdetr', iterations=args.iterations, data_yaml='configs/student_data.yaml', overrides=overrides)
    except Exception as e:
        print("\n[ERROR] Failed during RT-DETR SSL Pseudo Labels:")
        traceback.print_exc()
        
    try:
        print("[PIPELINE] 6. RT-DETR SSL Generative Negative Samples")
        rtdetr_gen_weights = run_ssl_generative(model_type='rtdetr', data_yaml='configs/generative_data.yaml', overrides=overrides)
    except Exception as e:
        print("\n[ERROR] Failed during RT-DETR SSL Generative Negative:")
        traceback.print_exc()

    print("\n=========================================================")
    print("   ALL TRAINING PIPELINES COMPLETED OR HANDLED ERRORS!")
    print("=========================================================")

if __name__ == '__main__':
    main()
