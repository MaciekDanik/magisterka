from ultralytics import YOLO, RTDETR
import copy
from config import get_config, get_absolute_path

def get_model(model_type, checkpoint_path):
    """Instantiates the correct model class based on model_type ('yolo' or 'rtdetr')."""
    if model_type.lower() == 'yolo':
        return YOLO(checkpoint_path)
    elif model_type.lower() == 'rtdetr':
        return RTDETR(checkpoint_path)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'yolo' or 'rtdetr'.")

def get_default_weights(model_type):
    if model_type.lower() == 'yolo':
        return get_absolute_path('models/pretrained/yolov8n-seg.pt')
    return get_absolute_path('models/pretrained/rtdetr-l.pt')

def run_baseline(model_type='yolo', initial_weights=None, data_yaml='configs/baseline_data.yaml', overrides=None):
    """
    Runs the baseline model training (no pseudo labels, no generated negatives).
    """
    if initial_weights is None:
        initial_weights = get_default_weights(model_type)
    
    print(f"--- Starting Baseline Training ({model_type.upper()}) ---")
    model = get_model(model_type, initial_weights)
    
    config = get_config(overrides)
    config['data'] = get_absolute_path(data_yaml)
    config['project'] = get_absolute_path(f'models/training_runs/{model_type.lower()}_Baseline')
    config['name'] = f'{model_type}_baseline'
    
    # Adjust augmentations for base usually (less aggressive)
    if model_type.lower() == 'rtdetr':
        config['mixup'] = 0.0
        config['degrees'] = 15.0
        config['hsv_h'] = 0.015
    
    results = model.train(**config)
    print(f"--- Baseline ({model_type.upper()}) Completed ---\n")
    # Return the path to the best weights
    best_weights_path = str(model.trainer.save_dir / 'weights' / 'best.pt') if hasattr(model, 'trainer') else None
    return best_weights_path

def run_ssl_pseudo_labels(model_type='yolo', iterations=2, initial_weights=None, data_yaml='configs/student_data.yaml', overrides=None):
    """
    Runs iterative SSL Pseudo Labels (Student v1, Student v2, etc.).
    """
    print(f"--- Starting SSL Pseudo Labels Training ({model_type.upper()}) over {iterations} iterations ---")
    
    # Track the current weights across iterations
    current_weights = initial_weights if initial_weights is not None else get_default_weights(model_type)
    
    for i in range(1, iterations + 1):
        print(f"  -> Iteration {i}/{iterations} ({model_type.upper()})")
        # Initialize model with the weights from the PREVIOUS iteration
        model = get_model(model_type, current_weights)
        
        config = get_config(overrides)
        config['data'] = get_absolute_path(data_yaml)
        config['project'] = get_absolute_path(f'models/training_runs/{model_type.lower()}_SSL_Pseudo')
        config['name'] = f'{model_type}_student_iter_{i}'
        
        # Optionally adjust generation args based on model_type to prevent inconsistencies
        if model_type.lower() == 'rtdetr':
            config['mixup'] = 0.0
            config['degrees'] = 15.0
            
        model.train(**config)
        
        # Grab best weights from this run to use in next iteration
        current_weights = str(model.trainer.save_dir / 'weights' / 'best.pt')
        print(f"  -> Obtained iteration {i} weights: {current_weights}")
        
    print(f"--- SSL Pseudo Labels ({model_type.upper()}) Completed ---\n")
    return current_weights

def run_ssl_generative(model_type='yolo', initial_weights=None, data_yaml='configs/generative_data.yaml', overrides=None):
    """
    Runs SSL with Generative Negative Samples.
    """
    print(f"--- Starting SSL Generative Training ({model_type.upper()}) ---")
    if initial_weights is None:
        initial_weights = get_default_weights(model_type)
        
    model = get_model(model_type, initial_weights)
    
    config = get_config(overrides)
    config['data'] = get_absolute_path(data_yaml)
    config['project'] = get_absolute_path(f'models/training_runs/{model_type.lower()}_SSL_Generative')
    config['name'] = f'{model_type}_generative'
    
    model.train(**config)
    
    best_weights = str(model.trainer.save_dir / 'weights' / 'best.pt') if hasattr(model, 'trainer') else None
    print(f"--- SSL Generative ({model_type.upper()}) Completed ---\n")
    return best_weights
