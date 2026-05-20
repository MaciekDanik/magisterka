import torch
from pathlib import Path

# Base directory setup (resolves to the project root: magisterka)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Check for GPU
DEFAULT_DEVICE = 0 if torch.cuda.is_available() else 'cpu'

BASE_TRAINING_CONFIG = {
    'task': 'detect',
    'epochs': 500, # Default to 100 for actual training
    'patience': 50,
    'imgsz': 640,
    'batch': 16,
    'save': True,
    'save_period': 20,
    'device': DEFAULT_DEVICE,
    'seed': 42, # Set seed for reproducibility
    
    # Standard SSL/Generative heavy augmentations
    'mixup': 0.2,       
    'mosaic': 1.0,      
    'hsv_h': 0.05,      
    'hsv_s': 0.7,       
    'hsv_v': 0.4,       
    'degrees': 25.0,    
    'translate': 0.2,   
    'scale': 0.6,
}

def get_config(overrides=None):
    """Returns a copy of the base config with optional overrides."""
    config = BASE_TRAINING_CONFIG.copy()
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                config[k] = v
    return config

def get_absolute_path(relative_path_str):
    """
    Resolves relative path concepts from old scripts to the absolute PROJECT_ROOT.
    Example: '../../configs/data.yaml' -> 'S:/.../magisterka/configs/data.yaml'
    """
    clean_path = relative_path_str.lstrip('./').lstrip('../')
    # If the path happens to point inside src via old ways, adjust logic if needed,
    # but configs/, datasets/, models/ are at the root level.
    return str(PROJECT_ROOT / clean_path)
