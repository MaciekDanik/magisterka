# Semi-Supervised Learning for Object Detection (Magisterka)

This repository contains the code, data structures, and models for my master's thesis. The project explores Semi-Supervised Learning (SSL) pipelines using YOLO and RT-DETR models, incorporating pseudo-labeling, image tiling, and generated negative sampling techniques.

## Project Layout

```text
├── configs/           # YAML configs for SSL and Generative datasets
├── datasets/          # Various training and testing datasets
│   ├── generative_negatives/ # Generated negative samples
│   ├── manual/               # Baseline manually labeled data
│   ├── pseudo/               # Pseudo-labeled student datasets
│   ├── test_sacred/          # Held-out testing data
│   └── unlabeled_raw/        # Raw images used for pseudo-labeling
├── models/            # Stored PyTorch weights (.pt) and training results
│   ├── pretrained/           # Foundation models and teacher.pt
│   ├── trained/              # Output student models
│   └── training_runs/        # CSV results, logs, artifacts
├── src/               # Python source code
│   ├── data_prep/     # Scripts for tiling and generating pseudo-labels
│   ├── inference/     # Model inference scripts
│   ├── training/      # ML Orchestrator and config algorithms
│   └── utils/         # Helper functions
└── README.md
```

## Setup & Requirements

1. **Clone the repository.**
2. **Setup the Python Environment.** It is highly recommended to use a virtual environment:
   ```bash
   python -m venv venv
   
   # Activate on Windows:
   venv\Scripts\activate
   ```
3. **Install Dependencies.** (Make sure you install the correct PyTorch version for your CUDA hardware along with `ultralytics`):
   ```bash
   pip install torch torchvision
   pip install ultralytics
   ```

## Configuration

Training hyperparameters (epochs, batch size, augmentations, device settings) are globally managed inside `src/training/config.py`.
* **Note:** To train on a GPU, update the `'device'` parameter in `config.py` from `'cpu'` to `0`. 

The data paths mapping class names to specific datasets (baseline, student, and generative datasets) are defined in the YAML files located in `configs/`.

## Running the Training Pipeline

The training process (Baseline $\rightarrow$ SSL Pseudo-Labeling $\rightarrow$ Generative SSL for YOLO and RT-DETR) is automated through a master orchestrator.

To run the orchestration pipeline:

```bash
# Run the master orchestrator from the project root
python src/training/orchestrator.py
```

### Breakdown of the Orchestrator Pipeline:
1. **Baseline Training:** Standard supervised learning using base manual data.
2. **SSL Pseudo Labels:** Iterative pseudo-label generating and training for Student models based off Teacher inferences.
3. **SSL Generative Negative Samples:** Fine-tuning on synthetically generated negative subsets to reduce False Positives.

Results and weights for each pipeline step are dynamically saved in categorized directories inside the `models/training_runs/` directory for safe keeping.
