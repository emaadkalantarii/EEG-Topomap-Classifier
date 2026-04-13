# EEG Topomap Classifier

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4-EE4C2C?logo=pytorch&logoColor=white)
![Task](https://img.shields.io/badge/Task-Binary%20Classification-4CAF50)
![License](https://img.shields.io/badge/License-MIT-blue)

A deep learning system that classifies brain responses to visual design stimuli as **good** or **bad** using EEG topomap images. The model is a custom Convolutional Neural Network (CNN) built in PyTorch, trained to distinguish between two neurological response patterns after 6 seconds of design exposure.

---

## Overview

Electroencephalography (EEG) topographic maps (topomaps) visualise the spatial distribution of brain electrical activity across the scalp at a given moment. This project applies computer vision to neuroscience: instead of a human expert interpreting the maps, a CNN learns to recognise the patterns that distinguish a positive brain response to a design from a negative one.

```
Input: EEG topomap image  →  CNN  →  Output: 0 (bad) or 1 (good)
```

---

## Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **90.96%** |
| Best Val Accuracy | 97.18% |
| Best Val Loss | 0.0808 |
| Best checkpoint saved at | Epoch 45 / 60 |
| Data split | 70 / 15 / 15 % (stratified) |
| Training images | 824 |
| Validation images | 177 |
| Test images | 177 |

> Test accuracy is measured on the **held-out test set** (177 images the model never saw during training or validation), making it the honest measure of generalisation.

---

## Model Architecture

The network is a custom CNN with four convolutional blocks followed by a fully connected classification head.

```
Input (3 × 128 × 128)
│
├── Conv Block 1:  Conv2d(3→16)    → ReLU → Conv2d(16→32)  → ReLU → MaxPool2d
├── Conv Block 2:  Conv2d(32→64)   → ReLU → Conv2d(64→128) → ReLU → MaxPool2d
├── Conv Block 3:  Conv2d(128→256) → ReLU → MaxPool2d
├── Conv Block 4:  Conv2d(256→512) → ReLU → MaxPool2d
│
├── AdaptiveAvgPool2d → (512 × 4 × 4)
├── Flatten → 8192
│
├── FC(8192→1024) → ReLU → Dropout(0.5)
├── FC(1024→1024) → ReLU → Dropout(0.5)
└── FC(1024→1)    → Sigmoid
│
Output: probability ∈ [0, 1]  →  threshold 0.5  →  class label {0, 1}
```

**Design choices:**
- Progressive filter doubling (16 → 512) extracts features from low-level edges up to high-level spatial patterns.
- `AdaptiveAvgPool2d` makes the model robust to small input size variations.
- Dropout (p=0.5) on both FC layers reduces overfitting on the small dataset.
- Sigmoid + BCELoss is the standard setup for binary classification with a single output neuron.

---

## Dataset

The dataset consists of EEG topomap images recorded after 6 seconds of exposure to visual design stimuli, split into two classes.

```
topomaps/
├── good/
│   ├── Good_6s_1.png
│   ├── Good_6s_2.png
│   └── ...           (500 images)
└── bad/
    ├── Bad_6s_1.png
    ├── Bad_6s_2.png
    └── ...           (678 images)
```

| Class | Label | Count | Description |
|-------|-------|-------|-------------|
| `bad` | `0` | 678 | Brain response to a poorly received design |
| `good` | `1` | 500 | Brain response to a well received design |

> The dataset is not included in this repository. Place the `topomaps/` folder in the project root before running training.

---

## Training Details

| Parameter | Value |
|-----------|-------|
| Optimiser | Adam |
| Learning rate | 0.001 |
| LR scheduler | ReduceLROnPlateau (factor 0.5, patience 5) |
| Loss function | BCELoss |
| Batch size | 32 (GPU) / 8 (CPU) — set automatically |
| Epochs | 60 |
| Input size | 128 × 128 |
| Normalisation | ImageNet mean/std |

### Data Augmentation

Augmentations are applied **only during training** to reduce overfitting:

| Transform | Parameters |
|-----------|------------|
| RandomHorizontalFlip | p = 0.5 |
| RandomRotation | ±10° |
| ColorJitter | brightness ±20%, contrast ±20% |

Validation and test sets use only resize and normalise — no augmentation.

### Data Split

Splits are **stratified** to preserve class balance across all three partitions.

```
Full dataset (1178 images)
├── 70%  →  Training set      (824 images)
├── 15%  →  Validation set    (177 images — used for checkpoint selection)
└── 15%  →  Test set          (177 images — held out; evaluated once after training)
```

---

## Project Structure

```
eeg-topomap-classifier/
├── train.py          # Training script: data loading, model, training loop, test eval
├── eval.py           # Inference script: load checkpoint, predict labels for new images
├── requirements.txt  # Additional dependencies (torchvision)
├── .gitignore
└── README.md
```

---

## Setup

**Prerequisites:** Python 3.8+, [Miniconda](https://www.anaconda.com/download/success) (recommended)

1. **Clone the repository:**

    ```bash
    git clone https://github.com/emaadkalantarii/eeg-topomap-classifier.git
    cd eeg-topomap-classifier
    ```

2. **Create the environment and install dependencies:**

    **Option A — conda (recommended, especially on Windows):**

    Conda resolves all native DLL dependencies automatically, avoiding common Windows issues with pip-installed PyTorch (e.g. `fbgemm.dll` errors).

    ```bash
    conda create -n brain-classifier python=3.11 -y
    conda activate brain-classifier
    conda install pytorch==2.4.0 torchvision==0.19.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
    pip install scikit-learn pillow numpy
    ```

    **Option B — pip + venv (Linux / macOS):**

    ```bash
    python -m venv venv-topo
    source venv-topo/bin/activate
    pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
    pip install scikit-learn pillow numpy
    ```

3. **Verify GPU is detected (optional but recommended):**

    ```bash
    python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
    ```

    Expected output:
    ```
    True
    NVIDIA GeForce RTX 4060 Laptop GPU
    ```

---

## Usage

### Training

Place the `topomaps/` dataset in the project root, then run:

```bash
python train.py
```

The script automatically detects and uses your GPU if available. Expected startup output:

```
GPU detected: NVIDIA GeForce RTX 4060 Laptop GPU
CUDA version: 12.1
VRAM available: 8.6 GB

Device: cuda
```

Training output per epoch:

```
Epoch   1/60 | Train Loss: 0.5099 | Val Loss: 0.3490 | Val Acc: 0.8701 | LR: 0.001000
  → Saved new best model.
...
Epoch  45/60 | Train Loss: 0.1544 | Val Loss: 0.0808 | Val Acc: 0.9718 | LR: 0.000125
  → Saved new best model.
...
Evaluating best model on held-out test set...
Test Accuracy: 0.9096
```

The best checkpoint (lowest validation loss) is saved as `model.pth`. After training completes, the script automatically loads the best checkpoint and reports accuracy on the held-out test set.

### Inference

Use the `load_and_predict` function from `eval.py` to run inference on any directory that follows the dataset structure:

```python
from eval import load_and_predict

results = load_and_predict("path/to/images", "model.pth")
# Returns:
# {
#   '/abs/path/good/Good_6s_1.png': 1,
#   '/abs/path/bad/Bad_6s_1.png':  0,
#   ...
# }
```

Or run the built-in self-test against the `topomaps/` folder:

```bash
python eval.py
```

```
GPU detected: NVIDIA GeForce RTX 4060 Laptop GPU
Model loaded successfully.
Found 1178 images.
Total predictions : 1178
Accuracy          : 0.9499
```

---

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.4.0 | Model definition and training |
| torchvision | ≥ 0.19.0 | Image transforms |
| Pillow | 11.0.0 | Image loading |
| NumPy | 2.2.0 | Numerical utilities |
| scikit-learn | 1.6 | Stratified train/test split |

---

## License

This project is licensed under the MIT License.
