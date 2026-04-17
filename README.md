# EEG Topomap Classifier

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4-EE4C2C?logo=pytorch&logoColor=white)
![Task](https://img.shields.io/badge/Task-Binary%20Classification-4CAF50)
![Domain](https://img.shields.io/badge/Domain-EEG%20%2F%20BCI-8A2BE2)
![License](https://img.shields.io/badge/License-MIT-blue)

A deep learning system that classifies brain responses to visual design stimuli as **good** or **bad** using EEG topographic map (topomap) images. The model is a custom Convolutional Neural Network (CNN) built from scratch in PyTorch, trained to distinguish between two neurological response patterns recorded after 6 seconds of design exposure.

---

## Overview

Electroencephalography (EEG) topographic maps (topomaps) visualise the spatial distribution of brain electrical activity across the scalp at a given moment. Traditionally, interpreting these maps to assess whether a design evokes a positive or negative neurological response requires domain expertise and is time-consuming at scale.

This project applies computer vision to that problem: instead of a human expert interpreting the maps, a CNN learns directly from topomap images to distinguish positive from negative brain responses to visual stimuli — enabling automated, scalable assessment for applications in UX research, design evaluation, and neuro-marketing.

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
- Progressive filter doubling (16 → 512) builds a feature hierarchy from low-level edges up to high-level spatial patterns.
- `AdaptiveAvgPool2d` makes the model robust to small input size variations.
- Dropout (p=0.5) on both FC layers is the primary regularisation mechanism, essential given the small dataset size.
- Sigmoid + BCELoss is the standard setup for binary classification with a single output neuron.

---

## Dataset

The dataset consists of EEG topomap images recorded after 6 seconds of exposure to visual design stimuli, labelled by the class of neurological response. It is included directly in this repository as no public download URL exists.

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

**Class imbalance:** The dataset is naturally imbalanced (678 bad vs. 500 good). All train/val/test splits are **stratified** to preserve this ratio across every partition, preventing any split from being accidentally skewed toward one class.

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
| Checkpoint strategy | Save on best validation loss |

### Data Augmentation

Augmentations are applied **only during training** to reduce overfitting. Validation and test sets use only resize and normalise — no augmentation.

| Transform | Parameters |
|-----------|------------|
| RandomHorizontalFlip | p = 0.5 |
| RandomRotation | ±10° |
| ColorJitter | brightness ±20%, contrast ±20% |

### Data Split

Splits are **stratified** to preserve class balance across all three partitions.

```
Full dataset (1178 images)
├── 70%  →  Training set      (824 images)
├── 15%  →  Validation set    (177 images — used for checkpoint selection)
└── 15%  →  Test set          (177 images — held out; evaluated once after training)
```

---

## Repository Structure

```
EEG-Topomap-Classifier/
├── topomaps/             # Full dataset (included — no public download URL exists)
│   ├── good/             # 500 images of positive brain responses
│   └── bad/              # 678 images of negative brain responses
├── train.py              # Training script: data loading, model definition, training loop, test eval
├── eval.py               # Inference script: load checkpoint, predict labels for new images
├── requirements.txt      # Python dependencies
├── .gitignore            # Excludes model checkpoint, venvs, and cache
└── README.md
```

> `model.pth` is not tracked — it is generated locally by running `train.py`.

---

## Setup

**Prerequisites:** Python 3.8+, [Miniconda](https://www.anaconda.com/download/success) (recommended)

**1. Clone the repository:**

```bash
git clone https://github.com/emaadkalantarii/EEG-Topomap-Classifier.git
cd EEG-Topomap-Classifier
```

The `topomaps/` dataset is included in the repository and will be available immediately after cloning.

**2. Create the environment and install dependencies:**

**Option A — conda (recommended, especially on Windows):**

Conda resolves all native DLL dependencies automatically, avoiding common Windows issues with pip-installed PyTorch (e.g. `fbgemm.dll` errors).

```bash
conda create -n brain-classifier python=3.11 -y
conda activate brain-classifier
conda install pytorch==2.4.0 torchvision==0.19.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install scikit-learn==1.6.0 pillow==11.0.0 numpy==2.2.0
```

**Option B — pip + venv (Linux / macOS):**

```bash
python -m venv venv-topo
source venv-topo/bin/activate
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
pip install scikit-learn==1.6.0 pillow==11.0.0 numpy==2.2.0
```

**3. Verify GPU is detected (optional but recommended):**

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

If a GPU is available, the output will be:
```
True
NVIDIA GeForce <your GPU name>
```
If no GPU is detected, the scripts automatically fall back to CPU.

---

## Usage

### Training

```bash
python train.py
```

The script automatically detects and uses your GPU if available. Example startup output on a CUDA-enabled machine:

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

The best checkpoint (lowest validation loss) is saved as `model.pth`. After training completes, the script automatically loads the best checkpoint and evaluates it on the held-out test set.

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

Or run the built-in self-test directly:

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

> **Note:** The self-test runs inference over all 1,178 images including those seen during training, which is why the reported accuracy (94.99%) is higher than the held-out test set accuracy (90.96%). The test set figure is the correct measure of generalisation.

---

## Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.4.0 | Model definition and training |
| torchvision | 0.19.0 | Image transforms |
| Pillow | 11.0.0 | Image loading |
| NumPy | 2.2.0 | Numerical utilities |
| scikit-learn | 1.6.0 | Stratified train/val/test split |

---

## License

This project is licensed under the MIT License.
