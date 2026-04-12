# Brain Response Classification: Good vs. Bad Design

This project trains a deep learning image classifier to distinguish between brain responses to **good** and **bad** designs. The model is a Convolutional Neural Network (CNN) implemented in PyTorch that processes brain topomap images and outputs a binary prediction.

---

## Dataset

The dataset is expected in a folder named `topomaps` with the following structure:

```
topomaps/
  good/
    Good_6s_1.png
    Good_6s_2.png
    ...
  bad/
    Bad_6s_1.png
    Bad_6s_2.png
    ...
```

Each `.png` file is a brain topomap image recorded after 6 seconds of exposure to a design stimulus.

---

## Objective

Train a binary image classifier with the following label encoding:

| Class | Label |
|-------|-------|
| bad   | `0`   |
| good  | `1`   |

When given an image as input, the model outputs either `0` or `1`.

---

## Methodology

| Item | Detail |
|------|--------|
| Language | Python 3 |
| Framework | PyTorch |
| Task | Binary image classification |
| Loss | Binary Cross-Entropy (`BCELoss`) |
| Evaluation metric | Accuracy |
| Input size | 128 × 128 RGB |
| Normalisation | ImageNet mean/std: `[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]` |

### Model Architecture

The CNN consists of four convolutional blocks with progressively increasing filter depths (16 → 32 → 64 → 128 → 256 → 512), each followed by ReLU activations and MaxPooling. An Adaptive Average Pooling layer reduces the spatial dimensions to 4×4, and two fully connected layers (1024 → 1024 → 1) with 0.5 Dropout produce the final output. A Sigmoid activation converts this to a probability; values ≥ 0.5 predict class `1` (good).

### Training Details

- **Optimiser:** Adam (`lr=0.001`)
- **LR Scheduler:** ReduceLROnPlateau (factor 0.5, patience 5)
- **Epochs:** 60
- **Batch size:** 8
- **Data split:** 70% train / 15% validation / 15% test (stratified)
- **Training augmentations:** RandomHorizontalFlip, RandomRotation (±10°), ColorJitter (brightness/contrast ±20%)
- **Checkpoint:** Best model (lowest validation loss) saved as `model.pth`

---

## Files

| File | Description |
|------|-------------|
| `train.py` | Data loading, model definition, training loop, final test evaluation |
| `eval.py` | Load a trained `.pth` checkpoint and run inference on a directory of images |
| `requirements.txt` | Third-party dependencies beyond the default grading environment |
| `README.md` | This file |

---

## Setup

1. **Create and activate a virtual environment:**

    ```bash
    python -m venv brain_env
    source brain_env/bin/activate   # Windows: brain_env\Scripts\activate
    ```

2. **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    > **Note:** The grading environment includes `Pytorch 2.4.0`, `Numpy 2.2.0`, `Pillow 11.0.0`, and `Scikit-learn 1.6` by default. The only additional dependency this project requires is `torchvision`, which is listed in `requirements.txt`.

---

## Usage

### Training

Place the `topomaps/` dataset folder in the same directory as `train.py`, then run:

```bash
python train.py
```

The script prints train loss, validation loss, and validation accuracy after each epoch, and saves the best checkpoint as `model.pth`. After training, it automatically evaluates the best checkpoint on the held-out test set.

### Evaluation

The `load_and_predict(directory, model_file)` function in `eval.py` is the grader-facing API. It accepts the path to an image directory (with the same `good/` / `bad/` structure) and a `.pth` model file, and returns a dictionary mapping absolute image paths to predicted integer labels:

```python
from eval import load_and_predict

results = load_and_predict("topomaps", "model.pth")
# {'/abs/path/good/Good_6s_1.png': 1, '/abs/path/bad/Bad_6s_1.png': 0, ...}
```

To run the self-test directly:

```bash
python eval.py
```
