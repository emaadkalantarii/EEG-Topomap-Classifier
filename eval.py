import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


# ---------------------------------------------------------------------------
# Model definition  (must be identical to the one in train.py)
# ---------------------------------------------------------------------------

class BrainModel(nn.Module):
    """
    CNN binary classifier for brain topomap images.
    Output is a single value in [0, 1] via Sigmoid; threshold 0.5 gives the
    predicted class (0 = bad, 1 = good).
    """

    def __init__(self):
        super(BrainModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)

        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.fc1 = nn.Linear(512 * 4 * 4, 1024)
        self.relu7 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 1024)
        self.relu8 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool1(self.relu2(self.conv2(self.relu1(self.conv1(x)))))   # block 1
        x = self.pool2(self.relu4(self.conv4(self.relu3(self.conv3(x)))))   # block 2
        x = self.pool3(self.relu5(self.conv5(x)))                           # block 3
        x = self.pool4(self.relu6(self.conv6(x)))                           # block 4
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout1(self.relu7(self.fc1(x)))
        x = self.dropout2(self.relu8(self.fc2(x)))
        x = self.sigmoid(self.fc3(x))
        return x


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        try:
            return torch.device('cuda')
        except Exception as e:
            print(f"CUDA error ({e}), falling back to CPU.")
    return torch.device('cpu')


# ---------------------------------------------------------------------------
# Public API  (called by the graders)
# ---------------------------------------------------------------------------

def load_and_predict(directory, model_file):
    """
    Loads a trained model and predicts class labels for brain response images.

    The directory argument is a folder with the same structure as the dataset:

        /path/to/some/images
          |_ good
             |_ Good_6s_1.png
             |_ Good_6s_2.png
             |_ ...
          |_ bad
             |_ Bad_6s_1.png
             |_ Bad_6s_2.png
             |_ ...

    The model_file argument is a trained model checkpoint in .pth format.

    This function:
    1. Reads all .png images from the good/ and bad/ sub-directories.
    2. Applies the same preprocessing as during training.
    3. Loads the model checkpoint.
    4. Runs inference and converts probabilities to integer labels:
       0 for "bad", 1 for "good".
    5. Returns a dictionary mapping absolute file paths to predicted labels.

    Returns
    -------
    dict
        {'/path/to/images/good/Good_6s_1.png': 1,
         '/path/to/images/bad/Bad_6s_1.png':  0, ...}
    """
    device = get_device()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Load model
    model = BrainModel()
    try:
        state_dict = torch.load(model_file, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model = model.to(device)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return {}

    model.eval()

    # Collect image paths from both sub-directories
    image_files = []
    for subfolder in ('good', 'bad'):
        folder_path = os.path.join(directory, subfolder)
        if os.path.exists(folder_path):
            for fname in os.listdir(folder_path):
                if fname.lower().endswith('.png'):
                    image_files.append(os.path.join(folder_path, fname))

    print(f"Found {len(image_files)} images.")

    labels_dict = {}
    batch_size  = 16 if device.type == 'cuda' else 4

    with torch.no_grad():
        for i in range(0, len(image_files), batch_size):
            batch_paths   = image_files[i : i + batch_size]
            batch_tensors = []
            valid_paths   = []

            for img_path in batch_paths:
                try:
                    img = Image.open(img_path).convert('RGB')
                    batch_tensors.append(transform(img))
                    valid_paths.append(img_path)
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

            if not batch_tensors:
                continue

            outputs = model(torch.stack(batch_tensors).to(device))

            for img_path, prob in zip(valid_paths, outputs.cpu().tolist()):
                # prob is a list with one element because the model output shape is (batch, 1)
                labels_dict[os.path.abspath(img_path)] = 1 if prob[0] >= 0.5 else 0

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return labels_dict


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_dir   = "topomaps"
    model_path = "model.pth"

    results = load_and_predict(test_dir, model_path)

    if not results:
        print("No predictions produced.")
    else:
        correct = 0
        for abs_path, pred_label in results.items():
            # Ground-truth is encoded in the parent folder name: 'good' -> 1, 'bad' -> 0
            parent_folder = os.path.basename(os.path.dirname(abs_path))
            true_label    = 1 if parent_folder == 'good' else 0
            if pred_label == true_label:
                correct += 1

        print(f"Total predictions : {len(results)}")
        print(f"Accuracy          : {correct / len(results):.4f}")
