import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BrainDataset(Dataset):
    """PyTorch Dataset for brain topomap images with binary class labels."""

    def __init__(self, img_paths, labels, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        if self.transform:
            img = self.transform(img)
        return img, label


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(data_dir):
    """
    Loads image paths and binary labels from the dataset directory.

    Expected structure:
        data_dir/
          good/  ->  label 1
          bad/   ->  label 0
    """
    img_paths = []
    labels = []

    good_dir = os.path.join(data_dir, 'good')
    for fname in os.listdir(good_dir):
        if fname.lower().endswith('.png'):
            img_paths.append(os.path.join(good_dir, fname))
            labels.append(1)  # good = 1

    bad_dir = os.path.join(data_dir, 'bad')
    for fname in os.listdir(bad_dir):
        if fname.lower().endswith('.png'):
            img_paths.append(os.path.join(bad_dir, fname))
            labels.append(0)  # bad = 0

    print(f"Total images : {len(img_paths)}")
    print(f"  Good (1)   : {labels.count(1)}")
    print(f"  Bad  (0)   : {labels.count(0)}")
    return img_paths, labels


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class BrainModel(nn.Module):
    """
    CNN binary classifier for brain topomap images.
    Output is a single value in [0, 1] via Sigmoid; threshold 0.5 gives the
    predicted class (0 = bad, 1 = good).
    """

    def __init__(self):
        super(BrainModel, self).__init__()

        # Convolutional blocks
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

        # Adaptive pooling to handle varying input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Fully connected classifier
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
# Device selection
# ---------------------------------------------------------------------------

def get_device():
    if torch.cuda.is_available():
        try:
            device = torch.device('cuda')
            print("GPU available – using CUDA.")
            return device
        except Exception as e:
            print(f"CUDA error ({e}), falling back to CPU.")
    print("No GPU found. Using CPU.")
    return torch.device('cpu')


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = get_device()
    print(f"Device: {device}\n")

    # Hyperparameters
    data_dir   = 'topomaps'
    batch_size = 8
    num_epochs = 60
    lr         = 0.001

    # Separate transforms: augmentation only during training
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Load and split: 70% train / 15% val / 15% test (stratified to preserve class balance)
    all_paths, all_labels = load_data(data_dir)

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths, all_labels, test_size=0.3, random_state=42, stratify=all_labels
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )

    print(f"Split – train: {len(train_paths)}  val: {len(val_paths)}  test: {len(test_paths)}\n")

    # Datasets and loaders
    train_dataset = BrainDataset(train_paths, train_labels, train_transform)
    val_dataset   = BrainDataset(val_paths,   val_labels,   eval_transform)
    test_dataset  = BrainDataset(test_paths,  test_labels,  eval_transform)

    num_workers = 2 if device.type == 'cuda' else 0

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Model, loss, optimiser, scheduler
    model     = BrainModel().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_loss = float('inf')

    print("Starting training...\n")

    for epoch in range(1, num_epochs + 1):

        # Training
        model.train()
        train_loss = 0.0

        for imgs, targets in train_loader:
            imgs    = imgs.to(device)
            targets = targets.unsqueeze(1).to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        correct  = 0
        total    = 0

        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs    = imgs.to(device)
                targets = targets.unsqueeze(1).to(device)

                outputs   = model(imgs)
                val_loss += criterion(outputs, targets).item() * imgs.size(0)

                predicted  = (outputs >= 0.5).float()
                total     += targets.size(0)
                correct   += (predicted == targets).sum().item()

        val_loss /= len(val_loader.dataset)
        val_acc   = correct / total

        scheduler.step(val_loss)
        curr_lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch:3d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | "
            f"LR: {curr_lr:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'model.pth')
            print("  → Saved new best model.")

    # Final evaluation on the held-out test set using the best checkpoint
    print("\nEvaluating best model on held-out test set...")
    model.load_state_dict(torch.load('model.pth', weights_only=True, map_location=device))
    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs    = imgs.to(device)
            targets = targets.unsqueeze(1).to(device)
            outputs = model(imgs)
            predicted = (outputs >= 0.5).float()
            total    += targets.size(0)
            correct  += (predicted == targets).sum().item()

    print(f"Test Accuracy: {correct / total:.4f}")

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
