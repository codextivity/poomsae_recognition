"""
Training script for Short Movement LSTM model

Trains a specialized model for detecting fast movements:
- 6_1 (right punch)
- 12_1 (left punch)
- 14_1 (right front kick)
- 16_1 (left front kick)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.lstm_classifier import PoomsaeLSTM
from configs.lstm_config_short import LSTMConfigShort
from configs.training_config import TrainingConfig
from configs.paths import Paths


class ShortMovementDataset(Dataset):
    """Dataset for short movement windows"""

    def __init__(self, npz_path, normalize=True):
        """
        Args:
            npz_path: Path to short_movements_windows.npz
            normalize: Whether to normalize features
        """
        data = np.load(npz_path, allow_pickle=True)
        self.X = data['X'].astype(np.float32)  # (N, 16, 78)
        self.y = data['y'].astype(np.int64)     # (N,)

        self.normalize = normalize
        if normalize:
            # Compute mean and std across all samples
            self.mean = self.X.mean(axis=(0, 1), keepdims=True)
            self.std = self.X.std(axis=(0, 1), keepdims=True) + 1e-8

        print(f"Loaded {len(self.X)} samples from {npz_path}")
        print(f"  X shape: {self.X.shape}")
        print(f"  y shape: {self.y.shape}")

        # Print class distribution
        unique, counts = np.unique(self.y, return_counts=True)
        class_names = ['6_1', '12_1', '14_1', '16_1', 'other']
        print("  Class distribution:")
        for u, c in zip(unique, counts):
            print(f"    Class {u} ({class_names[u]}): {c}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        if self.normalize:
            x = (x - self.mean.squeeze()) / self.std.squeeze()

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


class ShortMovementTrainer:
    """Trainer for short movement model"""

    def __init__(self, model, train_loader, val_loader, config, class_weights=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config

        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {self.device}")

        self.model = self.model.to(self.device)

        if class_weights is not None:
            class_weights = class_weights.to(self.device)
            print("[OK] Class weights applied")

        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=config.LABEL_SMOOTHING
        )

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=config.SCHEDULER_PATIENCE,
            factor=config.SCHEDULER_FACTOR,
        )

        self.best_val_acc = 0.0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1} [Train]")
        for data, target in pbar:
            data = data.to(self.device)
            target = target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

        return total_loss / len(self.train_loader), 100.0 * correct / total

    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        # Per-class accuracy tracking
        class_correct = [0] * 5
        class_total = [0] * 5

        with torch.no_grad():
            for data, target in self.val_loader:
                data = data.to(self.device)
                target = target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                # Per-class tracking
                for i in range(len(target)):
                    label = target[i].item()
                    class_total[label] += 1
                    if predicted[i] == target[i]:
                        class_correct[label] += 1

        # Print per-class accuracy
        class_names = ['6_1', '12_1', '14_1', '16_1', 'other']
        print("\n  Per-class accuracy:")
        for i in range(5):
            if class_total[i] > 0:
                acc = 100.0 * class_correct[i] / class_total[i]
                print(f"    {class_names[i]}: {acc:.1f}% ({class_correct[i]}/{class_total[i]})")

        return total_loss / len(self.val_loader), 100.0 * correct / total

    def train(self, epochs):
        print(f"\n{'='*60}")
        print("SHORT MOVEMENT MODEL TRAINING")
        print(f"{'='*60}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"{'='*60}\n")

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            print(f"\nEpoch {epoch + 1}/{epochs}:")
            print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

            self.scheduler.step(val_loss)

            if val_acc > self.best_val_acc:
                improvement = val_acc - self.best_val_acc
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.save_checkpoint('best')
                print(f"  [OK] New best! (+{improvement:.2f}%)")
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                print(f"  No improvement ({self.patience_counter}/{self.config.EARLY_STOPPING_PATIENCE})")

            if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"Best Val Accuracy: {self.best_val_acc:.2f}%")
        print(f"{'='*60}\n")

    def save_checkpoint(self, name):
        checkpoint_dir = Paths.CHECKPOINTS_DIR
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save with "short" prefix to distinguish from main model
        checkpoint_path = checkpoint_dir / f'lstm_short_{name}.pth'

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': {
                'sequence_length': 16,
                'num_classes': 5,
                'class_names': ['6_1', '12_1', '14_1', '16_1', 'other']
            }
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"  Saved: {checkpoint_path}")


def main():
    # Configs
    lstm_config = LSTMConfigShort()
    training_config = TrainingConfig()

    Paths.create_directories()

    # Load dataset
    data_path = Paths.WINDOWS_DIR / "short_movements_windows.npz"

    if not data_path.exists():
        print(f"[!] Dataset not found: {data_path}")
        print("Run preprocessing/create_windows_short.py first!")
        return

    dataset = ShortMovementDataset(data_path)

    # Split: 70% train, 15% val, 15% test
    n_total = len(dataset)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"\nDataset splits:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val: {len(val_dataset)}")
    print(f"  Test: {len(test_dataset)}")

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Calculate class weights
    train_labels = dataset.y[train_dataset.indices]
    class_counts = np.bincount(train_labels, minlength=5)
    total = len(train_labels)
    class_weights = total / (5 * class_counts + 1e-6)
    class_weights = torch.FloatTensor(class_weights)

    print("\nClass weights:")
    class_names = ['6_1', '12_1', '14_1', '16_1', 'other']
    for i, name in enumerate(class_names):
        print(f"  {name}: {class_weights[i]:.3f} ({class_counts[i]} samples)")

    # Create model
    model = PoomsaeLSTM(lstm_config)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    # Train
    trainer = ShortMovementTrainer(
        model, train_loader, val_loader,
        training_config, class_weights
    )
    trainer.train(training_config.EPOCHS)

    # Test evaluation
    print("\nEvaluating on test set...")
    best_checkpoint = torch.load(Paths.CHECKPOINTS_DIR / 'lstm_short_best.pth')
    model.load_state_dict(best_checkpoint['model_state_dict'])
    model = model.to(trainer.device)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(trainer.device)
            target = target.to(trainer.device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    test_acc = 100.0 * correct / total
    print(f"\nTest Accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
