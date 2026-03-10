"""
Main training script for Poomsae LSTM model
With class weights for imbalanced data
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
from tqdm import tqdm
import numpy as np
import sys

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from models.lstm_classifier import PoomsaeLSTM
from training.dataset import create_dataloaders
from configs.lstm_config import LSTMConfig
from configs.training_config import TrainingConfig
from configs.paths import Paths
from preprocessing.create_windows import CLASS_MAPPING, CLASS_NAMES


class Trainer:
    """Handles model training and validation"""

    def __init__(self, model, train_loader, val_loader, config, class_weights=None,
                 lstm_config=None, class_mapping=None):
        """
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            config: Training configuration
            class_weights: Optional tensor of class weights for imbalanced data
            lstm_config: LSTM model configuration (for saving)
            class_mapping: Class name mapping (for saving)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.lstm_config = lstm_config
        self.class_mapping = class_mapping

        # Setup device
        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
        print(f"\nUsing device: {self.device}")

        self.model = self.model.to(self.device)

        # Move class weights to device if provided
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
            print(f"[OK] Class weights applied for imbalanced data handling")

        # Loss function (with optional class weights)
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=config.LABEL_SMOOTHING
        )

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=config.SCHEDULER_PATIENCE,
            factor=config.SCHEDULER_FACTOR,
        )

        # Tracking variables
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.patience_counter = 0
        self.current_epoch = 0

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()

        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}/{self.config.EPOCHS} [Train]"
        )

        for batch_idx, (data, target) in enumerate(pbar):
            # Move to device
            data = data.to(self.device)
            target = target.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

        avg_loss = total_loss / len(self.train_loader)
        avg_acc = 100.0 * correct / total

        return avg_loss, avg_acc

    def validate(self):
        """Validate model"""
        self.model.eval()

        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc="Validating"):
                data = data.to(self.device)
                target = target.to(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)

                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        avg_acc = 100.0 * correct / total

        return avg_loss, avg_acc

    def train(self):
        """Full training loop"""
        print(f"\n{'=' * 60}")
        print("TRAINING START")
        print(f"{'=' * 60}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Device: {self.device}")
        print(f"Train samples: {len(self.train_loader.dataset)}")
        print(f"Val samples: {len(self.val_loader.dataset)}")
        print(f"Batch size: {self.config.BATCH_SIZE}")
        print(f"Epochs: {self.config.EPOCHS}")
        print(f"Learning rate: {self.config.LEARNING_RATE}")
        print(f"{'=' * 60}\n")

        for epoch in range(self.config.EPOCHS):
            self.current_epoch = epoch

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc = self.validate()

            # Track history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{self.config.EPOCHS} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

            # Learning rate scheduling
            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']

            if new_lr != old_lr:
                print(f"  Learning rate: {old_lr:.6f} → {new_lr:.6f}")

            # Save best model
            if val_acc > self.best_val_acc:
                improvement = val_acc - self.best_val_acc
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.save_checkpoint('best', self.lstm_config, self.config, self.class_mapping)
                print(f"  [OK] New best model! (+{improvement:.2f}%) Saved as 'best'")
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                print(f"  No improvement ({self.patience_counter}/{self.config.EARLY_STOPPING_PATIENCE})")

            # Early stopping check
            if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"\n{'=' * 60}")
                print(f"Early stopping triggered at epoch {epoch + 1}")
                print(f"{'=' * 60}")
                break

            # Regular checkpoint
            if (epoch + 1) % self.config.SAVE_EVERY == 0:
                self.save_checkpoint(f'epoch_{epoch + 1}', self.lstm_config, self.config, self.class_mapping)
                print(f"  [OK] Checkpoint saved: epoch_{epoch + 1}")

            print()  # Empty line between epochs

        # Training complete
        print(f"\n{'=' * 60}")
        print("TRAINING COMPLETE")
        print(f"{'=' * 60}")
        print(f"Best Validation Accuracy: {self.best_val_acc:.2f}%")
        print(f"Best Validation Loss: {self.best_val_loss:.4f}")
        print(f"Total Epochs: {self.current_epoch + 1}")
        print(f"{'=' * 60}\n")

    def save_checkpoint(self, name, lstm_config=None, training_config=None, class_mapping=None):
        """Save model checkpoint with full metadata"""
        checkpoint_dir = Paths.CHECKPOINTS_DIR
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = checkpoint_dir / f'lstm_taegeuk1_{name}.pth'

        # Build checkpoint with all important metadata
        checkpoint = {
            # Training state
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs,
        }

        # Add model architecture config
        if lstm_config:
            checkpoint['model_config'] = {
                'input_size': lstm_config.INPUT_SIZE,
                'hidden_size': lstm_config.HIDDEN_SIZE,
                'num_layers': lstm_config.NUM_LAYERS,
                'dropout': lstm_config.DROPOUT,
                'bidirectional': lstm_config.BIDIRECTIONAL,
                'num_classes': lstm_config.NUM_CLASSES,
                'sequence_length': lstm_config.SEQUENCE_LENGTH,
                'stride': lstm_config.STRIDE,
            }

        # Add training config
        if training_config:
            checkpoint['training_config'] = {
                'batch_size': training_config.BATCH_SIZE,
                'learning_rate': training_config.LEARNING_RATE,
                'weight_decay': training_config.WEIGHT_DECAY,
                'label_smoothing': training_config.LABEL_SMOOTHING,
                'short_movement_classes': getattr(training_config, 'SHORT_MOVEMENT_CLASSES', []),
                'short_movement_weight_multiplier': getattr(training_config, 'SHORT_MOVEMENT_WEIGHT_MULTIPLIER', 1.0),
            }

        # Add class mapping
        if class_mapping:
            checkpoint['class_mapping'] = class_mapping
            checkpoint['class_names'] = list(class_mapping.keys())
            checkpoint['num_classes'] = len(class_mapping)

        torch.save(checkpoint, checkpoint_path)

        # Save training history as JSON (human readable)
        history_path = checkpoint_dir / f'training_history_{name}.json'
        history = {
            'epoch': self.current_epoch + 1,
            'train_losses': [float(x) for x in self.train_losses],
            'val_losses': [float(x) for x in self.val_losses],
            'train_accs': [float(x) for x in self.train_accs],
            'val_accs': [float(x) for x in self.val_accs],
            'best_val_acc': float(self.best_val_acc),
            'best_val_loss': float(self.best_val_loss),
            # Include config in JSON too
            'model_config': checkpoint.get('model_config', {}),
            'training_config': checkpoint.get('training_config', {}),
            'num_classes': checkpoint.get('num_classes', 22),
        }

        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)


def main():
    """Main training function"""

    # Load configurations
    lstm_config = LSTMConfig()
    training_config = TrainingConfig()

    # Create necessary directories
    Paths.create_directories()

    print("Creating dataloaders...")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=Paths.WINDOWS_DIR,
        batch_size=training_config.BATCH_SIZE,
        train_split=training_config.TRAIN_SPLIT,
        val_split=training_config.VAL_SPLIT,
        num_workers=training_config.NUM_WORKERS
    )

    # Calculate class weights for imbalanced data
    print("\nCalculating class weights for imbalanced data...")
    train_labels = []
    for _, labels in train_loader:
        train_labels.extend(labels.numpy())
    train_labels = np.array(train_labels)

    class_counts = np.bincount(train_labels, minlength=lstm_config.NUM_CLASSES)
    total = len(train_labels)
    class_weights = total / (lstm_config.NUM_CLASSES * class_counts + 1e-6)

    # Apply extra weight multiplier for short movement classes
    short_classes = getattr(training_config, 'SHORT_MOVEMENT_CLASSES', [])
    short_multiplier = getattr(training_config, 'SHORT_MOVEMENT_WEIGHT_MULTIPLIER', 1.0)

    for class_idx in short_classes:
        if class_idx < len(class_weights):
            class_weights[class_idx] *= short_multiplier

    class_weights = torch.FloatTensor(class_weights)

    print("\nClass weights (higher weight = more attention):")
    print(f"{'Class':<8} {'ID':<8} {'Samples':>10} {'Weight':>10} {'Note':<15}")
    print("-" * 55)
    for i in range(lstm_config.NUM_CLASSES):
        note = "** SHORT **" if i in short_classes else ""
        print(f"{i:<8} {i:<8} {class_counts[i]:>10}   {class_weights[i]:>9.3f}  {note}")

    # Highlight extremes
    min_weight_idx = class_weights.argmin().item()
    max_weight_idx = class_weights.argmax().item()
    print(f"\nLowest weight: Class {min_weight_idx} (weight={class_weights[min_weight_idx]:.3f})")
    print(f"Highest weight: Class {max_weight_idx} (weight={class_weights[max_weight_idx]:.3f})")
    if short_classes:
        print(f"Short movement classes {short_classes} have {short_multiplier}x weight boost")

    # Create model
    print("\nCreating model...")
    model = PoomsaeLSTM(lstm_config)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Create trainer with class weights and metadata for saving
    trainer = Trainer(
        model, train_loader, val_loader, training_config, class_weights,
        lstm_config=lstm_config,
        class_mapping=CLASS_MAPPING
    )

    # Train
    trainer.train()

    # Evaluate on test set
    print("Evaluating on test set...")

    # Load best model
    best_checkpoint = torch.load(Paths.CHECKPOINTS_DIR / 'lstm_taegeuk1_best.pth')
    model.load_state_dict(best_checkpoint['model_state_dict'])
    model = model.to(trainer.device)

    # Create temporary trainer for evaluation (no class weights for unbiased test evaluation)
    test_trainer = Trainer(model, train_loader, test_loader, training_config, class_weights=None)
    test_loss, test_acc = test_trainer.validate()

    print(f"\n{'=' * 60}")
    print("FINAL TEST RESULTS")
    print(f"{'=' * 60}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()