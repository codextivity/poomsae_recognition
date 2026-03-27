"""Main training script for weighted LSTM movement classification."""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from configs.lstm_config import LSTMConfig
from configs.paths import Paths
from configs.training_config import TrainingConfig
from models.lstm_classifier import PoomsaeLSTM
from training.dataset import create_dataloaders
from utils.matplotlib_korean import configure_korean_font
from utils.save_normalization_stats import save_normalization_stats

configure_korean_font()


class Trainer:
    """Handles model training and validation."""

    def __init__(self, model, train_loader, val_loader, config, class_metadata, class_weights=None, lstm_config=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.lstm_config = lstm_config
        self.class_metadata = class_metadata

        self.device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
        print(f'\nUsing device: {self.device}')
        self.model = self.model.to(self.device)

        if class_weights is not None:
            class_weights = class_weights.to(self.device)
            print('[OK] Class weights applied for imbalanced data handling')

        self.criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=config.LABEL_SMOOTHING)
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            patience=config.SCHEDULER_PATIENCE,
            factor=config.SCHEDULER_FACTOR,
        )

        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.patience_counter = 0
        self.current_epoch = 0

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}/{self.config.EPOCHS} [Train]')
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

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.0 * correct / total:.2f}%'})

        return total_loss / len(self.train_loader), 100.0 * correct / total

    def validate(self, loader=None):
        self.model.eval()
        loader = loader or self.val_loader
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in tqdm(loader, desc='Validating'):
                data = data.to(self.device)
                target = target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        return total_loss / len(loader), 100.0 * correct / total

    def train(self):
        print(f"\n{'=' * 60}")
        print('TRAINING START')
        print(f"{'=' * 60}")
        print(f'Model: {self.model.__class__.__name__}')
        print(f'Device: {self.device}')
        print(f'Train samples: {len(self.train_loader.dataset)}')
        print(f'Val samples: {len(self.val_loader.dataset)}')
        print(f'Batch size: {self.config.BATCH_SIZE}')
        print(f'Epochs: {self.config.EPOCHS}')
        print(f'Learning rate: {self.config.LEARNING_RATE}')
        print(f"{'=' * 60}\n")

        for epoch in range(self.config.EPOCHS):
            self.current_epoch = epoch
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            print(f'\nEpoch {epoch + 1}/{self.config.EPOCHS} Summary:')
            print(f'  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%')

            old_lr = self.optimizer.param_groups[0]['lr']
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']
            if new_lr != old_lr:
                print(f'  Learning rate: {old_lr:.6f} -> {new_lr:.6f}')

            if val_acc > self.best_val_acc:
                improvement = val_acc - self.best_val_acc
                self.best_val_acc = val_acc
                self.best_val_loss = val_loss
                self.save_checkpoint('best')
                print(f"  [OK] New best model! (+{improvement:.2f}%) Saved as 'best'")
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                print(f'  No improvement ({self.patience_counter}/{self.config.EARLY_STOPPING_PATIENCE})')

            if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"\n{'=' * 60}")
                print(f'Early stopping triggered at epoch {epoch + 1}')
                print(f"{'=' * 60}")
                break

            if (epoch + 1) % self.config.SAVE_EVERY == 0:
                self.save_checkpoint(f'epoch_{epoch + 1}')
                print(f"  [OK] Checkpoint saved: epoch_{epoch + 1}")

        print(f"\n{'=' * 60}")
        print('TRAINING COMPLETE')
        print(f"{'=' * 60}")
        print(f'Best Validation Accuracy: {self.best_val_acc:.2f}%')
        print(f'Best Validation Loss: {self.best_val_loss:.4f}')
        print(f'Total Epochs: {self.current_epoch + 1}')
        print(f"{'=' * 60}\n")
        self.save_training_plots()
        self.save_normalization_stats_file()

    def save_training_plots(self):
        """Save loss/accuracy curves for the completed training run."""
        checkpoint_dir = Paths.CHECKPOINTS_DIR
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if not self.train_losses or not self.val_losses:
            print('[!] No training history available to plot')
            return

        epochs = list(range(1, len(self.train_losses) + 1))

        loss_fig = checkpoint_dir / 'train_val_loss.png'
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, self.train_losses, marker='o', label='Train Loss')
        plt.plot(epochs, self.val_losses, marker='o', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training vs Validation Loss')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(loss_fig, dpi=300, bbox_inches='tight')
        plt.close()

        if self.train_accs and self.val_accs:
            acc_fig = checkpoint_dir / 'train_val_accuracy.png'
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, self.train_accs, marker='o', label='Train Accuracy')
            plt.plot(epochs, self.val_accs, marker='o', label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.title('Training vs Validation Accuracy')
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(acc_fig, dpi=300, bbox_inches='tight')
            plt.close()
            print(f'[OK] Training accuracy plot saved: {acc_fig}')

        print(f'[OK] Training loss plot saved: {loss_fig}')

    def save_normalization_stats_file(self):
        """Persist normalization stats for inference-time z-score normalization."""
        try:
            output_path = save_normalization_stats(
                windows_dir=Paths.WINDOWS_DIR,
                checkpoint_dir=Paths.CHECKPOINTS_DIR,
            )
            print(f'[OK] Normalization stats saved: {output_path}')
        except Exception as exc:
            print(f'[WARN] Failed to save normalization stats automatically: {exc}')

    def save_checkpoint(self, name):
        checkpoint_dir = Paths.CHECKPOINTS_DIR
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f'lstm_{name}.pth'

        checkpoint = {
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
            'model_config': {
                'input_size': self.lstm_config.INPUT_SIZE,
                'hidden_size': self.lstm_config.HIDDEN_SIZE,
                'num_layers': self.lstm_config.NUM_LAYERS,
                'dropout': self.lstm_config.DROPOUT,
                'bidirectional': self.lstm_config.BIDIRECTIONAL,
                'num_classes': self.lstm_config.NUM_CLASSES,
                'sequence_length': self.lstm_config.SEQUENCE_LENGTH,
                'stride': self.lstm_config.STRIDE,
            },
            'training_config': {
                'batch_size': self.config.BATCH_SIZE,
                'learning_rate': self.config.LEARNING_RATE,
                'weight_decay': self.config.WEIGHT_DECAY,
                'label_smoothing': self.config.LABEL_SMOOTHING,
                'short_class_weight_multiplier': getattr(self.config, 'SHORT_MOVEMENT_WEIGHT_MULTIPLIER', 1.0),
            },
            'class_mapping': self.class_metadata['class_mapping'],
            'class_names': self.class_metadata['class_names'],
            'num_classes': self.class_metadata['num_classes'],
            'short_class_indices': self.class_metadata.get('short_class_indices', []),
            'short_movement_ids': self.class_metadata.get('short_movement_ids', []),
        }

        torch.save(checkpoint, checkpoint_path)

        history_path = checkpoint_dir / f'training_history_{name}.json'
        history = {
            'epoch': self.current_epoch + 1,
            'train_losses': [float(x) for x in self.train_losses],
            'val_losses': [float(x) for x in self.val_losses],
            'train_accs': [float(x) for x in self.train_accs],
            'val_accs': [float(x) for x in self.val_accs],
            'best_val_acc': float(self.best_val_acc),
            'best_val_loss': float(self.best_val_loss),
            'model_config': checkpoint['model_config'],
            'training_config': checkpoint['training_config'],
            'num_classes': checkpoint['num_classes'],
            'class_mapping': checkpoint['class_mapping'],
            'class_names': checkpoint['class_names'],
        }
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)


def main():
    lstm_config = LSTMConfig()
    training_config = TrainingConfig()
    Paths.create_directories()

    print('Creating dataloaders...')
    train_loader, val_loader, test_loader, class_metadata = create_dataloaders(
        data_dir=Paths.WINDOWS_DIR,
        batch_size=training_config.BATCH_SIZE,
        train_split=training_config.TRAIN_SPLIT,
        val_split=training_config.VAL_SPLIT,
        num_workers=training_config.NUM_WORKERS,
        return_metadata=True,
    )

    lstm_config.NUM_CLASSES = int(class_metadata['num_classes'])
    short_classes = list(class_metadata.get('short_class_indices', []))

    print('\nCalculating class weights for imbalanced data...')
    train_indices = getattr(train_loader.dataset, 'indices', None)
    if train_indices is None:
        raise ValueError('Train dataset does not expose indices for class-weight calculation')

    base_dataset = getattr(train_loader.dataset, 'base_dataset', None)
    if base_dataset is None:
        raise ValueError('Train dataset does not expose base_dataset for class-weight calculation')

    train_labels = np.array(base_dataset.y[train_indices], dtype=np.int64)

    class_counts = np.bincount(train_labels, minlength=lstm_config.NUM_CLASSES)
    total = len(train_labels)
    class_weights = total / (lstm_config.NUM_CLASSES * class_counts + 1e-6)
    short_multiplier = getattr(training_config, 'SHORT_MOVEMENT_WEIGHT_MULTIPLIER', 1.0)
    for class_idx in short_classes:
        if class_idx < len(class_weights):
            class_weights[class_idx] *= short_multiplier
    class_weights = torch.FloatTensor(class_weights)

    inverse = {idx: mov_id for mov_id, idx in class_metadata['class_mapping'].items()}
    print('\nClass weights (higher weight = more attention):')
    print(f"{'Class':<8} {'ID':<10} {'RawTrain':>10} {'Weight':>10} {'Note':<15}")
    print('-' * 62)
    for i in range(lstm_config.NUM_CLASSES):
        note = '** SHORT **' if i in short_classes else ''
        print(f"{i:<8} {inverse.get(i, f'class_{i}'):<10} {class_counts[i]:>10}   {class_weights[i]:>9.3f}  {note}")

    min_weight_idx = int(class_weights.argmin().item())
    max_weight_idx = int(class_weights.argmax().item())
    print(f"\nLowest weight: Class {min_weight_idx} ({inverse.get(min_weight_idx)}) weight={class_weights[min_weight_idx]:.3f}")
    print(f"Highest weight: Class {max_weight_idx} ({inverse.get(max_weight_idx)}) weight={class_weights[max_weight_idx]:.3f}")
    if short_classes:
        print(f'Short movement classes {short_classes} have {short_multiplier}x weight boost')

    print('\nCreating model...')
    model = PoomsaeLSTM(lstm_config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nModel Summary:')
    print(f'  Total parameters: {total_params:,}')
    print(f'  Trainable parameters: {trainable_params:,}')

    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        training_config,
        class_metadata=class_metadata,
        class_weights=class_weights,
        lstm_config=lstm_config,
    )
    trainer.train()

    print('Evaluating on test set...')
    best_checkpoint_path = Paths.CHECKPOINTS_DIR / 'lstm_best.pth'
    best_checkpoint = torch.load(best_checkpoint_path, map_location=trainer.device, weights_only=False)
    model.load_state_dict(best_checkpoint['model_state_dict'])
    model = model.to(trainer.device)

    test_trainer = Trainer(model, train_loader, test_loader, training_config, class_metadata=class_metadata, class_weights=None, lstm_config=lstm_config)
    test_loss, test_acc = test_trainer.validate(loader=test_loader)

    print(f"\n{'=' * 60}")
    print('FINAL TEST RESULTS')
    print(f"{'=' * 60}")
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_acc:.2f}%')
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()
