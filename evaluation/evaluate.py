"""Model evaluation with confusion matrix and per-class analysis."""

import json
import os
import sys
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from configs.class_metadata import metadata_from_checkpoint
from configs.lstm_config import LSTMConfig
from configs.paths import Paths
from configs.training_config import TrainingConfig
from models.lstm_classifier import PoomsaeLSTM
from training.dataset import create_dataloaders
from utils.matplotlib_korean import configure_korean_font

configure_korean_font()


class ModelEvaluator:
    """Comprehensive model evaluation with visualizations."""

    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        self.model, self.class_metadata = self._load_model(model_path)
        self.model.eval()
        self.class_names = self.class_metadata['class_names']
        self.num_classes = self.class_metadata['num_classes']

    def _load_model(self, model_path):
        print(f'\nLoading model from: {model_path}')
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        class_metadata = metadata_from_checkpoint(checkpoint)

        config = LSTMConfig()
        model_cfg = checkpoint.get('model_config', {})
        config.NUM_CLASSES = int(model_cfg.get('num_classes', class_metadata['num_classes']))
        config.SEQUENCE_LENGTH = int(model_cfg.get('sequence_length', config.SEQUENCE_LENGTH))
        config.INPUT_SIZE = int(model_cfg.get('input_size', config.INPUT_SIZE))
        config.HIDDEN_SIZE = int(model_cfg.get('hidden_size', config.HIDDEN_SIZE))
        config.NUM_LAYERS = int(model_cfg.get('num_layers', config.NUM_LAYERS))
        config.DROPOUT = float(model_cfg.get('dropout', config.DROPOUT))
        config.BIDIRECTIONAL = bool(model_cfg.get('bidirectional', config.BIDIRECTIONAL))

        model = PoomsaeLSTM(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)

        print('Model loaded successfully')
        print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Best val accuracy: {checkpoint.get('best_val_acc', 0):.2f}%")
        print(f"  num_classes: {class_metadata['num_classes']}")
        return model, class_metadata

    def evaluate(self, test_loader):
        print('\n' + '=' * 60)
        print('EVALUATING MODEL')
        print('=' * 60)

        all_predictions = []
        all_labels = []
        all_probabilities = []

        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0

        with torch.no_grad():
            for data, labels in tqdm(test_loader, desc='Evaluating'):
                data = data.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(data)
                loss = criterion(outputs, labels)
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, 1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                total_loss += loss.item()

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        accuracy = (all_predictions == all_labels).mean() * 100
        avg_loss = total_loss / len(test_loader)

        print('\nTest Results:')
        print(f'  Accuracy: {accuracy:.2f}%')
        print(f'  Loss: {avg_loss:.4f}')
        print(f'  Total samples: {len(all_labels)}')

        return {
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities,
            'accuracy': accuracy,
            'loss': avg_loss,
        }

    def plot_confusion_matrix(self, results, save_path=None):
        print('\nGenerating confusion matrix...')
        labels = list(range(self.num_classes))
        cm = confusion_matrix(results['labels'], results['predictions'], labels=labels)

        plt.figure(figsize=(max(10, self.num_classes * 0.6), max(8, self.num_classes * 0.6)))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'},
        )
        plt.title('Confusion Matrix', fontsize=16, pad=20)
        plt.xlabel('Predicted Movement', fontsize=12)
        plt.ylabel('True Movement', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f'Confusion matrix saved to: {save_path}')
        plt.close()
        return cm

    def analyze_per_movement_accuracy(self, results, save_path=None):
        print('\nAnalyzing per-movement accuracy...')
        labels = list(range(self.num_classes))
        cm = confusion_matrix(results['labels'], results['predictions'], labels=labels)

        per_class_stats = []
        for i in range(self.num_classes):
            true_positives = cm[i, i]
            total = cm[i, :].sum()
            accuracy = (true_positives / total) * 100 if total > 0 else 0.0
            cm_row = cm[i, :].copy()
            cm_row[i] = 0
            most_confused_with = int(cm_row.argmax()) if cm_row.size else i
            confusion_count = int(cm_row[most_confused_with]) if cm_row.size else 0
            per_class_stats.append({
                'movement': i,
                'movement_name': self.class_names[i],
                'accuracy': accuracy,
                'total_samples': int(total),
                'correct': int(true_positives),
                'most_confused_with': most_confused_with,
                'confusion_count': confusion_count,
            })

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(16, self.num_classes * 1.2), 6))
        movements = list(range(self.num_classes))
        accuracies = [s['accuracy'] for s in per_class_stats]
        colors = ['green' if acc >= 70 else 'orange' if acc >= 50 else 'red' for acc in accuracies]
        ax1.bar(movements, accuracies, color=colors, alpha=0.7)
        ax1.axhline(y=results['accuracy'], color='blue', linestyle='--', label=f'Overall: {results["accuracy"]:.1f}%')
        ax1.set_xlabel('Class Index')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Per-Class Accuracy')
        ax1.set_xticks(movements)
        ax1.set_xticklabels([self.class_names[i] for i in movements], rotation=45, ha='right')
        ax1.set_ylim(0, 105)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        totals = [s['total_samples'] for s in per_class_stats]
        ax2.bar(movements, totals, color='steelblue', alpha=0.7)
        ax2.set_xlabel('Class Index')
        ax2.set_ylabel('Sample Count')
        ax2.set_title('Samples Per Class')
        ax2.set_xticks(movements)
        ax2.set_xticklabels([self.class_names[i] for i in movements], rotation=45, ha='right')
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f'Per-class accuracy plot saved to: {save_path}')
        plt.close(fig)
        return per_class_stats

    def print_classification_report(self, results):
        print('\nClassification Report:')
        print('=' * 80)
        print(classification_report(results['labels'], results['predictions'], target_names=self.class_names, digits=3, zero_division=0))


def main():
    training_config = TrainingConfig()
    default_model_path = Paths.CHECKPOINTS_DIR / 'lstm_best.pth'
    model_path = default_model_path if default_model_path.exists() else (Paths.CHECKPOINTS_DIR / 'lstm_taegeuk1_best.pth')

    evaluator = ModelEvaluator(model_path)
    _, _, test_loader, dataset_meta = create_dataloaders(
        data_dir=Paths.WINDOWS_DIR,
        batch_size=training_config.BATCH_SIZE,
        train_split=training_config.TRAIN_SPLIT,
        val_split=training_config.VAL_SPLIT,
        num_workers=training_config.NUM_WORKERS,
        return_metadata=True,
    )

    print(f"\nDataset metadata classes: {dataset_meta['num_classes']}")
    results = evaluator.evaluate(test_loader)

    results_dir = Paths.ROOT / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    cm_path = results_dir / 'confusion_matrix.png'
    acc_path = results_dir / 'per_class_accuracy.png'
    json_path = results_dir / 'evaluation_results.json'

    evaluator.plot_confusion_matrix(results, save_path=cm_path)
    per_class_stats = evaluator.analyze_per_movement_accuracy(results, save_path=acc_path)
    evaluator.print_classification_report(results)

    serializable = {
        'accuracy': float(results['accuracy']),
        'loss': float(results['loss']),
        'num_classes': evaluator.num_classes,
        'class_names': evaluator.class_names,
        'per_class_stats': per_class_stats,
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f'\nSaved evaluation results to: {json_path}')


if __name__ == '__main__':
    main()
