"""
Model Evaluation Script with Confusion Matrix and Per-Movement Analysis

This script:
1. Loads the best trained model
2. Evaluates on test set
3. Generates confusion matrix
4. Shows per-movement accuracy
5. Identifies problematic movements
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import sys
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from models.lstm_classifier import PoomsaeLSTM
from training.dataset import create_dataloaders
from configs.lstm_config import LSTMConfig
from configs.training_config import TrainingConfig
from configs.paths import Paths


class ModelEvaluator:
    """Comprehensive model evaluation with visualizations"""

    def __init__(self, model_path, device='cuda'):
        """
        Args:
            model_path: Path to saved model checkpoint
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load model
        self.model = self._load_model(model_path)
        self.model.eval()

        # Movement names (Taegeuk 1)
        self.movement_names = [
            "01_준비자세",
            "02_아래막기",
            "03_몸통반대지르기",
            "04_아래막기",
            "05_몸통반대지르기",
            "06_아래막기",
            "07_몸통바로지르기",
            "08_몸통안막기",
            "09_몸통바로지르기",
            "10_몸통안막기",
            "11_몸통바로지르기",
            "12_아래막기",
            "13_몸통바로지르기",
            "14_올려막기",
            "15_앞차고몸통반대지르기",
            "16_뒤로돌아올려막기",
            "17_앞차고몸통반대지르기",
            "18_아래막기",
            "19_몸통지르기",
            "20_준비자세"
        ]

    def _load_model(self, model_path):
        """Load trained model from checkpoint"""
        print(f"\nLoading model from: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)

        # Create model
        config = LSTMConfig()
        model = PoomsaeLSTM(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)

        print(f"✓ Model loaded successfully")
        print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Best val accuracy: {checkpoint.get('best_val_acc', 'N/A'):.2f}%")

        return model

    def evaluate(self, test_loader):
        """
        Evaluate model on test set

        Returns:
            results: Dictionary with predictions, labels, and metrics
        """
        print("\n" + "=" * 60)
        print("EVALUATING MODEL")
        print("=" * 60)

        all_predictions = []
        all_labels = []
        all_probabilities = []

        criterion = nn.CrossEntropyLoss()
        total_loss = 0

        with torch.no_grad():
            for data, labels in tqdm(test_loader, desc="Evaluating"):
                data = data.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(data)
                loss = criterion(outputs, labels)

                # Get predictions
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, 1)

                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                total_loss += loss.item()

        # Calculate metrics
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)

        accuracy = (all_predictions == all_labels).mean() * 100
        avg_loss = total_loss / len(test_loader)

        results = {
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities,
            'accuracy': accuracy,
            'loss': avg_loss
        }

        print(f"\nTest Results:")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Total samples: {len(all_labels)}")

        return results

    def plot_confusion_matrix(self, results, save_path=None):
        """
        Create and plot confusion matrix

        Args:
            results: Results from evaluate()
            save_path: Where to save the plot (optional)
        """
        print("\nGenerating confusion matrix...")

        # Calculate confusion matrix
        cm = confusion_matrix(results['labels'], results['predictions'])

        # Calculate per-class accuracy
        class_accuracy = cm.diagonal() / cm.sum(axis=1) * 100

        # Create figure
        plt.figure(figsize=(16, 14))

        # Plot confusion matrix
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=range(1, 21),
            yticklabels=range(1, 21),
            cbar_kws={'label': 'Count'}
        )

        plt.title('Confusion Matrix - Taegeuk 1 Movement Recognition',
                  fontsize=16, pad=20)
        plt.xlabel('Predicted Movement', fontsize=12)
        plt.ylabel('True Movement', fontsize=12)

        # Add accuracy annotation
        plt.text(
            19.5, 9,
            f'Overall Accuracy: {results["accuracy"]:.2f}%',
            fontsize=14,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Confusion matrix saved to: {save_path}")

        plt.show()

        return cm

    def analyze_per_movement_accuracy(self, results, save_path=None):
        """
        Analyze and visualize per-movement accuracy

        Args:
            results: Results from evaluate()
            save_path: Where to save the plot (optional)
        """
        print("\nAnalyzing per-movement accuracy...")

        # Calculate confusion matrix
        cm = confusion_matrix(results['labels'], results['predictions'])

        # Per-movement metrics
        per_movement_stats = []

        for i in range(20):
            true_positives = cm[i, i]
            total = cm[i, :].sum()

            if total > 0:
                accuracy = (true_positives / total) * 100

                # Find most common confusion
                cm_row = cm[i, :].copy()
                cm_row[i] = 0  # Exclude correct predictions
                most_confused_with = cm_row.argmax()
                confusion_count = cm_row[most_confused_with]

                per_movement_stats.append({
                    'movement': i,
                    'accuracy': accuracy,
                    'total_samples': total,
                    'correct': true_positives,
                    'most_confused_with': most_confused_with,
                    'confusion_count': confusion_count
                })

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

        # Plot 1: Accuracy per movement
        movements = [s['movement'] + 1 for s in per_movement_stats]
        accuracies = [s['accuracy'] for s in per_movement_stats]

        colors = ['green' if acc >= 70 else 'orange' if acc >= 50 else 'red'
                  for acc in accuracies]

        bars = ax1.bar(movements, accuracies, color=colors, alpha=0.7)
        ax1.axhline(y=results['accuracy'], color='blue', linestyle='--',
                    label=f'Overall: {results["accuracy"]:.1f}%')
        ax1.set_xlabel('Movement Number', fontsize=12)
        ax1.set_ylabel('Accuracy (%)', fontsize=12)
        ax1.set_title('Per-Movement Accuracy', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 105)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{acc:.1f}%',
                     ha='center', va='bottom', fontsize=8)

        # Plot 2: Sample distribution
        sample_counts = [s['total_samples'] for s in per_movement_stats]
        ax2.bar(movements, sample_counts, color='skyblue', alpha=0.7)
        ax2.set_xlabel('Movement Number', fontsize=12)
        ax2.set_ylabel('Number of Samples', fontsize=12)
        ax2.set_title('Test Samples per Movement', fontsize=14, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Per-movement analysis saved to: {save_path}")

        plt.show()

        # Print detailed statistics
        print("\n" + "=" * 80)
        print("PER-MOVEMENT ACCURACY ANALYSIS")
        print("=" * 80)

        # Sort by accuracy (worst first)
        sorted_stats = sorted(per_movement_stats, key=lambda x: x['accuracy'])

        print("\n🔴 WORST PERFORMING MOVEMENTS:")
        for stat in sorted_stats[:5]:
            mov_num = stat['movement'] + 1
            print(f"\nMovement {mov_num}: {self.movement_names[stat['movement']]}")
            print(f"  Accuracy: {stat['accuracy']:.2f}%")
            print(f"  Correct: {stat['correct']}/{stat['total_samples']}")
            if stat['confusion_count'] > 0:
                confused_mov = stat['most_confused_with'] + 1
                print(f"  Most confused with: Movement {confused_mov} "
                      f"({stat['confusion_count']} times)")

        print("\n🟢 BEST PERFORMING MOVEMENTS:")
        for stat in sorted_stats[-5:]:
            mov_num = stat['movement'] + 1
            print(f"\nMovement {mov_num}: {self.movement_names[stat['movement']]}")
            print(f"  Accuracy: {stat['accuracy']:.2f}%")
            print(f"  Correct: {stat['correct']}/{stat['total_samples']}")

        return per_movement_stats

    def generate_classification_report(self, results):
        """Generate sklearn classification report"""
        print("\n" + "=" * 80)
        print("DETAILED CLASSIFICATION REPORT")
        print("=" * 80 + "\n")

        target_names = [f"Movement {i + 1}" for i in range(20)]

        report = classification_report(
            results['labels'],
            results['predictions'],
            target_names=target_names,
            digits=3
        )

        print(report)

        return report

    def save_results(self, results, output_dir):
        """Save evaluation results to JSON"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Prepare data for JSON (convert numpy to python types)
        json_results = {
            'accuracy': float(results['accuracy']),
            'loss': float(results['loss']),
            'total_samples': int(len(results['labels'])),
            'predictions': results['predictions'].tolist(),
            'labels': results['labels'].tolist()
        }

        # Save
        output_path = output_dir / 'evaluation_results.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2)

        print(f"\n✓ Results saved to: {output_path}")


def main():
    """Main evaluation function"""

    # Paths
    Paths.create_directories()
    results_dir = Paths.RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    # Model path
    model_path = Paths.CHECKPOINTS_DIR / 'lstm_taegeuk1_best.pth'

    if not model_path.exists():
        print(f"❌ Model not found at: {model_path}")
        print("Please train the model first!")
        return

    print("\n" + "=" * 80)
    print("POOMSAE RECOGNITION MODEL EVALUATION")
    print("=" * 80)

    # Create dataloaders
    print("\nLoading test data...")
    train_config = TrainingConfig()

    _, _, test_loader = create_dataloaders(
        data_dir=Paths.WINDOWS_DIR,
        batch_size=train_config.BATCH_SIZE,
        train_split=train_config.TRAIN_SPLIT,
        val_split=train_config.VAL_SPLIT,
        num_workers=0
    )

    # Create evaluator
    evaluator = ModelEvaluator(model_path)

    # Evaluate model
    results = evaluator.evaluate(test_loader)

    # Generate confusion matrix
    cm = evaluator.plot_confusion_matrix(
        results,
        save_path=results_dir / 'confusion_matrix.png'
    )

    # Per-movement analysis
    per_movement_stats = evaluator.analyze_per_movement_accuracy(
        results,
        save_path=results_dir / 'per_movement_accuracy.png'
    )

    # Classification report
    report = evaluator.generate_classification_report(results)

    # Save results
    evaluator.save_results(results, results_dir)

    # Final summary
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\n📊 Overall Test Accuracy: {results['accuracy']:.2f}%")
    print(f"📁 Results saved to: {results_dir}")
    print(f"\nGenerated files:")
    print(f"  - confusion_matrix.png")
    print(f"  - per_movement_accuracy.png")
    print(f"  - evaluation_results.json")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()