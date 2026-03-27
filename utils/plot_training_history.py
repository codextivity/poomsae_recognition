"""Generate training-history plots from a saved history JSON file."""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))
from utils.matplotlib_korean import configure_korean_font

configure_korean_font()


def load_history(history_path):
    with open(history_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_series(epochs, train_values, val_values, ylabel, title, output_path):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_values, marker='o', label=f'Train {ylabel}')
    plt.plot(epochs, val_values, marker='o', label=f'Validation {ylabel}')
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate PNG plots from training history JSON.')
    parser.add_argument('history_path', type=Path, help='Path to training_history_*.json')
    parser.add_argument('--output-dir', type=Path, default=None, help='Directory to save the generated plots')
    args = parser.parse_args()

    history_path = args.history_path
    history = load_history(history_path)
    output_dir = args.output_dir or history_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    train_losses = [float(x) for x in history.get('train_losses', [])]
    val_losses = [float(x) for x in history.get('val_losses', [])]
    train_accs = [float(x) for x in history.get('train_accs', [])]
    val_accs = [float(x) for x in history.get('val_accs', [])]

    if not train_losses or not val_losses:
        raise ValueError('History JSON does not contain train_losses/val_losses')

    epochs = list(range(1, len(train_losses) + 1))

    loss_path = output_dir / 'train_val_loss.png'
    plot_series(epochs, train_losses, val_losses, 'Loss', 'Training vs Validation Loss', loss_path)
    print(f'[OK] Saved loss plot: {loss_path}')

    if train_accs and val_accs:
        acc_path = output_dir / 'train_val_accuracy.png'
        plot_series(epochs, train_accs, val_accs, 'Accuracy (%)', 'Training vs Validation Accuracy', acc_path)
        print(f'[OK] Saved accuracy plot: {acc_path}')
    else:
        print('[!] Accuracy history not found; skipped accuracy plot')


if __name__ == '__main__':
    main()
