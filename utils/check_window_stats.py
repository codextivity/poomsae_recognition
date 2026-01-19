"""
Check Window Statistics

Analyzes the generated windows to see:
- Total windows per movement
- Class balance
- Which movements have most/least samples
"""

import numpy as np
from pathlib import Path
from collections import Counter


def analyze_windows():
    """Analyze window statistics"""

    # Path to windows
    windows_dir = Path('D:\All Docs\All Projects\Pycharm\poomsae_recognition\data\processed\windows')

    if not windows_dir.exists():
        print(f"❌ Windows directory not found: {windows_dir}")
        return

    window_files = list(windows_dir.glob('*.npz'))

    if not window_files:
        print(f"❌ No window files found in {windows_dir}")
        return

    print(f"\n{'=' * 70}")
    print(f"WINDOW STATISTICS")
    print(f"{'=' * 70}\n")

    print(f"Total window files: {len(window_files)}")

    # Collect all labels
    all_labels = []
    total_windows = 0

    for file in window_files:
        data = np.load(file)
        y = data['y']
        all_labels.extend(y)
        total_windows += len(y)

    print(f"Total windows: {total_windows}")

    # Count per movement
    label_counts = Counter(all_labels)

    # Movement names
    movement_names = [
        "01_준비자세", "02_아래막기", "03_몸통반대지르기", "04_아래막기",
        "05_몸통반대지르기", "06_아래막기", "07_몸통바로지르기", "08_몸통안막기",
        "09_몸통바로지르기", "10_몸통안막기", "11_몸통바로지르기", "12_아래막기",
        "13_몸통바로지르기", "14_올려막기", "15_앞차고몸통반대지르기", "16_뒤로돌아올려막기",
        "17_앞차고몸통반대지르기", "18_아래막기", "19_몸통지르기", "20_준비자세",
    ]

    # Display statistics
    print(f"\n{'=' * 70}")
    print(f"DISTRIBUTION BY MOVEMENT")
    print(f"{'=' * 70}\n")
    print(f"{'Movement':<30} {'Count':>10} {'Percentage':>12}")
    print(f"{'-' * 70}")

    for i in range(20):
        count = label_counts.get(i, 0)
        percentage = (count / total_windows * 100) if total_windows > 0 else 0
        status = "✅" if count > 0 else "❌"
        print(f"{movement_names[i]:<30} {count:>10} {percentage:>11.1f}% {status}")

    # Summary statistics
    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}\n")

    counts = [label_counts.get(i, 0) for i in range(20)]
    non_zero_counts = [c for c in counts if c > 0]

    print(f"Classes present: {len(non_zero_counts)}/20")
    print(f"Min samples: {min(counts)}")
    print(f"Max samples: {max(counts)}")
    print(f"Mean samples: {np.mean(counts):.1f}")
    print(f"Std samples: {np.std(counts):.1f}")

    # Balance ratio
    if max(counts) > 0:
        balance_ratio = min(non_zero_counts) / max(counts)
        print(f"Balance ratio: {balance_ratio:.2f} (1.0 = perfect balance)")

    # Missing movements
    missing = [i for i in range(20) if label_counts.get(i, 0) == 0]
    if missing:
        print(f"\n⚠️  WARNING: Missing movements!")
        for m in missing:
            print(f"   - Movement {m + 1}: {movement_names[m]}")

    # Check for extreme imbalance
    if max(counts) > 0 and min(non_zero_counts) / max(counts) < 0.3:
        print(f"\n⚠️  WARNING: Severe class imbalance detected!")
        print(f"   Consider data augmentation or resampling")

    print(f"\n{'=' * 70}\n")


if __name__ == "__main__":
    analyze_windows()