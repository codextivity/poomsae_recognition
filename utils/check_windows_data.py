import numpy as np
import pickle
from pathlib import Path

# Load training windows
data_path = Path('D:\All Docs\All Projects\Pycharm\poomsae_recognition\data\processed\windows')
files = list(data_path.glob('*.npz'))

all_data = []
for f in files:
    data = np.load(f)
    X = data['X']  # Shape: (N, 60, 78)
    all_data.append(X.reshape(-1, 78))

all_data = np.concatenate(all_data, axis=0)

# Calculate stats
mean = all_data.mean(axis=0)
std = all_data.std(axis=0)
std[std < 1e-6] = 1.0  # Prevent division by zero

# Save
stats = {'mean': mean, 'std': std}
with open('D:\All Docs\All Projects\Pycharm\poomsae_recognition\data\processed\keypoints\P002_keypoints.pkl', 'wb') as f:
    pickle.dump(stats, f)

print(f"✓ Saved normalization stats")
print(f"  Mean shape: {mean.shape}")
print(f"  Std shape: {std.shape}")
print(f"  Mean range: [{mean.min():.3f}, {mean.max():.3f}]")
print(f"  Std range: [{std.min():.3f}, {std.max():.3f}]")