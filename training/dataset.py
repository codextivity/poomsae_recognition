"""PyTorch dataset helpers for windowed poomsae training."""

import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from configs.class_metadata import get_short_class_indices, load_class_metadata_from_npz, load_dataset_class_metadata
from configs.policy_config import PolicyConfig


class PoomsaeDataset(Dataset):
    """Dataset for loading windowed poomsae keypoint sequences."""

    def __init__(self, windows_files, normalize=True, augment=False):
        self.normalize = normalize
        self.augment = augment

        self.X = []
        self.y = []
        self.metadata = []
        self.class_mapping = {}
        self.class_names = []
        self.num_classes = 0
        self.short_class_indices = set()

        expected_meta = None

        for file_path in windows_files:
            data = np.load(file_path, allow_pickle=True)
            self.X.append(data['X'])
            self.y.append(data['y'])

            file_meta = load_class_metadata_from_npz(file_path)
            if file_meta is not None:
                if expected_meta is None:
                    expected_meta = file_meta
                else:
                    if file_meta['class_mapping'] != expected_meta['class_mapping']:
                        raise ValueError(f'Class mapping mismatch in {file_path}')
                    if file_meta['class_names'] != expected_meta['class_names']:
                        raise ValueError(f'Class names mismatch in {file_path}')

            num_samples = len(data['y'])
            for i in range(num_samples):
                meta = {'source_file': str(file_path)}
                if 'movement_names' in data:
                    meta['movement_name'] = data['movement_names'][i]
                if 'movement_ids' in data:
                    meta['movement_id'] = data['movement_ids'][i]
                if 'quality' in data:
                    meta['quality'] = data['quality'][i]
                self.metadata.append(meta)

        self.X = np.concatenate(self.X, axis=0)
        self.y = np.concatenate(self.y, axis=0)

        if expected_meta is not None:
            self.class_mapping = expected_meta['class_mapping']
            self.class_names = expected_meta['class_names']
            self.num_classes = int(expected_meta['num_classes'])
            self.short_class_indices = set(expected_meta.get('short_class_indices', []))
        else:
            unique_labels = sorted({int(y) for y in self.y.tolist()})
            self.class_mapping = {f'class_{idx}': idx for idx in unique_labels}
            self.class_names = [f'class_{idx}' for idx in unique_labels]
            self.num_classes = len(unique_labels)
            self.short_class_indices = get_short_class_indices(self.class_mapping)

        print(f'Loaded {len(self)} samples from {len(windows_files)} files')
        print(f'  X shape: {self.X.shape}')
        print(f'  y shape: {self.y.shape}')
        print(f'  num_classes: {self.num_classes}')

        if self.normalize:
            self._calculate_stats()

    def _calculate_stats(self):
        self.mean = np.mean(self.X, axis=(0, 1))
        self.std = np.std(self.X, axis=(0, 1)) + 1e-8
        print('  Normalization stats calculated')
        print(f'    Mean shape: {self.mean.shape}')
        print(f'    Std shape: {self.std.shape}')

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        y = self.y[idx]

        if self.normalize:
            x = (x - self.mean) / self.std

        if self.augment:
            x = self._augment(x)

        return torch.FloatTensor(x), torch.LongTensor([y]).squeeze()

    def _augment(self, x):
        seq_len, num_features = x.shape

        if np.random.random() < 0.3:
            noise_level = np.random.uniform(0.005, 0.02)
            x = x + np.random.normal(0, noise_level, x.shape)

        if np.random.random() < 0.4:
            speed_factor = np.random.uniform(0.8, 1.2)
            new_len = int(seq_len * speed_factor)
            if new_len > 2:
                old_indices = np.linspace(0, seq_len - 1, seq_len)
                new_indices = np.linspace(0, seq_len - 1, new_len)
                x_warped = np.zeros((new_len, num_features))
                for f_idx in range(num_features):
                    x_warped[:, f_idx] = np.interp(new_indices, old_indices, x[:, f_idx])
                if new_len != seq_len:
                    final_indices = np.linspace(0, new_len - 1, seq_len)
                    x_final = np.zeros((seq_len, num_features))
                    for f_idx in range(num_features):
                        x_final[:, f_idx] = np.interp(final_indices, np.arange(new_len), x_warped[:, f_idx])
                    x = x_final

        if np.random.random() < 0.2:
            x = np.roll(x, np.random.randint(-3, 4), axis=0)

        if np.random.random() < 0.25:
            scale = np.random.uniform(0.9, 1.1)
            x_reshaped = x.reshape(seq_len, 26, 3)
            x_reshaped[:, :, :2] *= scale
            x = x_reshaped.reshape(seq_len, num_features)

        if np.random.random() < 0.2:
            x = self._rotate_skeleton(x, np.random.uniform(-15, 15))

        if np.random.random() < 0.15:
            x_reshaped = x.reshape(seq_len, 26, 3)
            num_drop = np.random.randint(1, 4)
            drop_indices = np.random.choice(26, num_drop, replace=False)
            x_reshaped[:, drop_indices, :] = 0
            x = x_reshaped.reshape(seq_len, num_features)

        return x

    def _horizontal_flip(self, x):
        seq_len = x.shape[0]
        x_reshaped = x.reshape(seq_len, 26, 3)
        x_reshaped[:, :, 0] = -x_reshaped[:, :, 0]
        swap_pairs = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (20, 21), (22, 23), (24, 25)]
        for left, right in swap_pairs:
            x_reshaped[:, [left, right], :] = x_reshaped[:, [right, left], :]
        return x_reshaped.reshape(seq_len, -1)

    def _rotate_skeleton(self, x, angle_degrees):
        seq_len = x.shape[0]
        x_reshaped = x.reshape(seq_len, 26, 3)
        angle_rad = np.radians(angle_degrees)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        x_coords = x_reshaped[:, :, 0].copy()
        y_coords = x_reshaped[:, :, 1].copy()
        x_reshaped[:, :, 0] = x_coords * cos_a - y_coords * sin_a
        x_reshaped[:, :, 1] = x_coords * sin_a + y_coords * cos_a
        return x_reshaped.reshape(seq_len, -1)

    def get_class_distribution(self):
        unique, counts = np.unique(self.y, return_counts=True)
        return dict(zip(unique, counts))


class PoomsaeSubset(Dataset):
    """View over a base dataset with independent augmentation behavior."""

    def __init__(self, base_dataset, indices, augment=False):
        self.base_dataset = base_dataset
        self.indices = list(int(i) for i in indices)
        self.augment = bool(augment)
        self.metadata = [self.base_dataset.metadata[i] for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        x = self.base_dataset.X[base_idx].copy()
        y = int(self.base_dataset.y[base_idx])

        if self.base_dataset.normalize:
            x = (x - self.base_dataset.mean) / self.base_dataset.std

        if self.augment:
            x = self.base_dataset._augment(x)

        return torch.FloatTensor(x), torch.LongTensor([y]).squeeze()


def _split_list_by_ratio(items, ratios):
    total = len(items)
    if total == 0:
        return [], [], []

    train_ratio, val_ratio, test_ratio = ratios
    raw_sizes = [total * train_ratio, total * val_ratio, total * test_ratio]
    sizes = [int(math.floor(x)) for x in raw_sizes]
    remainder = total - sum(sizes)

    fractional_order = sorted(
        range(len(raw_sizes)),
        key=lambda idx: raw_sizes[idx] - sizes[idx],
        reverse=True,
    )
    for idx in fractional_order[:remainder]:
        sizes[idx] += 1

    start = 0
    splits = []
    for size in sizes:
        splits.append(items[start:start + size])
        start += size
    return tuple(splits)


def create_dataloaders(data_dir, batch_size=32, train_split=0.7, val_split=0.15,
                       num_workers=4, seed=42, oversample_short=True, return_metadata=False):
    """Create train, validation, and test dataloaders."""
    data_dir = Path(data_dir)
    PolicyConfig.apply_profile()

    window_files = sorted(data_dir.glob('*_windows.npz'))
    if not window_files:
        raise ValueError(f'No window files found in {data_dir}')

    print(f'\nFound {len(window_files)} window files:')
    for file_path in window_files:
        print(f'  - {file_path.name}')

    full_dataset = PoomsaeDataset(window_files, normalize=True, augment=False)
    dataset_meta = load_dataset_class_metadata(data_dir) or {
        'class_mapping': full_dataset.class_mapping,
        'class_names': full_dataset.class_names,
        'num_classes': full_dataset.num_classes,
        'short_class_indices': sorted(full_dataset.short_class_indices),
        'short_movement_ids': list(PolicyConfig.SHORT_MOVEMENT_IDS),
    }
    PolicyConfig.SHORT_CLASS_INDICES = set(dataset_meta.get('short_class_indices', []))

    source_to_indices = {}
    for idx, meta in enumerate(full_dataset.metadata):
        source_to_indices.setdefault(meta['source_file'], []).append(idx)

    source_files = sorted(source_to_indices)
    rng = np.random.default_rng(seed)
    rng.shuffle(source_files)

    test_split = 1.0 - train_split - val_split
    if test_split <= 0:
        raise ValueError('train_split + val_split must be < 1.0')

    train_sources, val_sources, test_sources = _split_list_by_ratio(
        source_files,
        (train_split, val_split, test_split),
    )

    train_indices = [idx for source in train_sources for idx in source_to_indices[source]]
    val_indices = [idx for source in val_sources for idx in source_to_indices[source]]
    test_indices = [idx for source in test_sources for idx in source_to_indices[source]]

    train_dataset = PoomsaeSubset(full_dataset, train_indices, augment=True)
    val_dataset = PoomsaeSubset(full_dataset, val_indices, augment=False)
    test_dataset = PoomsaeSubset(full_dataset, test_indices, augment=False)

    sampler = None
    shuffle = True
    if oversample_short:
        train_labels = full_dataset.y[train_indices]
        train_metadata = [full_dataset.metadata[i] for i in train_indices]

        sample_weights = np.ones(len(train_labels))
        short_indices = set(dataset_meta.get('short_class_indices', []))
        for i, label in enumerate(train_labels):
            if int(label) in short_indices:
                sample_weights[i] *= float(PolicyConfig.SHORT_CLASS_WEIGHT_MULTIPLIER)
            if PolicyConfig.USE_QUALITY_AWARE_SAMPLING:
                quality = str(train_metadata[i].get('quality', 'none')).strip().lower()
                sample_weights[i] *= float(PolicyConfig.QUALITY_WEIGHT_MULTIPLIERS.get(quality, 1.0))

        sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_labels), replacement=True)
        shuffle = False

        short_count = sum(1 for label in train_labels if int(label) in short_indices)
        print('\n[OK] Oversampling enabled for short movements')
        print(f'  Policy profile: {PolicyConfig.PROFILE}')
        print(f'  Short movement samples: {short_count} ({100 * short_count / len(train_labels):.1f}%)')
        print(f'  Short class multiplier: {PolicyConfig.SHORT_CLASS_WEIGHT_MULTIPLIER}')
        print(f'  Short classes: {sorted(short_indices)}')
        if PolicyConfig.USE_QUALITY_AWARE_SAMPLING:
            print(f'  Quality-aware sampling: ON ({PolicyConfig.QUALITY_WEIGHT_MULTIPLIERS})')
        else:
            print('  Quality-aware sampling: OFF')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print('\nDataset splits:')
    print(f'  Train: {len(train_dataset)} samples ({train_split * 100:.0f}%)')
    print(f'  Val:   {len(val_dataset)} samples ({val_split * 100:.0f}%)')
    print(f'  Test:  {len(test_dataset)} samples ({(1 - train_split - val_split) * 100:.0f}%)')
    print(f'  Source videos: train={len(train_sources)}, val={len(val_sources)}, test={len(test_sources)}')

    print('\nClass distribution:')
    inverse = {idx: mov_id for mov_id, idx in dataset_meta['class_mapping'].items()}
    class_dist = full_dataset.get_class_distribution()
    for class_id, count in sorted(class_dist.items()):
        class_id = int(class_id)
        mov_id = inverse.get(class_id, f'class_{class_id}')
        print(f'  {class_id}: {mov_id} -> {count} samples')

    if return_metadata:
        return train_loader, val_loader, test_loader, dataset_meta
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    windows_dir = 'data/processed/windows'
    if not Path(windows_dir).exists():
        print(f'Error: Directory {windows_dir} does not exist')
        raise SystemExit(1)

    train_loader, val_loader, test_loader, metadata = create_dataloaders(
        data_dir=windows_dir,
        batch_size=16,
        train_split=0.7,
        val_split=0.15,
        return_metadata=True,
    )

    print(f'\nLoaded metadata: {metadata}')
    print('\nTesting batch loading...')
    for x_batch, y_batch in train_loader:
        print(f'  X: {x_batch.shape}')
        print(f'  y: {y_batch.shape}')
        print(f'  y values: {y_batch[:5]}')
        break

    print('\nDataset test successful!')
