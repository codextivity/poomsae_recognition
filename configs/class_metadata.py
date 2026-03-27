"""Helpers for movement/class metadata across different poomsae."""

import json
import re
from pathlib import Path


DEFAULT_SHORT_MOVEMENT_IDS = ('6_1', '12_1', '14_1', '16_1')


def parse_movement_id(movement_str):
    """Extract movement ID like '14_2' from a movement label."""
    match = re.match(r'^\s*(\d+)_(\d+)', str(movement_str).strip())
    if match:
        return f"{int(match.group(1))}_{int(match.group(2))}"
    return str(movement_str).strip()


def _resolve_movement_name(raw_name, movement_id):
    name = str(raw_name).strip()
    return name if name else movement_id


def build_class_metadata_from_annotations(annotations):
    """Build class mapping and ordered class names from annotation records."""
    class_mapping = {}
    class_names = []

    for ann in annotations:
        movement_id = parse_movement_id(ann.get('movement', ''))
        if not movement_id:
            continue
        if movement_id in class_mapping:
            continue
        class_mapping[movement_id] = len(class_mapping)
        class_names.append(_resolve_movement_name(ann.get('movement', movement_id), movement_id))

    return class_mapping, class_names


def build_class_metadata_from_annotation_files(annotation_files):
    """Build metadata from multiple annotation JSON files."""
    class_mapping = {}
    class_names = []

    for path in sorted(Path(p) for p in annotation_files):
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        anns = payload.get('annotations', [])
        file_mapping, file_names = build_class_metadata_from_annotations(anns)
        for movement_id, class_idx in file_mapping.items():
            if movement_id in class_mapping:
                continue
            class_mapping[movement_id] = len(class_mapping)
            class_names.append(file_names[class_idx])

    return class_mapping, class_names


def invert_class_mapping(class_mapping):
    return {int(idx): movement_id for movement_id, idx in class_mapping.items()}


def resolve_class_names(class_mapping, class_names=None):
    """Return class names ordered by class index."""
    inverse = invert_class_mapping(class_mapping)
    ordered_ids = [inverse[idx] for idx in range(len(inverse))]

    if class_names and len(class_names) == len(ordered_ids):
        return list(class_names)

    return ordered_ids


def metadata_payload(class_mapping, class_names, short_movement_ids=None):
    short_class_indices = get_short_class_indices(class_mapping, short_movement_ids)
    return {
        'num_classes': len(class_mapping),
        'class_mapping': dict(class_mapping),
        'class_names': list(class_names),
        'short_movement_ids': list(short_movement_ids or DEFAULT_SHORT_MOVEMENT_IDS),
        'short_class_indices': sorted(short_class_indices),
    }


def save_class_metadata_json(output_dir, class_mapping, class_names, short_movement_ids=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = metadata_payload(class_mapping, class_names, short_movement_ids)
    path = output_dir / 'class_metadata.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def load_class_metadata_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    class_mapping = {str(k): int(v) for k, v in payload.get('class_mapping', {}).items()}
    class_names = list(payload.get('class_names', []))
    payload['class_mapping'] = class_mapping
    payload['class_names'] = resolve_class_names(class_mapping, class_names)
    payload['num_classes'] = int(payload.get('num_classes', len(class_mapping)))
    payload['short_class_indices'] = [int(x) for x in payload.get('short_class_indices', [])]
    payload['short_movement_ids'] = list(payload.get('short_movement_ids', DEFAULT_SHORT_MOVEMENT_IDS))
    return payload


def load_dataset_class_metadata(windows_dir):
    windows_dir = Path(windows_dir)
    metadata_path = windows_dir / 'class_metadata.json'
    if metadata_path.exists():
        return load_class_metadata_json(metadata_path)

    npz_files = sorted(windows_dir.glob('*_windows.npz'))
    for npz_file in npz_files:
        payload = load_class_metadata_from_npz(npz_file)
        if payload is not None:
            return payload

    return None


def load_class_metadata_from_npz(npz_path):
    import numpy as np

    data = np.load(npz_path, allow_pickle=True)
    if 'class_mapping_json' not in data:
        return None

    mapping_json = str(data['class_mapping_json'].item())
    class_mapping = {str(k): int(v) for k, v in json.loads(mapping_json).items()}
    class_names = [str(x) for x in data['class_names'].tolist()] if 'class_names' in data else []
    short_indices = [int(x) for x in data['short_class_indices'].tolist()] if 'short_class_indices' in data else []
    short_ids = [str(x) for x in data['short_movement_ids'].tolist()] if 'short_movement_ids' in data else list(DEFAULT_SHORT_MOVEMENT_IDS)

    return {
        'num_classes': int(data['num_classes'].item()) if 'num_classes' in data else len(class_mapping),
        'class_mapping': class_mapping,
        'class_names': resolve_class_names(class_mapping, class_names),
        'short_class_indices': short_indices,
        'short_movement_ids': short_ids,
    }


def metadata_from_checkpoint(checkpoint):
    class_mapping = checkpoint.get('class_mapping') or {}
    class_mapping = {str(k): int(v) for k, v in class_mapping.items()}

    class_names = checkpoint.get('class_names') or []
    if not class_mapping:
        model_cfg = checkpoint.get('model_config', {})
        num_classes = int(checkpoint.get('num_classes', model_cfg.get('num_classes', len(class_names) or 0)))
        if class_names and len(class_names) == num_classes:
            class_mapping = {str(name): idx for idx, name in enumerate(class_names)}
        elif num_classes > 0:
            class_mapping = {f'class_{idx}': idx for idx in range(num_classes)}

    class_names = resolve_class_names(class_mapping, class_names)
    short_indices = checkpoint.get('short_class_indices')
    if short_indices is None:
        short_indices = sorted(get_short_class_indices(class_mapping))

    return {
        'num_classes': len(class_mapping),
        'class_mapping': class_mapping,
        'class_names': class_names,
        'short_class_indices': [int(x) for x in short_indices],
        'short_movement_ids': list(checkpoint.get('short_movement_ids', DEFAULT_SHORT_MOVEMENT_IDS)),
    }


def get_short_class_indices(class_mapping, short_movement_ids=None):
    short_ids = short_movement_ids or DEFAULT_SHORT_MOVEMENT_IDS
    return {int(class_mapping[mov_id]) for mov_id in short_ids if mov_id in class_mapping}


def resolve_annotation_file(annotations_dir, base_name):
    annotations_dir = Path(annotations_dir)
    candidates = [
        annotations_dir / f'{base_name}.json',
        annotations_dir / f'{base_name}_annotations.json',
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = list(annotations_dir.rglob(f'{base_name}.json'))
    if matches:
        return sorted(matches)[0]

    matches = list(annotations_dir.rglob(f'{base_name}_annotations.json'))
    if matches:
        return sorted(matches)[0]

    return None
