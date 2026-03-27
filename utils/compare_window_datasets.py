"""Compare old/new window datasets with focus on selected movement IDs."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


DEFAULT_FOCUS = ['5_1', '6_1', '7_1', '11_1', '12_1', '13_1']
DEFAULT_TRANSITIONS = [
    ('5_1', '6_1'),
    ('6_1', '7_1'),
    ('11_1', '12_1'),
    ('12_1', '13_1'),
    ('5_1', '12_1'),
    ('12_1', '6_1'),
    ('12_1', '7_1'),
]


def find_window_files(windows_dir: Path):
    return sorted(Path(windows_dir).glob('*_windows.npz'))


def load_npz(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)
    movement_ids = [str(x) for x in data['movement_ids'].tolist()] if 'movement_ids' in data else []
    quality = [str(x) for x in data['quality'].tolist()] if 'quality' in data else []
    percentage = [float(x) for x in data['percentage'].tolist()] if 'percentage' in data else []
    return {
        'path': npz_path,
        'stem': npz_path.stem.replace('_windows', ''),
        'movement_ids': movement_ids,
        'quality': quality,
        'percentage': percentage,
        'count': len(movement_ids),
    }


def compress_sequence(seq):
    compressed = []
    for item in seq:
        if not compressed or compressed[-1] != item:
            compressed.append(item)
    return compressed


def aggregate_dataset(npz_records):
    by_movement = defaultdict(lambda: {
        'count': 0,
        'quality': Counter(),
        'percentage_sum': 0.0,
        'percentage_count': 0,
    })
    transition_counts = Counter()

    total_windows = 0
    file_count = 0

    for rec in npz_records:
        file_count += 1
        total_windows += rec['count']
        movement_ids = rec['movement_ids']
        qualities = rec['quality']
        percentages = rec['percentage']

        for idx, mov_id in enumerate(movement_ids):
            stats = by_movement[mov_id]
            stats['count'] += 1
            if idx < len(qualities):
                stats['quality'][qualities[idx]] += 1
            if idx < len(percentages):
                stats['percentage_sum'] += percentages[idx]
                stats['percentage_count'] += 1

        compressed = compress_sequence(movement_ids)
        for a, b in zip(compressed, compressed[1:]):
            transition_counts[(a, b)] += 1

    summary = {
        'file_count': file_count,
        'total_windows': total_windows,
        'by_movement': {},
        'transitions': transition_counts,
    }

    for mov_id, stats in by_movement.items():
        mean_pct = stats['percentage_sum'] / stats['percentage_count'] if stats['percentage_count'] else 0.0
        summary['by_movement'][mov_id] = {
            'count': stats['count'],
            'quality': dict(stats['quality']),
            'mean_percentage': mean_pct,
        }

    return summary


def select_common(records_a, records_b):
    map_a = {rec['stem']: rec for rec in records_a}
    map_b = {rec['stem']: rec for rec in records_b}
    common = sorted(set(map_a) & set(map_b))
    return [map_a[k] for k in common], [map_b[k] for k in common], common


def compare_summaries(old_summary, new_summary, focus_ids, transitions):
    rows = []
    for mov_id in focus_ids:
        old_stats = old_summary['by_movement'].get(mov_id, {'count': 0, 'quality': {}, 'mean_percentage': 0.0})
        new_stats = new_summary['by_movement'].get(mov_id, {'count': 0, 'quality': {}, 'mean_percentage': 0.0})
        rows.append({
            'movement_id': mov_id,
            'old_count': old_stats['count'],
            'new_count': new_stats['count'],
            'delta_count': new_stats['count'] - old_stats['count'],
            'old_mean_percentage': round(old_stats['mean_percentage'], 2),
            'new_mean_percentage': round(new_stats['mean_percentage'], 2),
            'old_quality': old_stats['quality'],
            'new_quality': new_stats['quality'],
        })

    transition_rows = []
    for pair in transitions:
        transition_rows.append({
            'transition': f'{pair[0]} -> {pair[1]}',
            'old_count': int(old_summary['transitions'].get(pair, 0)),
            'new_count': int(new_summary['transitions'].get(pair, 0)),
            'delta_count': int(new_summary['transitions'].get(pair, 0) - old_summary['transitions'].get(pair, 0)),
        })

    return rows, transition_rows


def main():
    parser = argparse.ArgumentParser(description='Compare two window datasets.')
    parser.add_argument('old_dir', type=Path)
    parser.add_argument('new_dir', type=Path)
    parser.add_argument('--output', type=Path, default=None, help='Optional JSON output path')
    parser.add_argument('--focus', nargs='*', default=DEFAULT_FOCUS, help='Movement IDs to focus on')
    args = parser.parse_args()

    old_records = [load_npz(p) for p in find_window_files(args.old_dir)]
    new_records = [load_npz(p) for p in find_window_files(args.new_dir)]

    old_common, new_common, common_files = select_common(old_records, new_records)
    old_summary = aggregate_dataset(old_common)
    new_summary = aggregate_dataset(new_common)

    focus_rows, transition_rows = compare_summaries(old_summary, new_summary, args.focus, DEFAULT_TRANSITIONS)

    report = {
        'old_dir': str(args.old_dir),
        'new_dir': str(args.new_dir),
        'common_file_count': len(common_files),
        'common_files': common_files,
        'old_summary': {
            'file_count': old_summary['file_count'],
            'total_windows': old_summary['total_windows'],
        },
        'new_summary': {
            'file_count': new_summary['file_count'],
            'total_windows': new_summary['total_windows'],
        },
        'focus_rows': focus_rows,
        'transition_rows': transition_rows,
    }

    print('=' * 72)
    print('WINDOW DATASET COMPARISON')
    print('=' * 72)
    print(f'Old dir: {args.old_dir}')
    print(f'New dir: {args.new_dir}')
    print(f'Common files: {len(common_files)}')
    print(f'Old total windows (common files): {old_summary["total_windows"]}')
    print(f'New total windows (common files): {new_summary["total_windows"]}')
    print('=' * 72)
    print(f'{"ID":<8} {"Old":>8} {"New":>8} {"Delta":>8} {"OldPct":>10} {"NewPct":>10}')
    print('-' * 72)
    for row in focus_rows:
        print(
            f'{row["movement_id"]:<8} '
            f'{row["old_count"]:>8} {row["new_count"]:>8} {row["delta_count"]:>8} '
            f'{row["old_mean_percentage"]:>10.2f} {row["new_mean_percentage"]:>10.2f}'
        )
    print('-' * 72)
    print('Transition counts (run-level compressed sequence):')
    for row in transition_rows:
        print(f'  {row["transition"]:<16} old={row["old_count"]:>4} new={row["new_count"]:>4} delta={row["delta_count"]:>4}')

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f'\nSaved report: {args.output}')


if __name__ == '__main__':
    main()
