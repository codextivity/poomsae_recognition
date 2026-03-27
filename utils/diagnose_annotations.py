"""Validate annotation JSON files for the current start/end boundary schema."""

import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from configs.class_metadata import parse_movement_id
from configs.paths import Paths


class AnnotationDiagnostic:
    """Diagnose annotation file issues for explicit start/end annotations."""

    REQUIRED_FIELDS = ('movement', 'startTime', 'frame', 'endTime', 'endFrame')

    def __init__(self, annotations_dir):
        self.annotations_dir = Path(annotations_dir)
        self.issues = []

    def _load_files(self):
        files = sorted(self.annotations_dir.glob('*.json'))
        if not files:
            files = sorted(self.annotations_dir.glob('*_annotations.json'))
        return files

    def _infer_fps(self, data):
        total_duration = data.get('totalDuration')
        annotations = data.get('annotations', [])
        if not total_duration or not annotations:
            return None
        last_end_frame = max(int(float(ann.get('endFrame', 0) or 0)) for ann in annotations)
        if total_duration <= 0 or last_end_frame <= 0:
            return None
        return last_end_frame / float(total_duration)

    def analyze_single_file(self, annotation_path):
        print(f"\n{'=' * 70}")
        print(f"Analyzing: {annotation_path.name}")
        print(f"{'=' * 70}")

        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as exc:
            issue = f'Failed to load: {exc}'
            print(f'[ERROR] {issue}')
            self.issues.append(f'{annotation_path.name}: {issue}')
            return False

        annotations = data.get('annotations', [])
        expected_movements = int(data.get('expectedMovements', data.get('totalMovements', len(annotations))))
        actual_movements = len(annotations)
        print(f'Expected movements: {expected_movements}')
        print(f'Actual annotations: {actual_movements}')

        file_ok = True
        if actual_movements != expected_movements:
            issue = f'expected {expected_movements} annotations, found {actual_movements}'
            print(f'[ERROR] {issue}')
            self.issues.append(f'{annotation_path.name}: {issue}')
            file_ok = False

        fps_est = self._infer_fps(data)
        if fps_est is not None:
            print(f'Estimated FPS from annotations: {fps_est:.3f}')

        movement_ids = []
        prev_end_time = None
        prev_end_frame = None

        for idx, ann in enumerate(annotations):
            prefix = f'annotation {idx + 1}'
            for field in self.REQUIRED_FIELDS:
                if field not in ann:
                    issue = f'{prefix} missing {field}'
                    print(f'[ERROR] {issue}')
                    self.issues.append(f'{annotation_path.name}: {issue}')
                    file_ok = False
                    continue

            movement_raw = ann.get('movement', '')
            movement_id = parse_movement_id(movement_raw)
            movement_ids.append(movement_id)

            try:
                start_time = float(ann['startTime'])
                end_time = float(ann['endTime'])
                start_frame = int(float(ann['frame']))
                end_frame = int(float(ann['endFrame']))
            except Exception as exc:
                issue = f'{prefix} has invalid numeric field: {exc}'
                print(f'[ERROR] {issue}')
                self.issues.append(f'{annotation_path.name}: {issue}')
                file_ok = False
                continue

            if not movement_id or '_' not in movement_id:
                issue = f"{prefix} invalid movement format: {movement_raw}"
                print(f'[ERROR] {issue}')
                self.issues.append(f'{annotation_path.name}: {issue}')
                file_ok = False

            if end_time < start_time:
                issue = f'{prefix} endTime < startTime ({end_time} < {start_time})'
                print(f'[ERROR] {issue}')
                self.issues.append(f'{annotation_path.name}: {issue}')
                file_ok = False

            if end_frame < start_frame:
                issue = f'{prefix} endFrame < frame ({end_frame} < {start_frame})'
                print(f'[ERROR] {issue}')
                self.issues.append(f'{annotation_path.name}: {issue}')
                file_ok = False

            if prev_end_time is not None and start_time < prev_end_time:
                issue = f'{prefix} overlaps previous movement in time ({start_time:.3f} < {prev_end_time:.3f})'
                print(f'[ERROR] {issue}')
                self.issues.append(f'{annotation_path.name}: {issue}')
                file_ok = False
            elif prev_end_time is not None and start_time > prev_end_time:
                gap = start_time - prev_end_time
                print(f'[WARN] {prefix} has a time gap of {gap:.3f}s from previous movement')

            if prev_end_frame is not None and start_frame < prev_end_frame:
                issue = f'{prefix} overlaps previous movement in frames ({start_frame} < {prev_end_frame})'
                print(f'[ERROR] {issue}')
                self.issues.append(f'{annotation_path.name}: {issue}')
                file_ok = False
            elif prev_end_frame is not None and start_frame > prev_end_frame:
                gap_frames = start_frame - prev_end_frame
                print(f'[WARN] {prefix} has a frame gap of {gap_frames} from previous movement')

            if fps_est is not None:
                start_err = abs(start_time * fps_est - start_frame)
                end_err = abs(end_time * fps_est - end_frame)
                if start_err > 1.5:
                    issue = f'{prefix} startTime/frame mismatch ({start_err:.2f} frames)'
                    print(f'[WARN] {issue}')
                if end_err > 1.5:
                    issue = f'{prefix} endTime/endFrame mismatch ({end_err:.2f} frames)'
                    print(f'[WARN] {issue}')

            prev_end_time = end_time
            prev_end_frame = end_frame

        unique_ids = []
        seen = set()
        for movement_id in movement_ids:
            if movement_id not in seen:
                seen.add(movement_id)
                unique_ids.append(movement_id)

        print(f'Unique movement labels: {len(unique_ids)}')
        print(f'Movement order: {unique_ids}')

        duplicates = sorted({movement_id for movement_id in movement_ids if movement_ids.count(movement_id) > 1})
        if duplicates:
            print(f'[WARN] Duplicate movement IDs present: {duplicates}')

        if file_ok:
            print('[OK] File passed required validation')
        else:
            print('[FAIL] File has blocking issues')

        return file_ok

    def analyze_all(self):
        annotation_files = self._load_files()
        if not annotation_files:
            print(f'No annotation files found in {self.annotations_dir}')
            return False

        print(f"\n{'=' * 70}")
        print('ANNOTATION DIAGNOSTIC REPORT')
        print(f"{'=' * 70}")
        print(f'Found {len(annotation_files)} annotation files')

        ok_count = 0
        fail_count = 0
        for annotation_file in annotation_files:
            if self.analyze_single_file(annotation_file):
                ok_count += 1
            else:
                fail_count += 1

        print(f"\n{'=' * 70}")
        print('SUMMARY')
        print(f"{'=' * 70}")
        print(f'Total files: {len(annotation_files)}')
        print(f'OK: {ok_count}')
        print(f'FAIL: {fail_count}')
        if self.issues:
            print('\nBlocking issues:')
            for issue in self.issues:
                print(f'  - {issue}')

        return fail_count == 0


def main():
    diagnostic = AnnotationDiagnostic(Paths.RAW_ANNOTATIONS)
    success = diagnostic.analyze_all()
    raise SystemExit(0 if success else 1)


if __name__ == '__main__':
    main()
