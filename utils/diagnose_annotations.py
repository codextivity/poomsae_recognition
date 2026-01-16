"""
Annotation Diagnostic Tool

This script analyzes all annotation files to identify issues:
- Missing movements
- Duplicate movements
- Incorrect formatting
- Time overlap issues
"""

import json
from pathlib import Path
from collections import defaultdict
import sys

sys.path.append(str(Path(__file__).parent))
from configs.paths import Paths


class AnnotationDiagnostic:
    """Diagnose annotation file issues"""

    def __init__(self, annotations_dir):
        self.annotations_dir = Path(annotations_dir)
        self.issues = []

    def analyze_single_file(self, annotation_path):
        """Analyze a single annotation file"""
        print(f"\n{'=' * 60}")
        print(f"Analyzing: {annotation_path.name}")
        print(f"{'=' * 60}")

        # Load annotations
        try:
            with open(annotation_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ ERROR loading file: {e}")
            self.issues.append(f"{annotation_path.name}: Failed to load - {e}")
            return None

        annotations = data.get('annotations', [])

        # Basic info
        print(f"Total annotations: {len(annotations)}")

        if len(annotations) != 20:
            issue = f"❌ Expected 20 movements, found {len(annotations)}"
            print(issue)
            self.issues.append(f"{annotation_path.name}: {issue}")

        # Check format and extract movement numbers
        movement_numbers = []
        movement_times = []

        for i, ann in enumerate(annotations):
            # Check required fields
            if 'movement' not in ann:
                issue = f"❌ Annotation {i} missing 'movement' field"
                print(issue)
                self.issues.append(f"{annotation_path.name}: {issue}")
                continue

            if 'startTime' not in ann:
                issue = f"❌ Annotation {i} missing 'startTime' field"
                print(issue)
                self.issues.append(f"{annotation_path.name}: {issue}")
                continue

            # Extract movement number from format "X. Name"
            movement_str = ann['movement']

            # Try to parse movement number
            try:
                # Format: "2. 아래막기" -> extract "2"
                if '.' in movement_str:
                    number_str = movement_str.split('.')[0].strip()
                    movement_num = int(number_str)
                else:
                    # Maybe just a number
                    movement_num = int(movement_str)

                movement_numbers.append(movement_num)

            except ValueError:
                issue = f"❌ Cannot parse movement number from: '{movement_str}'"
                print(issue)
                self.issues.append(f"{annotation_path.name}: {issue}")
                continue

            # Parse time
            try:
                start_time = float(ann['startTime'])
                movement_times.append((movement_num, start_time))
            except ValueError:
                issue = f"❌ Cannot parse startTime: '{ann['startTime']}'"
                print(issue)
                self.issues.append(f"{annotation_path.name}: {issue}")

        # Analysis results
        print(f"\nMovement numbers found: {sorted(movement_numbers)}")

        # Check for completeness (should be 1-20)
        expected = set(range(1, 21))
        found = set(movement_numbers)

        missing = expected - found
        extra = found - expected

        if missing:
            issue = f"❌ Missing movements: {sorted(missing)}"
            print(issue)
            self.issues.append(f"{annotation_path.name}: {issue}")

        if extra:
            issue = f"❌ Extra/invalid movements: {sorted(extra)}"
            print(issue)
            self.issues.append(f"{annotation_path.name}: {issue}")

        # Check for duplicates
        duplicates = [num for num in found if movement_numbers.count(num) > 1]
        if duplicates:
            issue = f"❌ Duplicate movements: {sorted(set(duplicates))}"
            print(issue)
            self.issues.append(f"{annotation_path.name}: {issue}")

        # Check time ordering
        sorted_times = sorted(movement_times, key=lambda x: x[1])
        if sorted_times != movement_times:
            issue = f"⚠️  Movements not in chronological order"
            print(issue)
            # Show expected vs actual order
            print(f"   Expected order: {[x[0] for x in sorted_times]}")
            print(f"   Actual order:   {[x[0] for x in movement_times]}")

        # Check for very short/long durations
        for i in range(len(movement_times) - 1):
            curr_num, curr_time = movement_times[i]
            next_num, next_time = movement_times[i + 1]
            duration = next_time - curr_time

            if duration < 0.5:
                issue = f"⚠️  Movement {curr_num} duration very short: {duration:.2f}s"
                print(issue)
            elif duration > 10:
                issue = f"⚠️  Movement {curr_num} duration very long: {duration:.2f}s"
                print(issue)
                self.issues.append(f"{annotation_path.name}: {issue}")

        # Summary
        if missing or extra or duplicates:
            print(f"\n❌ FILE HAS ISSUES")
            return False
        else:
            print(f"\n✓ FILE OK")
            return True

    def analyze_all(self):
        """Analyze all annotation files"""
        annotation_files = sorted(self.annotations_dir.glob('*_annotations.json'))

        if not annotation_files:
            print(f"❌ No annotation files found in {self.annotations_dir}")
            return

        print(f"\n{'=' * 70}")
        print(f"ANNOTATION DIAGNOSTIC REPORT")
        print(f"{'=' * 70}")
        print(f"Found {len(annotation_files)} annotation files\n")

        ok_count = 0
        problem_count = 0

        for ann_file in annotation_files:
            result = self.analyze_single_file(ann_file)
            if result:
                ok_count += 1
            else:
                problem_count += 1

        # Overall summary
        print(f"\n{'=' * 70}")
        print(f"SUMMARY")
        print(f"{'=' * 70}")
        print(f"Total files: {len(annotation_files)}")
        print(f"✓ OK: {ok_count}")
        print(f"❌ Problems: {problem_count}")

        if self.issues:
            print(f"\n{'=' * 70}")
            print(f"ALL ISSUES FOUND ({len(self.issues)} total):")
            print(f"{'=' * 70}")
            for issue in self.issues:
                print(f"  • {issue}")

        # Aggregate statistics
        print(f"\n{'=' * 70}")
        print(f"AGGREGATE STATISTICS")
        print(f"{'=' * 70}")

        self.check_aggregate_statistics()

    def check_aggregate_statistics(self):
        """Check overall patterns across all files"""
        annotation_files = sorted(self.annotations_dir.glob('*_annotations.json'))

        all_movement_numbers = defaultdict(int)

        for ann_file in annotation_files:
            try:
                with open(ann_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                annotations = data.get('annotations', [])

                for ann in annotations:
                    movement_str = ann.get('movement', '')

                    # Parse movement number
                    try:
                        if '.' in movement_str:
                            number_str = movement_str.split('.')[0].strip()
                            movement_num = int(number_str)
                        else:
                            movement_num = int(movement_str)

                        all_movement_numbers[movement_num] += 1
                    except:
                        pass
            except:
                pass

        print("\nMovement frequency across all files:")
        for i in range(1, 21):
            count = all_movement_numbers.get(i, 0)
            expected = len(annotation_files)
            status = "✓" if count == expected else "❌"
            print(f"  {status} Movement {i:2d}: {count:3d} files (expected {expected})")

        # Check which movements are problematic
        missing_movements = [i for i in range(1, 21) if all_movement_numbers.get(i, 0) < len(annotation_files)]
        if missing_movements:
            print(f"\n❌ Movements not in all files: {missing_movements}")


def main():
    """Main diagnostic"""
    Paths.create_directories()

    diagnostic = AnnotationDiagnostic(Paths.RAW_ANNOTATIONS)
    diagnostic.analyze_all()

    print(f"\n{'=' * 70}")
    print(f"NEXT STEPS")
    print(f"{'=' * 70}")
    print(f"1. Fix any annotation files with errors")
    print(f"2. Ensure all files have exactly 20 movements (1-20)")
    print(f"3. Ensure movements are in chronological order")
    print(f"4. Re-run: python preprocessing/create_windows.py")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()