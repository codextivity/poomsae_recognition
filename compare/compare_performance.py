# """
# Compare Student Performance with Reference
#
# Usage:
#     python compare_performance.py \
#         --student-video P001.mp4 \
#         --student-segments P001_segments.csv \
#         --reference references/master_reference.pkl \
#         --output results/comparison/P001_comparison.json
# """
#
# import numpy as np
# import pandas as pd
# import json
# from pathlib import Path
# from scipy.spatial.distance import euclidean
# from scipy.spatial import procrustes
# from dtaidistance import dtw
# import pickle
#
#
# def convert_to_serializable(obj):
#     """Convert numpy types to Python native types for JSON serialization"""
#     if isinstance(obj, np.integer):
#         return int(obj)
#     elif isinstance(obj, np.floating):
#         return float(obj)
#     elif isinstance(obj, np.ndarray):
#         return obj.tolist()
#     elif isinstance(obj, dict):
#         return {key: convert_to_serializable(value) for key, value in obj.items()}
#     elif isinstance(obj, list):
#         return [convert_to_serializable(item) for item in obj]
#     else:
#         return obj
#
#
# class PerformanceComparator:
#     """Compare student performance with master reference"""
#
#     def __init__(self, reference_path):
#         """Load reference data"""
#         with open(reference_path, 'rb') as f:
#             self.reference = pickle.load(f)
#
#         print(f"✓ Loaded reference: {self.reference['video_name']}")
#         print(f"  Movements: {len(self.reference['movements'])}")
#
#     def calculate_temporal_iou(self, student_start, student_end, ref_start, ref_end):
#         """Calculate Intersection over Union for temporal alignment"""
#         intersection_start = max(student_start, ref_start)
#         intersection_end = min(student_end, ref_end)
#
#         if intersection_start >= intersection_end:
#             return 0.0
#
#         intersection = intersection_end - intersection_start
#         union = max(student_end, ref_end) - min(student_start, ref_start)
#
#         return intersection / union if union > 0 else 0.0
#
#     def calculate_pose_similarity(self, pose1, pose2):
#         """
#         Calculate pose similarity using Procrustes distance
#
#         Returns: similarity score (0-100, higher is better)
#         """
#         # Extract coordinates (ignore confidence)
#         coords1 = pose1[:, :2]
#         coords2 = pose2[:, :2]
#
#         # Filter out zero/invalid keypoints
#         valid_mask = (np.linalg.norm(coords1, axis=1) > 1e-6) & \
#                      (np.linalg.norm(coords2, axis=1) > 1e-6)
#
#         if valid_mask.sum() < 5:
#             return 0.0
#
#         coords1_valid = coords1[valid_mask]
#         coords2_valid = coords2[valid_mask]
#
#         # Procrustes analysis (removes translation, rotation, scaling)
#         _, _, disparity = procrustes(coords1_valid, coords2_valid)
#
#         # Convert disparity to similarity score (0-100)
#         similarity = max(0, 100 * (1 - disparity))
#
#         return similarity
#
#     def calculate_movement_score(self, student_movement, ref_movement):
#         """Calculate comprehensive score for a single movement"""
#
#         # 1. TEMPORAL SCORE (30%)
#         student_duration = student_movement['duration']
#         ref_duration = ref_movement['duration']
#
#         duration_ratio = min(student_duration, ref_duration) / max(student_duration, ref_duration)
#         temporal_score = duration_ratio * 100
#
#         # 2. POSE SIMILARITY SCORE (70%)
#         student_kps = student_movement['all_keypoints']
#         ref_kps = ref_movement['all_keypoints']
#
#         # Use Dynamic Time Warping to align sequences
#         # Then compare aligned poses
#         pose_scores = []
#
#         # Compare key poses (start, middle, end)
#         key_pose_scores = {}
#         for pose_name in ['start', 'end', 'middle']:
#             if pose_name in student_movement['key_poses'] and \
#                     pose_name in ref_movement['key_poses']:
#                 similarity = self.calculate_pose_similarity(
#                     student_movement['key_poses'][pose_name],
#                     ref_movement['key_poses'][pose_name]
#                 )
#                 key_pose_scores[pose_name] = similarity
#                 pose_scores.append(similarity)
#
#         # Average pose similarity
#         avg_pose_score = np.mean(pose_scores) if pose_scores else 0.0
#
#         # 3. OVERALL SCORE
#         overall_score = temporal_score * 0.3 + avg_pose_score * 0.7
#
#         return {
#             'overall_score': overall_score,
#             'temporal_score': temporal_score,
#             'pose_score': avg_pose_score,
#             'key_pose_scores': key_pose_scores,
#             'duration_diff': student_duration - ref_duration,
#             'duration_ratio': duration_ratio
#         }
#
#     def compare(self, student_segments_csv, student_keypoints_path):
#         """Compare student with reference"""
#
#         # Load student data
#         student_segments = pd.read_csv(student_segments_csv)
#
#         with open(student_keypoints_path, 'rb') as f:
#             student_data = pickle.load(f)
#
#         print(f"\n{'=' * 70}")
#         print(f"COMPARING PERFORMANCE")
#         print(f"{'=' * 70}\n")
#
#         comparison_results = {
#             'student_video': student_data['video_name'],
#             'reference_video': self.reference['video_name'],
#             'movement_scores': [],
#             'overall_summary': {}
#         }
#
#         all_scores = []
#
#         for student_mov, ref_mov in zip(student_data['movements'],
#                                         self.reference['movements']):
#             movement_num = student_mov['movement_number']
#
#             print(f"Movement {movement_num}: {student_mov['movement_name']}")
#
#             # Calculate scores
#             scores = self.calculate_movement_score(student_mov, ref_mov)
#
#             scores['movement_number'] = movement_num
#             scores['movement_name'] = student_mov['movement_name']
#
#             comparison_results['movement_scores'].append(scores)
#             all_scores.append(scores['overall_score'])
#
#             print(f"  Overall: {scores['overall_score']:.1f}/100")
#             print(f"  Temporal: {scores['temporal_score']:.1f}/100")
#             print(f"  Pose: {scores['pose_score']:.1f}/100")
#             print(f"  Duration diff: {scores['duration_diff']:+.2f}s")
#             print()
#
#         # Overall summary
#         comparison_results['overall_summary'] = {
#             'average_score': np.mean(all_scores),
#             'min_score': np.min(all_scores),
#             'max_score': np.max(all_scores),
#             'std_score': np.std(all_scores),
#             'movements_above_80': sum(s >= 80 for s in all_scores),
#             'movements_below_60': sum(s < 60 for s in all_scores)
#         }
#
#         print(f"{'=' * 70}")
#         print(f"OVERALL SUMMARY")
#         print(f"{'=' * 70}")
#         print(f"Average Score: {comparison_results['overall_summary']['average_score']:.1f}/100")
#         print(f"Range: {comparison_results['overall_summary']['min_score']:.1f} - "
#               f"{comparison_results['overall_summary']['max_score']:.1f}")
#         print(f"Movements ≥80: {comparison_results['overall_summary']['movements_above_80']}/20")
#         print(f"Movements <60: {comparison_results['overall_summary']['movements_below_60']}/20")
#         print(f"{'=' * 70}\n")
#
#         return comparison_results
#
#
# def main():
#     import argparse
#
#     parser = argparse.ArgumentParser(description='Compare with reference')
#     parser.add_argument('--student-video', required=True)
#     parser.add_argument('--student-segments', required=True)
#     parser.add_argument('--student-keypoints', required=True)
#     parser.add_argument('--reference', required=True)
#     parser.add_argument('--output', default='comparison_result.json')
#
#     args = parser.parse_args()
#
#     comparator = PerformanceComparator(args.reference)
#     results = comparator.compare(args.student_segments, args.student_keypoints)
#
#     # Save results (convert numpy types to Python types)
#     output_path = Path(args.output)
#     output_path.parent.mkdir(parents=True, exist_ok=True)
#
#     # Convert all numpy types to serializable types
#     serializable_results = convert_to_serializable(results)
#
#     with open(output_path, 'w', encoding='utf-8') as f:
#         json.dump(serializable_results, f, indent=2, ensure_ascii=False)
#
#     print(f"✅ Comparison results saved: {output_path}")
#
#
# if __name__ == "__main__":
#     main()
#
# # python compare_performance.py --student-video "D:\All Docs\All Projects\Pycharm\poomsae_recognition\data\reference\videos\P011.mp4" --student-segments "D:\All Docs\All Projects\Pycharm\poomsae_recognition\results\movement_segments\P011_segments_20260121_115122.csv" --student-keypoints "D:\All Docs\All Projects\Pycharm\poomsae_recognition\compare\students\keypoints\P001_keypoints.pkl" --reference "D:\All Docs\All Projects\Pycharm\poomsae_recognition\compare\references\master_reference.pkl" --output results/comparison/P011_comparison.json

"""
Compare Student Performance with Reference

Usage:
    python compare_performance.py \
        --student-segments P001_segments.csv \
        --student-keypoints P001_keypoints.pkl \
        --reference master_reference.pkl \
        --output P001_comparison.json
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from scipy.spatial import procrustes
import pickle


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


class PerformanceComparator:
    """Compare student performance with master reference"""

    def __init__(self, reference_path):
        """Load reference data"""
        with open(reference_path, 'rb') as f:
            self.reference = pickle.load(f)

        print(f"✓ Loaded reference: {self.reference['video_name']}")
        print(f"  Movements: {len(self.reference['movements'])}")

    def calculate_pose_similarity(self, pose1, pose2):
        """
        Calculate pose similarity using Procrustes distance

        Returns: similarity score (0-100, higher is better)
        """
        # Extract coordinates (ignore confidence)
        coords1 = pose1[:, :2]
        coords2 = pose2[:, :2]

        # Filter out zero/invalid keypoints
        valid_mask = (np.linalg.norm(coords1, axis=1) > 1e-6) & \
                     (np.linalg.norm(coords2, axis=1) > 1e-6)

        if valid_mask.sum() < 5:
            return 0.0

        coords1_valid = coords1[valid_mask]
        coords2_valid = coords2[valid_mask]

        try:
            # Procrustes analysis (removes translation, rotation, scaling)
            _, _, disparity = procrustes(coords1_valid, coords2_valid)

            # Convert disparity to similarity score (0-100)
            # Lower disparity = higher similarity
            similarity = max(0, min(100, 100 * (1 - disparity * 2)))

            return similarity
        except:
            return 0.0

    def calculate_temporal_score(self, student_duration, ref_duration):
        """Calculate timing score"""
        if ref_duration == 0:
            return 0.0

        duration_ratio = min(student_duration, ref_duration) / max(student_duration, ref_duration)
        temporal_score = duration_ratio * 100

        return temporal_score

    def calculate_movement_score(self, student_movement, ref_movement):
        """Calculate comprehensive score for a single movement"""

        # 1. TEMPORAL SCORE (30%)
        temporal_score = self.calculate_temporal_score(
            student_movement['duration'],
            ref_movement['duration']
        )

        # 2. POSE SIMILARITY SCORE (70%)
        pose_scores = []
        key_pose_scores = {}

        # Compare key poses (start, middle, end)
        for pose_name in ['start', 'middle', 'end']:
            if pose_name in student_movement['key_poses'] and \
                    pose_name in ref_movement['key_poses']:
                similarity = self.calculate_pose_similarity(
                    student_movement['key_poses'][pose_name],
                    ref_movement['key_poses'][pose_name]
                )
                key_pose_scores[pose_name] = float(similarity)
                pose_scores.append(similarity)

        # Average pose similarity
        avg_pose_score = float(np.mean(pose_scores)) if pose_scores else 0.0

        # 3. OVERALL SCORE
        overall_score = temporal_score * 0.3 + avg_pose_score * 0.7

        # 4. GRADE
        if overall_score >= 90:
            grade = "A+"
        elif overall_score >= 85:
            grade = "A"
        elif overall_score >= 80:
            grade = "B+"
        elif overall_score >= 75:
            grade = "B"
        elif overall_score >= 70:
            grade = "C+"
        elif overall_score >= 65:
            grade = "C"
        elif overall_score >= 60:
            grade = "D"
        else:
            grade = "F"

        # 5. FEEDBACK
        feedback = []
        duration_diff = student_movement['duration'] - ref_movement['duration']

        if abs(duration_diff) < 0.1:
            feedback.append("Perfect timing!")
        elif duration_diff > 0.5:
            feedback.append("Movement too slow")
        elif duration_diff < -0.5:
            feedback.append("Movement too fast")

        if avg_pose_score >= 90:
            feedback.append("Excellent form!")
        elif avg_pose_score >= 75:
            feedback.append("Good form")
        elif avg_pose_score < 60:
            feedback.append("Form needs improvement")

        return {
            'overall_score': float(overall_score),
            'temporal_score': float(temporal_score),
            'pose_score': float(avg_pose_score),
            'key_pose_scores': key_pose_scores,
            'duration_diff': float(duration_diff),
            'duration_ratio': float(min(student_movement['duration'], ref_movement['duration']) /
                                    max(student_movement['duration'], ref_movement['duration'])),
            'grade': grade,
            'feedback': ' | '.join(feedback) if feedback else "Keep practicing!"
        }

    def compare(self, student_segments_csv, student_keypoints_path):
        """Compare student with reference"""

        # Load student data
        student_segments = pd.read_csv(student_segments_csv)

        with open(student_keypoints_path, 'rb') as f:
            student_data = pickle.load(f)

        print(f"\n{'=' * 70}")
        print(f"COMPARING PERFORMANCE")
        print(f"{'=' * 70}\n")
        print(f"Student: {student_data['video_name']}")
        print(f"Reference: {self.reference['video_name']}")
        print()

        comparison_results = {
            'student_video': student_data['video_name'],
            'reference_video': self.reference['video_name'],
            'timestamp': pd.Timestamp.now().isoformat(),
            'movement_scores': [],
            'overall_summary': {}
        }

        all_scores = []

        for student_mov, ref_mov in zip(student_data['movements'],
                                        self.reference['movements']):
            movement_num = student_mov['movement_number']

            print(f"Movement {movement_num}: {student_mov['movement_name']}")

            # Calculate scores
            scores = self.calculate_movement_score(student_mov, ref_mov)

            scores['movement_number'] = int(movement_num)
            scores['movement_name'] = student_mov['movement_name']

            comparison_results['movement_scores'].append(scores)
            all_scores.append(scores['overall_score'])

            print(f"  Overall: {scores['overall_score']:.1f}/100 (Grade: {scores['grade']})")
            print(f"  Temporal: {scores['temporal_score']:.1f}/100")
            print(f"  Pose: {scores['pose_score']:.1f}/100")
            print(f"  Feedback: {scores['feedback']}")
            print()

        # Overall summary
        avg_score = float(np.mean(all_scores))

        if avg_score >= 90:
            overall_grade = "A+"
        elif avg_score >= 85:
            overall_grade = "A"
        elif avg_score >= 80:
            overall_grade = "B+"
        elif avg_score >= 75:
            overall_grade = "B"
        elif avg_score >= 70:
            overall_grade = "C+"
        elif avg_score >= 65:
            overall_grade = "C"
        elif avg_score >= 60:
            overall_grade = "D"
        else:
            overall_grade = "F"

        # Find strengths and weaknesses
        sorted_scores = sorted(comparison_results['movement_scores'],
                               key=lambda x: x['overall_score'], reverse=True)

        strengths = [f"Movement {s['movement_number']}" for s in sorted_scores[:3]]
        weaknesses = [f"Movement {s['movement_number']}" for s in sorted_scores[-3:]]

        comparison_results['overall_summary'] = {
            'average_score': float(avg_score),
            'overall_grade': overall_grade,
            'min_score': float(np.min(all_scores)),
            'max_score': float(np.max(all_scores)),
            'std_score': float(np.std(all_scores)),
            'movements_above_80': int(sum(s >= 80 for s in all_scores)),
            'movements_below_60': int(sum(s < 60 for s in all_scores)),
            'strengths': strengths,
            'weaknesses': weaknesses
        }

        print(f"{'=' * 70}")
        print(f"OVERALL SUMMARY")
        print(f"{'=' * 70}")
        print(f"Average Score: {avg_score:.1f}/100")
        print(f"Overall Grade: {overall_grade}")
        print(f"Range: {comparison_results['overall_summary']['min_score']:.1f} - "
              f"{comparison_results['overall_summary']['max_score']:.1f}")
        print(f"Movements ≥80: {comparison_results['overall_summary']['movements_above_80']}/20")
        print(f"Movements <60: {comparison_results['overall_summary']['movements_below_60']}/20")
        print(f"\nStrengths: {', '.join(strengths)}")
        print(f"Areas for Improvement: {', '.join(weaknesses)}")
        print(f"{'=' * 70}\n")

        return comparison_results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Compare with reference')
    parser.add_argument('--student-segments', required=True, help='Student segments CSV')
    parser.add_argument('--student-keypoints', required=True, help='Student keypoints PKL')
    parser.add_argument('--reference', required=True, help='Reference PKL')
    parser.add_argument('--output', default='comparison_result.json', help='Output JSON')

    args = parser.parse_args()

    comparator = PerformanceComparator(args.reference)
    results = comparator.compare(args.student_segments, args.student_keypoints)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert all numpy types to serializable types
    serializable_results = convert_to_serializable(results)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)

    print(f"✅ Comparison results saved: {output_path}")


if __name__ == "__main__":
    main()