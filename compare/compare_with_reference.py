"""
Compare Student Performance with Reference

Compares student keypoints with reference keypoints and generates
scores and feedback for each movement.

Features:
- Dynamic Time Warping (DTW) for sequence alignment
- Procrustes analysis for pose similarity
- Handles different movement speeds/durations

Usage:
    python compare_with_reference.py \
        --student compare/students/student1 \
        --reference compare/references \
        --output compare/students/student1/comparison.json
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
from pathlib import Path
import json
import argparse
from scipy.spatial import procrustes
from scipy.spatial.distance import cdist
from datetime import datetime


def compute_dtw(seq1, seq2, dist_func):
    """
    Compute Dynamic Time Warping distance and alignment path

    Args:
        seq1: First sequence (N, ...)
        seq2: Second sequence (M, ...)
        dist_func: Function to compute distance between two frames

    Returns:
        dtw_distance: Normalized DTW distance
        path: List of (i, j) tuples representing alignment
    """
    n, m = len(seq1), len(seq2)

    # Compute cost matrix
    cost_matrix = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            cost_matrix[i, j] = dist_func(seq1[i], seq2[j])

    # Compute accumulated cost matrix
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = cost_matrix[i - 1, j - 1]
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],      # insertion
                dtw_matrix[i, j - 1],      # deletion
                dtw_matrix[i - 1, j - 1]   # match
            )

    # Backtrack to find path
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        candidates = [
            (dtw_matrix[i - 1, j - 1], i - 1, j - 1),
            (dtw_matrix[i - 1, j], i - 1, j),
            (dtw_matrix[i, j - 1], i, j - 1)
        ]
        _, i, j = min(candidates, key=lambda x: x[0])

    path.reverse()

    # Normalize by path length
    dtw_distance = dtw_matrix[n, m] / len(path)

    return dtw_distance, path


class PerformanceComparator:
    """Compare student performance with reference using DTW alignment"""

    # Scoring weights
    WEIGHT_POSE = 0.7       # 70% for pose similarity
    WEIGHT_TIMING = 0.3     # 30% for timing

    # Grade thresholds
    GRADE_THRESHOLDS = [
        (95, 'A+', 'Excellent'),
        (90, 'A', 'Very Good'),
        (85, 'B+', 'Good'),
        (80, 'B', 'Above Average'),
        (75, 'C+', 'Average'),
        (70, 'C', 'Below Average'),
        (65, 'D+', 'Needs Improvement'),
        (60, 'D', 'Poor'),
        (0, 'F', 'Fail')
    ]

    # Use DTW for these movements (kicks and fast movements)
    DTW_MOVEMENTS = ['14_1', '16_1', '6_1', '12_1']

    def __init__(self, reference_dir):
        """
        Args:
            reference_dir: Directory containing reference data
        """
        self.reference_dir = Path(reference_dir)
        self.reference_frames_dir = self.reference_dir / 'frames'

        # Load reference index
        index_path = self.reference_frames_dir / 'index.json'
        if not index_path.exists():
            raise FileNotFoundError(f"Reference index not found: {index_path}")

        with open(index_path, 'r', encoding='utf-8') as f:
            self.reference_index = json.load(f)

        print(f"Loaded reference: {self.reference_index['video']}")
        print(f"  Movements: {len(self.reference_index['movements'])}")
        print(f"  FPS: {self.reference_index['fps']}")

        # Load all reference keypoints
        self.reference_keypoints = {}
        for mov in self.reference_index['movements']:
            movement_id = mov['movement_id']
            npz_path = self.reference_frames_dir / movement_id / 'keypoints.npz'
            if npz_path.exists():
                self.reference_keypoints[movement_id] = np.load(npz_path)

        print(f"  Loaded keypoints for {len(self.reference_keypoints)} movements")

    def calculate_pose_similarity(self, pose1, pose2):
        """
        Calculate pose similarity using Procrustes analysis

        Args:
            pose1, pose2: (26, 3) keypoint arrays [x, y, conf]

        Returns:
            similarity: 0-100 score (higher is better)
        """
        # Extract coordinates (ignore confidence)
        coords1 = pose1[:, :2].copy()
        coords2 = pose2[:, :2].copy()

        # Filter out invalid keypoints (zero or low confidence)
        conf1 = pose1[:, 2]
        conf2 = pose2[:, 2]
        valid_mask = (conf1 > 0.3) & (conf2 > 0.3) & \
                     (np.linalg.norm(coords1, axis=1) > 1e-6) & \
                     (np.linalg.norm(coords2, axis=1) > 1e-6)

        if valid_mask.sum() < 5:
            return 0.0

        coords1_valid = coords1[valid_mask]
        coords2_valid = coords2[valid_mask]

        try:
            # Procrustes analysis (removes translation, rotation, scaling differences)
            _, _, disparity = procrustes(coords1_valid, coords2_valid)

            # Convert disparity to similarity (0-100)
            # disparity is sum of squared differences after alignment
            similarity = max(0, min(100, 100 * (1 - disparity * 2)))

            return similarity
        except Exception:
            return 0.0

    def calculate_sequence_similarity_simple(self, student_kps, reference_kps):
        """
        Calculate similarity using simple key frame sampling (no DTW)
        """
        n_student = len(student_kps)
        n_ref = len(reference_kps)

        if n_student == 0 or n_ref == 0:
            return {'overall': 0, 'start': 0, 'middle': 0, 'end': 0, 'method': 'simple'}

        # Sample key frames from both sequences
        def get_key_indices(n):
            if n < 3:
                return [0, n-1] if n > 1 else [0]
            return [0, n // 4, n // 2, 3 * n // 4, n - 1]

        student_indices = get_key_indices(n_student)
        ref_indices = get_key_indices(n_ref)

        # Compare corresponding key frames
        similarities = []
        for s_idx, r_idx in zip(student_indices, ref_indices):
            if s_idx < n_student and r_idx < n_ref:
                sim = self.calculate_pose_similarity(
                    student_kps[s_idx],
                    reference_kps[r_idx]
                )
                similarities.append(sim)

        # Also compare start, middle, end specifically
        start_sim = self.calculate_pose_similarity(student_kps[0], reference_kps[0])
        mid_s, mid_r = n_student // 2, n_ref // 2
        middle_sim = self.calculate_pose_similarity(student_kps[mid_s], reference_kps[mid_r])
        end_sim = self.calculate_pose_similarity(student_kps[-1], reference_kps[-1])

        overall_sim = np.mean(similarities) if similarities else 0

        return {
            'overall': float(overall_sim),
            'start': float(start_sim),
            'middle': float(middle_sim),
            'end': float(end_sim),
            'sampled_count': len(similarities),
            'method': 'simple'
        }

    def calculate_sequence_similarity_dtw(self, student_kps, reference_kps):
        """
        Calculate similarity using Dynamic Time Warping alignment

        DTW finds the optimal alignment between sequences of different lengths,
        making it robust to speed variations.
        """
        n_student = len(student_kps)
        n_ref = len(reference_kps)

        if n_student == 0 or n_ref == 0:
            return {'overall': 0, 'start': 0, 'middle': 0, 'end': 0, 'method': 'dtw'}

        # Distance function for DTW
        def pose_distance(pose1, pose2):
            # Use 1 - similarity as distance
            sim = self.calculate_pose_similarity(pose1, pose2)
            return (100 - sim) / 100  # Normalize to 0-1

        # Compute DTW
        dtw_distance, path = compute_dtw(student_kps, reference_kps, pose_distance)

        # Calculate similarity from aligned frames
        aligned_similarities = []
        for s_idx, r_idx in path:
            sim = self.calculate_pose_similarity(student_kps[s_idx], reference_kps[r_idx])
            aligned_similarities.append(sim)

        # Overall score from DTW alignment
        overall_sim = np.mean(aligned_similarities) if aligned_similarities else 0

        # Get start, middle, end from aligned path
        start_pair = path[0]
        mid_pair = path[len(path) // 2]
        end_pair = path[-1]

        start_sim = self.calculate_pose_similarity(
            student_kps[start_pair[0]], reference_kps[start_pair[1]]
        )
        middle_sim = self.calculate_pose_similarity(
            student_kps[mid_pair[0]], reference_kps[mid_pair[1]]
        )
        end_sim = self.calculate_pose_similarity(
            student_kps[end_pair[0]], reference_kps[end_pair[1]]
        )

        return {
            'overall': float(overall_sim),
            'start': float(start_sim),
            'middle': float(middle_sim),
            'end': float(end_sim),
            'dtw_distance': float(dtw_distance),
            'alignment_length': len(path),
            'method': 'dtw'
        }

    def calculate_sequence_similarity(self, student_kps, reference_kps, movement_id=None):
        """
        Calculate similarity between two keypoint sequences

        Uses DTW for movements with variable timing (kicks),
        simple sampling for others.

        Args:
            student_kps: (N, 26, 3) student keypoints
            reference_kps: (M, 26, 3) reference keypoints
            movement_id: Movement ID to determine method

        Returns:
            dict with similarity scores
        """
        # Use DTW for specific movements or when duration differs significantly
        n_student = len(student_kps)
        n_ref = len(reference_kps)

        use_dtw = False

        # Use DTW for kick movements
        if movement_id in self.DTW_MOVEMENTS:
            use_dtw = True

        # Also use DTW if duration ratio is significantly different (>1.5x or <0.67x)
        if n_ref > 0:
            ratio = n_student / n_ref
            if ratio > 1.5 or ratio < 0.67:
                use_dtw = True

        if use_dtw:
            return self.calculate_sequence_similarity_dtw(student_kps, reference_kps)
        else:
            return self.calculate_sequence_similarity_simple(student_kps, reference_kps)

    def calculate_timing_score(self, student_duration, reference_duration):
        """
        Calculate timing/duration score

        Args:
            student_duration: Student movement duration in seconds
            reference_duration: Reference movement duration in seconds

        Returns:
            score: 0-100 (higher is better)
        """
        if reference_duration <= 0:
            return 0.0

        # Calculate ratio
        ratio = student_duration / reference_duration

        # Perfect timing = ratio of 1.0
        # Score decreases as ratio deviates from 1.0
        if ratio > 1:
            # Too slow
            score = max(0, 100 - (ratio - 1) * 100)
        else:
            # Too fast
            score = max(0, 100 - (1 - ratio) * 100)

        return float(score)

    def get_grade(self, score):
        """Get grade and description from score"""
        for threshold, grade, description in self.GRADE_THRESHOLDS:
            if score >= threshold:
                return grade, description
        return 'F', 'Fail'

    def generate_feedback(self, pose_score, timing_score, duration_diff):
        """Generate specific feedback based on scores"""
        feedback = []

        # Pose feedback
        if pose_score >= 90:
            feedback.append("Excellent form!")
        elif pose_score >= 80:
            feedback.append("Good form")
        elif pose_score >= 70:
            feedback.append("Form is acceptable but could be improved")
        elif pose_score >= 60:
            feedback.append("Form needs significant improvement")
        else:
            feedback.append("Form needs major correction")

        # Timing feedback
        if abs(duration_diff) < 0.2:
            feedback.append("Perfect timing!")
        elif duration_diff > 0.5:
            feedback.append(f"Too slow (by {duration_diff:.1f}s)")
        elif duration_diff < -0.5:
            feedback.append(f"Too fast (by {-duration_diff:.1f}s)")
        elif duration_diff > 0:
            feedback.append("Slightly slow")
        else:
            feedback.append("Slightly fast")

        return feedback

    def compare_movement(self, student_movement, movement_id):
        """
        Compare a single student movement with reference

        Args:
            student_movement: Dict with student movement data
            movement_id: Movement ID to compare

        Returns:
            dict with comparison results
        """
        if movement_id not in self.reference_keypoints:
            return {
                'error': f'Reference not found for {movement_id}',
                'overall_score': 0,
                'grade': 'N/A'
            }

        # Load keypoints
        ref_data = self.reference_keypoints[movement_id]
        ref_kps = ref_data['keypoints_norm']
        ref_meta = json.loads(str(ref_data['meta']))

        # Find reference duration
        ref_mov = None
        for mov in self.reference_index['movements']:
            if mov['movement_id'] == movement_id:
                ref_mov = mov
                break

        ref_duration = ref_mov['duration'] if ref_mov else 1.0

        # Load student keypoints
        student_npz_path = student_movement.get('keypoints_file')
        if not student_npz_path:
            return {
                'error': 'Student keypoints file not specified',
                'overall_score': 0,
                'grade': 'N/A'
            }

        # Calculate pose similarity (use DTW for variable-timing movements)
        pose_scores = self.calculate_sequence_similarity(
            student_movement['keypoints_norm'],
            ref_kps,
            movement_id=movement_id
        )

        # Calculate timing score
        student_duration = student_movement.get('duration', 0)
        timing_score = self.calculate_timing_score(student_duration, ref_duration)
        duration_diff = student_duration - ref_duration

        # Overall score
        pose_score = pose_scores['overall']
        overall_score = (pose_score * self.WEIGHT_POSE +
                        timing_score * self.WEIGHT_TIMING)

        # Grade
        grade, grade_desc = self.get_grade(overall_score)

        # Feedback
        feedback = self.generate_feedback(pose_score, timing_score, duration_diff)

        return {
            'movement_id': movement_id,
            'movement_name': student_movement.get('name', movement_id),
            'overall_score': round(overall_score, 1),
            'pose_score': round(pose_score, 1),
            'pose_details': {
                'start': round(pose_scores['start'], 1),
                'middle': round(pose_scores['middle'], 1),
                'end': round(pose_scores['end'], 1)
            },
            'timing_score': round(timing_score, 1),
            'duration': {
                'student': round(student_duration, 2),
                'reference': round(ref_duration, 2),
                'difference': round(duration_diff, 2)
            },
            'grade': grade,
            'grade_description': grade_desc,
            'feedback': feedback,
            'comparison_method': pose_scores.get('method', 'simple')
        }

    def compare(self, student_dir, output_path=None):
        """
        Compare all student movements with reference

        Args:
            student_dir: Directory containing student results
            output_path: Optional output path for comparison JSON

        Returns:
            dict with full comparison results
        """
        student_dir = Path(student_dir)

        # Load student results
        results_path = student_dir / 'results.json'
        if not results_path.exists():
            raise FileNotFoundError(f"Student results not found: {results_path}")

        with open(results_path, 'r', encoding='utf-8') as f:
            student_results = json.load(f)

        print(f"\n{'='*70}")
        print("COMPARING STUDENT WITH REFERENCE")
        print(f"{'='*70}")
        print(f"Student: {student_results['video_name']}")
        print(f"Reference: {self.reference_index['video']}")
        print(f"Detected movements: {student_results['num_detected']}/22")
        print(f"{'='*70}\n")

        # Comparison results
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'student': {
                'video': student_results['video_name'],
                'detected': student_results['num_detected'],
                'skipped': student_results['num_skipped']
            },
            'reference': {
                'video': self.reference_index['video']
            },
            'movements': [],
            'summary': {}
        }

        all_scores = []
        all_pose_scores = []
        all_timing_scores = []

        # Compare each movement
        for mov in student_results['movements']:
            movement_id = mov['movement_id']
            print(f"Comparing: {movement_id} - {mov['name'][:30]}")

            # Load student keypoints
            npz_path = student_dir / mov['keypoints_file']
            if npz_path.exists():
                student_npz = np.load(npz_path)
                mov['keypoints_norm'] = student_npz['keypoints_norm']
            else:
                print(f"  WARNING: Keypoints file not found: {npz_path}")
                continue

            # Compare
            result = self.compare_movement(mov, movement_id)
            comparison['movements'].append(result)

            if 'error' not in result:
                all_scores.append(result['overall_score'])
                all_pose_scores.append(result['pose_score'])
                all_timing_scores.append(result['timing_score'])

                method = result.get('comparison_method', 'simple')
                method_str = f" [DTW]" if method == 'dtw' else ""
                print(f"  Overall: {result['overall_score']:.1f}/100 ({result['grade']}){method_str}")
                print(f"  Pose: {result['pose_score']:.1f} | Timing: {result['timing_score']:.1f}")
                print(f"  Feedback: {', '.join(result['feedback'])}")
            else:
                print(f"  ERROR: {result['error']}")

            print()

        # Calculate summary
        if all_scores:
            avg_score = np.mean(all_scores)
            avg_grade, avg_desc = self.get_grade(avg_score)

            comparison['summary'] = {
                'overall_score': round(avg_score, 1),
                'overall_grade': avg_grade,
                'overall_description': avg_desc,
                'pose_score_avg': round(np.mean(all_pose_scores), 1),
                'timing_score_avg': round(np.mean(all_timing_scores), 1),
                'score_range': {
                    'min': round(min(all_scores), 1),
                    'max': round(max(all_scores), 1)
                },
                'movements_evaluated': len(all_scores),
                'movements_above_80': sum(1 for s in all_scores if s >= 80),
                'movements_below_60': sum(1 for s in all_scores if s < 60)
            }

            # Find best and worst movements
            scored_movements = [(m['movement_id'], m['overall_score'])
                              for m in comparison['movements'] if 'overall_score' in m]
            scored_movements.sort(key=lambda x: x[1], reverse=True)

            comparison['summary']['best_movements'] = [m[0] for m in scored_movements[:3]]
            comparison['summary']['needs_improvement'] = [m[0] for m in scored_movements[-3:]]

        # Print summary
        print(f"{'='*70}")
        print("COMPARISON SUMMARY")
        print(f"{'='*70}")
        if comparison['summary']:
            s = comparison['summary']
            print(f"Overall Score: {s['overall_score']:.1f}/100")
            print(f"Overall Grade: {s['overall_grade']} ({s['overall_description']})")
            print(f"Pose Score (avg): {s['pose_score_avg']:.1f}")
            print(f"Timing Score (avg): {s['timing_score_avg']:.1f}")
            print(f"Score Range: {s['score_range']['min']:.1f} - {s['score_range']['max']:.1f}")
            print(f"Movements >= 80: {s['movements_above_80']}/{s['movements_evaluated']}")
            print(f"Movements < 60: {s['movements_below_60']}/{s['movements_evaluated']}")
            print(f"\nBest: {', '.join(s['best_movements'])}")
            print(f"Needs Work: {', '.join(s['needs_improvement'])}")
        print(f"{'='*70}\n")

        # Save results
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, indent=2, ensure_ascii=False)
            print(f"Results saved: {output_path}")

        return comparison


def main():
    parser = argparse.ArgumentParser(description='Compare student with reference')
    parser.add_argument('--student', required=True, help='Student results directory')
    parser.add_argument('--reference', default='compare/references',
                        help='Reference directory')
    parser.add_argument('--output', help='Output comparison JSON path')

    args = parser.parse_args()

    # Default output path
    if not args.output:
        args.output = Path(args.student) / 'comparison.json'

    comparator = PerformanceComparator(args.reference)
    comparator.compare(args.student, args.output)


if __name__ == "__main__":
    main()
