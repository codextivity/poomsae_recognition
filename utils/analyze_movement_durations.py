"""
Movement Duration Analyzer

Analyzes all annotation files to determine optimal window size and stride
based on actual movement durations in your dataset.
"""

import json
from pathlib import Path
import numpy as np
import sys
from collections import defaultdict
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent))
from configs.paths import Paths


class MovementDurationAnalyzer:
    """Analyze movement durations across all videos"""

    def __init__(self, annotations_dir):
        self.annotations_dir = Path(annotations_dir)
        self.all_durations = defaultdict(list)  # movement_num -> [durations]
        self.video_durations = []  # total video durations

    def analyze_single_file(self, annotation_path):
        """Analyze durations in a single file"""
        with open(annotation_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        annotations = data['annotations']

        # Parse and sort by movement number
        movements = []
        for ann in annotations:
            movement_str = ann['movement']
            if '.' in movement_str:
                number_str = movement_str.split('.')[0].strip()
                movement_num = int(number_str)
            else:
                movement_num = int(movement_str)

            movements.append({
                'number': movement_num,
                'start_time': float(ann['startTime'])
            })

        movements.sort(key=lambda x: x['number'])

        # Calculate durations
        for i in range(len(movements) - 1):
            curr = movements[i]
            next_mov = movements[i + 1]

            duration = next_mov['start_time'] - curr['start_time']
            self.all_durations[curr['number']].append(duration)

        # Last movement - estimate or use remaining time
        # For now, use average of other movements
        if len(movements) > 1:
            avg_duration = np.mean([next_mov['start_time'] - movements[i]['start_time']
                                    for i in range(len(movements) - 1)])
            self.all_durations[movements[-1]['number']].append(avg_duration)

        # Total video duration
        total_duration = movements[-1]['start_time']
        self.video_durations.append(total_duration)

    def analyze_all(self):
        """Analyze all annotation files"""
        annotation_files = sorted(self.annotations_dir.glob('*_annotations.json'))

        print(f"\n{'=' * 70}")
        print(f"MOVEMENT DURATION ANALYSIS")
        print(f"{'=' * 70}")
        print(f"Analyzing {len(annotation_files)} videos...\n")

        for ann_file in annotation_files:
            self.analyze_single_file(ann_file)

        # Statistics per movement
        print(f"\n{'=' * 70}")
        print(f"PER-MOVEMENT DURATION STATISTICS")
        print(f"{'=' * 70}\n")

        all_stats = []

        for mov_num in range(1, 21):
            durations = self.all_durations[mov_num]

            if durations:
                mean_dur = np.mean(durations)
                median_dur = np.median(durations)
                min_dur = np.min(durations)
                max_dur = np.max(durations)
                std_dur = np.std(durations)

                # Frames at 30 FPS
                mean_frames = mean_dur * 30
                min_frames = min_dur * 30
                max_frames = max_dur * 30

                all_stats.append({
                    'movement': mov_num,
                    'mean': mean_dur,
                    'median': median_dur,
                    'min': min_dur,
                    'max': max_dur,
                    'std': std_dur,
                    'mean_frames': mean_frames,
                    'min_frames': min_frames,
                    'max_frames': max_frames,
                    'count': len(durations)
                })

                print(f"Movement {mov_num:2d}:")
                print(f"  Duration: {mean_dur:.2f}s ± {std_dur:.2f}s "
                      f"(min: {min_dur:.2f}s, max: {max_dur:.2f}s)")
                print(f"  Frames@30fps: {mean_frames:.0f} ± {std_dur * 30:.0f} "
                      f"(min: {min_frames:.0f}, max: {max_frames:.0f})")
                print(f"  Samples: {len(durations)}")
                print()

        # Overall statistics
        print(f"\n{'=' * 70}")
        print(f"OVERALL STATISTICS")
        print(f"{'=' * 70}\n")

        all_movement_durations = []
        for durations in self.all_durations.values():
            all_movement_durations.extend(durations)

        overall_mean = np.mean(all_movement_durations)
        overall_median = np.median(all_movement_durations)
        overall_min = np.min(all_movement_durations)
        overall_max = np.max(all_movement_durations)
        overall_std = np.std(all_movement_durations)

        print(f"All Movements Combined:")
        print(f"  Mean duration: {overall_mean:.2f}s ({overall_mean * 30:.0f} frames)")
        print(f"  Median duration: {overall_median:.2f}s ({overall_median * 30:.0f} frames)")
        print(f"  Min duration: {overall_min:.2f}s ({overall_min * 30:.0f} frames)")
        print(f"  Max duration: {overall_max:.2f}s ({overall_max * 30:.0f} frames)")
        print(f"  Std deviation: {overall_std:.2f}s ({overall_std * 30:.0f} frames)")
        print(f"  Total samples: {len(all_movement_durations)}")

        # Percentiles
        p25 = np.percentile(all_movement_durations, 25)
        p75 = np.percentile(all_movement_durations, 75)
        p90 = np.percentile(all_movement_durations, 90)

        print(f"\nPercentiles:")
        print(f"  25th percentile: {p25:.2f}s ({p25 * 30:.0f} frames)")
        print(f"  75th percentile: {p75:.2f}s ({p75 * 30:.0f} frames)")
        print(f"  90th percentile: {p90:.2f}s ({p90 * 30:.0f} frames)")

        # Video statistics
        print(f"\n{'=' * 70}")
        print(f"VIDEO DURATION STATISTICS")
        print(f"{'=' * 70}\n")

        avg_video_dur = np.mean(self.video_durations)
        print(f"Average total video duration: {avg_video_dur:.2f}s ({avg_video_dur * 30:.0f} frames)")
        print(f"Min video duration: {np.min(self.video_durations):.2f}s")
        print(f"Max video duration: {np.max(self.video_durations):.2f}s")

        # Recommendations
        print(f"\n{'=' * 70}")
        print(f"WINDOW SIZE RECOMMENDATIONS")
        print(f"{'=' * 70}\n")

        self.make_recommendations(all_stats, overall_mean, overall_median,
                                  overall_min, overall_max, p25, p75, p90)

        # Create visualization
        self.create_visualizations(all_stats)

        return all_stats

    def make_recommendations(self, all_stats, mean, median, min_dur, max_dur, p25, p75, p90):
        """Make recommendations for window size and stride"""

        print("Current Settings:")
        print(f"  Window size: 90 frames (3.0s @ 30fps)")
        print(f"  Stride: 15 frames (0.5s @ 30fps)")

        print(f"\nAnalysis:")

        # Check if current window size is appropriate
        mean_frames = mean * 30
        median_frames = median * 30

        # Count movements shorter than window
        short_movements = sum(1 for s in all_stats if s['mean_frames'] < 90)

        print(f"  Mean movement duration: {mean:.2f}s ({mean_frames:.0f} frames)")
        print(f"  Median movement duration: {median:.2f}s ({median_frames:.0f} frames)")
        print(f"  Movements shorter than 90 frames: {short_movements}/20")

        # Window size should capture most of a movement
        # Ideal: mean + 0.5*std to mean + std
        recommended_window_duration = median  # Use median as baseline
        recommended_window_frames = int(median * 30)

        # Round to nearest 15 frames for compatibility
        recommended_window_frames = round(recommended_window_frames / 15) * 15

        # Stride should be 1/4 to 1/6 of window for good overlap
        recommended_stride = recommended_window_frames // 5  # 20% of window
        recommended_stride = max(15, recommended_stride)  # At least 15 frames

        print(f"\n{'=' * 70}")
        print(f"RECOMMENDATIONS:")
        print(f"{'=' * 70}")

        # Option 1: Keep current
        print(f"\nOption 1: KEEP CURRENT (Safe)")
        print(f"  SEQUENCE_LENGTH = 90  # 3.0s @ 30fps")
        print(f"  STRIDE = 15  # 0.5s @ 30fps")
        print(f"  Pros: ")
        print(f"    - Already tested")
        print(f"    - Good for movements up to {90 / 30:.1f}s")
        print(f"    - High overlap (83% overlap)")
        print(f"  Cons:")
        if mean_frames < 90:
            print(f"    - Longer than average movement ({mean:.2f}s)")
            print(f"    - Windows will span multiple movements")

        # Option 2: Optimized based on data
        print(f"\nOption 2: OPTIMIZED FOR YOUR DATA")
        print(f"  SEQUENCE_LENGTH = {recommended_window_frames}  # {recommended_window_frames / 30:.1f}s @ 30fps")
        print(f"  STRIDE = {recommended_stride}  # {recommended_stride / 30:.1f}s @ 30fps")
        print(f"  Pros:")
        print(f"    - Matches median movement duration")
        print(f"    - Better temporal resolution")
        if recommended_window_frames < 90:
            print(f"    - Less overlap between movements")
        print(f"  Cons:")
        print(f"    - Need to retrain from scratch")

        # Option 3: Shorter window for quick movements
        short_window = 60  # 2 seconds
        short_stride = 10  # 0.33 seconds
        print(f"\nOption 3: SHORTER WINDOW (For quick movements)")
        print(f"  SEQUENCE_LENGTH = {short_window}  # {short_window / 30:.1f}s @ 30fps")
        print(f"  STRIDE = {short_stride}  # {short_stride / 30:.1f}s @ 30fps")
        print(f"  Pros:")
        print(f"    - Better for movements < 2s")
        print(f"    - More training samples")
        print(f"    - Less movement overlap")
        print(f"  Cons:")
        print(f"    - Might miss long movements")
        print(f"    - Need to retrain")

        # Specific movement analysis
        print(f"\n{'=' * 70}")
        print(f"PROBLEMATIC MOVEMENTS (if keeping 90 frames):")
        print(f"{'=' * 70}")

        for stat in all_stats:
            if stat['mean_frames'] < 90:
                coverage = (stat['mean_frames'] / 90) * 100
                print(f"  Movement {stat['movement']:2d}: {stat['mean']:.2f}s ({stat['mean_frames']:.0f} frames)")
                print(f"    - Only {coverage:.0f}% of window")
                print(f"    - Windows will include adjacent movements")

        print(f"\n{'=' * 70}")
        print(f"MY RECOMMENDATION:")
        print(f"{'=' * 70}")

        if mean_frames >= 80:
            print(f"\n✅ KEEP CURRENT (90 frames, stride 15)")
            print(f"   Your mean movement duration ({mean:.2f}s) is close to window size.")
            print(f"   Current settings are appropriate.")
        elif mean_frames >= 60:
            print(f"\n⚠️  CONSIDER OPTIMIZING")
            print(f"   Use: SEQUENCE_LENGTH = {recommended_window_frames}, STRIDE = {recommended_stride}")
            print(f"   This will better match your movement durations.")
        else:
            print(f"\n🔴 REDUCE WINDOW SIZE")
            print(f"   Your movements are SHORT (mean: {mean:.2f}s)")
            print(f"   Use: SEQUENCE_LENGTH = 60, STRIDE = 10")
            print(f"   This will reduce overlap and increase samples.")

    def create_visualizations(self, all_stats):
        """Create duration distribution plots"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))

            # Plot 1: Mean duration per movement
            movements = [s['movement'] for s in all_stats]
            means = [s['mean'] for s in all_stats]

            axes[0, 0].bar(movements, means, color='steelblue', alpha=0.7)
            axes[0, 0].axhline(y=3.0, color='red', linestyle='--', label='Current window (3.0s)')
            axes[0, 0].axhline(y=np.mean(means), color='green', linestyle='--', label=f'Mean ({np.mean(means):.2f}s)')
            axes[0, 0].set_xlabel('Movement Number')
            axes[0, 0].set_ylabel('Duration (seconds)')
            axes[0, 0].set_title('Mean Duration per Movement')
            axes[0, 0].legend()
            axes[0, 0].grid(axis='y', alpha=0.3)

            # Plot 2: Duration distribution (histogram)
            all_durations = []
            for s in all_stats:
                all_durations.extend(self.all_durations[s['movement']])

            axes[0, 1].hist(all_durations, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
            axes[0, 1].axvline(x=3.0, color='red', linestyle='--', label='Current window (3.0s)')
            axes[0, 1].axvline(x=np.mean(all_durations), color='green', linestyle='--',
                               label=f'Mean ({np.mean(all_durations):.2f}s)')
            axes[0, 1].set_xlabel('Duration (seconds)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Overall Duration Distribution')
            axes[0, 1].legend()
            axes[0, 1].grid(axis='y', alpha=0.3)

            # Plot 3: Min/Max range per movement
            mins = [s['min'] for s in all_stats]
            maxs = [s['max'] for s in all_stats]

            axes[1, 0].fill_between(movements, mins, maxs, alpha=0.3, color='steelblue')
            axes[1, 0].plot(movements, means, 'o-', color='navy', label='Mean')
            axes[1, 0].axhline(y=3.0, color='red', linestyle='--', label='Current window (3.0s)')
            axes[1, 0].set_xlabel('Movement Number')
            axes[1, 0].set_ylabel('Duration (seconds)')
            axes[1, 0].set_title('Duration Range per Movement (Min-Max)')
            axes[1, 0].legend()
            axes[1, 0].grid(axis='y', alpha=0.3)

            # Plot 4: Frame counts
            mean_frames = [s['mean_frames'] for s in all_stats]

            axes[1, 1].bar(movements, mean_frames, color='steelblue', alpha=0.7)
            axes[1, 1].axhline(y=90, color='red', linestyle='--', label='Current window (90 frames)')
            axes[1, 1].axhline(y=60, color='orange', linestyle='--', label='Alternative (60 frames)')
            axes[1, 1].set_xlabel('Movement Number')
            axes[1, 1].set_ylabel('Frames @ 30fps')
            axes[1, 1].set_title('Mean Duration in Frames')
            axes[1, 1].legend()
            axes[1, 1].grid(axis='y', alpha=0.3)

            plt.tight_layout()

            output_path = Path('movement_duration_analysis.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Visualization saved: {output_path}")

            plt.show()

        except Exception as e:
            print(f"\n⚠️  Could not create visualizations: {e}")


def main():
    """Main analysis"""
    Paths.create_directories()

    analyzer = MovementDurationAnalyzer(Paths.RAW_ANNOTATIONS)
    stats = analyzer.analyze_all()


if __name__ == "__main__":
    main()