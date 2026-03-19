"""
Movement Duration Analyzer (supports sub-movements like 1_1, 1_2, ...)

Analyzes annotation JSON files to determine window size/stride based on
actual movement segment durations in your dataset.

Expected annotation format (example):
{
  "annotations": [
    {"movement": "1_1_왼_앞서기_왼_아래막기", "startTime": "30.033", "frame": 901},
    ...
  ]
}
"""

import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# Optional project-specific Paths import (kept compatible with your original)
try:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from configs.paths import Paths
except Exception:
    Paths = None


def parse_movement_id(movement_str: str) -> str:
    """
    Convert movement string into an ID:
      "1_2_..." -> "1_2"
      "10_3_..." -> "10_3"
      "4" -> "4"
      "4_..." -> "4"
    """
    parts = str(movement_str).split('_')

    # Major + Sub pattern
    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
        return f"{int(parts[0])}_{int(parts[1])}"

    # Major only
    if len(parts) >= 1 and parts[0].isdigit():
        return str(int(parts[0]))

    # Fallback (rare)
    return str(movement_str)


def parse_movement_name(movement_str: str) -> str:
    """
    Extract movement name from format "X_Y_description"
      "6_1_오른 지르기" -> "오른 지르기"
      "14_2_오른 앞서기 오른 지르기" -> "오른 앞서기 오른 지르기"
    """
    parts = str(movement_str).split('_')

    # Skip the numeric parts (X_Y) and join the rest
    name_parts = []
    skip_count = 0

    for i, part in enumerate(parts):
        if i < 2 and part.isdigit():
            skip_count += 1
        else:
            name_parts.append(part)

    if name_parts:
        return '_'.join(name_parts).replace('_', ' ').strip()

    return movement_str


def sort_key_movement_id(mid: str):
    """
    Sorting key for IDs like:
      "1_2" -> (1,2)
      "10_3" -> (10,3)
      "4" -> (4,0)
    """
    parts = mid.split('_')
    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
        return (int(parts[0]), int(parts[1]))
    if len(parts) == 1 and parts[0].isdigit():
        return (int(parts[0]), 0)
    return (10**9, 10**9)  # unknowns go to the end


class MovementDurationAnalyzer:
    """Analyze movement durations across all videos"""

    def __init__(self, annotations_dir: str | Path, fps: float = 30.0):
        self.annotations_dir = Path(annotations_dir)
        self.fps = fps
        self.all_durations = defaultdict(list)  # movement_id -> [durations]
        self.movement_names = {}  # movement_id -> name (e.g., "6_1" -> "오른 지르기")
        self.video_durations = []  # proxy: last startTime (not true end)

    def analyze_single_file(self, annotation_path: Path):
        """Analyze durations in a single file"""
        with open(annotation_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        annotations = data.get("annotations", [])
        if not annotations:
            return

        segments = []
        for ann in annotations:
            movement_str = ann.get("movement", "")
            movement_id = parse_movement_id(movement_str)
            movement_name = parse_movement_name(movement_str)

            # Store the movement name (first occurrence wins)
            if movement_id not in self.movement_names:
                self.movement_names[movement_id] = movement_name

            try:
                start_time = float(ann.get("startTime", 0.0))
            except Exception:
                start_time = 0.0

            try:
                end_time_raw = ann.get("endTime", None)
                end_time = float(end_time_raw) if end_time_raw not in (None, "") else None
            except Exception:
                end_time = None

            try:
                frame = int(ann.get("frame", 0))
            except Exception:
                frame = 0

            try:
                end_frame_raw = ann.get("endFrame", None)
                end_frame = int(end_frame_raw) if end_frame_raw not in (None, "") else None
            except Exception:
                end_frame = None

            segments.append({
                "id": movement_id,
                "start_time": start_time,
                "end_time": end_time,
                "frame": frame,
                "end_frame": end_frame,
            })

        # IMPORTANT: Sort by time, not by movement number
        segments.sort(key=lambda x: x["start_time"])

        # Prefer explicit end_time from the new annotation schema. Fall back to
        # next start_time for the legacy start-only format.
        for i, curr in enumerate(segments):
            duration = None

            if curr["end_time"] is not None:
                duration = curr["end_time"] - curr["start_time"]
            elif i < len(segments) - 1:
                nxt = segments[i + 1]
                duration = nxt["start_time"] - curr["start_time"]
            elif curr["end_frame"] is not None and curr["frame"] >= 0:
                frame_diff = curr["end_frame"] - curr["frame"] + 1
                if frame_diff > 0:
                    duration = frame_diff / self.fps

            if duration is not None and duration >= 0:
                self.all_durations[curr["id"]].append(duration)

        # Prefer explicit totalDuration from the file. Fall back to the old proxy
        # behavior for legacy annotations.
        try:
            self.video_durations.append(float(data.get("totalDuration")))
        except Exception:
            self.video_durations.append(segments[-1]["start_time"])

    def analyze_all(self):
        """Analyze all annotation files"""
        annotation_files = sorted(self.annotations_dir.glob("*_annotations.json"))

        # Newer exports may use plain .json names like G001_TG1_front.json
        # instead of the older *_annotations.json pattern.
        if not annotation_files:
            annotation_files = sorted(self.annotations_dir.glob("*.json"))

        print(f"\n{'=' * 70}")
        print("MOVEMENT DURATION ANALYSIS")
        print(f"{'=' * 70}")
        print(f"Analyzing {len(annotation_files)} videos from: {self.annotations_dir}\n")

        for ann_file in annotation_files:
            self.analyze_single_file(ann_file)

        # Class summary
        print(f"\n{'=' * 70}")
        print("CLASS SUMMARY")
        print(f"{'=' * 70}\n")

        movement_ids = sorted(self.movement_names.keys(), key=sort_key_movement_id)
        print(f"Total unique classes: {len(movement_ids)}\n")
        print(f"{'Class':<10} {'ID':<8} {'Name':<40}")
        print(f"{'-' * 60}")

        for idx, mov_id in enumerate(movement_ids):
            mov_name = self.movement_names.get(mov_id, "Unknown")
            print(f"{idx:<10} {mov_id:<8} {mov_name:<40}")

        print(f"\n[INFO] Your model should have NUM_CLASSES = {len(movement_ids)}")

        # Per-movement statistics
        print(f"\n{'=' * 70}")
        print("PER-MOVEMENT DURATION STATISTICS")
        print(f"{'=' * 70}\n")

        all_stats = []
        movement_ids = sorted(self.all_durations.keys(), key=sort_key_movement_id)

        for mov_id in movement_ids:
            durations = self.all_durations[mov_id]
            if not durations:
                continue

            mean_dur = float(np.mean(durations))
            median_dur = float(np.median(durations))
            min_dur = float(np.min(durations))
            max_dur = float(np.max(durations))
            std_dur = float(np.std(durations))

            mean_frames = mean_dur * self.fps
            std_frames = std_dur * self.fps
            min_frames = min_dur * self.fps
            max_frames = max_dur * self.fps

            mov_name = self.movement_names.get(mov_id, "Unknown")

            all_stats.append({
                "movement": mov_id,
                "name": mov_name,
                "mean": mean_dur,
                "median": median_dur,
                "min": min_dur,
                "max": max_dur,
                "std": std_dur,
                "mean_frames": mean_frames,
                "min_frames": min_frames,
                "max_frames": max_frames,
                "count": len(durations),
            })

            print(f"Movement {mov_id}: {mov_name}")
            print(f"  Duration: {mean_dur:.2f}s +/- {std_dur:.2f}s (min: {min_dur:.2f}s, max: {max_dur:.2f}s)")
            print(f"  Frames@{int(self.fps)}fps: {mean_frames:.0f} +/- {std_frames:.0f} (min: {min_frames:.0f}, max: {max_frames:.0f})")
            print(f"  Samples: {len(durations)}")
            print()

        # Overall statistics
        all_movement_durations = []
        for durs in self.all_durations.values():
            all_movement_durations.extend(durs)

        if all_movement_durations:
            mean = float(np.mean(all_movement_durations))
            median = float(np.median(all_movement_durations))
            mn = float(np.min(all_movement_durations))
            mx = float(np.max(all_movement_durations))
            std = float(np.std(all_movement_durations))

            p25 = float(np.percentile(all_movement_durations, 25))
            p75 = float(np.percentile(all_movement_durations, 75))
            p90 = float(np.percentile(all_movement_durations, 90))
        else:
            mean = median = mn = mx = std = p25 = p75 = p90 = 0.0

        print(f"\n{'=' * 70}")
        print("OVERALL STATISTICS")
        print(f"{'=' * 70}\n")

        print("All Movements Combined:")
        print(f"  Mean duration: {mean:.2f}s ({mean * self.fps:.0f} frames)")
        print(f"  Median duration: {median:.2f}s ({median * self.fps:.0f} frames)")
        print(f"  Min duration: {mn:.2f}s ({mn * self.fps:.0f} frames)")
        print(f"  Max duration: {mx:.2f}s ({mx * self.fps:.0f} frames)")
        print(f"  Std deviation: {std:.2f}s ({std * self.fps:.0f} frames)")
        print(f"  Total samples: {len(all_movement_durations)}")

        print("\nPercentiles:")
        print(f"  25th percentile: {p25:.2f}s ({p25 * self.fps:.0f} frames)")
        print(f"  75th percentile: {p75:.2f}s ({p75 * self.fps:.0f} frames)")
        print(f"  90th percentile: {p90:.2f}s ({p90 * self.fps:.0f} frames)")

        # Video statistics (proxy)
        print(f"\n{'=' * 70}")
        print("VIDEO DURATION STATISTICS (proxy)")
        print(f"{'=' * 70}\n")

        if self.video_durations:
            avg_video = float(np.mean(self.video_durations))
            print(f"Average total video duration (proxy): {avg_video:.2f}s ({avg_video * self.fps:.0f} frames)")
            print(f"Min video duration (proxy): {float(np.min(self.video_durations)):.2f}s")
            print(f"Max video duration (proxy): {float(np.max(self.video_durations)):.2f}s")
        else:
            print("No video durations available (no annotations found).")

        # Recommendations + Visualization
        print(f"\n{'=' * 70}")
        print("WINDOW SIZE RECOMMENDATIONS")
        print(f"{'=' * 70}\n")

        self.make_recommendations(all_stats, mean, median)
        self.create_visualizations(all_stats)

        return all_stats

    def make_recommendations(self, all_stats, mean, median):
        """Make simple recommendations (same idea as your original)"""
        current_window = 24
        current_stride = 2

        print("Current Settings:")
        print(f"  Window size: {current_window} frames ({current_window / self.fps:.1f}s @ {int(self.fps)}fps)")
        print(f"  Stride: {current_stride} frames ({current_stride / self.fps:.1f}s @ {int(self.fps)}fps)\n")

        mean_frames = mean * self.fps
        median_frames = median * self.fps

        short_movements = sum(1 for s in all_stats if s["mean_frames"] < current_window)

        print("Analysis:")
        print(f"  Mean movement duration: {mean:.2f}s ({mean_frames:.0f} frames)")
        print(f"  Median movement duration: {median:.2f}s ({median_frames:.0f} frames)")
        print(f"  Movements shorter than {current_window} frames: {short_movements}/{len(all_stats)}")

        recommended_window_frames = int(round((median * self.fps) / 15) * 15)
        recommended_stride = max(15, recommended_window_frames // 5)

        print(f"\n{'=' * 70}")
        print("RECOMMENDATIONS:")
        print(f"{'=' * 70}")

        print("\nOption 1: KEEP CURRENT (Safe)")
        print(f"  SEQUENCE_LENGTH = {current_window}  # {current_window / self.fps:.1f}s")
        print(f"  STRIDE = {current_stride}  # {current_stride / self.fps:.1f}s")

        print("\nOption 2: OPTIMIZED FOR YOUR DATA")
        print(f"  SEQUENCE_LENGTH = {recommended_window_frames}  # {recommended_window_frames / self.fps:.1f}s")
        print(f"  STRIDE = {recommended_stride}  # {recommended_stride / self.fps:.1f}s")

        print("\nOption 3: LONGER WINDOW (For slower movements)")
        print("  SEQUENCE_LENGTH = 48  # 1.6s")
        print("  STRIDE = 8  # 0.27s")

    def create_visualizations(self, all_stats):
        """Create duration distribution plots (supports string movement IDs)"""
        try:
            if not all_stats:
                print("\n[!] No stats to visualize.")
                return

            labels = [s["movement"] for s in all_stats]
            x = np.arange(len(labels))

            means = np.array([s["mean"] for s in all_stats], dtype=float)
            mins = np.array([s["min"] for s in all_stats], dtype=float)
            maxs = np.array([s["max"] for s in all_stats], dtype=float)
            mean_frames = np.array([s["mean_frames"] for s in all_stats], dtype=float)

            fig, axes = plt.subplots(2, 2, figsize=(18, 12))

            # Plot 1: Mean duration per movement
            axes[0, 0].bar(x, means, alpha=0.7)
            axes[0, 0].axhline(y=1.07, linestyle="--", label="Current window (1.07s)")
            axes[0, 0].axhline(y=float(np.mean(means)), linestyle="--", label=f"Mean ({np.mean(means):.2f}s)")
            axes[0, 0].set_xlabel("Movement ID")
            axes[0, 0].set_ylabel("Duration (seconds)")
            axes[0, 0].set_title("Mean Duration per Movement")
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(labels, rotation=45, ha="right")
            axes[0, 0].legend()
            axes[0, 0].grid(axis="y", alpha=0.3)

            # Plot 2: Overall duration distribution
            all_durations = []
            for s in all_stats:
                all_durations.extend(self.all_durations[s["movement"]])

            axes[0, 1].hist(all_durations, bins=30, alpha=0.7, edgecolor="black")
            axes[0, 1].axvline(x=1.07, linestyle="--", label="Current window (1.07s)")
            axes[0, 1].axvline(x=float(np.mean(all_durations)), linestyle="--", label=f"Mean ({np.mean(all_durations):.2f}s)")
            axes[0, 1].set_xlabel("Duration (seconds)")
            axes[0, 1].set_ylabel("Frequency")
            axes[0, 1].set_title("Overall Duration Distribution")
            axes[0, 1].legend()
            axes[0, 1].grid(axis="y", alpha=0.3)

            # Plot 3: Min/Max range per movement
            axes[1, 0].fill_between(x, mins, maxs, alpha=0.3)
            axes[1, 0].plot(x, means, "o-", label="Mean")
            axes[1, 0].axhline(y=1.07, linestyle="--", label="Current window (1.07s)")
            axes[1, 0].set_xlabel("Movement ID")
            axes[1, 0].set_ylabel("Duration (seconds)")
            axes[1, 0].set_title("Duration Range per Movement (Min-Max)")
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(labels, rotation=45, ha="right")
            axes[1, 0].legend()
            axes[1, 0].grid(axis="y", alpha=0.3)

            # Plot 4: Mean duration in frames
            axes[1, 1].bar(x, mean_frames, alpha=0.7)
            axes[1, 1].axhline(y=32, linestyle="--", label="Current window (32 frames)")
            axes[1, 1].axhline(y=48, linestyle="--", label="Alternative (48 frames)")
            axes[1, 1].set_xlabel("Movement ID")
            axes[1, 1].set_ylabel(f"Frames @ {int(self.fps)}fps")
            axes[1, 1].set_title("Mean Duration in Frames")
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(labels, rotation=45, ha="right")
            axes[1, 1].legend()
            axes[1, 1].grid(axis="y", alpha=0.3)

            plt.tight_layout()

            output_path = Path("movement_duration_analysis.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"\n[OK] Visualization saved: {output_path}")

            # plt.show()  # Disabled for non-interactive use

        except Exception as e:
            print(f"\n[!] Could not create visualizations: {e}")


def main():
    if Paths is not None:
        Paths.create_directories()
        annotations_dir = Paths.RAW_ANNOTATIONS
    else:
        # If not using your project Paths, set folder here:
        annotations_dir = "./annotations"

    analyzer = MovementDurationAnalyzer(annotations_dir, fps=30.0)
    analyzer.analyze_all()


if __name__ == "__main__":
    main()
