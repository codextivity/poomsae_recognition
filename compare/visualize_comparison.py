"""
Create Visual Comparison Charts

Usage:
    python visualize_comparison.py --comparison P001_comparison.json --output results/charts/
"""

import json
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
from pathlib import Path
import pandas as pd

# Configure matplotlib for Korean font
try:
    # Windows
    font_path = "C:/Windows/Fonts/malgun.ttf"
    font_prop = fm.FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
except:
    try:
        # Linux
        plt.rcParams['font.family'] = 'NanumGothic'
    except:
        print("⚠️  Korean font not found, using default")

plt.rcParams['axes.unicode_minus'] = False


class ComparisonVisualizer:
    """Create visual comparison charts"""

    def __init__(self, comparison_json_path):
        """Load comparison results"""
        with open(comparison_json_path, 'r', encoding='utf-8') as f:
            self.results = json.load(f)

        print(f"✓ Loaded comparison results")
        print(f"  Student: {self.results['student_video']}")
        print(f"  Reference: {self.results['reference_video']}")
        print(f"  Average Score: {self.results['overall_summary']['average_score']:.1f}/100")

    def create_all_charts(self, output_dir):
        """Create all visualization charts"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'=' * 70}")
        print(f"CREATING VISUALIZATIONS")
        print(f"{'=' * 70}\n")

        # 1. Overall score chart
        print("Creating overall score chart...")
        self.plot_overall_scores(output_dir / "01_overall_scores.png")

        # 2. Movement-by-movement comparison
        print("Creating movement comparison chart...")
        self.plot_movement_comparison(output_dir / "02_movement_comparison.png")

        # 3. Score breakdown (temporal vs pose)
        print("Creating score breakdown...")
        self.plot_score_breakdown(output_dir / "03_score_breakdown.png")

        # 4. Key pose similarities
        print("Creating key pose chart...")
        self.plot_key_poses(output_dir / "04_key_poses.png")

        # 5. Grade distribution
        print("Creating grade distribution...")
        self.plot_grade_distribution(output_dir / "05_grade_distribution.png")

        print(f"\n✅ All charts saved to: {output_dir}")
        print(f"{'=' * 70}\n")

    def plot_overall_scores(self, output_path):
        """Plot overall score bar chart"""
        movements = [m['movement_number'] for m in self.results['movement_scores']]
        scores = [m['overall_score'] for m in self.results['movement_scores']]
        grades = [m['grade'] for m in self.results['movement_scores']]

        fig, ax = plt.subplots(figsize=(16, 8))

        # Color code by grade
        colors = []
        for score in scores:
            if score >= 90:
                colors.append('#2ecc71')  # Green
            elif score >= 80:
                colors.append('#3498db')  # Blue
            elif score >= 70:
                colors.append('#f39c12')  # Orange
            else:
                colors.append('#e74c3c')  # Red

        bars = ax.bar(movements, scores, color=colors, alpha=0.8, edgecolor='black')

        # Add score labels on bars
        for bar, score, grade in zip(bars, scores, grades):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                    f'{score:.1f}\n{grade}',
                    ha='center', va='bottom', fontsize=9, weight='bold')

        # Add threshold lines
        ax.axhline(y=90, color='green', linestyle='--', alpha=0.3, label='A+ threshold (90)')
        ax.axhline(y=80, color='blue', linestyle='--', alpha=0.3, label='B+ threshold (80)')
        ax.axhline(y=70, color='orange', linestyle='--', alpha=0.3, label='C+ threshold (70)')
        ax.axhline(y=60, color='red', linestyle='--', alpha=0.3, label='D threshold (60)')

        ax.set_xlabel('Movement Number', fontsize=14, weight='bold')
        ax.set_ylabel('Score', fontsize=14, weight='bold')
        ax.set_title(
            f'Movement Scores - Overall: {self.results["overall_summary"]["average_score"]:.1f}/100 ({self.results["overall_summary"]["overall_grade"]})',
            fontsize=16, weight='bold')
        ax.set_ylim(0, 105)
        ax.set_xticks(movements)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(loc='lower right')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_movement_comparison(self, output_path):
        """Plot temporal vs pose scores"""
        movements = [m['movement_number'] for m in self.results['movement_scores']]
        temporal_scores = [m['temporal_score'] for m in self.results['movement_scores']]
        pose_scores = [m['pose_score'] for m in self.results['movement_scores']]

        fig, ax = plt.subplots(figsize=(16, 8))

        x = np.arange(len(movements))
        width = 0.35

        bars1 = ax.bar(x - width / 2, temporal_scores, width, label='Temporal (30%)',
                       color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width / 2, pose_scores, width, label='Pose (70%)',
                       color='#e74c3c', alpha=0.8)

        ax.set_xlabel('Movement Number', fontsize=14, weight='bold')
        ax.set_ylabel('Score', fontsize=14, weight='bold')
        ax.set_title('Temporal vs Pose Scores by Movement', fontsize=16, weight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(movements)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 105)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_score_breakdown(self, output_path):
        """Plot score components as stacked bars"""
        movements = [m['movement_number'] for m in self.results['movement_scores']]
        temporal_weighted = [m['temporal_score'] * 0.3 for m in self.results['movement_scores']]
        pose_weighted = [m['pose_score'] * 0.7 for m in self.results['movement_scores']]

        fig, ax = plt.subplots(figsize=(16, 8))

        ax.bar(movements, temporal_weighted, label='Temporal (30%)',
               color='#3498db', alpha=0.8)
        ax.bar(movements, pose_weighted, bottom=temporal_weighted,
               label='Pose (70%)', color='#e74c3c', alpha=0.8)

        ax.set_xlabel('Movement Number', fontsize=14, weight='bold')
        ax.set_ylabel('Weighted Score Contribution', fontsize=14, weight='bold')
        ax.set_title('Score Breakdown (Weighted)', fontsize=16, weight='bold')
        ax.set_xticks(movements)
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 105)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_key_poses(self, output_path):
        """Plot key pose similarities"""
        movements = [m['movement_number'] for m in self.results['movement_scores']]

        start_scores = []
        middle_scores = []
        end_scores = []

        for m in self.results['movement_scores']:
            kp_scores = m['key_pose_scores']
            start_scores.append(kp_scores.get('start', 0))
            middle_scores.append(kp_scores.get('middle', 0))
            end_scores.append(kp_scores.get('end', 0))

        fig, ax = plt.subplots(figsize=(16, 8))

        ax.plot(movements, start_scores, 'o-', label='Start Pose',
                color='#2ecc71', linewidth=2, markersize=8)
        ax.plot(movements, middle_scores, 's-', label='Middle Pose',
                color='#3498db', linewidth=2, markersize=8)
        ax.plot(movements, end_scores, '^-', label='End Pose',
                color='#e74c3c', linewidth=2, markersize=8)

        ax.set_xlabel('Movement Number', fontsize=14, weight='bold')
        ax.set_ylabel('Similarity Score', fontsize=14, weight='bold')
        ax.set_title('Key Pose Similarities', fontsize=16, weight='bold')
        ax.set_xticks(movements)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_grade_distribution(self, output_path):
        """Plot grade distribution pie chart"""
        grades = [m['grade'] for m in self.results['movement_scores']]

        grade_counts = {}
        for grade in grades:
            grade_counts[grade] = grade_counts.get(grade, 0) + 1

        # Sort grades
        grade_order = ['A+', 'A', 'B+', 'B', 'C+', 'C', 'D', 'F']
        sorted_grades = [g for g in grade_order if g in grade_counts]
        counts = [grade_counts[g] for g in sorted_grades]

        colors = ['#2ecc71', '#27ae60', '#3498db', '#2980b9',
                  '#f39c12', '#e67e22', '#e74c3c', '#c0392b']
        colors = colors[:len(sorted_grades)]

        fig, ax = plt.subplots(figsize=(10, 8))

        wedges, texts, autotexts = ax.pie(counts, labels=sorted_grades,
                                          autopct='%1.1f%%', startangle=90,
                                          colors=colors, textprops={'fontsize': 12, 'weight': 'bold'})

        ax.set_title(f'Grade Distribution\nOverall: {self.results["overall_summary"]["overall_grade"]}',
                     fontsize=16, weight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Create visual comparison charts')
    parser.add_argument('--comparison', required=True, help='Comparison JSON file')
    parser.add_argument('--output', required=True, help='Output directory for charts')

    args = parser.parse_args()

    visualizer = ComparisonVisualizer(args.comparison)
    visualizer.create_all_charts(args.output)


if __name__ == "__main__":
    main()


# python visualize_comparison.py --comparison "D:\All Docs\All Projects\Pycharm\poomsae_recognition\compare\results\comparison\comparison.json" --output results/charts/
# """