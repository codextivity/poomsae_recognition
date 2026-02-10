"""
Generate Detailed HTML Report

Usage:
    python generate_report.py --comparison P001_comparison.json --charts results/charts/ --output P001_report.html
"""

import json
from pathlib import Path
import base64
from datetime import datetime


class ReportGenerator:
    """Generate HTML report"""

    def __init__(self, comparison_json_path, charts_dir):
        """Load comparison results and charts"""
        with open(comparison_json_path, 'r', encoding='utf-8') as f:
            self.results = json.load(f)

        self.charts_dir = Path(charts_dir)

        print(f"✓ Loaded comparison results")

    def image_to_base64(self, image_path):
        """Convert image to base64 for embedding"""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode()

    def generate_html_report(self, output_path):
        """Generate complete HTML report"""

        summary = self.results['overall_summary']
        timestamp = datetime.fromisoformat(self.results['timestamp']).strftime('%Y-%m-%d %H:%M:%S')

        html = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Poomsae Performance Report</title>
    <style>
        body {{
            font-family: 'Malgun Gothic', sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 40px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 5px solid #3498db;
            padding-left: 15px;
        }}
        .summary {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin: 20px 0;
        }}
        .summary h2 {{
            color: white;
            border-left: 5px solid white;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        .summary-item {{
            text-align: center;
        }}
        .summary-item .value {{
            font-size: 36px;
            font-weight: bold;
            margin: 10px 0;
        }}
        .summary-item .label {{
            font-size: 14px;
            opacity: 0.9;
        }}
        .chart {{
            margin: 30px 0;
            text-align: center;
        }}
        .chart img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .grade {{
            font-weight: bold;
            padding: 5px 10px;
            border-radius: 5px;
            display: inline-block;
        }}
        .grade-a {{ background-color: #2ecc71; color: white; }}
        .grade-b {{ background-color: #3498db; color: white; }}
        .grade-c {{ background-color: #f39c12; color: white; }}
        .grade-d {{ background-color: #e74c3c; color: white; }}
        .grade-f {{ background-color: #c0392b; color: white; }}
        .feedback {{
            background-color: #ecf0f1;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        .strength {{
            color: #27ae60;
            font-weight: bold;
        }}
        .weakness {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🥋 Taekwondo Poomsae Performance Report</h1>

        <div style="margin: 20px 0;">
            <strong>Student Video:</strong> {self.results['student_video']}<br>
            <strong>Reference Video:</strong> {self.results['reference_video']}<br>
            <strong>Analysis Date:</strong> {timestamp}
        </div>

        <div class="summary">
            <h2>📊 Overall Performance Summary</h2>
            <div class="summary-grid">
                <div class="summary-item">
                    <div class="label">Average Score</div>
                    <div class="value">{summary['average_score']:.1f}/100</div>
                </div>
                <div class="summary-item">
                    <div class="label">Overall Grade</div>
                    <div class="value">{summary['overall_grade']}</div>
                </div>
                <div class="summary-item">
                    <div class="label">Movements ≥80</div>
                    <div class="value">{summary['movements_above_80']}/20</div>
                </div>
                <div class="summary-item">
                    <div class="label">Score Range</div>
                    <div class="value" style="font-size: 20px;">{summary['min_score']:.1f} - {summary['max_score']:.1f}</div>
                </div>
            </div>

            <div class="feedback">
                <p><span class="strength">✓ Strengths:</span> {', '.join(summary['strengths'])}</p>
                <p><span class="weakness">⚠ Areas for Improvement:</span> {', '.join(summary['weaknesses'])}</p>
            </div>
        </div>
"""

        # Add charts
        chart_files = [
            ('01_overall_scores.png', 'Overall Scores by Movement'),
            ('02_movement_comparison.png', 'Temporal vs Pose Scores'),
            ('03_score_breakdown.png', 'Score Breakdown'),
            ('04_key_poses.png', 'Key Pose Similarities'),
            ('05_grade_distribution.png', 'Grade Distribution')
        ]

        html += "\n<h2>📈 Visual Analysis</h2>\n"

        for filename, title in chart_files:
            chart_path = self.charts_dir / filename
            if chart_path.exists():
                img_base64 = self.image_to_base64(chart_path)
                html += f"""
        <div class="chart">
            <h3>{title}</h3>
            <img src="data:image/png;base64,{img_base64}" alt="{title}">
        </div>
"""

        # Movement details table
        html += """
        <h2>📋 Detailed Movement Analysis</h2>
        <table>
            <tr>
                <th>Movement</th>
                <th>Overall Score</th>
                <th>Grade</th>
                <th>Temporal</th>
                <th>Pose</th>
                <th>Duration Diff</th>
                <th>Feedback</th>
            </tr>
"""

        for m in self.results['movement_scores']:
            grade_class = m['grade'][0].lower()
            if grade_class in ['a', 'b']:
                grade_class = f'grade-{grade_class}'
            else:
                grade_class = f'grade-{grade_class}'

            html += f"""
            <tr>
                <td><strong>Movement {m['movement_number']}</strong><br>{m['movement_name']}</td>
                <td><strong>{m['overall_score']:.1f}/100</strong></td>
                <td><span class="{grade_class} grade">{m['grade']}</span></td>
                <td>{m['temporal_score']:.1f}</td>
                <td>{m['pose_score']:.1f}</td>
                <td>{m['duration_diff']:+.2f}s</td>
                <td>{m['feedback']}</td>
            </tr>
"""

        html += """
        </table>

        <div class="footer">
            <p>Generated by Poomsae Recognition System</p>
            <p>© 2025 Taekwondo Performance Analysis</p>
        </div>

    </div>
</body>
</html>
"""

        # Save report
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        print(f"\n{'=' * 70}")
        print(f"✅ HTML report generated: {output_path}")
        print(f"   Open in browser to view")
        print(f"{'=' * 70}\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate HTML report')
    parser.add_argument('--comparison', required=True, help='Comparison JSON file')
    parser.add_argument('--charts', required=True, help='Charts directory')
    parser.add_argument('--output', required=True, help='Output HTML file')

    args = parser.parse_args()

    generator = ReportGenerator(args.comparison, args.charts)
    generator.generate_html_report(args.output)


if __name__ == "__main__":
    main()
#  python generate_report.py --comparison "D:\All Docs\All Projects\Pycharm\poomsae_recognition\compare\results\comparison\comparison.json" --charts "D:\All Docs\All Projects\Pycharm\poomsae_recognition\compare\results\charts\" --output P001_report.html