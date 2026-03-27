"""
Generate HTML Report for Student Comparison

Creates a visual HTML report showing comparison results,
scores, and feedback for each movement.

Usage:
    python generate_html_report.py --comparison compare/students/P011/comparison.json
    python generate_html_report.py --comparison compare/students/P011/comparison.json --output report.html
"""
import json
from pathlib import Path
import argparse
from datetime import datetime


def get_score_color(score):
    """Get color based on score"""
    if score >= 90:
        return '#4CAF50'  # Green
    elif score >= 80:
        return '#8BC34A'  # Light green
    elif score >= 70:
        return '#FFC107'  # Yellow
    elif score >= 60:
        return '#FF9800'  # Orange
    else:
        return '#F44336'  # Red


def get_grade_color(grade):
    """Get color based on grade"""
    grade_colors = {
        'A+': '#4CAF50', 'A': '#4CAF50',
        'B+': '#8BC34A', 'B': '#8BC34A',
        'C+': '#FFC107', 'C': '#FFC107',
        'D+': '#FF9800', 'D': '#FF9800',
        'F': '#F44336'
    }
    return grade_colors.get(grade, '#9E9E9E')


def generate_html_report(comparison_path, output_path=None):
    """Generate HTML report from comparison JSON"""

    comparison_path = Path(comparison_path)
    if not comparison_path.exists():
        print(f"Comparison file not found: {comparison_path}")
        return

    with open(comparison_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if output_path is None:
        output_path = comparison_path.parent / 'report.html'
    else:
        output_path = Path(output_path)

    # Extract data
    summary = data.get('summary', {})
    movements = data.get('movements', [])
    student = data.get('student', {})
    reference = data.get('reference', {})

    overall_score = summary.get('overall_score', 0)
    overall_grade = summary.get('overall_grade', 'N/A')

    # Generate movement rows
    movement_rows = ""
    for mov in movements:
        score = mov.get('overall_score', 0)
        grade = mov.get('grade', 'N/A')
        pose = mov.get('pose_score', 0)
        timing = mov.get('timing_score', 0)

        pose_details = mov.get('pose_details', {})
        duration = mov.get('duration', {})
        feedback = mov.get('feedback', [])

        score_color = get_score_color(score)
        grade_color = get_grade_color(grade)

        # Duration info
        dur_student = duration.get('student', 0)
        dur_ref = duration.get('reference', 0)
        dur_diff = duration.get('difference', 0)
        dur_class = 'timing-fast' if dur_diff < -0.2 else 'timing-slow' if dur_diff > 0.2 else 'timing-ok'

        movement_rows += f"""
        <tr class="movement-row">
            <td class="movement-id">{mov.get('movement_id', '?')}</td>
            <td class="movement-name">{mov.get('movement_name', 'Unknown')}</td>
            <td class="score-cell">
                <div class="score-bar-container">
                    <div class="score-bar" style="width: {score}%; background: {score_color};"></div>
                    <span class="score-value">{score:.1f}</span>
                </div>
            </td>
            <td class="grade-cell" style="background: {grade_color};">{grade}</td>
            <td class="pose-cell">{pose:.1f}</td>
            <td class="timing-cell">{timing:.1f}</td>
            <td class="duration-cell {dur_class}">{dur_student:.2f}s / {dur_ref:.2f}s ({dur_diff:+.2f}s)</td>
            <td class="feedback-cell">{'<br>'.join(feedback)}</td>
        </tr>
        """

    # Generate best/worst sections
    best_movements = summary.get('best_movements', [])
    needs_improvement = summary.get('needs_improvement', [])

    best_html = ''.join([f'<span class="tag tag-best">{m}</span>' for m in best_movements])
    worst_html = ''.join([f'<span class="tag tag-worst">{m}</span>' for m in needs_improvement])

    # HTML template
    html = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Poomsae Comparison Report - {student.get('video', 'Student')}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Malgun Gothic', 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #fff;
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        h1 {{
            text-align: center;
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #4CAF50, #2196F3);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}

        .subtitle {{
            text-align: center;
            color: #888;
            margin-bottom: 30px;
        }}

        /* Summary Cards */
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}

        .summary-card {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 25px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
        }}

        .summary-card.main {{
            grid-column: span 2;
            background: linear-gradient(135deg, rgba(76,175,80,0.2), rgba(33,150,243,0.2));
        }}

        .summary-label {{
            font-size: 0.9em;
            color: #888;
            margin-bottom: 10px;
        }}

        .summary-value {{
            font-size: 2.5em;
            font-weight: bold;
        }}

        .summary-value.grade {{
            font-size: 3em;
        }}

        .summary-subtext {{
            font-size: 0.85em;
            color: #666;
            margin-top: 5px;
        }}

        /* Score Ring */
        .score-ring {{
            width: 150px;
            height: 150px;
            margin: 0 auto 15px;
            position: relative;
        }}

        .score-ring svg {{
            transform: rotate(-90deg);
        }}

        .score-ring-bg {{
            fill: none;
            stroke: rgba(255,255,255,0.1);
            stroke-width: 10;
        }}

        .score-ring-progress {{
            fill: none;
            stroke: {get_score_color(overall_score)};
            stroke-width: 10;
            stroke-linecap: round;
            stroke-dasharray: {overall_score * 4.4} 440;
            transition: stroke-dasharray 1s ease;
        }}

        .score-ring-text {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 2em;
            font-weight: bold;
        }}

        /* Tags */
        .tags-section {{
            margin-bottom: 30px;
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }}

        .tags-group {{
            flex: 1;
            min-width: 200px;
        }}

        .tags-title {{
            font-size: 0.9em;
            color: #888;
            margin-bottom: 10px;
        }}

        .tag {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            margin: 3px;
            font-size: 0.9em;
        }}

        .tag-best {{
            background: rgba(76,175,80,0.3);
            border: 1px solid #4CAF50;
        }}

        .tag-worst {{
            background: rgba(244,67,54,0.3);
            border: 1px solid #F44336;
        }}

        /* Table */
        .table-container {{
            background: rgba(255,255,255,0.03);
            border-radius: 15px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.1);
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
        }}

        th {{
            background: rgba(255,255,255,0.05);
            padding: 15px 10px;
            text-align: left;
            font-weight: 600;
            font-size: 0.85em;
            color: #888;
            text-transform: uppercase;
        }}

        td {{
            padding: 12px 10px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }}

        .movement-row:hover {{
            background: rgba(255,255,255,0.03);
        }}

        .movement-id {{
            font-weight: bold;
            color: #4CAF50;
            width: 60px;
        }}

        .movement-name {{
            max-width: 200px;
        }}

        .score-cell {{
            width: 150px;
        }}

        .score-bar-container {{
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            height: 25px;
            position: relative;
            overflow: hidden;
        }}

        .score-bar {{
            height: 100%;
            border-radius: 10px;
            transition: width 0.5s ease;
        }}

        .score-value {{
            position: absolute;
            right: 10px;
            top: 50%;
            transform: translateY(-50%);
            font-size: 0.85em;
            font-weight: bold;
        }}

        .grade-cell {{
            width: 50px;
            text-align: center;
            font-weight: bold;
            color: #fff;
            border-radius: 5px;
        }}

        .pose-cell, .timing-cell {{
            text-align: center;
            width: 70px;
        }}

        .duration-cell {{
            font-size: 0.85em;
            width: 150px;
        }}

        .timing-fast {{ color: #FF9800; }}
        .timing-slow {{ color: #2196F3; }}
        .timing-ok {{ color: #4CAF50; }}

        .feedback-cell {{
            font-size: 0.85em;
            color: #aaa;
            max-width: 200px;
        }}

        /* Footer */
        .footer {{
            text-align: center;
            margin-top: 30px;
            padding: 20px;
            color: #666;
            font-size: 0.85em;
        }}

        /* Responsive */
        @media (max-width: 1000px) {{
            .table-container {{
                overflow-x: auto;
            }}

            .summary-card.main {{
                grid-column: span 1;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Poomsae Performance Report</h1>
        <p class="subtitle">
            Student: {student.get('video', 'Unknown')} |
            Reference: {reference.get('video', 'Unknown')} |
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        </p>

        <!-- Summary Cards -->
        <div class="summary-grid">
            <div class="summary-card main">
                <div class="summary-label">Overall Score</div>
                <div class="score-ring">
                    <svg viewBox="0 0 160 160">
                        <circle class="score-ring-bg" cx="80" cy="80" r="70"></circle>
                        <circle class="score-ring-progress" cx="80" cy="80" r="70"></circle>
                    </svg>
                    <div class="score-ring-text">{overall_score:.1f}</div>
                </div>
                <div class="summary-value grade" style="color: {get_grade_color(overall_grade)};">{overall_grade}</div>
                <div class="summary-subtext">{summary.get('overall_description', '')}</div>
            </div>

            <div class="summary-card">
                <div class="summary-label">Pose Score (Avg)</div>
                <div class="summary-value" style="color: {get_score_color(summary.get('pose_score_avg', 0))};">
                    {summary.get('pose_score_avg', 0):.1f}
                </div>
                <div class="summary-subtext">Form & Technique</div>
            </div>

            <div class="summary-card">
                <div class="summary-label">Timing Score (Avg)</div>
                <div class="summary-value" style="color: {get_score_color(summary.get('timing_score_avg', 0))};">
                    {summary.get('timing_score_avg', 0):.1f}
                </div>
                <div class="summary-subtext">Speed & Rhythm</div>
            </div>

            <div class="summary-card">
                <div class="summary-label">Movements Detected</div>
                <div class="summary-value">{student.get('detected', 0)}/{len(movements)}</div>
                <div class="summary-subtext">Skipped: {student.get('skipped', 0)}</div>
            </div>

            <div class="summary-card">
                <div class="summary-label">Above 80 Points</div>
                <div class="summary-value" style="color: #4CAF50;">{summary.get('movements_above_80', 0)}</div>
                <div class="summary-subtext">movements</div>
            </div>

            <div class="summary-card">
                <div class="summary-label">Below 60 Points</div>
                <div class="summary-value" style="color: #F44336;">{summary.get('movements_below_60', 0)}</div>
                <div class="summary-subtext">movements</div>
            </div>
        </div>

        <!-- Best/Worst Tags -->
        <div class="tags-section">
            <div class="tags-group">
                <div class="tags-title">🏆 Best Movements</div>
                {best_html}
            </div>
            <div class="tags-group">
                <div class="tags-title">⚠️ Needs Improvement</div>
                {worst_html}
            </div>
        </div>

        <!-- Movement Details Table -->
        <div class="table-container">
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Movement</th>
                        <th>Score</th>
                        <th>Grade</th>
                        <th>Pose</th>
                        <th>Timing</th>
                        <th>Duration (Student / Ref)</th>
                        <th>Feedback</th>
                    </tr>
                </thead>
                <tbody>
                    {movement_rows}
                </tbody>
            </table>
        </div>

        <div class="footer">
            <p>Generated by Poomsae Recognition System</p>
            <p>Score weights: Pose {int(0.7*100)}% | Timing {int(0.3*100)}%</p>
        </div>
    </div>
</body>
</html>
"""

    # Save HTML
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"HTML report generated: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Generate HTML report')
    parser.add_argument('--comparison', required=True, help='Comparison JSON path')
    parser.add_argument('--output', help='Output HTML path')

    args = parser.parse_args()
    generate_html_report(args.comparison, args.output)


if __name__ == "__main__":
    main()
