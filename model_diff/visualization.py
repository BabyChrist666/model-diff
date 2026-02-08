"""Visualization of model diffs."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import json
import html

from .diff import ParameterDiff, ChangeType, DiffSummary, LayerDiff
from .analysis import ChangePattern, TrainingInsight, LayerAnalysis


@dataclass
class VisualizationConfig:
    """Configuration for visualization."""

    max_layers_display: int = 50
    color_scheme: str = "default"  # default, dark, light
    include_raw_data: bool = False
    chart_width: int = 800
    chart_height: int = 400


class DiffVisualizer:
    """Visualize model differences."""

    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize visualizer.

        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()

    def create_html_report(
        self,
        diffs: List[ParameterDiff],
        summary: DiffSummary,
        analysis: Dict[str, Any],
        old_name: str = "Model A",
        new_name: str = "Model B",
    ) -> str:
        """
        Create HTML report of differences.

        Args:
            diffs: Parameter diffs
            summary: Diff summary
            analysis: Analysis results
            old_name: Name for old model
            new_name: Name for new model

        Returns:
            HTML string
        """
        insights_html = self._render_insights(analysis.get("insights", []))
        summary_html = self._render_summary(summary, old_name, new_name)
        layer_analysis_html = self._render_layer_analysis(analysis.get("layer_analysis", {}))
        changes_html = self._render_changes(diffs)
        chart_data = self._prepare_chart_data(diffs, analysis)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Diff Report: {html.escape(old_name)} vs {html.escape(new_name)}</title>
    <style>
        :root {{
            --primary: #3b82f6;
            --secondary: #64748b;
            --success: #22c55e;
            --warning: #f59e0b;
            --danger: #ef4444;
            --bg: #0f172a;
            --bg-light: #1e293b;
            --text: #f1f5f9;
            --text-muted: #94a3b8;
            --border: #334155;
        }}

        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            line-height: 1.6;
        }}

        .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}

        header {{
            background: var(--bg-light);
            padding: 30px 0;
            border-bottom: 1px solid var(--border);
            margin-bottom: 30px;
        }}

        h1 {{ font-size: 2rem; margin-bottom: 10px; }}
        h2 {{ font-size: 1.5rem; margin: 30px 0 20px; color: var(--primary); }}
        h3 {{ font-size: 1.2rem; margin-bottom: 15px; }}

        .subtitle {{ color: var(--text-muted); font-size: 1.1rem; }}

        .card {{
            background: var(--bg-light);
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 20px;
            border: 1px solid var(--border);
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}

        .stat-card {{
            background: var(--bg);
            border-radius: 8px;
            padding: 20px;
            text-align: center;
        }}

        .stat-value {{
            font-size: 2rem;
            font-weight: bold;
            color: var(--primary);
        }}

        .stat-label {{
            color: var(--text-muted);
            font-size: 0.9rem;
            margin-top: 5px;
        }}

        .insight {{
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 15px;
            border-left: 4px solid;
        }}

        .insight-info {{ background: rgba(59, 130, 246, 0.1); border-color: var(--primary); }}
        .insight-warning {{ background: rgba(245, 158, 11, 0.1); border-color: var(--warning); }}
        .insight-critical {{ background: rgba(239, 68, 68, 0.1); border-color: var(--danger); }}

        .insight-title {{ font-weight: 600; margin-bottom: 5px; }}
        .insight-desc {{ color: var(--text-muted); font-size: 0.95rem; }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.9rem;
        }}

        th, td {{
            padding: 12px 15px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}

        th {{ background: var(--bg); color: var(--text-muted); font-weight: 600; }}
        tr:hover {{ background: rgba(59, 130, 246, 0.05); }}

        .change-added {{ color: var(--success); }}
        .change-removed {{ color: var(--danger); }}
        .change-modified {{ color: var(--warning); }}

        .progress-bar {{
            height: 8px;
            background: var(--bg);
            border-radius: 4px;
            overflow: hidden;
        }}

        .progress-fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s;
        }}

        .chart-container {{
            width: 100%;
            height: 300px;
            margin: 20px 0;
        }}

        .badge {{
            display: inline-block;
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
        }}

        .badge-pattern {{ background: rgba(59, 130, 246, 0.2); color: var(--primary); }}

        @media (max-width: 768px) {{
            .stats-grid {{ grid-template-columns: repeat(2, 1fr); }}
        }}
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1>Model Diff Report</h1>
            <p class="subtitle">{html.escape(old_name)} → {html.escape(new_name)}</p>
        </div>
    </header>

    <div class="container">
        {summary_html}

        <h2>Training Insights</h2>
        <div class="card">
            <span class="badge badge-pattern">{analysis.get('change_pattern', 'unknown').replace('_', ' ').title()}</span>
            {insights_html}
        </div>

        <h2>Layer Type Analysis</h2>
        <div class="card">
            {layer_analysis_html}
        </div>

        <h2>Parameter Changes</h2>
        <div class="card">
            {changes_html}
        </div>
    </div>

    <script>
        const chartData = {json.dumps(chart_data)};
        // Chart rendering would go here with a library like Chart.js
    </script>
</body>
</html>"""

    def _render_summary(
        self,
        summary: DiffSummary,
        old_name: str,
        new_name: str,
    ) -> str:
        """Render summary section."""
        return f"""
        <h2>Summary</h2>
        <div class="card">
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{summary.total_parameters_old:,}</div>
                    <div class="stat-label">Parameters ({html.escape(old_name)})</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{summary.total_parameters_new:,}</div>
                    <div class="stat-label">Parameters ({html.escape(new_name)})</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{summary.modified_parameters}</div>
                    <div class="stat-label">Modified Tensors</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{summary.avg_cosine_similarity:.3f}</div>
                    <div class="stat-label">Avg Cosine Similarity</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{summary.added_parameters}</div>
                    <div class="stat-label">Added Tensors</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{summary.removed_parameters}</div>
                    <div class="stat-label">Removed Tensors</div>
                </div>
            </div>
        </div>
        """

    def _render_insights(self, insights: List[Dict]) -> str:
        """Render insights section."""
        if not insights:
            return "<p>No significant insights detected.</p>"

        html_parts = []
        for insight in insights:
            severity = insight.get("severity", "info")
            html_parts.append(f"""
            <div class="insight insight-{severity}">
                <div class="insight-title">{html.escape(insight.get('title', ''))}</div>
                <div class="insight-desc">{html.escape(insight.get('description', ''))}</div>
            </div>
            """)

        return "\n".join(html_parts)

    def _render_layer_analysis(self, layer_analysis: Dict) -> str:
        """Render layer analysis section."""
        if not layer_analysis:
            return "<p>No layer analysis available.</p>"

        rows = []
        for layer_type, data in layer_analysis.items():
            avg_change = data.get("avg_change", 0)
            max_change = data.get("max_change", 0)
            count = data.get("count", 0)

            bar_width = min(avg_change * 100, 100)
            bar_color = "#ef4444" if avg_change > 0.5 else "#f59e0b" if avg_change > 0.1 else "#22c55e"

            rows.append(f"""
            <tr>
                <td><strong>{html.escape(layer_type.title())}</strong></td>
                <td>{count}</td>
                <td>{avg_change:.4f}</td>
                <td>{max_change:.4f}</td>
                <td>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {bar_width}%; background: {bar_color};"></div>
                    </div>
                </td>
            </tr>
            """)

        return f"""
        <table>
            <thead>
                <tr>
                    <th>Layer Type</th>
                    <th>Count</th>
                    <th>Avg Change</th>
                    <th>Max Change</th>
                    <th>Relative</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        """

    def _render_changes(self, diffs: List[ParameterDiff]) -> str:
        """Render changes table."""
        # Sort by relative change
        sorted_diffs = sorted(
            diffs,
            key=lambda x: x.relative_change,
            reverse=True,
        )[:self.config.max_layers_display]

        rows = []
        for diff in sorted_diffs:
            change_class = {
                ChangeType.ADDED: "change-added",
                ChangeType.REMOVED: "change-removed",
                ChangeType.MODIFIED: "change-modified",
                ChangeType.SHAPE_CHANGED: "change-modified",
                ChangeType.UNCHANGED: "",
            }.get(diff.change_type, "")

            shape = ""
            if diff.old_info:
                shape = str(list(diff.old_info.shape))
            elif diff.new_info:
                shape = str(list(diff.new_info.shape))

            rows.append(f"""
            <tr>
                <td>{html.escape(diff.name)}</td>
                <td><span class="{change_class}">{diff.change_type.value}</span></td>
                <td>{shape}</td>
                <td>{diff.relative_change:.6f}</td>
                <td>{diff.cosine_sim:.4f}</td>
                <td>{diff.l2_diff:.4f}</td>
            </tr>
            """)

        return f"""
        <table>
            <thead>
                <tr>
                    <th>Parameter Name</th>
                    <th>Change Type</th>
                    <th>Shape</th>
                    <th>Relative Change</th>
                    <th>Cosine Sim</th>
                    <th>L2 Diff</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        <p style="color: var(--text-muted); margin-top: 15px;">
            Showing top {len(rows)} of {len(diffs)} parameters
        </p>
        """

    def _prepare_chart_data(
        self,
        diffs: List[ParameterDiff],
        analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Prepare data for charts."""
        # Layer change distribution
        layer_changes = []
        for diff in diffs:
            if diff.change_type == ChangeType.MODIFIED:
                layer_changes.append({
                    "name": diff.name,
                    "relative_change": diff.relative_change,
                    "cosine_sim": diff.cosine_sim,
                    "l2_diff": diff.l2_diff,
                })

        # Layer type breakdown
        layer_analysis = analysis.get("layer_analysis", {})
        type_breakdown = [
            {"type": k, "avg_change": v.get("avg_change", 0)}
            for k, v in layer_analysis.items()
        ]

        return {
            "layer_changes": layer_changes[:100],  # Limit for performance
            "type_breakdown": type_breakdown,
        }

    def create_json_report(
        self,
        diffs: List[ParameterDiff],
        summary: DiffSummary,
        analysis: Dict[str, Any],
    ) -> str:
        """Create JSON report."""
        return json.dumps({
            "summary": summary.to_dict(),
            "analysis": analysis,
            "parameters": [d.to_dict() for d in diffs],
        }, indent=2)

    def create_text_report(
        self,
        diffs: List[ParameterDiff],
        summary: DiffSummary,
        analysis: Dict[str, Any],
        old_name: str = "Model A",
        new_name: str = "Model B",
    ) -> str:
        """Create text report."""
        lines = [
            "=" * 60,
            f"MODEL DIFF REPORT: {old_name} → {new_name}",
            "=" * 60,
            "",
            "SUMMARY",
            "-" * 40,
            f"Parameters ({old_name}): {summary.total_parameters_old:,}",
            f"Parameters ({new_name}): {summary.total_parameters_new:,}",
            f"Modified tensors: {summary.modified_parameters}",
            f"Added tensors: {summary.added_parameters}",
            f"Removed tensors: {summary.removed_parameters}",
            f"Avg cosine similarity: {summary.avg_cosine_similarity:.4f}",
            f"Total L2 diff: {summary.total_l2_diff:.4f}",
            "",
            "CHANGE PATTERN",
            "-" * 40,
            f"Pattern: {analysis.get('change_pattern', 'unknown')}",
            "",
            "INSIGHTS",
            "-" * 40,
        ]

        for insight in analysis.get("insights", []):
            lines.append(f"[{insight.get('severity', 'info').upper()}] {insight.get('title', '')}")
            lines.append(f"  {insight.get('description', '')}")
            lines.append("")

        lines.extend([
            "TOP CHANGED PARAMETERS",
            "-" * 40,
        ])

        sorted_diffs = sorted(
            [d for d in diffs if d.change_type == ChangeType.MODIFIED],
            key=lambda x: x.relative_change,
            reverse=True,
        )[:20]

        for diff in sorted_diffs:
            lines.append(f"{diff.name}: {diff.relative_change:.6f} (cos={diff.cosine_sim:.4f})")

        return "\n".join(lines)


def create_diff_report(
    diffs: List[ParameterDiff],
    summary: DiffSummary,
    analysis: Dict[str, Any],
    format: str = "html",
    old_name: str = "Model A",
    new_name: str = "Model B",
    config: Optional[VisualizationConfig] = None,
) -> str:
    """
    Create a diff report.

    Args:
        diffs: Parameter diffs
        summary: Diff summary
        analysis: Analysis results
        format: Output format (html, json, text)
        old_name: Name for old model
        new_name: Name for new model
        config: Visualization config

    Returns:
        Report string
    """
    visualizer = DiffVisualizer(config)

    if format == "html":
        return visualizer.create_html_report(diffs, summary, analysis, old_name, new_name)
    elif format == "json":
        return visualizer.create_json_report(diffs, summary, analysis)
    elif format == "text":
        return visualizer.create_text_report(diffs, summary, analysis, old_name, new_name)
    else:
        raise ValueError(f"Unknown format: {format}")
