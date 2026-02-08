"""Tests for visualization."""

import pytest
import json

from model_diff.checkpoint import TensorInfo
from model_diff.diff import (
    ParameterDiff,
    ChangeType,
    DiffSummary,
)
from model_diff.visualization import (
    VisualizationConfig,
    DiffVisualizer,
    create_diff_report,
)


def create_param_diff(
    name: str,
    change_type: ChangeType = ChangeType.MODIFIED,
    relative_change: float = 0.5,
) -> ParameterDiff:
    """Helper to create parameter diff."""
    info = TensorInfo(name, (64, 64), "float32", 4096, 16384)
    return ParameterDiff(
        name=name,
        change_type=change_type,
        old_info=info,
        new_info=info,
        relative_change=relative_change,
        cosine_sim=0.95,
        l2_diff=relative_change * 10,
    )


def create_test_summary() -> DiffSummary:
    """Create test summary."""
    return DiffSummary(
        total_parameters_old=1000000,
        total_parameters_new=1000000,
        added_parameters=5,
        removed_parameters=2,
        modified_parameters=100,
        unchanged_parameters=893,
        total_l2_diff=50.0,
        avg_cosine_similarity=0.95,
        most_changed_layers=["layer1", "layer2"],
    )


def create_test_analysis() -> dict:
    """Create test analysis results."""
    return {
        "layer_analysis": {
            "attention": {
                "layer_type": "attention",
                "count": 10,
                "total_change": 5.0,
                "avg_change": 0.5,
                "max_change": 0.9,
            },
            "mlp": {
                "layer_type": "mlp",
                "count": 5,
                "total_change": 1.5,
                "avg_change": 0.3,
                "max_change": 0.5,
            },
        },
        "change_pattern": "attention_heavy",
        "insights": [
            {
                "category": "training_dynamics",
                "title": "Attention Layers Changed",
                "description": "Attention layers have significant changes",
                "severity": "info",
            },
            {
                "category": "stability",
                "title": "Stable Training",
                "description": "High cosine similarity indicates stable training",
                "severity": "info",
            },
        ],
    }


class TestVisualizationConfig:
    """Tests for VisualizationConfig."""

    def test_default_config(self):
        config = VisualizationConfig()
        assert config.max_layers_display == 50
        assert config.color_scheme == "default"
        assert config.include_raw_data is False

    def test_custom_config(self):
        config = VisualizationConfig(
            max_layers_display=100,
            color_scheme="dark",
        )
        assert config.max_layers_display == 100
        assert config.color_scheme == "dark"


class TestDiffVisualizer:
    """Tests for DiffVisualizer."""

    def test_create_visualizer(self):
        viz = DiffVisualizer()
        assert viz.config is not None

    def test_create_html_report(self):
        viz = DiffVisualizer()
        diffs = [
            create_param_diff("layer1.weight", ChangeType.MODIFIED, 0.5),
            create_param_diff("layer2.weight", ChangeType.ADDED, 0.0),
        ]
        summary = create_test_summary()
        analysis = create_test_analysis()

        html = viz.create_html_report(
            diffs, summary, analysis,
            old_name="Model v1",
            new_name="Model v2",
        )

        assert "<html" in html
        assert "Model v1" in html
        assert "Model v2" in html
        assert "layer1.weight" in html

    def test_html_report_contains_summary(self):
        viz = DiffVisualizer()
        diffs = [create_param_diff("test.weight")]
        summary = create_test_summary()
        analysis = create_test_analysis()

        html = viz.create_html_report(diffs, summary, analysis)

        assert "1,000,000" in html or "1000000" in html
        assert "Modified" in html or "modified" in html

    def test_html_report_contains_insights(self):
        viz = DiffVisualizer()
        diffs = [create_param_diff("test.weight")]
        summary = create_test_summary()
        analysis = create_test_analysis()

        html = viz.create_html_report(diffs, summary, analysis)

        assert "Attention Layers Changed" in html
        assert "Stable Training" in html

    def test_html_report_layer_analysis(self):
        viz = DiffVisualizer()
        diffs = [create_param_diff("test.weight")]
        summary = create_test_summary()
        analysis = create_test_analysis()

        html = viz.create_html_report(diffs, summary, analysis)

        assert "Attention" in html
        assert "Mlp" in html

    def test_html_report_escapes_html(self):
        viz = DiffVisualizer()
        diffs = [create_param_diff("test.weight")]
        summary = create_test_summary()
        analysis = create_test_analysis()

        html = viz.create_html_report(
            diffs, summary, analysis,
            old_name="<script>alert('xss')</script>",
        )

        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html

    def test_create_json_report(self):
        viz = DiffVisualizer()
        diffs = [create_param_diff("layer1.weight")]
        summary = create_test_summary()
        analysis = create_test_analysis()

        json_str = viz.create_json_report(diffs, summary, analysis)
        data = json.loads(json_str)

        assert "summary" in data
        assert "analysis" in data
        assert "parameters" in data
        assert len(data["parameters"]) == 1

    def test_json_report_valid_structure(self):
        viz = DiffVisualizer()
        diffs = [
            create_param_diff("layer1.weight", ChangeType.MODIFIED),
            create_param_diff("layer2.weight", ChangeType.ADDED),
        ]
        summary = create_test_summary()
        analysis = create_test_analysis()

        json_str = viz.create_json_report(diffs, summary, analysis)
        data = json.loads(json_str)

        # Check summary fields
        assert data["summary"]["total_parameters_old"] == 1000000
        assert data["summary"]["modified_parameters"] == 100

        # Check parameters
        assert len(data["parameters"]) == 2
        assert data["parameters"][0]["name"] == "layer1.weight"

    def test_create_text_report(self):
        viz = DiffVisualizer()
        diffs = [create_param_diff("layer1.weight")]
        summary = create_test_summary()
        analysis = create_test_analysis()

        text = viz.create_text_report(
            diffs, summary, analysis,
            old_name="Old Model",
            new_name="New Model",
        )

        assert "Old Model" in text
        assert "New Model" in text
        assert "SUMMARY" in text
        assert "INSIGHTS" in text

    def test_text_report_contains_stats(self):
        viz = DiffVisualizer()
        diffs = [create_param_diff("layer1.weight")]
        summary = create_test_summary()
        analysis = create_test_analysis()

        text = viz.create_text_report(diffs, summary, analysis)

        assert "1,000,000" in text or "1000000" in text
        assert "Modified tensors" in text or "modified" in text.lower()

    def test_text_report_contains_parameters(self):
        viz = DiffVisualizer()
        diffs = [
            create_param_diff("layer1.weight", relative_change=0.9),
            create_param_diff("layer2.weight", relative_change=0.1),
        ]
        summary = create_test_summary()
        analysis = create_test_analysis()

        text = viz.create_text_report(diffs, summary, analysis)

        assert "layer1.weight" in text
        assert "layer2.weight" in text

    def test_render_insights_empty(self):
        viz = DiffVisualizer()
        html = viz._render_insights([])
        assert "No significant insights" in html

    def test_render_layer_analysis_empty(self):
        viz = DiffVisualizer()
        html = viz._render_layer_analysis({})
        assert "No layer analysis" in html

    def test_prepare_chart_data(self):
        viz = DiffVisualizer()
        diffs = [
            create_param_diff("layer1.weight", relative_change=0.5),
            create_param_diff("layer2.weight", relative_change=0.3),
        ]
        analysis = create_test_analysis()

        chart_data = viz._prepare_chart_data(diffs, analysis)

        assert "layer_changes" in chart_data
        assert "type_breakdown" in chart_data
        assert len(chart_data["layer_changes"]) == 2


class TestCreateDiffReport:
    """Tests for create_diff_report helper."""

    def test_create_html_format(self):
        diffs = [create_param_diff("test.weight")]
        summary = create_test_summary()
        analysis = create_test_analysis()

        report = create_diff_report(
            diffs, summary, analysis,
            format="html",
        )

        assert "<html" in report

    def test_create_json_format(self):
        diffs = [create_param_diff("test.weight")]
        summary = create_test_summary()
        analysis = create_test_analysis()

        report = create_diff_report(
            diffs, summary, analysis,
            format="json",
        )

        data = json.loads(report)
        assert "summary" in data

    def test_create_text_format(self):
        diffs = [create_param_diff("test.weight")]
        summary = create_test_summary()
        analysis = create_test_analysis()

        report = create_diff_report(
            diffs, summary, analysis,
            format="text",
        )

        assert "MODEL DIFF REPORT" in report

    def test_unknown_format_raises(self):
        diffs = [create_param_diff("test.weight")]
        summary = create_test_summary()
        analysis = create_test_analysis()

        with pytest.raises(ValueError):
            create_diff_report(diffs, summary, analysis, format="pdf")

    def test_custom_config(self):
        config = VisualizationConfig(max_layers_display=10)
        diffs = [create_param_diff(f"layer{i}.weight") for i in range(20)]
        summary = create_test_summary()
        analysis = create_test_analysis()

        report = create_diff_report(
            diffs, summary, analysis,
            format="html",
            config=config,
        )

        # Should only show 10 layers in detail
        assert "Showing top 10 of 20" in report


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_diffs(self):
        viz = DiffVisualizer()
        summary = create_test_summary()
        analysis = {"layer_analysis": {}, "change_pattern": "sparse", "insights": []}

        html = viz.create_html_report([], summary, analysis)
        assert "<html" in html

    def test_many_diffs_truncated(self):
        viz = DiffVisualizer(VisualizationConfig(max_layers_display=5))
        diffs = [create_param_diff(f"layer{i}.weight") for i in range(100)]
        summary = create_test_summary()
        analysis = create_test_analysis()

        html = viz.create_html_report(diffs, summary, analysis)
        assert "Showing top 5 of 100" in html

    def test_special_characters_in_names(self):
        viz = DiffVisualizer()
        diffs = [create_param_diff("model/layer:0/weight")]
        summary = create_test_summary()
        analysis = create_test_analysis()

        html = viz.create_html_report(diffs, summary, analysis)
        assert "model/layer:0/weight" in html

    def test_all_change_types(self):
        viz = DiffVisualizer()
        diffs = [
            create_param_diff("added", ChangeType.ADDED),
            create_param_diff("removed", ChangeType.REMOVED),
            create_param_diff("modified", ChangeType.MODIFIED),
            create_param_diff("shape_changed", ChangeType.SHAPE_CHANGED),
            create_param_diff("unchanged", ChangeType.UNCHANGED, 0.0),
        ]
        summary = create_test_summary()
        analysis = create_test_analysis()

        html = viz.create_html_report(diffs, summary, analysis)

        assert "added" in html.lower()
        assert "removed" in html.lower()
        assert "modified" in html.lower()
