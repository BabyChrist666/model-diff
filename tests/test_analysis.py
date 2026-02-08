"""Tests for diff analysis."""

import pytest

from model_diff.checkpoint import TensorInfo
from model_diff.diff import (
    ParameterDiff,
    ChangeType,
    DiffSummary,
)
from model_diff.analysis import (
    ChangePattern,
    TrainingInsight,
    LayerAnalysis,
    AnalysisConfig,
    DiffAnalyzer,
)


def create_param_diff(name: str, relative_change: float = 0.0, cosine_sim: float = 1.0) -> ParameterDiff:
    """Helper to create parameter diff."""
    info = TensorInfo(name, (64, 64), "float32", 4096, 16384)
    change_type = ChangeType.MODIFIED if relative_change > 0 else ChangeType.UNCHANGED
    return ParameterDiff(
        name=name,
        change_type=change_type,
        old_info=info,
        new_info=info,
        relative_change=relative_change,
        cosine_sim=cosine_sim,
        l2_diff=relative_change * 10,
    )


class TestChangePattern:
    """Tests for ChangePattern enum."""

    def test_patterns_exist(self):
        assert ChangePattern.UNIFORM
        assert ChangePattern.GRADIENT
        assert ChangePattern.INVERSE_GRADIENT
        assert ChangePattern.FOCUSED
        assert ChangePattern.SPARSE
        assert ChangePattern.ATTENTION_HEAVY
        assert ChangePattern.MLP_HEAVY
        assert ChangePattern.EMBEDDING_HEAVY

    def test_pattern_values(self):
        assert ChangePattern.UNIFORM.value == "uniform"
        assert ChangePattern.GRADIENT.value == "gradient"


class TestTrainingInsight:
    """Tests for TrainingInsight."""

    def test_create_insight(self):
        insight = TrainingInsight(
            category="training_dynamics",
            title="Test Insight",
            description="This is a test",
            severity="warning",
        )
        assert insight.category == "training_dynamics"
        assert insight.severity == "warning"

    def test_to_dict(self):
        insight = TrainingInsight(
            category="test",
            title="Test",
            description="Description",
            evidence={"key": "value"},
        )
        d = insight.to_dict()
        assert d["category"] == "test"
        assert d["evidence"]["key"] == "value"


class TestLayerAnalysis:
    """Tests for LayerAnalysis."""

    def test_create_analysis(self):
        analysis = LayerAnalysis(
            layer_type="attention",
            count=10,
            total_change=5.0,
            avg_change=0.5,
            max_change=1.0,
            layers=["attn.0", "attn.1"],
        )
        assert analysis.layer_type == "attention"
        assert analysis.count == 10

    def test_to_dict(self):
        analysis = LayerAnalysis(
            layer_type="mlp",
            count=5,
            total_change=2.5,
            avg_change=0.5,
            max_change=0.8,
        )
        d = analysis.to_dict()
        assert d["layer_type"] == "mlp"
        assert d["avg_change"] == 0.5


class TestAnalysisConfig:
    """Tests for AnalysisConfig."""

    def test_default_config(self):
        config = AnalysisConfig()
        assert config.significant_change_threshold == 0.1
        assert "attention" in config.layer_patterns
        assert "mlp" in config.layer_patterns

    def test_custom_threshold(self):
        config = AnalysisConfig(significant_change_threshold=0.2)
        assert config.significant_change_threshold == 0.2


class TestDiffAnalyzer:
    """Tests for DiffAnalyzer."""

    def test_create_analyzer(self):
        analyzer = DiffAnalyzer()
        assert analyzer.config is not None

    def test_analyze_empty(self):
        analyzer = DiffAnalyzer()
        summary = DiffSummary(
            total_parameters_old=0,
            total_parameters_new=0,
            added_parameters=0,
            removed_parameters=0,
            modified_parameters=0,
            unchanged_parameters=0,
            total_l2_diff=0,
            avg_cosine_similarity=1.0,
        )

        result = analyzer.analyze([], summary)

        assert "layer_analysis" in result
        assert "change_pattern" in result
        assert "insights" in result

    def test_analyze_layer_types(self):
        analyzer = DiffAnalyzer()
        diffs = [
            create_param_diff("model.attention.q_proj.weight", 0.5),
            create_param_diff("model.attention.k_proj.weight", 0.4),
            create_param_diff("model.mlp.fc1.weight", 0.2),
        ]

        layer_analysis = analyzer.analyze_layer_types(diffs)

        assert "attention" in layer_analysis
        assert layer_analysis["attention"].count == 2

    def test_detect_sparse_pattern(self):
        analyzer = DiffAnalyzer()
        # No modified parameters at all - should be sparse
        diffs = [create_param_diff(f"layer{i}.weight", 0.0) for i in range(10)]

        pattern = analyzer.detect_change_pattern(diffs, {})
        assert pattern == ChangePattern.SPARSE

    def test_detect_attention_heavy_pattern(self):
        analyzer = DiffAnalyzer()
        diffs = [
            create_param_diff("model.attention.q_proj.weight", 0.8),
            create_param_diff("model.attention.k_proj.weight", 0.7),
            create_param_diff("model.mlp.fc1.weight", 0.1),
        ]

        layer_analysis = analyzer.analyze_layer_types(diffs)
        pattern = analyzer.detect_change_pattern(diffs, layer_analysis)

        assert pattern == ChangePattern.ATTENTION_HEAVY

    def test_detect_mlp_heavy_pattern(self):
        analyzer = DiffAnalyzer()
        diffs = [
            create_param_diff("model.attention.q_proj.weight", 0.1),
            create_param_diff("model.mlp.fc1.weight", 0.8),
            create_param_diff("model.mlp.fc2.weight", 0.7),
        ]

        layer_analysis = analyzer.analyze_layer_types(diffs)
        pattern = analyzer.detect_change_pattern(diffs, layer_analysis)

        assert pattern == ChangePattern.MLP_HEAVY

    def test_generate_insights_architecture_change(self):
        analyzer = DiffAnalyzer()
        summary = DiffSummary(
            total_parameters_old=1000,
            total_parameters_new=1100,
            added_parameters=100,
            removed_parameters=0,
            modified_parameters=50,
            unchanged_parameters=850,
            total_l2_diff=10.0,
            avg_cosine_similarity=0.95,
        )

        insights = analyzer.generate_insights([], summary, {}, ChangePattern.UNIFORM)

        arch_insights = [i for i in insights if i.category == "architecture"]
        assert len(arch_insights) == 1
        assert "Added" in arch_insights[0].description

    def test_generate_insights_low_cosine(self):
        analyzer = DiffAnalyzer()
        summary = DiffSummary(
            total_parameters_old=1000,
            total_parameters_new=1000,
            added_parameters=0,
            removed_parameters=0,
            modified_parameters=100,
            unchanged_parameters=900,
            total_l2_diff=50.0,
            avg_cosine_similarity=0.75,  # Low cosine sim
        )

        insights = analyzer.generate_insights([], summary, {}, ChangePattern.UNIFORM)

        stability_insights = [i for i in insights if i.category == "stability"]
        assert len(stability_insights) == 1
        assert stability_insights[0].severity == "warning"

    def test_generate_insights_frozen_layers(self):
        analyzer = DiffAnalyzer()
        diffs = [create_param_diff(f"layer{i}.weight", 0.0) for i in range(10)]

        summary = DiffSummary(
            total_parameters_old=1000,
            total_parameters_new=1000,
            added_parameters=0,
            removed_parameters=0,
            modified_parameters=0,
            unchanged_parameters=10,
            total_l2_diff=0.0,
            avg_cosine_similarity=1.0,
        )

        insights = analyzer.generate_insights(diffs, summary, {}, ChangePattern.SPARSE)

        frozen_insights = [i for i in insights if "Frozen" in i.title]
        assert len(frozen_insights) == 1

    def test_get_most_changed_parameters(self):
        analyzer = DiffAnalyzer()
        diffs = [
            create_param_diff("layer1.weight", 0.9),
            create_param_diff("layer2.weight", 0.5),
            create_param_diff("layer3.weight", 0.1),
        ]

        top = analyzer.get_most_changed_parameters(diffs, top_k=2)

        assert len(top) == 2
        assert top[0].name == "layer1.weight"
        assert top[1].name == "layer2.weight"

    def test_get_layer_change_ranking(self):
        analyzer = DiffAnalyzer()
        diffs = [
            create_param_diff("layer1.weight", 0.5),
            create_param_diff("layer1.bias", 0.3),
            create_param_diff("layer2.weight", 0.1),
        ]

        ranking = analyzer.get_layer_change_ranking(diffs)

        assert ranking[0][0] == "layer1"
        assert ranking[0][1] == 0.8  # 0.5 + 0.3

    def test_compare_training_stages(self):
        analyzer = DiffAnalyzer()

        stage1 = [
            create_param_diff("layer1.weight", 0.5),
            create_param_diff("layer2.weight", 0.3),
        ]
        stage2 = [
            create_param_diff("layer1.weight", 0.1),  # Decelerating
            create_param_diff("layer2.weight", 0.6),  # Accelerating
        ]

        comparison = analyzer.compare_training_stages(stage1, stage2)

        assert "accelerating" in comparison
        assert "decelerating" in comparison


class TestFullAnalysis:
    """Integration tests for full analysis."""

    def test_full_analysis_workflow(self):
        analyzer = DiffAnalyzer()

        diffs = [
            create_param_diff("model.layer.0.attention.q_proj.weight", 0.6),
            create_param_diff("model.layer.0.attention.k_proj.weight", 0.5),
            create_param_diff("model.layer.0.mlp.fc1.weight", 0.1),
            create_param_diff("model.layer.1.attention.q_proj.weight", 0.4),
            create_param_diff("model.layer.1.mlp.fc1.weight", 0.2),
        ]

        summary = DiffSummary(
            total_parameters_old=100000,
            total_parameters_new=100000,
            added_parameters=0,
            removed_parameters=0,
            modified_parameters=5,
            unchanged_parameters=95,
            total_l2_diff=20.0,
            avg_cosine_similarity=0.92,
        )

        result = analyzer.analyze(diffs, summary)

        # Check structure
        assert "layer_analysis" in result
        assert "change_pattern" in result
        assert "insights" in result
        assert "summary" in result

        # Check layer analysis
        assert "attention" in result["layer_analysis"]
        assert result["layer_analysis"]["attention"]["count"] == 3

    def test_gradient_pattern_detection(self):
        analyzer = DiffAnalyzer()

        # Earlier layers change more
        diffs = [
            create_param_diff("model.layer.0.weight", 0.9),
            create_param_diff("model.layer.1.weight", 0.7),
            create_param_diff("model.layer.2.weight", 0.5),
            create_param_diff("model.layer.3.weight", 0.3),
            create_param_diff("model.layer.4.weight", 0.1),
        ]

        pattern = analyzer.detect_change_pattern(diffs, {})

        assert pattern == ChangePattern.GRADIENT

    def test_inverse_gradient_pattern_detection(self):
        analyzer = DiffAnalyzer()

        # Later layers change more
        diffs = [
            create_param_diff("model.layer.0.weight", 0.1),
            create_param_diff("model.layer.1.weight", 0.2),
            create_param_diff("model.layer.2.weight", 0.4),
            create_param_diff("model.layer.3.weight", 0.6),
            create_param_diff("model.layer.4.weight", 0.9),
        ]

        pattern = analyzer.detect_change_pattern(diffs, {})

        assert pattern == ChangePattern.INVERSE_GRADIENT
