"""Analysis of model differences."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any
import re

from .diff import ParameterDiff, ChangeType, LayerDiff, DiffSummary


class ChangePattern(Enum):
    """Patterns of change during training."""

    UNIFORM = "uniform"  # All layers changed equally
    GRADIENT = "gradient"  # Earlier layers changed more
    INVERSE_GRADIENT = "inverse_gradient"  # Later layers changed more
    FOCUSED = "focused"  # Only specific layers changed
    SPARSE = "sparse"  # Most layers unchanged
    ATTENTION_HEAVY = "attention_heavy"  # Attention layers changed most
    MLP_HEAVY = "mlp_heavy"  # MLP layers changed most
    EMBEDDING_HEAVY = "embedding_heavy"  # Embedding layers changed most


@dataclass
class TrainingInsight:
    """Insight derived from model diff."""

    category: str
    title: str
    description: str
    severity: str = "info"  # info, warning, critical
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "category": self.category,
            "title": self.title,
            "description": self.description,
            "severity": self.severity,
            "evidence": self.evidence,
        }


@dataclass
class LayerAnalysis:
    """Analysis of changes to a specific layer type."""

    layer_type: str
    count: int
    total_change: float
    avg_change: float
    max_change: float
    layers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "layer_type": self.layer_type,
            "count": self.count,
            "total_change": self.total_change,
            "avg_change": self.avg_change,
            "max_change": self.max_change,
            "layers": self.layers,
        }


@dataclass
class AnalysisConfig:
    """Configuration for diff analysis."""

    significant_change_threshold: float = 0.1
    layer_patterns: Dict[str, str] = field(default_factory=lambda: {
        "attention": r"(attention|attn|self_attn|q_proj|k_proj|v_proj|o_proj)",
        "mlp": r"(mlp|feed_forward|fc1|fc2|dense|linear)",
        "embedding": r"(embed|wte|wpe|token_embedding|position_embedding)",
        "norm": r"(norm|layer_norm|ln_|rmsnorm)",
        "output": r"(lm_head|output|classifier)",
    })
    analyze_training_dynamics: bool = True


class DiffAnalyzer:
    """Analyze model differences for insights."""

    def __init__(self, config: Optional[AnalysisConfig] = None):
        """
        Initialize analyzer.

        Args:
            config: Analysis configuration
        """
        self.config = config or AnalysisConfig()

    def analyze(
        self,
        diffs: List[ParameterDiff],
        summary: DiffSummary,
    ) -> Dict[str, Any]:
        """
        Analyze diffs and generate insights.

        Args:
            diffs: List of parameter diffs
            summary: Diff summary

        Returns:
            Analysis results
        """
        layer_analysis = self.analyze_layer_types(diffs)
        change_pattern = self.detect_change_pattern(diffs, layer_analysis)
        insights = self.generate_insights(diffs, summary, layer_analysis, change_pattern)

        return {
            "layer_analysis": {k: v.to_dict() for k, v in layer_analysis.items()},
            "change_pattern": change_pattern.value,
            "insights": [i.to_dict() for i in insights],
            "summary": summary.to_dict(),
        }

    def analyze_layer_types(
        self,
        diffs: List[ParameterDiff],
    ) -> Dict[str, LayerAnalysis]:
        """Analyze changes by layer type."""
        type_stats: Dict[str, Dict[str, Any]] = {}

        for layer_type, pattern in self.config.layer_patterns.items():
            regex = re.compile(pattern, re.IGNORECASE)
            matching_diffs = [
                d for d in diffs
                if regex.search(d.name) and d.change_type == ChangeType.MODIFIED
            ]

            if matching_diffs:
                changes = [d.relative_change for d in matching_diffs]
                type_stats[layer_type] = LayerAnalysis(
                    layer_type=layer_type,
                    count=len(matching_diffs),
                    total_change=sum(changes),
                    avg_change=sum(changes) / len(changes),
                    max_change=max(changes),
                    layers=[d.name for d in matching_diffs],
                )

        return type_stats

    def detect_change_pattern(
        self,
        diffs: List[ParameterDiff],
        layer_analysis: Dict[str, LayerAnalysis],
    ) -> ChangePattern:
        """Detect the pattern of changes."""
        # Filter to modified diffs
        modified = [d for d in diffs if d.change_type == ChangeType.MODIFIED]

        if not modified:
            return ChangePattern.SPARSE

        # Calculate change ratios
        changes = [d.relative_change for d in modified]
        avg_change = sum(changes) / len(changes) if changes else 0
        max_change = max(changes) if changes else 0

        # Check for sparse changes
        significant = sum(1 for c in changes if c > self.config.significant_change_threshold)
        if significant < len(changes) * 0.1:
            return ChangePattern.SPARSE

        # Check layer type dominance
        if layer_analysis:
            type_changes = {
                k: v.avg_change for k, v in layer_analysis.items()
            }
            if type_changes:
                max_type = max(type_changes, key=type_changes.get)
                max_type_change = type_changes[max_type]

                # If one type dominates
                other_avg = sum(v for k, v in type_changes.items() if k != max_type)
                other_count = len([k for k in type_changes if k != max_type])
                if other_count > 0:
                    other_avg /= other_count
                    if max_type_change > other_avg * 2:
                        if max_type == "attention":
                            return ChangePattern.ATTENTION_HEAVY
                        elif max_type == "mlp":
                            return ChangePattern.MLP_HEAVY
                        elif max_type == "embedding":
                            return ChangePattern.EMBEDDING_HEAVY

        # Check for gradient patterns
        # Extract layer indices
        layer_indices = []
        for d in modified:
            # Try to extract layer number
            match = re.search(r"\.(\d+)\.", d.name)
            if match:
                layer_indices.append((int(match.group(1)), d.relative_change))

        if layer_indices:
            layer_indices.sort(key=lambda x: x[0])

            # Check correlation with layer depth
            early_avg = sum(c for i, c in layer_indices if i < len(layer_indices) / 2)
            late_avg = sum(c for i, c in layer_indices if i >= len(layer_indices) / 2)
            early_count = sum(1 for i, c in layer_indices if i < len(layer_indices) / 2)
            late_count = sum(1 for i, c in layer_indices if i >= len(layer_indices) / 2)

            if early_count > 0 and late_count > 0:
                early_avg /= early_count
                late_avg /= late_count

                if early_avg > late_avg * 1.5:
                    return ChangePattern.GRADIENT
                elif late_avg > early_avg * 1.5:
                    return ChangePattern.INVERSE_GRADIENT

        # Check for focused changes
        if significant < len(changes) * 0.3:
            return ChangePattern.FOCUSED

        return ChangePattern.UNIFORM

    def generate_insights(
        self,
        diffs: List[ParameterDiff],
        summary: DiffSummary,
        layer_analysis: Dict[str, LayerAnalysis],
        change_pattern: ChangePattern,
    ) -> List[TrainingInsight]:
        """Generate training insights."""
        insights = []

        # Size change insight
        if summary.added_parameters > 0 or summary.removed_parameters > 0:
            insights.append(TrainingInsight(
                category="architecture",
                title="Model Architecture Changed",
                description=(
                    f"Added {summary.added_parameters} parameters, "
                    f"removed {summary.removed_parameters} parameters"
                ),
                severity="warning" if summary.removed_parameters > 0 else "info",
                evidence={
                    "added": summary.added_parameters,
                    "removed": summary.removed_parameters,
                },
            ))

        # Change pattern insight
        pattern_descriptions = {
            ChangePattern.UNIFORM: "Changes are distributed uniformly across all layers",
            ChangePattern.GRADIENT: "Earlier layers changed more than later layers",
            ChangePattern.INVERSE_GRADIENT: "Later layers changed more than earlier layers",
            ChangePattern.FOCUSED: "Changes are concentrated in specific layers",
            ChangePattern.SPARSE: "Very few layers have significant changes",
            ChangePattern.ATTENTION_HEAVY: "Attention layers have the most significant changes",
            ChangePattern.MLP_HEAVY: "MLP/feed-forward layers have the most changes",
            ChangePattern.EMBEDDING_HEAVY: "Embedding layers changed significantly",
        }

        insights.append(TrainingInsight(
            category="training_dynamics",
            title=f"Change Pattern: {change_pattern.value.replace('_', ' ').title()}",
            description=pattern_descriptions.get(change_pattern, "Unknown pattern"),
            severity="info",
            evidence={"pattern": change_pattern.value},
        ))

        # Low cosine similarity warning
        if summary.avg_cosine_similarity < 0.9:
            insights.append(TrainingInsight(
                category="stability",
                title="Significant Weight Direction Changes",
                description=(
                    f"Average cosine similarity is {summary.avg_cosine_similarity:.3f}, "
                    "indicating substantial directional changes in weights"
                ),
                severity="warning" if summary.avg_cosine_similarity < 0.8 else "info",
                evidence={"cosine_similarity": summary.avg_cosine_similarity},
            ))

        # Large L2 change warning
        if summary.total_l2_diff > 100:
            insights.append(TrainingInsight(
                category="magnitude",
                title="Large Weight Magnitude Changes",
                description=f"Total L2 difference is {summary.total_l2_diff:.2f}",
                severity="warning",
                evidence={"total_l2_diff": summary.total_l2_diff},
            ))

        # Layer-specific insights
        for layer_type, analysis in layer_analysis.items():
            if analysis.avg_change > self.config.significant_change_threshold:
                insights.append(TrainingInsight(
                    category="layer_changes",
                    title=f"{layer_type.title()} Layers Changed Significantly",
                    description=(
                        f"{analysis.count} {layer_type} layers changed with "
                        f"average relative change of {analysis.avg_change:.3f}"
                    ),
                    severity="info",
                    evidence={
                        "layer_type": layer_type,
                        "count": analysis.count,
                        "avg_change": analysis.avg_change,
                    },
                ))

        # Check for frozen layers
        unchanged_count = sum(1 for d in diffs if d.change_type == ChangeType.UNCHANGED)
        total = len(diffs)
        if total > 0 and unchanged_count / total > 0.5:
            insights.append(TrainingInsight(
                category="training_config",
                title="Many Layers Appear Frozen",
                description=(
                    f"{unchanged_count} out of {total} parameters "
                    f"({100 * unchanged_count / total:.1f}%) are unchanged"
                ),
                severity="info",
                evidence={
                    "unchanged": unchanged_count,
                    "total": total,
                    "ratio": unchanged_count / total,
                },
            ))

        return insights

    def get_most_changed_parameters(
        self,
        diffs: List[ParameterDiff],
        top_k: int = 20,
    ) -> List[ParameterDiff]:
        """Get the most changed parameters."""
        modified = [d for d in diffs if d.change_type == ChangeType.MODIFIED]
        return sorted(modified, key=lambda x: x.relative_change, reverse=True)[:top_k]

    def get_layer_change_ranking(
        self,
        diffs: List[ParameterDiff],
    ) -> List[tuple]:
        """Get layers ranked by total change."""
        layer_changes: Dict[str, float] = {}

        for d in diffs:
            if d.change_type != ChangeType.MODIFIED:
                continue

            # Extract layer name
            parts = d.name.rsplit(".", 1)
            layer_name = parts[0] if len(parts) > 1 else d.name

            if layer_name not in layer_changes:
                layer_changes[layer_name] = 0.0
            layer_changes[layer_name] += d.relative_change

        return sorted(layer_changes.items(), key=lambda x: x[1], reverse=True)

    def compare_training_stages(
        self,
        stage1_diffs: List[ParameterDiff],
        stage2_diffs: List[ParameterDiff],
    ) -> Dict[str, Any]:
        """Compare changes between two training stages."""
        stage1_changes = {d.name: d.relative_change for d in stage1_diffs}
        stage2_changes = {d.name: d.relative_change for d in stage2_diffs}

        all_names = set(stage1_changes.keys()) | set(stage2_changes.keys())

        accelerating = []  # Changed more in stage 2
        decelerating = []  # Changed less in stage 2
        stable = []  # Similar change rate

        for name in all_names:
            c1 = stage1_changes.get(name, 0)
            c2 = stage2_changes.get(name, 0)

            if c2 > c1 * 1.5:
                accelerating.append((name, c1, c2))
            elif c1 > c2 * 1.5:
                decelerating.append((name, c1, c2))
            else:
                stable.append((name, c1, c2))

        return {
            "accelerating": accelerating[:10],
            "decelerating": decelerating[:10],
            "stable_count": len(stable),
            "total_stage1_change": sum(stage1_changes.values()),
            "total_stage2_change": sum(stage2_changes.values()),
        }
