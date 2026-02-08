"""Model Diff - Compare model checkpoints to understand training changes."""

from .checkpoint import (
    Checkpoint,
    CheckpointLoader,
    CheckpointFormat,
    load_checkpoint,
)
from .diff import (
    ModelDiff,
    DiffConfig,
    LayerDiff,
    ParameterDiff,
    DiffSummary,
    compare_checkpoints,
)
from .analysis import (
    DiffAnalyzer,
    AnalysisConfig,
    ChangePattern,
    LayerAnalysis,
    TrainingInsight,
)
from .visualization import (
    DiffVisualizer,
    VisualizationConfig,
    create_diff_report,
)

__version__ = "0.1.0"
__all__ = [
    # Checkpoint
    "Checkpoint",
    "CheckpointLoader",
    "CheckpointFormat",
    "load_checkpoint",
    # Diff
    "ModelDiff",
    "DiffConfig",
    "LayerDiff",
    "ParameterDiff",
    "DiffSummary",
    "compare_checkpoints",
    # Analysis
    "DiffAnalyzer",
    "AnalysisConfig",
    "ChangePattern",
    "LayerAnalysis",
    "TrainingInsight",
    # Visualization
    "DiffVisualizer",
    "VisualizationConfig",
    "create_diff_report",
]
