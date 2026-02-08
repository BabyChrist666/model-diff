# Model Diff

Compare two model checkpoints to understand training changes.

[![Tests](https://github.com/BabyChrist666/model-diff/actions/workflows/tests.yml/badge.svg)](https://github.com/BabyChrist666/model-diff/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/BabyChrist666/model-diff/branch/master/graph/badge.svg)](https://codecov.io/gh/BabyChrist666/model-diff)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Model Diff is a powerful tool for understanding what changes between model checkpoints during training. Whether you're debugging training runs, comparing fine-tuned models to their base versions, or analyzing the effects of different training strategies, Model Diff provides comprehensive insights into parameter-level changes.

## Features

- **Multi-Format Support**: Load PyTorch (.pt, .pth, .bin), SafeTensors, NumPy, and HuggingFace checkpoints
- **Comprehensive Metrics**: L2 diff, L1 diff, cosine similarity, relative change, and more
- **Change Pattern Detection**: Identify training patterns (gradient, uniform, attention-heavy, etc.)
- **Layer Type Analysis**: Automatic categorization of attention, MLP, embedding, and other layers
- **Training Insights**: Automated detection of frozen layers, stability issues, and architectural changes
- **Rich Visualization**: HTML, JSON, and text report formats

## Installation

```bash
pip install model-diff
```

Or from source:

```bash
git clone https://github.com/BabyChrist666/model-diff.git
cd model-diff
pip install -e .
```

## Quick Start

```python
from model_diff import compare_checkpoints, DiffAnalyzer, create_diff_report

# Compare two checkpoints
diffs, summary = compare_checkpoints("model_v1.pt", "model_v2.pt")

# Analyze the differences
analyzer = DiffAnalyzer()
analysis = analyzer.analyze(diffs, summary)

# Generate a report
report = create_diff_report(
    diffs, summary, analysis,
    format="html",
    old_name="v1",
    new_name="v2",
)

# Save report
with open("diff_report.html", "w") as f:
    f.write(report)
```

## Detailed Usage

### Loading Checkpoints

```python
from model_diff import load_checkpoint, Checkpoint

# Load a single checkpoint
ckpt = load_checkpoint("model.pt")
print(f"Parameters: {ckpt.num_parameters:,}")
print(f"Size: {ckpt.size_bytes / 1e6:.1f} MB")
print(f"Layers: {len(ckpt.layer_names)}")

# Access tensor info
for name, info in ckpt.tensor_info.items():
    print(f"{name}: {info.shape} (mean={info.mean:.4f}, std={info.std:.4f})")
```

### Computing Differences

```python
from model_diff import ModelDiff, DiffConfig

config = DiffConfig(
    compute_element_wise=True,
    change_threshold=1e-6,
    include_unchanged=False,
    top_k_layers=20,
)

differ = ModelDiff(config)
diffs, summary = differ.diff(old_checkpoint, new_checkpoint)

# Summary statistics
print(f"Modified: {summary.modified_parameters}")
print(f"Added: {summary.added_parameters}")
print(f"Removed: {summary.removed_parameters}")
print(f"Avg cosine similarity: {summary.avg_cosine_similarity:.4f}")
```

### Analyzing Changes

```python
from model_diff import DiffAnalyzer, AnalysisConfig

config = AnalysisConfig(
    significant_change_threshold=0.1,
    analyze_training_dynamics=True,
)

analyzer = DiffAnalyzer(config)
analysis = analyzer.analyze(diffs, summary)

# Change pattern
print(f"Pattern: {analysis['change_pattern']}")

# Layer analysis
for layer_type, data in analysis['layer_analysis'].items():
    print(f"{layer_type}: {data['count']} layers, avg change: {data['avg_change']:.4f}")

# Insights
for insight in analysis['insights']:
    print(f"[{insight['severity']}] {insight['title']}: {insight['description']}")
```

### Generating Reports

```python
from model_diff import create_diff_report, VisualizationConfig

config = VisualizationConfig(
    max_layers_display=50,
    color_scheme="dark",
)

# HTML report
html_report = create_diff_report(
    diffs, summary, analysis,
    format="html",
    old_name="Base Model",
    new_name="Fine-tuned",
    config=config,
)

# JSON report
json_report = create_diff_report(
    diffs, summary, analysis,
    format="json",
)

# Text report
text_report = create_diff_report(
    diffs, summary, analysis,
    format="text",
)
```

## Change Patterns

Model Diff detects several patterns of change:

| Pattern | Description |
|---------|-------------|
| **Uniform** | All layers changed equally |
| **Gradient** | Earlier layers changed more than later layers |
| **Inverse Gradient** | Later layers changed more than earlier layers |
| **Focused** | Only specific layers changed significantly |
| **Sparse** | Very few layers have significant changes |
| **Attention Heavy** | Attention layers changed most |
| **MLP Heavy** | MLP/feed-forward layers changed most |
| **Embedding Heavy** | Embedding layers changed significantly |

## Metrics

For each parameter, the following metrics are computed:

| Metric | Description |
|--------|-------------|
| `l2_diff` | L2 (Euclidean) distance between old and new |
| `l1_diff` | L1 (Manhattan) distance |
| `max_diff` | Maximum element-wise difference |
| `mean_diff` | Mean element-wise difference |
| `cosine_sim` | Cosine similarity (direction preservation) |
| `relative_change` | L2 diff normalized by old parameter norm |
| `change_ratio` | Fraction of elements that changed |

## Insights Generated

Model Diff automatically detects:

- **Architecture changes**: Added/removed parameters
- **Training dynamics**: Change patterns across layers
- **Stability issues**: Low cosine similarity warnings
- **Frozen layers**: Parameters that didn't change
- **Layer-specific changes**: Which layer types changed most

## Supported Formats

| Format | Extensions | Notes |
|--------|------------|-------|
| PyTorch | .pt, .pth, .bin | Supports nested state dicts |
| SafeTensors | .safetensors | Memory-efficient format |
| NumPy | .npy, .npz | Raw array format |
| HuggingFace | directory | Loads config.json + model files |

## API Reference

### DiffConfig

```python
@dataclass
class DiffConfig:
    compute_element_wise: bool = True   # Compute per-element stats
    change_threshold: float = 1e-6      # Threshold for "unchanged"
    include_unchanged: bool = False     # Include unchanged params
    normalize_by_size: bool = True      # Normalize by param count
    top_k_layers: int = 10              # Top layers in summary
```

### AnalysisConfig

```python
@dataclass
class AnalysisConfig:
    significant_change_threshold: float = 0.1  # For insight generation
    layer_patterns: Dict[str, str]             # Regex patterns for layer types
    analyze_training_dynamics: bool = True     # Detect training patterns
```

### VisualizationConfig

```python
@dataclass
class VisualizationConfig:
    max_layers_display: int = 50       # Max layers in report
    color_scheme: str = "default"      # default, dark, light
    include_raw_data: bool = False     # Include full data
    chart_width: int = 800
    chart_height: int = 400
```

## Use Cases

- **Training debugging**: Understand why training diverged
- **Fine-tuning analysis**: See what changed during fine-tuning
- **Checkpoint comparison**: Compare checkpoints at different training steps
- **Model merging**: Understand differences before merging
- **Architecture validation**: Verify expected changes occurred

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=model_diff

# Run specific test file
pytest tests/test_diff.py -v
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details.
