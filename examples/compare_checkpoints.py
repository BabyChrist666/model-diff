#!/usr/bin/env python3
"""
Model Diff - Checkpoint Comparison Example

This example demonstrates how to compare model checkpoints
to understand what changed during training.
"""

from model_diff import (
    ModelDiff,
    DiffConfig,
    CheckpointLoader,
    DiffReport,
)


def main():
    print("=" * 60)
    print("Model Diff - Checkpoint Comparison")
    print("=" * 60)

    # Example 1: Basic comparison
    print("\n1. Basic checkpoint comparison...")

    # Create a differ with default settings
    differ = ModelDiff()

    # Create mock checkpoints for demonstration
    checkpoint_v1 = {
        "model.layer1.weight": [[1.0, 2.0], [3.0, 4.0]],
        "model.layer1.bias": [0.1, 0.2],
        "model.layer2.weight": [[0.5, 0.5], [0.5, 0.5]],
    }

    checkpoint_v2 = {
        "model.layer1.weight": [[1.1, 2.1], [3.0, 4.0]],  # Slightly changed
        "model.layer1.bias": [0.1, 0.2],  # Unchanged
        "model.layer2.weight": [[0.6, 0.4], [0.5, 0.5]],  # Changed
        "model.layer3.weight": [[1.0]],  # New layer
    }

    # Compare checkpoints
    diff = differ.compare(checkpoint_v1, checkpoint_v2)

    print(f"\n   Parameters in v1: {diff.params_v1}")
    print(f"   Parameters in v2: {diff.params_v2}")
    print(f"   Added layers: {diff.added_layers}")
    print(f"   Removed layers: {diff.removed_layers}")
    print(f"   Modified layers: {diff.modified_layers}")

    # Example 2: Detailed layer analysis
    print("\n" + "-" * 60)
    print("2. Detailed layer analysis...")

    for layer_name, layer_diff in diff.layer_diffs.items():
        print(f"\n   Layer: {layer_name}")
        print(f"     Change magnitude: {layer_diff.magnitude:.4f}")
        print(f"     Relative change: {layer_diff.relative_change:.2%}")
        print(f"     Max element change: {layer_diff.max_change:.4f}")

    # Example 3: Custom configuration
    print("\n" + "-" * 60)
    print("3. Using custom configuration...")

    config = DiffConfig(
        ignore_layers=["model.layer1.bias"],  # Ignore bias terms
        threshold=0.05,  # Only report changes > 5%
        compute_statistics=True,  # Compute detailed stats
        output_format="detailed",  # Detailed output
    )

    differ = ModelDiff(config)
    diff = differ.compare(checkpoint_v1, checkpoint_v2)

    print(f"   Significant changes: {len(diff.significant_changes)}")
    print(f"   Ignored layers: {len(diff.ignored_layers)}")

    # Example 4: Generate diff report
    print("\n" + "-" * 60)
    print("4. Generating diff report...")

    report = differ.generate_report(diff)

    print(f"\n   Report Summary:")
    print(f"     Total parameters: {report.total_parameters:,}")
    print(f"     Changed parameters: {report.changed_parameters:,}")
    print(f"     Change ratio: {report.change_ratio:.2%}")
    print(f"     Average magnitude: {report.avg_magnitude:.6f}")

    # Example 5: Training progress tracking
    print("\n" + "-" * 60)
    print("5. Tracking training progress...")

    # Simulate checkpoints at different training steps
    checkpoints = [
        {"step": 0, "weights": [[0.0, 0.0]]},
        {"step": 100, "weights": [[0.5, 0.3]]},
        {"step": 200, "weights": [[0.8, 0.6]]},
        {"step": 300, "weights": [[0.9, 0.85]]},
    ]

    print("\n   Training progression:")
    for i in range(1, len(checkpoints)):
        prev = {"w": checkpoints[i - 1]["weights"]}
        curr = {"w": checkpoints[i]["weights"]}
        step_diff = differ.compare(prev, curr)

        print(f"     Step {checkpoints[i-1]['step']} -> {checkpoints[i]['step']}: "
              f"change = {step_diff.total_change:.4f}")

    # Example 6: Export comparison
    print("\n" + "-" * 60)
    print("6. Exporting comparison results...")

    # Export to different formats
    json_output = differ.export_diff(diff, format="json")
    print(f"   JSON export: {len(json_output)} characters")

    csv_output = differ.export_diff(diff, format="csv")
    print(f"   CSV export: {len(csv_output)} characters")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
