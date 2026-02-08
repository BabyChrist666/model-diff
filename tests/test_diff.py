"""Tests for model diffing."""

import pytest

from model_diff.checkpoint import (
    Checkpoint,
    CheckpointFormat,
    TensorInfo,
    MockArray,
)
from model_diff.diff import (
    ChangeType,
    ParameterDiff,
    LayerDiff,
    DiffSummary,
    DiffConfig,
    ModelDiff,
    compare_checkpoints,
)


def create_test_checkpoint(tensors_data: dict) -> Checkpoint:
    """Create a test checkpoint from tensor data."""
    tensors = {}
    tensor_info = {}

    for name, data in tensors_data.items():
        arr = MockArray(data)
        tensors[name] = arr

        shape = arr.shape
        num_elements = 1
        for dim in shape:
            num_elements *= dim

        tensor_info[name] = TensorInfo(
            name=name,
            shape=shape,
            dtype="float32",
            num_elements=num_elements,
            size_bytes=num_elements * 4,
        )

    return Checkpoint(
        path="/test",
        format=CheckpointFormat.PYTORCH,
        tensors=tensors,
        tensor_info=tensor_info,
    )


class TestChangeType:
    """Tests for ChangeType enum."""

    def test_types_exist(self):
        assert ChangeType.UNCHANGED
        assert ChangeType.MODIFIED
        assert ChangeType.ADDED
        assert ChangeType.REMOVED
        assert ChangeType.SHAPE_CHANGED

    def test_type_values(self):
        assert ChangeType.UNCHANGED.value == "unchanged"
        assert ChangeType.MODIFIED.value == "modified"


class TestParameterDiff:
    """Tests for ParameterDiff."""

    def test_create_diff(self):
        diff = ParameterDiff(
            name="layer1.weight",
            change_type=ChangeType.MODIFIED,
            l2_diff=0.5,
            cosine_sim=0.99,
        )
        assert diff.name == "layer1.weight"
        assert diff.l2_diff == 0.5

    def test_change_ratio(self):
        diff = ParameterDiff(
            name="test",
            change_type=ChangeType.MODIFIED,
            changed_elements=50,
            total_elements=100,
        )
        assert diff.change_ratio == 0.5

    def test_change_ratio_zero_total(self):
        diff = ParameterDiff(
            name="test",
            change_type=ChangeType.MODIFIED,
            changed_elements=0,
            total_elements=0,
        )
        assert diff.change_ratio == 0.0

    def test_to_dict(self):
        info = TensorInfo("test", (64, 64), "float32", 4096, 16384)
        diff = ParameterDiff(
            name="test",
            change_type=ChangeType.MODIFIED,
            old_info=info,
            new_info=info,
            l2_diff=1.0,
            cosine_sim=0.95,
        )
        d = diff.to_dict()
        assert d["name"] == "test"
        assert d["change_type"] == "modified"
        assert d["l2_diff"] == 1.0


class TestLayerDiff:
    """Tests for LayerDiff."""

    def test_create_layer_diff(self):
        layer = LayerDiff(name="layer1")
        assert layer.name == "layer1"
        assert len(layer.parameters) == 0

    def test_total_l2_diff(self):
        layer = LayerDiff(
            name="layer1",
            parameters=[
                ParameterDiff("p1", ChangeType.MODIFIED, l2_diff=1.0),
                ParameterDiff("p2", ChangeType.MODIFIED, l2_diff=2.0),
            ],
        )
        assert layer.total_l2_diff == 3.0

    def test_avg_cosine_sim(self):
        layer = LayerDiff(
            name="layer1",
            parameters=[
                ParameterDiff("p1", ChangeType.MODIFIED, cosine_sim=0.9),
                ParameterDiff("p2", ChangeType.MODIFIED, cosine_sim=1.0),
            ],
        )
        assert layer.avg_cosine_sim == 0.95

    def test_avg_cosine_sim_no_modified(self):
        layer = LayerDiff(
            name="layer1",
            parameters=[
                ParameterDiff("p1", ChangeType.UNCHANGED),
            ],
        )
        assert layer.avg_cosine_sim == 1.0

    def test_num_changes(self):
        layer = LayerDiff(
            name="layer1",
            parameters=[
                ParameterDiff("p1", ChangeType.MODIFIED),
                ParameterDiff("p2", ChangeType.UNCHANGED),
                ParameterDiff("p3", ChangeType.ADDED),
            ],
        )
        assert layer.num_changes == 2


class TestDiffSummary:
    """Tests for DiffSummary."""

    def test_create_summary(self):
        summary = DiffSummary(
            total_parameters_old=1000000,
            total_parameters_new=1000000,
            added_parameters=0,
            removed_parameters=0,
            modified_parameters=100,
            unchanged_parameters=900,
            total_l2_diff=50.0,
            avg_cosine_similarity=0.98,
        )
        assert summary.total_parameters_old == 1000000
        assert summary.modified_parameters == 100

    def test_to_dict(self):
        summary = DiffSummary(
            total_parameters_old=1000,
            total_parameters_new=1000,
            added_parameters=10,
            removed_parameters=5,
            modified_parameters=100,
            unchanged_parameters=885,
            total_l2_diff=25.0,
            avg_cosine_similarity=0.95,
            most_changed_layers=["layer1", "layer2"],
        )
        d = summary.to_dict()
        assert d["added_parameters"] == 10
        assert "layer1" in d["most_changed_layers"]


class TestDiffConfig:
    """Tests for DiffConfig."""

    def test_default_config(self):
        config = DiffConfig()
        assert config.compute_element_wise is True
        assert config.change_threshold == 1e-6
        assert config.include_unchanged is False

    def test_custom_config(self):
        config = DiffConfig(
            change_threshold=1e-4,
            include_unchanged=True,
        )
        assert config.change_threshold == 1e-4
        assert config.include_unchanged is True


class TestModelDiff:
    """Tests for ModelDiff."""

    def test_create_differ(self):
        differ = ModelDiff()
        assert differ.config is not None

    def test_diff_identical(self):
        data = {"layer1.weight": [[1, 2], [3, 4]]}
        ckpt1 = create_test_checkpoint(data)
        ckpt2 = create_test_checkpoint(data)

        differ = ModelDiff()
        diffs, summary = differ.diff(ckpt1, ckpt2)

        # Should have no changes (or all unchanged)
        modified = [d for d in diffs if d.change_type == ChangeType.MODIFIED]
        assert len(modified) == 0

    def test_diff_modified(self):
        ckpt1 = create_test_checkpoint({"layer1.weight": [[1, 2], [3, 4]]})
        ckpt2 = create_test_checkpoint({"layer1.weight": [[2, 3], [4, 5]]})

        differ = ModelDiff()
        diffs, summary = differ.diff(ckpt1, ckpt2)

        modified = [d for d in diffs if d.change_type == ChangeType.MODIFIED]
        assert len(modified) == 1
        assert modified[0].l2_diff > 0

    def test_diff_added(self):
        ckpt1 = create_test_checkpoint({"layer1.weight": [[1, 2], [3, 4]]})
        ckpt2 = create_test_checkpoint({
            "layer1.weight": [[1, 2], [3, 4]],
            "layer2.weight": [[5, 6], [7, 8]],
        })

        differ = ModelDiff()
        diffs, summary = differ.diff(ckpt1, ckpt2)

        added = [d for d in diffs if d.change_type == ChangeType.ADDED]
        assert len(added) == 1
        assert added[0].name == "layer2.weight"

    def test_diff_removed(self):
        ckpt1 = create_test_checkpoint({
            "layer1.weight": [[1, 2], [3, 4]],
            "layer2.weight": [[5, 6], [7, 8]],
        })
        ckpt2 = create_test_checkpoint({"layer1.weight": [[1, 2], [3, 4]]})

        differ = ModelDiff()
        diffs, summary = differ.diff(ckpt1, ckpt2)

        removed = [d for d in diffs if d.change_type == ChangeType.REMOVED]
        assert len(removed) == 1
        assert removed[0].name == "layer2.weight"

    def test_diff_shape_changed(self):
        ckpt1 = create_test_checkpoint({"layer1.weight": [[1, 2], [3, 4]]})
        ckpt2 = create_test_checkpoint({"layer1.weight": [[1, 2, 3], [4, 5, 6]]})

        differ = ModelDiff()
        diffs, summary = differ.diff(ckpt1, ckpt2)

        shape_changed = [d for d in diffs if d.change_type == ChangeType.SHAPE_CHANGED]
        assert len(shape_changed) == 1

    def test_compute_l2(self):
        differ = ModelDiff()
        arr = MockArray([3, 4])  # L2 norm = 5
        l2 = differ._compute_l2(arr)
        assert abs(l2 - 5.0) < 0.001

    def test_compute_cosine_similarity(self):
        differ = ModelDiff()
        arr1 = MockArray([1, 0])
        arr2 = MockArray([1, 0])
        sim = differ._compute_cosine_similarity(arr1, arr2)
        assert abs(sim - 1.0) < 0.001

    def test_compute_cosine_similarity_orthogonal(self):
        differ = ModelDiff()
        arr1 = MockArray([1, 0])
        arr2 = MockArray([0, 1])
        sim = differ._compute_cosine_similarity(arr1, arr2)
        assert abs(sim) < 0.001

    def test_get_layer_diffs(self):
        differ = ModelDiff()
        param_diffs = [
            ParameterDiff("layer1.weight", ChangeType.MODIFIED),
            ParameterDiff("layer1.bias", ChangeType.MODIFIED),
            ParameterDiff("layer2.weight", ChangeType.MODIFIED),
        ]

        layer_diffs = differ.get_layer_diffs(param_diffs)

        assert "layer1" in layer_diffs
        assert "layer2" in layer_diffs
        assert len(layer_diffs["layer1"].parameters) == 2
        assert len(layer_diffs["layer2"].parameters) == 1

    def test_summary_most_changed(self):
        ckpt1 = create_test_checkpoint({
            "layer1.weight": [[1, 2], [3, 4]],
            "layer2.weight": [[1, 2], [3, 4]],
        })
        ckpt2 = create_test_checkpoint({
            "layer1.weight": [[10, 20], [30, 40]],  # Big change
            "layer2.weight": [[1.1, 2.1], [3.1, 4.1]],  # Small change
        })

        differ = ModelDiff()
        diffs, summary = differ.diff(ckpt1, ckpt2)

        # layer1 should have more change
        assert len(summary.most_changed_layers) > 0

    def test_include_unchanged(self):
        data = {"layer1.weight": [[1, 2], [3, 4]]}
        ckpt1 = create_test_checkpoint(data)
        ckpt2 = create_test_checkpoint(data)

        config = DiffConfig(include_unchanged=True)
        differ = ModelDiff(config)
        diffs, summary = differ.diff(ckpt1, ckpt2)

        # Should include the unchanged parameter
        assert len(diffs) >= 1


class TestCompareCheckpoints:
    """Tests for compare_checkpoints helper."""

    def test_compare_mock(self):
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f1:
            path1 = f1.name
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f2:
            path2 = f2.name

        try:
            diffs, summary = compare_checkpoints(path1, path2)
            assert summary is not None
        finally:
            os.unlink(path1)
            os.unlink(path2)
