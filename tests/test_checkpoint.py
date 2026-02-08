"""Tests for checkpoint loading."""

import pytest
import tempfile
import os
from pathlib import Path

from model_diff.checkpoint import (
    Checkpoint,
    CheckpointLoader,
    CheckpointFormat,
    TensorInfo,
    MockNumpy,
    MockArray,
    load_checkpoint,
)


class TestCheckpointFormat:
    """Tests for CheckpointFormat enum."""

    def test_formats_exist(self):
        assert CheckpointFormat.PYTORCH
        assert CheckpointFormat.SAFETENSORS
        assert CheckpointFormat.NUMPY
        assert CheckpointFormat.HUGGINGFACE
        assert CheckpointFormat.GGUF
        assert CheckpointFormat.UNKNOWN

    def test_format_values(self):
        assert CheckpointFormat.PYTORCH.value == "pytorch"
        assert CheckpointFormat.SAFETENSORS.value == "safetensors"


class TestTensorInfo:
    """Tests for TensorInfo."""

    def test_create_info(self):
        info = TensorInfo(
            name="layer1.weight",
            shape=(64, 64),
            dtype="float32",
            num_elements=4096,
            size_bytes=16384,
            mean=0.0,
            std=1.0,
            min_val=-3.0,
            max_val=3.0,
            sparsity=0.1,
        )
        assert info.name == "layer1.weight"
        assert info.shape == (64, 64)
        assert info.num_elements == 4096

    def test_to_dict(self):
        info = TensorInfo(
            name="test",
            shape=(32, 32),
            dtype="float32",
            num_elements=1024,
            size_bytes=4096,
        )
        d = info.to_dict()
        assert d["name"] == "test"
        assert d["shape"] == [32, 32]
        assert d["num_elements"] == 1024


class TestCheckpoint:
    """Tests for Checkpoint."""

    def test_create_checkpoint(self):
        ckpt = Checkpoint(
            path="/path/to/model.pt",
            format=CheckpointFormat.PYTORCH,
        )
        assert ckpt.path == "/path/to/model.pt"
        assert ckpt.format == CheckpointFormat.PYTORCH

    def test_num_parameters(self):
        info1 = TensorInfo("l1", (64, 64), "float32", 4096, 16384)
        info2 = TensorInfo("l2", (64,), "float32", 64, 256)

        ckpt = Checkpoint(
            path="/test",
            format=CheckpointFormat.PYTORCH,
            tensor_info={"l1": info1, "l2": info2},
        )
        assert ckpt.num_parameters == 4160

    def test_size_bytes(self):
        info1 = TensorInfo("l1", (64, 64), "float32", 4096, 16384)
        info2 = TensorInfo("l2", (64,), "float32", 64, 256)

        ckpt = Checkpoint(
            path="/test",
            format=CheckpointFormat.PYTORCH,
            tensor_info={"l1": info1, "l2": info2},
        )
        assert ckpt.size_bytes == 16640

    def test_layer_names(self):
        info1 = TensorInfo("layer1.weight", (64, 64), "float32", 4096, 16384)
        info2 = TensorInfo("layer1.bias", (64,), "float32", 64, 256)

        ckpt = Checkpoint(
            path="/test",
            format=CheckpointFormat.PYTORCH,
            tensor_info={"layer1.weight": info1, "layer1.bias": info2},
        )
        assert "layer1.weight" in ckpt.layer_names
        assert "layer1.bias" in ckpt.layer_names

    def test_get_layers_by_pattern(self):
        info1 = TensorInfo("layer1.weight", (64, 64), "float32", 4096, 16384)
        info2 = TensorInfo("layer1.bias", (64,), "float32", 64, 256)
        info3 = TensorInfo("embed.weight", (1000, 64), "float32", 64000, 256000)

        ckpt = Checkpoint(
            path="/test",
            format=CheckpointFormat.PYTORCH,
            tensor_info={
                "layer1.weight": info1,
                "layer1.bias": info2,
                "embed.weight": info3,
            },
        )

        weight_layers = ckpt.get_layers_by_pattern(r"\.weight$")
        assert len(weight_layers) == 2
        assert "layer1.weight" in weight_layers
        assert "embed.weight" in weight_layers

    def test_to_dict(self):
        info = TensorInfo("layer1.weight", (64, 64), "float32", 4096, 16384)

        ckpt = Checkpoint(
            path="/test",
            format=CheckpointFormat.PYTORCH,
            tensor_info={"layer1.weight": info},
            metadata={"version": "1.0"},
        )

        d = ckpt.to_dict()
        assert d["path"] == "/test"
        assert d["format"] == "pytorch"
        assert d["num_parameters"] == 4096
        assert "layer1.weight" in d["layers"]


class TestCheckpointLoader:
    """Tests for CheckpointLoader."""

    def test_create_loader(self):
        loader = CheckpointLoader()
        assert loader.compute_stats is True

    def test_create_loader_no_stats(self):
        loader = CheckpointLoader(compute_stats=False)
        assert loader.compute_stats is False

    def test_detect_format_pytorch(self):
        loader = CheckpointLoader()
        assert loader.detect_format("model.pt") == CheckpointFormat.PYTORCH
        assert loader.detect_format("model.pth") == CheckpointFormat.PYTORCH
        assert loader.detect_format("model.bin") == CheckpointFormat.PYTORCH

    def test_detect_format_safetensors(self):
        loader = CheckpointLoader()
        assert loader.detect_format("model.safetensors") == CheckpointFormat.SAFETENSORS

    def test_detect_format_numpy(self):
        loader = CheckpointLoader()
        assert loader.detect_format("weights.npy") == CheckpointFormat.NUMPY
        assert loader.detect_format("weights.npz") == CheckpointFormat.NUMPY

    def test_detect_format_gguf(self):
        loader = CheckpointLoader()
        assert loader.detect_format("model.gguf") == CheckpointFormat.GGUF

    def test_detect_format_unknown(self):
        loader = CheckpointLoader()
        assert loader.detect_format("model.xyz") == CheckpointFormat.UNKNOWN

    def test_compute_tensor_info(self):
        loader = CheckpointLoader()
        # Use the mock state dict which creates proper arrays
        state_dict = loader._create_mock_state_dict(Path("/test"))
        tensor = state_dict["layer1.weight"]

        info = loader._compute_tensor_info("test", tensor)

        assert info.name == "test"
        assert info.shape == (64, 64)
        assert info.num_elements == 4096

    def test_create_mock_state_dict(self):
        loader = CheckpointLoader()
        state_dict = loader._create_mock_state_dict(Path("/test"))

        assert "layer1.weight" in state_dict
        assert "layer1.bias" in state_dict
        assert "layer2.weight" in state_dict


class TestMockNumpy:
    """Tests for MockNumpy."""

    def test_randn(self):
        np = MockNumpy()
        arr = np.randn(10)
        assert len(arr.data) == 10

    def test_randn_2d(self):
        np = MockNumpy()
        arr = np.randn(3, 4)
        assert len(arr.data) == 3
        assert len(arr.data[0]) == 4

    def test_mean(self):
        np = MockNumpy()
        arr = MockArray([1, 2, 3, 4, 5])
        mean = np.mean(arr)
        assert mean == 3.0

    def test_std(self):
        np = MockNumpy()
        arr = MockArray([1, 1, 1, 1])
        std = np.std(arr)
        assert std == 0.0

    def test_min_max(self):
        np = MockNumpy()
        arr = MockArray([1, 5, 3, 2, 4])
        assert np.min(arr) == 1
        assert np.max(arr) == 5

    def test_flatten(self):
        np = MockNumpy()
        data = [[1, 2], [3, 4]]
        flat = np._flatten(data)
        assert flat == [1, 2, 3, 4]


class TestMockArray:
    """Tests for MockArray."""

    def test_create_1d(self):
        arr = MockArray([1, 2, 3])
        assert arr.shape == (3,)
        assert len(arr) == 3

    def test_create_2d(self):
        arr = MockArray([[1, 2], [3, 4]])
        assert arr.shape == (2, 2)

    def test_astype(self):
        arr = MockArray([1, 2, 3])
        arr2 = arr.astype("float64")
        assert arr2._dtype == "float64"

    def test_flatten(self):
        arr = MockArray([[1, 2], [3, 4]])
        flat = arr.flatten()
        assert flat.data == [1, 2, 3, 4]

    def test_subtraction(self):
        arr1 = MockArray([1, 2, 3])
        arr2 = MockArray([1, 1, 1])
        result = arr1 - arr2
        assert result.data == [0, 1, 2]


class TestLoadCheckpoint:
    """Tests for load_checkpoint helper."""

    def test_load_mock_pytorch(self):
        # This will use the mock loader
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            ckpt = load_checkpoint(path)
            assert ckpt.format == CheckpointFormat.PYTORCH
            assert len(ckpt.tensor_info) > 0
        finally:
            os.unlink(path)

    def test_load_with_stats(self):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name

        try:
            ckpt = load_checkpoint(path, compute_stats=True)
            # Check that stats were computed
            for info in ckpt.tensor_info.values():
                assert hasattr(info, "mean")
                assert hasattr(info, "std")
        finally:
            os.unlink(path)
