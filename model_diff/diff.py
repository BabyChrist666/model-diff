"""Model diffing functionality."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import math

from .checkpoint import Checkpoint, TensorInfo


class ChangeType(Enum):
    """Type of change between checkpoints."""

    UNCHANGED = "unchanged"
    MODIFIED = "modified"
    ADDED = "added"
    REMOVED = "removed"
    SHAPE_CHANGED = "shape_changed"


@dataclass
class ParameterDiff:
    """Difference for a single parameter."""

    name: str
    change_type: ChangeType
    old_info: Optional[TensorInfo] = None
    new_info: Optional[TensorInfo] = None
    l2_diff: float = 0.0
    l1_diff: float = 0.0
    max_diff: float = 0.0
    mean_diff: float = 0.0
    cosine_sim: float = 1.0
    relative_change: float = 0.0
    changed_elements: int = 0
    total_elements: int = 0

    @property
    def change_ratio(self) -> float:
        """Ratio of changed elements."""
        if self.total_elements == 0:
            return 0.0
        return self.changed_elements / self.total_elements

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "change_type": self.change_type.value,
            "l2_diff": self.l2_diff,
            "l1_diff": self.l1_diff,
            "max_diff": self.max_diff,
            "mean_diff": self.mean_diff,
            "cosine_sim": self.cosine_sim,
            "relative_change": self.relative_change,
            "change_ratio": self.change_ratio,
            "old_shape": list(self.old_info.shape) if self.old_info else None,
            "new_shape": list(self.new_info.shape) if self.new_info else None,
        }


@dataclass
class LayerDiff:
    """Aggregated difference for a layer."""

    name: str
    parameters: List[ParameterDiff] = field(default_factory=list)

    @property
    def total_l2_diff(self) -> float:
        """Total L2 difference across parameters."""
        return sum(p.l2_diff for p in self.parameters)

    @property
    def avg_cosine_sim(self) -> float:
        """Average cosine similarity."""
        if not self.parameters:
            return 1.0
        sims = [p.cosine_sim for p in self.parameters if p.change_type == ChangeType.MODIFIED]
        return sum(sims) / len(sims) if sims else 1.0

    @property
    def num_changes(self) -> int:
        """Number of changed parameters."""
        return sum(1 for p in self.parameters if p.change_type != ChangeType.UNCHANGED)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "total_l2_diff": self.total_l2_diff,
            "avg_cosine_sim": self.avg_cosine_sim,
            "num_changes": self.num_changes,
            "parameters": [p.to_dict() for p in self.parameters],
        }


@dataclass
class DiffSummary:
    """Summary of differences between checkpoints."""

    total_parameters_old: int
    total_parameters_new: int
    added_parameters: int
    removed_parameters: int
    modified_parameters: int
    unchanged_parameters: int
    total_l2_diff: float
    avg_cosine_similarity: float
    most_changed_layers: List[str] = field(default_factory=list)
    least_changed_layers: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_parameters_old": self.total_parameters_old,
            "total_parameters_new": self.total_parameters_new,
            "added_parameters": self.added_parameters,
            "removed_parameters": self.removed_parameters,
            "modified_parameters": self.modified_parameters,
            "unchanged_parameters": self.unchanged_parameters,
            "total_l2_diff": self.total_l2_diff,
            "avg_cosine_similarity": self.avg_cosine_similarity,
            "most_changed_layers": self.most_changed_layers,
            "least_changed_layers": self.least_changed_layers,
        }


@dataclass
class DiffConfig:
    """Configuration for diff computation."""

    compute_element_wise: bool = True
    change_threshold: float = 1e-6
    include_unchanged: bool = False
    normalize_by_size: bool = True
    top_k_layers: int = 10


class ModelDiff:
    """Compare two model checkpoints."""

    def __init__(self, config: Optional[DiffConfig] = None):
        """
        Initialize differ.

        Args:
            config: Diff configuration
        """
        self.config = config or DiffConfig()
        self._numpy = None

    def _get_numpy(self):
        """Lazy load numpy."""
        if self._numpy is None:
            try:
                import numpy as np
                self._numpy = np
            except ImportError:
                from .checkpoint import MockNumpy
                self._numpy = MockNumpy()
        return self._numpy

    def diff(
        self,
        old_checkpoint: Checkpoint,
        new_checkpoint: Checkpoint,
    ) -> Tuple[List[ParameterDiff], DiffSummary]:
        """
        Compute differences between checkpoints.

        Args:
            old_checkpoint: Old/base checkpoint
            new_checkpoint: New checkpoint

        Returns:
            Tuple of (parameter diffs, summary)
        """
        np = self._get_numpy()

        old_names = set(old_checkpoint.layer_names)
        new_names = set(new_checkpoint.layer_names)

        parameter_diffs = []

        # Added parameters
        for name in new_names - old_names:
            diff = ParameterDiff(
                name=name,
                change_type=ChangeType.ADDED,
                new_info=new_checkpoint.get_info(name),
            )
            parameter_diffs.append(diff)

        # Removed parameters
        for name in old_names - new_names:
            diff = ParameterDiff(
                name=name,
                change_type=ChangeType.REMOVED,
                old_info=old_checkpoint.get_info(name),
            )
            parameter_diffs.append(diff)

        # Common parameters
        for name in old_names & new_names:
            old_info = old_checkpoint.get_info(name)
            new_info = new_checkpoint.get_info(name)

            # Check shape change
            if old_info.shape != new_info.shape:
                diff = ParameterDiff(
                    name=name,
                    change_type=ChangeType.SHAPE_CHANGED,
                    old_info=old_info,
                    new_info=new_info,
                )
                parameter_diffs.append(diff)
                continue

            # Compute element-wise diff
            old_tensor = old_checkpoint.get_tensor(name)
            new_tensor = new_checkpoint.get_tensor(name)

            if old_tensor is not None and new_tensor is not None:
                diff = self._compute_parameter_diff(
                    name, old_tensor, new_tensor, old_info, new_info
                )
                if diff.change_type != ChangeType.UNCHANGED or self.config.include_unchanged:
                    parameter_diffs.append(diff)
            else:
                # Can't compute diff without tensors
                diff = ParameterDiff(
                    name=name,
                    change_type=ChangeType.MODIFIED,
                    old_info=old_info,
                    new_info=new_info,
                )
                parameter_diffs.append(diff)

        # Compute summary
        summary = self._compute_summary(
            parameter_diffs,
            old_checkpoint,
            new_checkpoint,
        )

        return parameter_diffs, summary

    def _compute_parameter_diff(
        self,
        name: str,
        old_tensor: Any,
        new_tensor: Any,
        old_info: TensorInfo,
        new_info: TensorInfo,
    ) -> ParameterDiff:
        """Compute diff for a single parameter."""
        np = self._get_numpy()

        # Convert to numpy
        if hasattr(old_tensor, "numpy"):
            old_arr = old_tensor.numpy()
        elif hasattr(old_tensor, "cpu"):
            old_arr = old_tensor.cpu().numpy()
        else:
            old_arr = np.asarray(old_tensor)

        if hasattr(new_tensor, "numpy"):
            new_arr = new_tensor.numpy()
        elif hasattr(new_tensor, "cpu"):
            new_arr = new_tensor.cpu().numpy()
        else:
            new_arr = np.asarray(new_tensor)

        # Flatten for comparison
        old_flat = old_arr.flatten()
        new_flat = new_arr.flatten()

        if hasattr(old_flat, "astype"):
            old_flat = old_flat.astype(np.float64)
            new_flat = new_flat.astype(np.float64)

        # Compute difference
        diff_arr = new_flat - old_flat

        # Check if unchanged
        if hasattr(diff_arr, "data"):
            diff_data = diff_arr.data if hasattr(diff_arr, "data") else [diff_arr]
            np_obj = MockNumpy() if isinstance(np, MockNumpy) else np
            flat_diff = np_obj._flatten(diff_data) if hasattr(np_obj, "_flatten") else list(diff_arr.data)
            max_abs_diff = max(abs(x) for x in flat_diff) if flat_diff else 0
        else:
            max_abs_diff = float(np.max(np.abs(diff_arr)))

        if max_abs_diff < self.config.change_threshold:
            return ParameterDiff(
                name=name,
                change_type=ChangeType.UNCHANGED,
                old_info=old_info,
                new_info=new_info,
                cosine_sim=1.0,
            )

        # Compute metrics
        l2_diff = self._compute_l2(diff_arr)
        l1_diff = self._compute_l1(diff_arr)
        mean_diff = self._compute_mean(diff_arr)
        max_diff = max_abs_diff
        cosine_sim = self._compute_cosine_similarity(old_flat, new_flat)

        # Relative change
        old_norm = self._compute_l2(old_flat)
        relative_change = l2_diff / old_norm if old_norm > 0 else float('inf')

        # Count changed elements
        total_elements = old_info.num_elements
        changed_elements = self._count_changed(diff_arr)

        return ParameterDiff(
            name=name,
            change_type=ChangeType.MODIFIED,
            old_info=old_info,
            new_info=new_info,
            l2_diff=l2_diff,
            l1_diff=l1_diff,
            max_diff=max_diff,
            mean_diff=mean_diff,
            cosine_sim=cosine_sim,
            relative_change=relative_change,
            changed_elements=changed_elements,
            total_elements=total_elements,
        )

    def _compute_l2(self, arr) -> float:
        """Compute L2 norm."""
        np = self._get_numpy()
        if hasattr(arr, "data"):
            np_obj = MockNumpy() if isinstance(np, MockNumpy) else np
            flat = np_obj._flatten(arr.data) if hasattr(np_obj, "_flatten") else list(arr.data)
            return math.sqrt(sum(x * x for x in flat))
        return float(np.sqrt(np.sum(arr * arr)))

    def _compute_l1(self, arr) -> float:
        """Compute L1 norm."""
        np = self._get_numpy()
        if hasattr(arr, "data"):
            np_obj = MockNumpy() if isinstance(np, MockNumpy) else np
            flat = np_obj._flatten(arr.data) if hasattr(np_obj, "_flatten") else list(arr.data)
            return sum(abs(x) for x in flat)
        return float(np.sum(np.abs(arr)))

    def _compute_mean(self, arr) -> float:
        """Compute mean."""
        np = self._get_numpy()
        if hasattr(arr, "data"):
            np_obj = MockNumpy() if isinstance(np, MockNumpy) else np
            flat = np_obj._flatten(arr.data) if hasattr(np_obj, "_flatten") else list(arr.data)
            return sum(flat) / len(flat) if flat else 0
        return float(np.mean(arr))

    def _compute_cosine_similarity(self, arr1, arr2) -> float:
        """Compute cosine similarity."""
        np = self._get_numpy()

        if hasattr(arr1, "data"):
            np_obj = MockNumpy() if isinstance(np, MockNumpy) else np
            flat1 = np_obj._flatten(arr1.data) if hasattr(np_obj, "_flatten") else list(arr1.data)
            flat2 = np_obj._flatten(arr2.data) if hasattr(np_obj, "_flatten") else list(arr2.data)

            dot = sum(a * b for a, b in zip(flat1, flat2))
            norm1 = math.sqrt(sum(x * x for x in flat1))
            norm2 = math.sqrt(sum(x * x for x in flat2))

            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot / (norm1 * norm2)

        dot = float(np.sum(arr1 * arr2))
        norm1 = float(np.sqrt(np.sum(arr1 * arr1)))
        norm2 = float(np.sqrt(np.sum(arr2 * arr2)))

        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def _count_changed(self, diff_arr) -> int:
        """Count elements that changed."""
        np = self._get_numpy()
        threshold = self.config.change_threshold

        if hasattr(diff_arr, "data"):
            np_obj = MockNumpy() if isinstance(np, MockNumpy) else np
            flat = np_obj._flatten(diff_arr.data) if hasattr(np_obj, "_flatten") else list(diff_arr.data)
            return sum(1 for x in flat if abs(x) > threshold)

        return int(np.sum(np.abs(diff_arr) > threshold))

    def _compute_summary(
        self,
        diffs: List[ParameterDiff],
        old_ckpt: Checkpoint,
        new_ckpt: Checkpoint,
    ) -> DiffSummary:
        """Compute diff summary."""
        added = sum(1 for d in diffs if d.change_type == ChangeType.ADDED)
        removed = sum(1 for d in diffs if d.change_type == ChangeType.REMOVED)
        modified = sum(1 for d in diffs if d.change_type in [
            ChangeType.MODIFIED, ChangeType.SHAPE_CHANGED
        ])
        unchanged = sum(1 for d in diffs if d.change_type == ChangeType.UNCHANGED)

        total_l2 = sum(d.l2_diff for d in diffs)

        # Average cosine similarity for modified
        modified_diffs = [d for d in diffs if d.change_type == ChangeType.MODIFIED]
        avg_cos = (
            sum(d.cosine_sim for d in modified_diffs) / len(modified_diffs)
            if modified_diffs else 1.0
        )

        # Find most and least changed layers
        layer_changes: Dict[str, float] = {}
        for d in diffs:
            # Extract layer name (remove .weight, .bias, etc.)
            parts = d.name.rsplit(".", 1)
            layer_name = parts[0] if len(parts) > 1 else d.name

            if layer_name not in layer_changes:
                layer_changes[layer_name] = 0.0
            layer_changes[layer_name] += d.relative_change

        sorted_layers = sorted(
            layer_changes.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        k = self.config.top_k_layers
        most_changed = [name for name, _ in sorted_layers[:k] if layer_changes[name] > 0]
        least_changed = [name for name, _ in sorted_layers[-k:] if layer_changes[name] == 0] if sorted_layers else []

        return DiffSummary(
            total_parameters_old=old_ckpt.num_parameters,
            total_parameters_new=new_ckpt.num_parameters,
            added_parameters=added,
            removed_parameters=removed,
            modified_parameters=modified,
            unchanged_parameters=unchanged,
            total_l2_diff=total_l2,
            avg_cosine_similarity=avg_cos,
            most_changed_layers=most_changed,
            least_changed_layers=least_changed,
        )

    def get_layer_diffs(
        self,
        parameter_diffs: List[ParameterDiff],
    ) -> Dict[str, LayerDiff]:
        """Group parameter diffs by layer."""
        layers: Dict[str, LayerDiff] = {}

        for param_diff in parameter_diffs:
            # Extract layer name
            parts = param_diff.name.rsplit(".", 1)
            layer_name = parts[0] if len(parts) > 1 else param_diff.name

            if layer_name not in layers:
                layers[layer_name] = LayerDiff(name=layer_name)

            layers[layer_name].parameters.append(param_diff)

        return layers


def compare_checkpoints(
    old_path: str,
    new_path: str,
    config: Optional[DiffConfig] = None,
) -> Tuple[List[ParameterDiff], DiffSummary]:
    """
    Compare two checkpoints.

    Args:
        old_path: Path to old checkpoint
        new_path: Path to new checkpoint
        config: Diff configuration

    Returns:
        Tuple of (parameter diffs, summary)
    """
    from .checkpoint import load_checkpoint

    old_ckpt = load_checkpoint(old_path)
    new_ckpt = load_checkpoint(new_path)

    differ = ModelDiff(config)
    return differ.diff(old_ckpt, new_ckpt)


class MockNumpy:
    """Mock numpy - reuse from checkpoint module."""

    def __init__(self):
        self.float64 = "float64"

    def asarray(self, data):
        from .checkpoint import MockArray
        if isinstance(data, MockArray):
            return data
        return MockArray(data)

    def _flatten(self, data):
        if isinstance(data, (list, tuple)):
            result = []
            for item in data:
                if isinstance(item, (list, tuple)):
                    result.extend(self._flatten(item))
                else:
                    result.append(item)
            return result
        return [data]

    def abs(self, arr):
        from .checkpoint import MockArray
        if hasattr(arr, "data"):
            flat = self._flatten(arr.data)
            return MockArray([abs(x) for x in flat])
        return abs(arr)

    def sum(self, arr):
        if hasattr(arr, "data"):
            flat = self._flatten(arr.data)
            return sum(flat)
        return arr

    def sqrt(self, val):
        return math.sqrt(val) if val >= 0 else 0

    def mean(self, arr):
        if hasattr(arr, "data"):
            flat = self._flatten(arr.data)
            return sum(flat) / len(flat) if flat else 0
        return arr

    def max(self, arr):
        if hasattr(arr, "data"):
            flat = self._flatten(arr.data)
            return max(flat) if flat else 0
        return arr
