import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

INLINE_VALUE_LIMIT = 256
MAX_BLOCK_GROUPS = 8
MAX_SLICE_ELEMENTS = 4096


def json_values(array: NDArray) -> list[Any]:
    values = np.asarray(array).astype(object)
    if np.issubdtype(array.dtype, np.floating):
        values[~np.isfinite(array)] = None
    return values.tolist()


@dataclass(frozen=True)
class MemoryTensorSpec:
    """Shape-first description of persistent runtime state."""

    id: str
    name: str
    kind: str
    role: str
    value: NDArray
    axes: tuple[str, ...]
    growth_axis: Optional[int]

    def to_dict(self) -> dict[str, Any]:
        array = np.asarray(self.value)
        payload: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "kind": self.kind,
            "role": self.role,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "numel": int(array.size),
            "bytes": int(array.nbytes),
            "axes": list(self.axes),
            "growth_axis": self.growth_axis,
            "growth_axis_name": (
                self.axes[self.growth_axis] if self.growth_axis is not None else None
            ),
            "values_loaded": array.size <= INLINE_VALUE_LIMIT,
            "blocks": self._blocks(array),
        }
        if array.size <= INLINE_VALUE_LIMIT:
            payload["values"] = json_values(array)
        return payload

    def _blocks(self, array: NDArray) -> list[dict[str, Any]]:
        if self.growth_axis is None or array.size == 0:
            return []

        growth_size = array.shape[self.growth_axis]
        growth_step = max(1, math.ceil(growth_size / MAX_BLOCK_GROUPS))
        head_axis = self.axes.index("head") if "head" in self.axes else None
        head_size = array.shape[head_axis] if head_axis is not None else 1
        head_step = max(1, math.ceil(head_size / MAX_BLOCK_GROUPS))
        blocks: list[dict[str, Any]] = []

        for head_start in range(0, head_size, head_step):
            head_end = min(head_start + head_step, head_size)
            for start in range(0, growth_size, growth_step):
                end = min(start + growth_step, growth_size)
                slices = [slice(None)] * array.ndim
                slices[self.growth_axis] = slice(start, end)
                if head_axis is not None:
                    slices[head_axis] = slice(head_start, head_end)
                block = array[tuple(slices)]
                finite = (
                    block[np.isfinite(block)]
                    if np.issubdtype(
                        block.dtype,
                        np.floating,
                    )
                    else block.reshape(-1)
                )
                blocks.append(
                    {
                        "start": start,
                        "end": end,
                        "head_start": head_start if head_axis is not None else None,
                        "head_end": head_end if head_axis is not None else None,
                        "numel": int(block.size),
                        "min": float(np.min(finite)) if finite.size else None,
                        "max": float(np.max(finite)) if finite.size else None,
                        "mean_abs": (
                            float(np.mean(np.abs(finite))) if finite.size else None
                        ),
                        "l2": (float(np.linalg.norm(finite)) if finite.size else None),
                    }
                )
        return blocks


@dataclass(frozen=True)
class MemorySpec:
    kind: str
    tensors: tuple[MemoryTensorSpec, ...]

    def to_dict(self) -> dict[str, Any]:
        tensor_payloads = [tensor.to_dict() for tensor in self.tensors]
        return {
            "kind": self.kind,
            "tensors": tensor_payloads,
            "total_numel": sum(item["numel"] for item in tensor_payloads),
            "total_bytes": sum(item["bytes"] for item in tensor_payloads),
            "growth_axes": sorted(
                {
                    item["growth_axis_name"]
                    for item in tensor_payloads
                    if item["growth_axis_name"] is not None
                }
            ),
        }


def infer_axes(
    array: NDArray,
    memory_id: str = "",
) -> tuple[tuple[str, ...], Optional[int]]:
    if "state" in memory_id and array.ndim == 4:
        return (
            ("batch", "head", "key_feature", "value_feature"),
            None,
        )
    if array.ndim == 4:
        return ("batch", "head", "token", "feature"), 2
    if array.ndim == 3:
        return ("batch", "token", "feature"), 1
    if array.ndim == 2:
        return ("batch", "state"), None
    return tuple(f"axis_{index}" for index in range(array.ndim)), None


def slice_memory(
    array: NDArray,
    axes: Sequence[str],
    growth_axis: Optional[int],
    start: int,
    end: int,
    head: Optional[int],
) -> NDArray:
    slices = [slice(None)] * array.ndim
    if growth_axis is not None:
        slices[growth_axis] = slice(start, end)
    if head is not None and "head" in axes:
        head_axis = axes.index("head")
        slices[head_axis] = slice(head, head + 1)
    result = array[tuple(slices)]
    if result.size > MAX_SLICE_ELEMENTS:
        raise ValueError(
            f"Requested memory slice has {result.size} values; "
            f"limit is {MAX_SLICE_ELEMENTS}."
        )
    return result
