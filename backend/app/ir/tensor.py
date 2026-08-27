from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TensorSpec:
    """Static metadata for a tensor carried by the attention graph."""

    id: str
    name: str
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Tensor id must not be empty.")
        if not self.name:
            raise ValueError("Tensor name must not be empty.")
        if any(dimension < 0 for dimension in self.shape):
            raise ValueError("Tensor dimensions must be non-negative.")
        if not self.dtype:
            raise ValueError("Tensor dtype must not be empty.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
        }
