from dataclasses import dataclass, field
from typing import Any


@dataclass
class Node:
    """An executable operation in an attention graph."""

    id: str
    op: str
    label: str
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Node id must not be empty.")
        if not self.op:
            raise ValueError("Node op must not be empty.")
        if not self.label:
            raise ValueError("Node label must not be empty.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "op": self.op,
            "label": self.label,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "attrs": dict(self.attrs),
        }
