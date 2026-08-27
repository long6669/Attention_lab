from dataclasses import dataclass, field
from typing import Any, Optional

from .node import Node
from .tensor import TensorSpec


@dataclass(frozen=True)
class Edge:
    """A directed data-flow edge between two graph nodes."""

    source: str
    target: str
    tensor_id: str

    def __post_init__(self) -> None:
        if not self.source or not self.target:
            raise ValueError("Edge endpoints must not be empty.")
        if not self.tensor_id:
            raise ValueError("Edge tensor id must not be empty.")

    def to_dict(self) -> dict[str, str]:
        return {
            "source": self.source,
            "target": self.target,
            "tensor_id": self.tensor_id,
        }


@dataclass
class Graph:
    """The framework-independent intermediate representation for attention."""

    nodes: list[Node] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    tensors: dict[str, TensorSpec] = field(default_factory=dict)

    def add_tensor(self, tensor: TensorSpec) -> TensorSpec:
        if tensor.id in self.tensors:
            raise ValueError(f"Tensor id already exists: {tensor.id}")
        self.tensors[tensor.id] = tensor
        return tensor

    def add_node(self, node: Node) -> Node:
        if self.get_node(node.id) is not None:
            raise ValueError(f"Node id already exists: {node.id}")

        unknown_tensors = [
            tensor_id
            for tensor_id in (*node.inputs, *node.outputs)
            if tensor_id not in self.tensors
        ]
        if unknown_tensors:
            unknown = ", ".join(dict.fromkeys(unknown_tensors))
            raise ValueError(f"Node references unknown tensors: {unknown}")

        self.nodes.append(node)
        return node

    def add_edge(self, edge: Edge) -> Edge:
        if self.get_node(edge.source) is None:
            raise ValueError(f"Unknown edge source node: {edge.source}")
        if self.get_node(edge.target) is None:
            raise ValueError(f"Unknown edge target node: {edge.target}")
        if edge.tensor_id not in self.tensors:
            raise ValueError(f"Unknown edge tensor: {edge.tensor_id}")
        if edge.tensor_id not in self.get_node(edge.source).outputs:
            raise ValueError("Edge tensor is not produced by its source node.")
        if edge.tensor_id not in self.get_node(edge.target).inputs:
            raise ValueError("Edge tensor is not consumed by its target node.")
        if edge in self.edges:
            raise ValueError("Edge already exists.")

        self.edges.append(edge)
        return edge

    def get_node(self, node_id: str) -> Optional[Node]:
        return next((node for node in self.nodes if node.id == node_id), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "tensors": {
                tensor_id: tensor.to_dict()
                for tensor_id, tensor in self.tensors.items()
            },
        }
