from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from app.ir import Edge, Graph, Node, TensorSpec
from app.runtime import NumPyRuntime
from app.tracing import TraceRecorder


@dataclass(frozen=True)
class TensorValue:
    id: str
    name: str
    value: NDArray

    def to_dict(self) -> dict[str, Any]:
        values = self.value.astype(object)
        if np.issubdtype(self.value.dtype, np.floating):
            values[~np.isfinite(self.value)] = None
        return {
            "id": self.id,
            "name": self.name,
            "shape": list(self.value.shape),
            "dtype": str(self.value.dtype),
            "values": values.tolist(),
        }


class PrimitiveExecutor:
    """Executes primitives while building graph, trace, and tensor state."""

    def __init__(
        self,
        runtime: Optional[NumPyRuntime] = None,
        recorder: Optional[TraceRecorder] = None,
    ) -> None:
        self.runtime = runtime or NumPyRuntime()
        self.recorder = recorder or TraceRecorder()
        self.graph = Graph()
        self.tensors: dict[str, TensorValue] = {}
        self._producers: dict[str, str] = {}

    def value(self, tensor_id: str) -> NDArray:
        try:
            return self.tensors[tensor_id].value
        except KeyError as error:
            raise KeyError(f"Unknown tensor value: {tensor_id}") from error

    def input(
        self,
        values: ArrayLike,
        node_id: str = "input",
        output_id: str = "token_ids",
        label: str = "Input Tokens",
        output_name: str = "Token IDs",
    ) -> str:
        return self._register(
            node_id=node_id,
            op="input",
            label=label,
            input_ids=[],
            output_id=output_id,
            output_name=output_name,
            output=np.asarray(values, dtype=np.int32),
        )

    def embedding(
        self,
        input_id: str,
        values: ArrayLike,
        node_id: str = "embedding",
        output_id: str = "x",
        label: str = "Toy Embedding",
        output_name: str = "Embedding",
    ) -> str:
        return self._register(
            node_id=node_id,
            op="embedding",
            label=label,
            input_ids=[input_id],
            output_id=output_id,
            output_name=output_name,
            output=np.asarray(values, dtype=np.float32),
        )

    def linear(
        self,
        input_id: str,
        weight: ArrayLike,
        node_id: str,
        output_id: str,
        label: str,
        output_name: str,
        bias: Optional[ArrayLike] = None,
        parameter_name: Optional[str] = None,
        op: str = "linear",
    ) -> str:
        weight_array = np.asarray(weight)
        weight_id = f"{node_id}_weight"
        parameter_ids = [weight_id]
        self._store_tensor(
            weight_id,
            parameter_name or f"{label} Weight",
            weight_array,
        )
        if bias is not None:
            bias_id = f"{node_id}_bias"
            self._store_tensor(
                bias_id,
                f"{label} Bias",
                np.asarray(bias),
            )
            parameter_ids.append(bias_id)

        output = self.runtime.linear(self.value(input_id), weight_array, bias)
        return self._register(
            node_id,
            op,
            label,
            [input_id],
            output_id,
            output_name,
            output,
            {
                "parameter_ids": parameter_ids,
                "has_bias": bias is not None,
            },
        )

    def repeat_kv(
        self,
        input_id: str,
        repeats: int,
        node_id: str,
        output_id: str,
        label: str,
        output_name: str,
    ) -> str:
        output = self.runtime.repeat_kv(self.value(input_id), repeats)
        return self._register(
            node_id,
            "repeat_kv",
            label,
            [input_id],
            output_id,
            output_name,
            output,
            {"repeats": repeats},
        )

    def rope(
        self,
        input_id: str,
        positions: ArrayLike,
        node_id: str,
        output_id: str,
        label: str,
        output_name: str,
        base: float = 10000.0,
    ) -> str:
        position_array = np.asarray(positions, dtype=np.int32)
        output = self.runtime.rope(
            self.value(input_id),
            position_array,
            base,
        )
        return self._register(
            node_id,
            "rope",
            label,
            [input_id],
            output_id,
            output_name,
            output,
            {
                "base": base,
                "positions": position_array.tolist(),
            },
        )

    def split_heads(
        self,
        input_id: str,
        num_heads: int,
        node_id: str,
        output_id: str,
        label: str,
        output_name: str,
    ) -> str:
        values = self.value(input_id)
        if values.ndim != 3:
            raise ValueError("SplitHeads expects [batch, sequence, model].")
        if values.shape[-1] % num_heads != 0:
            raise ValueError("Model dimension must be divisible by num_heads.")

        batch, sequence, model = values.shape
        head_dim = model // num_heads
        split = self.runtime.reshape(
            values,
            (batch, sequence, num_heads, head_dim),
        )
        output = self.runtime.transpose(split, (0, 2, 1, 3))
        return self._register(
            node_id,
            "split_heads",
            label,
            [input_id],
            output_id,
            output_name,
            output,
            {"num_heads": num_heads, "head_dim": head_dim},
        )

    def transpose(
        self,
        input_id: str,
        axes: Sequence[int],
        node_id: str,
        output_id: str,
        label: str,
        output_name: str,
    ) -> str:
        output = self.runtime.transpose(self.value(input_id), axes)
        return self._register(
            node_id,
            "transpose",
            label,
            [input_id],
            output_id,
            output_name,
            output,
            {"axes": list(axes)},
        )

    def matmul(
        self,
        left_id: str,
        right_id: str,
        node_id: str,
        output_id: str,
        label: str,
        output_name: str,
        visualization: Optional[str] = None,
    ) -> str:
        output = self.runtime.matmul(
            self.value(left_id),
            self.value(right_id),
        )
        attrs = {} if visualization is None else {"visualization": visualization}
        return self._register(
            node_id,
            "matmul",
            label,
            [left_id, right_id],
            output_id,
            output_name,
            output,
            attrs,
        )

    def scale(
        self,
        input_id: str,
        divisor: float,
        node_id: str,
        output_id: str,
        label: str,
        output_name: str,
    ) -> str:
        output = self.runtime.scale(self.value(input_id), divisor)
        return self._register(
            node_id,
            "scale",
            label,
            [input_id],
            output_id,
            output_name,
            output,
            {"divisor": divisor, "visualization": "attention_matrix"},
        )

    def causal_mask(
        self,
        input_id: str,
        node_id: str,
        output_id: str,
        label: str,
        output_name: str,
    ) -> str:
        output = self.runtime.causal_mask(self.value(input_id))
        return self._register(
            node_id,
            "causal_mask",
            label,
            [input_id],
            output_id,
            output_name,
            output,
            {"visualization": "attention_matrix"},
        )

    def softmax(
        self,
        input_id: str,
        node_id: str,
        output_id: str,
        label: str,
        output_name: str,
    ) -> str:
        output = self.runtime.softmax(self.value(input_id))
        return self._register(
            node_id,
            "softmax",
            label,
            [input_id],
            output_id,
            output_name,
            output,
            {"axis": -1, "visualization": "attention_matrix"},
        )

    def merge_heads(
        self,
        input_id: str,
        node_id: str,
        output_id: str,
        label: str,
        output_name: str,
    ) -> str:
        values = self.value(input_id)
        if values.ndim != 4:
            raise ValueError("MergeHeads expects [batch, heads, sequence, dim].")

        batch, heads, sequence, head_dim = values.shape
        transposed = self.runtime.transpose(values, (0, 2, 1, 3))
        output = self.runtime.reshape(
            transposed,
            (batch, sequence, heads * head_dim),
        )
        return self._register(
            node_id,
            "merge_heads",
            label,
            [input_id],
            output_id,
            output_name,
            output,
        )

    def cache_append(
        self,
        input_id: str,
        node_id: str,
        output_id: str,
        label: str,
        output_name: str,
        existing: Optional[ArrayLike] = None,
        existing_id: Optional[str] = None,
    ) -> str:
        current = self.value(input_id)
        if existing is not None and existing_id is not None:
            raise ValueError("Provide existing cache data or an existing tensor id, not both.")
        if existing_id is not None:
            existing = self.value(existing_id)
        previous_tokens = 0 if existing is None else int(np.asarray(existing).shape[-2])
        output = (
            current.copy()
            if existing is None
            else np.concatenate((np.asarray(existing), current), axis=-2)
        )
        return self._register(
            node_id,
            "cache_append",
            label,
            ([existing_id] if existing_id is not None else []) + [input_id],
            output_id,
            output_name,
            output,
            {
                "previous_tokens": previous_tokens,
                "appended_tokens": int(current.shape[-2]),
                "cached_tokens": int(output.shape[-2]),
            },
        )

    def cache_read(
        self,
        values: ArrayLike,
        node_id: str,
        output_id: str,
        label: str,
        output_name: str,
    ) -> str:
        output = np.asarray(values).copy()
        return self._register(
            node_id,
            "cache_read",
            label,
            [],
            output_id,
            output_name,
            output,
            {
                "cached_tokens": int(output.shape[-2]),
                "bytes": int(output.nbytes),
            },
        )

    def output(
        self,
        input_id: str,
        node_id: str = "output",
        output_id: str = "output",
    ) -> str:
        return self._register(
            node_id,
            "output",
            "Output",
            [input_id],
            output_id,
            "Attention Output",
            self.value(input_id).copy(),
        )

    def tensor_payload(self) -> dict[str, dict[str, Any]]:
        return {
            tensor_id: tensor.to_dict()
            for tensor_id, tensor in self.tensors.items()
        }

    def _store_tensor(
        self,
        tensor_id: str,
        tensor_name: str,
        value: NDArray,
    ) -> None:
        array = np.asarray(value)
        self.graph.add_tensor(
            TensorSpec(
                id=tensor_id,
                name=tensor_name,
                shape=array.shape,
                dtype=str(array.dtype),
            )
        )
        self.tensors[tensor_id] = TensorValue(
            id=tensor_id,
            name=tensor_name,
            value=array,
        )

    def _register(
        self,
        node_id: str,
        op: str,
        label: str,
        input_ids: list[str],
        output_id: str,
        output_name: str,
        output: NDArray,
        attrs: Optional[dict[str, Any]] = None,
    ) -> str:
        output_array = np.asarray(output)
        self._store_tensor(output_id, output_name, output_array)
        input_shapes = [
            list(self.graph.tensors[tensor_id].shape)
            for tensor_id in input_ids
        ]
        node_attrs = {
            **(attrs or {}),
            "input_shapes": input_shapes,
            "output_shapes": [list(output_array.shape)],
        }
        node = self.graph.add_node(
            Node(
                id=node_id,
                op=op,
                label=label,
                inputs=list(input_ids),
                outputs=[output_id],
                attrs=node_attrs,
            )
        )
        for tensor_id in input_ids:
            producer = self._producers.get(tensor_id)
            if producer is not None:
                self.graph.add_edge(Edge(producer, node.id, tensor_id))

        self._producers[output_id] = node.id
        self.recorder.record(
            node_id=node.id,
            op=node.op,
            inputs=node.inputs,
            outputs=node.outputs,
            title=node.label,
        )
        return output_id
