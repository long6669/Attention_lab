from typing import Optional, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


class NumPyRuntime:
    """NumPy implementations of the tensor operations used by AttnLab."""

    def linear(
        self,
        inputs: ArrayLike,
        weight: ArrayLike,
        bias: Optional[ArrayLike] = None,
    ) -> NDArray:
        inputs_array = np.asarray(inputs)
        weight_array = np.asarray(weight)

        if inputs_array.ndim == 0:
            raise ValueError("Linear inputs must have at least one dimension.")
        if weight_array.ndim != 2:
            raise ValueError("Linear weight must be a 2D array.")
        if inputs_array.shape[-1] != weight_array.shape[0]:
            raise ValueError(
                "Linear input dimension must match the first weight dimension."
            )

        output = np.matmul(inputs_array, weight_array)
        if bias is None:
            return output

        bias_array = np.asarray(bias)
        if bias_array.ndim != 1 or bias_array.shape[0] != weight_array.shape[1]:
            raise ValueError("Linear bias must match the output dimension.")
        return output + bias_array

    def matmul(self, left: ArrayLike, right: ArrayLike) -> NDArray:
        return np.matmul(np.asarray(left), np.asarray(right))

    def reshape(self, tensor: ArrayLike, shape: Sequence[int]) -> NDArray:
        return np.reshape(np.asarray(tensor), tuple(shape))

    def transpose(
        self,
        tensor: ArrayLike,
        axes: Optional[Sequence[int]] = None,
    ) -> NDArray:
        normalized_axes = None if axes is None else tuple(axes)
        return np.transpose(np.asarray(tensor), axes=normalized_axes)

    def repeat_kv(self, tensor: ArrayLike, repeats: int) -> NDArray:
        tensor_array = np.asarray(tensor)
        if tensor_array.ndim != 4:
            raise ValueError("RepeatKV expects [batch, heads, sequence, dim].")
        if repeats <= 0:
            raise ValueError("RepeatKV repeats must be positive.")
        return np.repeat(tensor_array, repeats, axis=1)

    def rope(
        self,
        tensor: ArrayLike,
        positions: ArrayLike,
        base: float = 10000.0,
    ) -> NDArray:
        tensor_array = np.asarray(tensor)
        position_array = np.asarray(positions)
        if tensor_array.ndim != 4:
            raise ValueError("RoPE expects [batch, heads, sequence, dim].")
        if tensor_array.shape[-1] % 2 != 0:
            raise ValueError("RoPE head dimension must be even.")
        if position_array.ndim != 1:
            raise ValueError("RoPE positions must be a 1D array.")
        if position_array.shape[0] != tensor_array.shape[-2]:
            raise ValueError("RoPE positions must match sequence length.")
        if base <= 0:
            raise ValueError("RoPE base must be positive.")

        head_dim = tensor_array.shape[-1]
        frequencies = 1.0 / (
            base ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim)
        )
        angles = position_array.astype(np.float32)[:, None] * frequencies[None, :]
        cosine = np.cos(angles)[None, None, :, :]
        sine = np.sin(angles)[None, None, :, :]
        even = tensor_array[..., 0::2]
        odd = tensor_array[..., 1::2]
        output = np.empty_like(tensor_array)
        output[..., 0::2] = even * cosine - odd * sine
        output[..., 1::2] = even * sine + odd * cosine
        return output

    def scale(self, tensor: ArrayLike, divisor: float) -> NDArray:
        if divisor == 0:
            raise ValueError("Scale divisor must not be zero.")
        return np.asarray(tensor) / divisor

    def causal_mask(
        self,
        scores: ArrayLike,
        mask_value: float = -np.inf,
    ) -> NDArray:
        scores_array = np.asarray(scores)
        if scores_array.ndim < 2:
            raise ValueError("Attention scores must have at least two dimensions.")
        if not np.issubdtype(scores_array.dtype, np.floating):
            raise TypeError("Attention scores must use a floating-point dtype.")

        query_length, key_length = scores_array.shape[-2:]
        query_offset = max(key_length - query_length, 0)
        query_positions = np.arange(query_length)[:, None] + query_offset
        key_positions = np.arange(key_length)[None, :]
        mask = key_positions > query_positions

        return np.where(mask, mask_value, scores_array)

    def softmax(self, tensor: ArrayLike, axis: int = -1) -> NDArray:
        tensor_array = np.asarray(tensor)
        if tensor_array.ndim == 0:
            raise ValueError("Softmax input must have at least one dimension.")
        if not np.issubdtype(tensor_array.dtype, np.floating):
            tensor_array = tensor_array.astype(np.float32)

        shifted = tensor_array - np.max(tensor_array, axis=axis, keepdims=True)
        exponentials = np.exp(shifted)
        denominator = np.sum(exponentials, axis=axis, keepdims=True)
        return exponentials / denominator

    def kda_scan(
        self,
        queries: ArrayLike,
        keys: ArrayLike,
        values: ArrayLike,
        initial_state: ArrayLike,
        decay: float,
        write_rate: float,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray, NDArray]:
        q = np.asarray(queries)
        k = np.asarray(keys)
        v = np.asarray(values)
        state = np.asarray(initial_state).copy()
        if q.ndim != 4 or q.shape != k.shape or q.shape != v.shape:
            raise ValueError("KDA Q/K/V must share [batch, heads, sequence, dim].")
        expected_state = (q.shape[0], q.shape[1], q.shape[3], q.shape[3])
        if state.shape != expected_state:
            raise ValueError("KDA state must be [batch, heads, dim, dim].")
        if not 0.0 <= decay <= 1.0:
            raise ValueError("KDA decay must be between zero and one.")
        if not 0.0 <= write_rate <= 1.0:
            raise ValueError("KDA write_rate must be between zero and one.")

        outputs = []
        decayed_states = []
        erased_states = []
        written_states = []
        for position in range(q.shape[2]):
            key = k[:, :, position, :]
            key = key / np.maximum(
                np.linalg.norm(key, axis=-1, keepdims=True),
                1e-6,
            )
            value = v[:, :, position, :]
            decayed = decay * state
            prediction = np.einsum("bhd,bhdv->bhv", key, decayed)
            erased = decayed - write_rate * np.einsum(
                "bhd,bhv->bhdv",
                key,
                prediction,
            )
            written = erased + write_rate * np.einsum(
                "bhd,bhv->bhdv",
                key,
                value,
            )
            output = np.einsum(
                "bhd,bhdv->bhv",
                q[:, :, position, :],
                written,
            )
            decayed_states.append(decayed)
            erased_states.append(erased)
            written_states.append(written)
            outputs.append(output)
            state = written

        return (
            np.stack(outputs, axis=2),
            np.stack(decayed_states, axis=2),
            np.stack(erased_states, axis=2),
            np.stack(written_states, axis=2),
            state,
        )

    def sequence_compress(
        self,
        tensor: ArrayLike,
        window_sizes: Sequence[int],
    ) -> tuple[NDArray, list[tuple[int, int, int]]]:
        tensor_array = np.asarray(tensor)
        if tensor_array.ndim != 4:
            raise ValueError(
                "SequenceCompression expects [batch, heads, sequence, dim]."
            )
        if not window_sizes or any(size <= 0 for size in window_sizes):
            raise ValueError("Compression window sizes must be positive.")

        sequence = tensor_array.shape[2]
        summaries = []
        spans: list[tuple[int, int, int]] = []
        for level, window_size in enumerate(window_sizes):
            for end in range(1, sequence + 1):
                start = max(0, end - window_size)
                summaries.append(
                    np.mean(
                        tensor_array[:, :, start:end, :],
                        axis=2,
                        keepdims=True,
                    )
                )
                spans.append((start, end, level))
        return np.concatenate(summaries, axis=2), spans

    def index_scores(
        self,
        queries: ArrayLike,
        compressed_keys: ArrayLike,
        spans: Sequence[Sequence[int]],
        query_offset: int,
    ) -> NDArray:
        q = np.asarray(queries)
        k = np.asarray(compressed_keys)
        if q.ndim != 4 or k.ndim != 4:
            raise ValueError("Indexer expects rank-four Q and compressed K.")
        if q.shape[:2] != k.shape[:2] or q.shape[-1] != k.shape[-1]:
            raise ValueError("Indexer Q and K dimensions are incompatible.")
        if len(spans) != k.shape[2]:
            raise ValueError("Compression spans must match compressed sequence.")

        scores = np.matmul(q, np.swapaxes(k, -1, -2))
        scores = scores / np.sqrt(q.shape[-1])
        span_ends = np.asarray([span[1] for span in spans])
        query_ends = query_offset + np.arange(q.shape[2]) + 1
        mask = span_ends[None, :] > query_ends[:, None]
        return np.where(mask[None, None, :, :], -np.inf, scores)

    def topk(self, tensor: ArrayLike, k: int) -> NDArray:
        values = np.asarray(tensor)
        if values.ndim == 0:
            raise ValueError("TopK input must have at least one dimension.")
        if k <= 0:
            raise ValueError("TopK k must be positive.")
        selected = min(k, values.shape[-1])
        order = np.argsort(values, axis=-1)
        return np.flip(order[..., -selected:], axis=-1).astype(np.int32)

    def route_values(
        self,
        values: ArrayLike,
        indices: ArrayLike,
    ) -> NDArray:
        value_array = np.asarray(values)
        index_array = np.asarray(indices)
        if value_array.ndim != 4 or index_array.ndim != 4:
            raise ValueError("Routing expects values [B,H,S,D] and indices [B,H,Q,K].")
        expanded = np.broadcast_to(
            value_array[:, :, None, :, :],
            (
                value_array.shape[0],
                value_array.shape[1],
                index_array.shape[2],
                value_array.shape[2],
                value_array.shape[3],
            ),
        )
        return np.take_along_axis(
            expanded,
            index_array[..., None],
            axis=3,
        )

    def route_scores(
        self,
        scores: ArrayLike,
        indices: ArrayLike,
    ) -> NDArray:
        return np.take_along_axis(
            np.asarray(scores),
            np.asarray(indices),
            axis=-1,
        )

    def weighted_route(
        self,
        probabilities: ArrayLike,
        routed_values: ArrayLike,
    ) -> NDArray:
        weights = np.asarray(probabilities)
        values = np.asarray(routed_values)
        if values.shape[:-1] != weights.shape:
            raise ValueError("Routed values must align with routing probabilities.")
        return np.sum(weights[..., None] * values, axis=-2)
