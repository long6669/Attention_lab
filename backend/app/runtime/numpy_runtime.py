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
