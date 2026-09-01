import hashlib
import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.memory import MemorySpec, MemoryTensorSpec
from app.ops import PrimitiveExecutor


@dataclass(frozen=True)
class MHAConfig:
    architecture: str = "mha"
    d_model: int = 8
    num_q_heads: int = 2
    num_kv_heads: int = 2
    dtype: str = "float32"
    seed: int = 42
    use_rope: bool = False
    rope_base: float = 10000.0
    kv_lora_rank: int = 4
    state_decay: float = 0.95
    state_write_rate: float = 0.5
    compression_window: int = 2
    routing_top_k: int = 2

    def __post_init__(self) -> None:
        if self.d_model <= 0:
            raise ValueError("d_model must be positive.")
        if self.num_q_heads <= 0 or self.num_kv_heads <= 0:
            raise ValueError("Attention head counts must be positive.")
        if self.d_model % self.num_q_heads != 0:
            raise ValueError("d_model must be divisible by num_q_heads.")
        if self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError("num_q_heads must be divisible by num_kv_heads.")
        if self.dtype != "float32":
            raise ValueError("The MVP runtime supports float32 only.")
        if self.rope_base <= 0:
            raise ValueError("rope_base must be positive.")
        if self.kv_lora_rank <= 0:
            raise ValueError("kv_lora_rank must be positive.")
        if not 0.0 <= self.state_decay <= 1.0:
            raise ValueError("state_decay must be between zero and one.")
        if not 0.0 <= self.state_write_rate <= 1.0:
            raise ValueError("state_write_rate must be between zero and one.")
        if self.compression_window <= 0 or self.routing_top_k <= 0:
            raise ValueError("Compression and routing sizes must be positive.")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.num_q_heads

    @property
    def num_heads(self) -> int:
        return self.num_q_heads


@dataclass(frozen=True)
class AttentionState:
    """Persistent K/V state retained between prefill and decode calls."""

    tokens: tuple[str, ...]
    k_cache: NDArray
    v_cache: NDArray


@dataclass(frozen=True)
class AttentionRun:
    tokens: list[str]
    executor: PrimitiveExecutor
    config: MHAConfig
    state: AttentionState
    phase: str
    previous_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        k_cache = self.state.k_cache
        v_cache = self.state.v_cache
        appended_tokens = len(self.tokens) - self.previous_tokens
        memory_spec = MemorySpec(
            kind="attention_cache",
            tensors=(
                MemoryTensorSpec(
                    id="k_cache",
                    name="Key Cache",
                    kind="kv_cache",
                    role="key",
                    value=k_cache,
                    axes=("batch", "head", "token", "feature"),
                    growth_axis=2,
                ),
                MemoryTensorSpec(
                    id="v_cache",
                    name="Value Cache",
                    kind="kv_cache",
                    role="value",
                    value=v_cache,
                    axes=("batch", "head", "token", "feature"),
                    growth_axis=2,
                ),
            ),
        ).to_dict()
        return {
            "tokens": list(self.tokens),
            "phase": self.phase,
            "config": {
                **asdict(self.config),
                "batch_size": 1,
                "seq_len": len(self.tokens),
                "head_dim": self.config.head_dim,
                "num_heads": self.config.num_q_heads,
            },
            "graph": self.executor.graph.to_dict(),
            "trace": self.executor.recorder.to_list(),
            "tensors": self.executor.tensor_payload(),
            "memory": {
                "cache_kind": "kv",
                "tokens": len(self.tokens),
                "k_cache": self._cache_summary(k_cache),
                "v_cache": self._cache_summary(v_cache),
                "total_elements": int(k_cache.size + v_cache.size),
                "total_bytes": int(k_cache.nbytes + v_cache.nbytes),
                "spec": memory_spec,
            },
            "cache_activity": {
                "phase": self.phase,
                "read_tokens": self.previous_tokens if self.phase == "decode" else 0,
                "appended_tokens": appended_tokens,
                "resulting_tokens": len(self.tokens),
            },
        }

    @staticmethod
    def _cache_summary(cache: NDArray) -> dict[str, Any]:
        return {
            "shape": list(cache.shape),
            "dtype": str(cache.dtype),
            "elements": int(cache.size),
            "bytes": int(cache.nbytes),
        }


class MultiHeadAttention:
    """Deterministic toy MHA expressed entirely through graph primitives."""

    def __init__(self, config: MHAConfig = MHAConfig()) -> None:
        self.config = config

    def run(self, tokens: list[str]) -> AttentionRun:
        """Compatibility alias for the prefill execution path."""
        return self.prefill(tokens)

    def prefill(self, tokens: list[str]) -> AttentionRun:
        if not tokens:
            raise ValueError("At least one token is required.")

        executor = PrimitiveExecutor()
        token_ids = executor.input(np.arange(len(tokens))[None, :])
        embeddings = executor.embedding(
            token_ids,
            self._embed_tokens(tokens)[None, :, :],
        )
        weights = self._projection_weights()
        q = executor.linear(
            embeddings,
            weights["q"],
            "q_proj",
            "q",
            "Q Projection",
            "Q",
            parameter_name="Wq",
        )
        k = executor.linear(
            embeddings,
            weights["k"],
            "k_proj",
            "k",
            "K Projection",
            "K",
            parameter_name="Wk",
        )
        v = executor.linear(
            embeddings,
            weights["v"],
            "v_proj",
            "v",
            "V Projection",
            "V",
            parameter_name="Wv",
        )
        q_heads = executor.split_heads(
            q,
            self.config.num_q_heads,
            "q_split",
            "q_heads",
            "Split Q Heads",
            "Q Heads",
        )
        k_heads = executor.split_heads(
            k,
            self.config.num_kv_heads,
            "k_split",
            "k_heads",
            "Split K Heads",
            "K Heads",
        )
        v_heads = executor.split_heads(
            v,
            self.config.num_kv_heads,
            "v_split",
            "v_heads",
            "Split V Heads",
            "V Heads",
        )
        q_attention, k_for_cache = self._apply_rope(
            executor,
            q_heads,
            k_heads,
            np.arange(len(tokens), dtype=np.int32),
        )
        k_cache = executor.cache_append(
            k_for_cache,
            "k_cache_append",
            "k_cache",
            "K Cache Append",
            "K Cache",
        )
        v_cache = executor.cache_append(
            v_heads,
            "v_cache_append",
            "v_cache",
            "V Cache Append",
            "V Cache",
        )
        attention_k, attention_v = self._expand_kv(
            executor,
            k_cache,
            v_cache,
        )
        self._attention(
            executor,
            q_attention,
            attention_k,
            attention_v,
            score_label="QK^T",
        )

        state = AttentionState(
            tokens=tuple(tokens),
            k_cache=executor.value(k_cache).copy(),
            v_cache=executor.value(v_cache).copy(),
        )
        return AttentionRun(
            tokens=list(tokens),
            executor=executor,
            config=self.config,
            state=state,
            phase="prefill",
        )

    def decode(self, state: AttentionState, new_token: str) -> AttentionRun:
        if not new_token.strip():
            raise ValueError("Decode token must not be empty.")
        self._validate_state(state)

        executor = PrimitiveExecutor()
        token_position = len(state.tokens)
        token_id = executor.input(
            np.array([[token_position]], dtype=np.int32),
            node_id="decode_input",
            output_id="new_token_id",
            label="Decode Token",
            output_name="New Token ID",
        )
        embedding = executor.embedding(
            token_id,
            self._embed_tokens([new_token])[None, :, :],
            node_id="decode_embedding",
            output_id="x_new",
            label="New Token Embedding",
            output_name="New Token Embedding",
        )
        weights = self._projection_weights()
        q_new = executor.linear(
            embedding,
            weights["q"],
            "q_new_proj",
            "q_new",
            "Q_new Projection",
            "Q New",
            parameter_name="Wq",
        )
        k_new = executor.linear(
            embedding,
            weights["k"],
            "k_new_proj",
            "k_new",
            "K_new Projection",
            "K New",
            parameter_name="Wk",
        )
        v_new = executor.linear(
            embedding,
            weights["v"],
            "v_new_proj",
            "v_new",
            "V_new Projection",
            "V New",
            parameter_name="Wv",
        )
        q_new_heads = executor.split_heads(
            q_new,
            self.config.num_q_heads,
            "q_new_split",
            "q_new_heads",
            "Split Q_new Heads",
            "Q New Heads",
        )
        k_new_heads = executor.split_heads(
            k_new,
            self.config.num_kv_heads,
            "k_new_split",
            "k_new_heads",
            "Split K_new Heads",
            "K New Heads",
        )
        v_new_heads = executor.split_heads(
            v_new,
            self.config.num_kv_heads,
            "v_new_split",
            "v_new_heads",
            "Split V_new Heads",
            "V New Heads",
        )
        q_attention, k_for_cache = self._apply_rope(
            executor,
            q_new_heads,
            k_new_heads,
            np.array([token_position], dtype=np.int32),
        )

        k_previous = executor.cache_read(
            state.k_cache,
            "k_cache_read",
            "k_cache_previous",
            "K Cache Read",
            "Previous K Cache",
        )
        v_previous = executor.cache_read(
            state.v_cache,
            "v_cache_read",
            "v_cache_previous",
            "V Cache Read",
            "Previous V Cache",
        )
        k_cache = executor.cache_append(
            k_for_cache,
            "k_cache_append",
            "k_cache",
            "K Cache Append",
            "K Cache",
            existing_id=k_previous,
        )
        v_cache = executor.cache_append(
            v_new_heads,
            "v_cache_append",
            "v_cache",
            "V Cache Append",
            "V Cache",
            existing_id=v_previous,
        )
        attention_k, attention_v = self._expand_kv(
            executor,
            k_cache,
            v_cache,
        )
        self._attention(
            executor,
            q_attention,
            attention_k,
            attention_v,
            score_label="Q_new K_cache^T",
        )

        tokens = [*state.tokens, new_token]
        next_state = AttentionState(
            tokens=tuple(tokens),
            k_cache=executor.value(k_cache).copy(),
            v_cache=executor.value(v_cache).copy(),
        )
        return AttentionRun(
            tokens=tokens,
            executor=executor,
            config=self.config,
            state=next_state,
            phase="decode",
            previous_tokens=len(state.tokens),
        )

    def _apply_rope(
        self,
        executor: PrimitiveExecutor,
        q_heads: str,
        k_heads: str,
        positions: NDArray,
    ) -> tuple[str, str]:
        if not self.config.use_rope:
            return q_heads, k_heads

        q_rotated = executor.rope(
            q_heads,
            positions,
            "q_rope",
            "q_rotated",
            "Apply RoPE to Q",
            "Rotated Q",
            self.config.rope_base,
        )
        k_rotated = executor.rope(
            k_heads,
            positions,
            "k_rope",
            "k_rotated",
            "Apply RoPE to K",
            "Rotated K",
            self.config.rope_base,
        )
        return q_rotated, k_rotated

    def _expand_kv(
        self,
        executor: PrimitiveExecutor,
        k_cache: str,
        v_cache: str,
    ) -> tuple[str, str]:
        repeats = self.config.num_q_heads // self.config.num_kv_heads
        if repeats == 1:
            return k_cache, v_cache

        k_attention = executor.repeat_kv(
            k_cache,
            repeats,
            "k_repeat",
            "k_attention",
            "Map K Heads to Q Heads",
            "Expanded K",
        )
        v_attention = executor.repeat_kv(
            v_cache,
            repeats,
            "v_repeat",
            "v_attention",
            "Map V Heads to Q Heads",
            "Expanded V",
        )
        return k_attention, v_attention

    def _attention(
        self,
        executor: PrimitiveExecutor,
        q_heads: str,
        k_cache: str,
        v_cache: str,
        score_label: str,
    ) -> None:
        k_transposed = executor.transpose(
            k_cache,
            (0, 1, 3, 2),
            "k_transpose",
            "k_transposed",
            "Transpose K Cache",
            "K Cache Transposed",
        )
        raw_scores = executor.matmul(
            q_heads,
            k_transposed,
            "qk_matmul",
            "raw_scores",
            score_label,
            "Raw Attention Scores",
            visualization="attention_matrix",
        )
        scaled_scores = executor.scale(
            raw_scores,
            math.sqrt(self.config.head_dim),
            "scale",
            "scaled_scores",
            "Scale",
            "Scaled Attention Scores",
        )
        masked_scores = executor.causal_mask(
            scaled_scores,
            "causal_mask",
            "masked_scores",
            "Causal Mask",
            "Masked Attention Scores",
        )
        probabilities = executor.softmax(
            masked_scores,
            "softmax",
            "attention_probs",
            "Softmax",
            "Attention Probabilities",
        )
        context = executor.matmul(
            probabilities,
            v_cache,
            "attention_value",
            "context_heads",
            "Attention x V Cache",
            "Context Heads",
        )
        merged = executor.merge_heads(
            context,
            "merge_heads",
            "merged_context",
            "Merge Heads",
            "Merged Context",
        )
        executor.output(merged)

    def _validate_state(self, state: AttentionState) -> None:
        expected_prefix = (1, self.config.num_kv_heads)
        expected_suffix = (self.config.head_dim,)
        for name, cache in (("K", state.k_cache), ("V", state.v_cache)):
            if cache.ndim != 4:
                raise ValueError(f"{name} cache must have four dimensions.")
            if cache.shape[:2] != expected_prefix or cache.shape[-1:] != expected_suffix:
                raise ValueError(f"{name} cache shape does not match MHA config.")
            if cache.shape[-2] != len(state.tokens):
                raise ValueError(f"{name} cache length does not match tokens.")

    def _embed_tokens(self, tokens: list[str]) -> NDArray:
        embeddings = [self._embedding_for_token(token) for token in tokens]
        return np.stack(embeddings).astype(self.config.dtype)

    def _embedding_for_token(self, token: str) -> NDArray:
        key = f"{self.config.seed}:{token}".encode("utf-8")
        digest = hashlib.sha256(key).digest()
        token_seed = int.from_bytes(digest[:8], byteorder="big", signed=False)
        rng = np.random.default_rng(token_seed)
        return rng.normal(0.0, 0.5, self.config.d_model).astype(
            self.config.dtype
        )

    def _projection_weights(self) -> dict[str, NDArray]:
        rng = np.random.default_rng(self.config.seed)
        scale = 1.0 / math.sqrt(self.config.d_model)
        kv_width = self.config.num_kv_heads * self.config.head_dim
        return {
            "q": rng.normal(
                0.0,
                scale,
                (self.config.d_model, self.config.d_model),
            ).astype(self.config.dtype),
            "k": rng.normal(
                0.0,
                scale,
                (self.config.d_model, kv_width),
            ).astype(self.config.dtype),
            "v": rng.normal(
                0.0,
                scale,
                (self.config.d_model, kv_width),
            ).astype(self.config.dtype),
        }
