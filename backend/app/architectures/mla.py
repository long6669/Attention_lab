import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.architectures.mha import MHAConfig, MultiHeadAttention
from app.ops import PrimitiveExecutor


@dataclass(frozen=True)
class LatentAttentionState:
    """Persistent compressed KV state used by the toy MLA path."""

    tokens: tuple[str, ...]
    latent_cache: NDArray


@dataclass(frozen=True)
class LatentAttentionRun:
    tokens: list[str]
    executor: PrimitiveExecutor
    config: MHAConfig
    state: LatentAttentionState
    phase: str
    previous_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        latent_cache = self.state.latent_cache
        appended_tokens = len(self.tokens) - self.previous_tokens
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
                "cache_kind": "latent",
                "tokens": len(self.tokens),
                "latent_cache": {
                    "shape": list(latent_cache.shape),
                    "dtype": str(latent_cache.dtype),
                    "elements": int(latent_cache.size),
                    "bytes": int(latent_cache.nbytes),
                },
                "total_elements": int(latent_cache.size),
                "total_bytes": int(latent_cache.nbytes),
            },
            "cache_activity": {
                "phase": self.phase,
                "read_tokens": self.previous_tokens if self.phase == "decode" else 0,
                "appended_tokens": appended_tokens,
                "resulting_tokens": len(self.tokens),
            },
        }


class MultiHeadLatentAttention(MultiHeadAttention):
    """Toy MLA that persists a low-rank latent cache instead of K/V."""

    def __init__(
        self,
        config: MHAConfig = MHAConfig(
            architecture="mla",
            num_q_heads=2,
            num_kv_heads=2,
            kv_lora_rank=4,
        ),
    ) -> None:
        super().__init__(config)

    def prefill(self, tokens: list[str]) -> LatentAttentionRun:
        if not tokens:
            raise ValueError("At least one token is required.")

        executor = PrimitiveExecutor()
        token_ids = executor.input(np.arange(len(tokens))[None, :])
        embeddings = executor.embedding(
            token_ids,
            self._embed_tokens(tokens)[None, :, :],
        )
        q_weight = self._projection_weights()["q"]
        q = executor.linear(
            embeddings,
            q_weight,
            "q_proj",
            "q",
            "Q Projection",
            "Q",
            parameter_name="Wq",
        )
        q_heads = executor.split_heads(
            q,
            self.config.num_q_heads,
            "q_split",
            "q_heads",
            "Split Q Heads",
            "Q Heads",
        )
        weights = self._latent_weights()
        latent = executor.linear(
            embeddings,
            weights["down"],
            "kv_compress",
            "kv_latent",
            "Low-Rank KV Compression",
            "KV Latent",
            parameter_name="W_DKV",
            op="low_rank_compression",
        )
        latent_cache = executor.cache_append(
            latent,
            "latent_cache_append",
            "latent_cache",
            "Latent Cache Append",
            "Latent Cache",
        )
        k_heads, v_heads = self._decompress(executor, latent_cache, weights)
        self._attention(
            executor,
            q_heads,
            k_heads,
            v_heads,
            score_label="QK_latent^T",
        )

        state = LatentAttentionState(
            tokens=tuple(tokens),
            latent_cache=executor.value(latent_cache).copy(),
        )
        return LatentAttentionRun(
            tokens=list(tokens),
            executor=executor,
            config=self.config,
            state=state,
            phase="prefill",
        )

    def run(self, tokens: list[str]) -> LatentAttentionRun:
        return self.prefill(tokens)

    def decode(
        self,
        state: LatentAttentionState,
        new_token: str,
    ) -> LatentAttentionRun:
        if not new_token.strip():
            raise ValueError("Decode token must not be empty.")
        self._validate_latent_state(state)

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
        q = executor.linear(
            embedding,
            self._projection_weights()["q"],
            "q_new_proj",
            "q_new",
            "Q_new Projection",
            "Q New",
            parameter_name="Wq",
        )
        q_heads = executor.split_heads(
            q,
            self.config.num_q_heads,
            "q_new_split",
            "q_new_heads",
            "Split Q_new Heads",
            "Q New Heads",
        )
        weights = self._latent_weights()
        latent_new = executor.linear(
            embedding,
            weights["down"],
            "kv_new_compress",
            "kv_latent_new",
            "Compress New KV",
            "New KV Latent",
            parameter_name="W_DKV",
            op="low_rank_compression",
        )
        latent_previous = executor.cache_read(
            state.latent_cache,
            "latent_cache_read",
            "latent_cache_previous",
            "Latent Cache Read",
            "Previous Latent Cache",
        )
        latent_cache = executor.cache_append(
            latent_new,
            "latent_cache_append",
            "latent_cache",
            "Latent Cache Append",
            "Latent Cache",
            existing_id=latent_previous,
        )
        k_heads, v_heads = self._decompress(executor, latent_cache, weights)
        self._attention(
            executor,
            q_heads,
            k_heads,
            v_heads,
            score_label="Q_new K_latent^T",
        )

        tokens = [*state.tokens, new_token]
        next_state = LatentAttentionState(
            tokens=tuple(tokens),
            latent_cache=executor.value(latent_cache).copy(),
        )
        return LatentAttentionRun(
            tokens=tokens,
            executor=executor,
            config=self.config,
            state=next_state,
            phase="decode",
            previous_tokens=len(state.tokens),
        )

    def _decompress(
        self,
        executor: PrimitiveExecutor,
        latent_cache: str,
        weights: dict[str, NDArray],
    ) -> tuple[str, str]:
        k = executor.linear(
            latent_cache,
            weights["k_up"],
            "k_up_proj",
            "k_reconstructed",
            "Reconstruct K",
            "Reconstructed K",
            parameter_name="W_UK",
        )
        v = executor.linear(
            latent_cache,
            weights["v_up"],
            "v_up_proj",
            "v_reconstructed",
            "Reconstruct V",
            "Reconstructed V",
            parameter_name="W_UV",
        )
        k_heads = executor.split_heads(
            k,
            self.config.num_q_heads,
            "k_split",
            "k_heads",
            "Split Reconstructed K",
            "K Heads",
        )
        v_heads = executor.split_heads(
            v,
            self.config.num_q_heads,
            "v_split",
            "v_heads",
            "Split Reconstructed V",
            "V Heads",
        )
        return k_heads, v_heads

    def _latent_weights(self) -> dict[str, NDArray]:
        rng = np.random.default_rng(self.config.seed + 1000)
        down_scale = 1.0 / math.sqrt(self.config.d_model)
        up_scale = 1.0 / math.sqrt(self.config.kv_lora_rank)
        return {
            "down": rng.normal(
                0.0,
                down_scale,
                (self.config.d_model, self.config.kv_lora_rank),
            ).astype(self.config.dtype),
            "k_up": rng.normal(
                0.0,
                up_scale,
                (self.config.kv_lora_rank, self.config.d_model),
            ).astype(self.config.dtype),
            "v_up": rng.normal(
                0.0,
                up_scale,
                (self.config.kv_lora_rank, self.config.d_model),
            ).astype(self.config.dtype),
        }

    def _validate_latent_state(self, state: LatentAttentionState) -> None:
        expected_shape = (
            1,
            len(state.tokens),
            self.config.kv_lora_rank,
        )
        if state.latent_cache.shape != expected_shape:
            raise ValueError("Latent cache shape does not match MLA config.")
