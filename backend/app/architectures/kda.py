from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.architectures.mha import MHAConfig, MultiHeadAttention
from app.memory import MemorySpec, MemoryTensorSpec
from app.ops import PrimitiveExecutor


@dataclass(frozen=True)
class KDAState:
    tokens: tuple[str, ...]
    recurrent_state: NDArray


@dataclass(frozen=True)
class KDARun:
    tokens: list[str]
    executor: PrimitiveExecutor
    config: MHAConfig
    state: KDAState
    phase: str
    previous_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        state = self.state.recurrent_state
        memory_spec = MemorySpec(
            kind="recurrent_state",
            tensors=(
                MemoryTensorSpec(
                    id="recurrent_state",
                    name="KDA Recurrent State",
                    kind="recurrent_state",
                    role="fast_weight",
                    value=state,
                    axes=("batch", "head", "key_feature", "value_feature"),
                    growth_axis=None,
                ),
            ),
        ).to_dict()
        processed = len(self.tokens) - self.previous_tokens
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
                "cache_kind": "recurrent",
                "tokens": len(self.tokens),
                "recurrent_state": {
                    "shape": list(state.shape),
                    "dtype": str(state.dtype),
                    "elements": int(state.size),
                    "bytes": int(state.nbytes),
                },
                "total_elements": int(state.size),
                "total_bytes": int(state.nbytes),
                "spec": memory_spec,
            },
            "cache_activity": {
                "phase": self.phase,
                "read_tokens": self.previous_tokens if self.phase == "decode" else 0,
                "appended_tokens": processed,
                "resulting_tokens": len(self.tokens),
                "update_kind": "state_update",
            },
        }


class KimiDeltaAttention(MultiHeadAttention):
    """Toy gated delta-rule attention with a fixed-size recurrent state."""

    def __init__(
        self,
        config: MHAConfig = MHAConfig(
            architecture="kda",
            num_q_heads=2,
            num_kv_heads=2,
        ),
    ) -> None:
        super().__init__(config)

    def prefill(self, tokens: list[str]) -> KDARun:
        if not tokens:
            raise ValueError("At least one token is required.")
        executor = PrimitiveExecutor()
        q_heads, k_heads, v_heads = self._project(
            executor,
            tokens,
            decode=False,
        )
        state_id = executor.state_init(
            (
                1,
                self.config.num_q_heads,
                self.config.head_dim,
                self.config.head_dim,
            )
        )
        context, final_state = executor.kda_scan(
            q_heads,
            k_heads,
            v_heads,
            state_id,
            self.config.state_decay,
            self.config.state_write_rate,
        )
        self._finish(executor, context)
        state = KDAState(
            tokens=tuple(tokens),
            recurrent_state=executor.value(final_state).copy(),
        )
        return KDARun(
            tokens=list(tokens),
            executor=executor,
            config=self.config,
            state=state,
            phase="prefill",
        )

    def run(self, tokens: list[str]) -> KDARun:
        return self.prefill(tokens)

    def decode(self, state: KDAState, new_token: str) -> KDARun:
        if not new_token.strip():
            raise ValueError("Decode token must not be empty.")
        self._validate_kda_state(state)
        executor = PrimitiveExecutor()
        q_heads, k_heads, v_heads = self._project(
            executor,
            [new_token],
            decode=True,
            token_position=len(state.tokens),
        )
        state_id = executor.state_read(state.recurrent_state)
        context, final_state = executor.kda_scan(
            q_heads,
            k_heads,
            v_heads,
            state_id,
            self.config.state_decay,
            self.config.state_write_rate,
        )
        self._finish(executor, context)
        tokens = [*state.tokens, new_token]
        next_state = KDAState(
            tokens=tuple(tokens),
            recurrent_state=executor.value(final_state).copy(),
        )
        return KDARun(
            tokens=tokens,
            executor=executor,
            config=self.config,
            state=next_state,
            phase="decode",
            previous_tokens=len(state.tokens),
        )

    def _project(
        self,
        executor: PrimitiveExecutor,
        tokens: list[str],
        decode: bool,
        token_position: int = 0,
    ) -> tuple[str, str, str]:
        prefix = "new_" if decode else ""
        token_ids = executor.input(
            np.arange(token_position, token_position + len(tokens))[None, :],
            node_id=f"{prefix}input",
            output_id=f"{prefix}token_ids",
            label="Decode Token" if decode else "Input Tokens",
            output_name="New Token ID" if decode else "Token IDs",
        )
        embeddings = executor.embedding(
            token_ids,
            self._embed_tokens(tokens)[None, :, :],
            node_id=f"{prefix}embedding",
            output_id=f"{prefix}x",
            label="New Token Embedding" if decode else "Toy Embedding",
            output_name="New Token Embedding" if decode else "Embedding",
        )
        weights = self._projection_weights()
        projected = []
        for name, display_name in (("q", "Q"), ("k", "K"), ("v", "V")):
            projection = executor.linear(
                embeddings,
                weights[name],
                f"{prefix}{name}_proj",
                f"{prefix}{name}",
                f"{display_name}{'_new' if decode else ''} Projection",
                f"{display_name}{' New' if decode else ''}",
                parameter_name=f"W{name}",
            )
            projected.append(
                executor.split_heads(
                    projection,
                    self.config.num_q_heads,
                    f"{prefix}{name}_split",
                    f"{prefix}{name}_heads",
                    f"Split {display_name} Heads",
                    f"{display_name} Heads",
                )
            )
        return projected[0], projected[1], projected[2]

    def _finish(self, executor: PrimitiveExecutor, context: str) -> None:
        merged = executor.merge_heads(
            context,
            "merge_heads",
            "merged_context",
            "Merge Heads",
            "Merged Context",
        )
        executor.output(merged)

    def _validate_kda_state(self, state: KDAState) -> None:
        expected = (
            1,
            self.config.num_q_heads,
            self.config.head_dim,
            self.config.head_dim,
        )
        if state.recurrent_state.shape != expected:
            raise ValueError("KDA recurrent state shape does not match config.")
