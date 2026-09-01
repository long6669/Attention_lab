import numpy as np

from app.architectures.mha import (
    AttentionRun,
    AttentionState,
    MHAConfig,
    MultiHeadAttention,
)
from app.ops import PrimitiveExecutor


class CompressedSparseAttention(MultiHeadAttention):
    """Sparse attention routed over causal compressed sequence summaries."""

    def __init__(
        self,
        config: MHAConfig = MHAConfig(architecture="csa"),
    ) -> None:
        if config.architecture not in {"csa", "hca"}:
            raise ValueError("CompressedSparseAttention requires CSA or HCA.")
        super().__init__(config)

    @property
    def window_sizes(self) -> tuple[int, ...]:
        if self.config.architecture == "hca":
            return (
                self.config.compression_window,
                self.config.compression_window * 2,
            )
        return (self.config.compression_window,)

    def prefill(self, tokens: list[str]) -> AttentionRun:
        if not tokens:
            raise ValueError("At least one token is required.")
        executor = PrimitiveExecutor()
        q_heads, k_heads, v_heads = self._project(
            executor,
            tokens,
            decode=False,
        )
        k_cache = executor.cache_append(
            k_heads,
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
        self._routed_attention(
            executor,
            q_heads,
            k_cache,
            v_cache,
            query_offset=0,
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

    def run(self, tokens: list[str]) -> AttentionRun:
        return self.prefill(tokens)

    def decode(self, state: AttentionState, new_token: str) -> AttentionRun:
        if not new_token.strip():
            raise ValueError("Decode token must not be empty.")
        self._validate_state(state)
        executor = PrimitiveExecutor()
        q_heads, k_heads, v_heads = self._project(
            executor,
            [new_token],
            decode=True,
            token_position=len(state.tokens),
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
            k_heads,
            "k_cache_append",
            "k_cache",
            "K Cache Append",
            "K Cache",
            existing_id=k_previous,
        )
        v_cache = executor.cache_append(
            v_heads,
            "v_cache_append",
            "v_cache",
            "V Cache Append",
            "V Cache",
            existing_id=v_previous,
        )
        self._routed_attention(
            executor,
            q_heads,
            k_cache,
            v_cache,
            query_offset=len(state.tokens),
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
        outputs = []
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
            outputs.append(
                executor.split_heads(
                    projection,
                    self.config.num_q_heads,
                    f"{prefix}{name}_split",
                    f"{prefix}{name}_heads",
                    f"Split {display_name} Heads",
                    f"{display_name} Heads",
                )
            )
        return outputs[0], outputs[1], outputs[2]

    def _routed_attention(
        self,
        executor: PrimitiveExecutor,
        q_heads: str,
        k_cache: str,
        v_cache: str,
        query_offset: int,
    ) -> None:
        compressed_k, spans = executor.sequence_compress(
            k_cache,
            self.window_sizes,
            "sequence_compress_k",
            "compressed_k",
            "Compress K Sequence",
            "Compressed K",
        )
        compressed_v, _ = executor.sequence_compress(
            v_cache,
            self.window_sizes,
            "sequence_compress_v",
            "compressed_v",
            "Compress V Sequence",
            "Compressed V",
        )
        scores = executor.indexer(
            q_heads,
            compressed_k,
            spans,
            query_offset,
        )
        indices = executor.topk(
            scores,
            self.config.routing_top_k,
        )
        selected_scores = executor.route_scores(scores, indices)
        probabilities = executor.softmax(
            selected_scores,
            "routing_softmax",
            "routing_probs",
            "Routing Softmax",
            "Routing Probabilities",
        )
        routed_values = executor.route_values(
            compressed_v,
            indices,
            "route_values",
            "routed_values",
            "Route Compressed Values",
            "Routed Values",
        )
        context = executor.weighted_route(probabilities, routed_values)
        merged = executor.merge_heads(
            context,
            "merge_heads",
            "merged_context",
            "Merge Heads",
            "Merged Context",
        )
        executor.output(merged)
