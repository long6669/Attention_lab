# Attention implementations and fidelity

AttnLab is an executable teaching tool, not a training framework or a drop-in
implementation of production model kernels. Every architecture uses small,
deterministic NumPy tensors so its graph, values, and persistent memory can be
inspected.

## Fidelity labels

- **Core reference**: implements the defining inference equation, with toy
  dimensions and deterministic synthetic weights.
- **Concept model**: demonstrates the main state or routing idea while omitting
  learned gates, optimized kernels, model-specific projections, or other
  production details.

## Implementation matrix

| Architecture | Fidelity | What AttnLab executes | Important omissions |
| --- | --- | --- | --- |
| MHA | Core reference | Q/K/V projections, head split, scaled causal attention, softmax, value aggregation, and KV-cache decode | Training, dropout, output projection, model weights, fused kernels |
| MQA | Core reference | One shared KV head expanded only for attention | Trained checkpoint conversion and quality evaluation |
| GQA | Core reference | Fewer KV heads than query heads with deterministic group mapping | Trained checkpoint conversion and quality evaluation |
| RoPE | Core reference | Pairwise rotary transform of Q and K using absolute decode positions | Partial rotary dimensions and model-specific frequency scaling |
| MLA | Concept model | Joint low-rank KV latent, persistent latent cache, and K/V reconstruction | Decoupled RoPE, absorbed decode matrices, query compression, normalization, production dimensions |
| KDA | Concept model | Fixed recurrent matrix, decay, delta erase/write, sequential scan, and one-token state update | KDA's learned channel-wise decay, learned gates, chunkwise algorithm, normalization, convolution, hybrid layers |
| CSA | Concept model | Causal sequence summaries, query-based index scores, Top-K selection, and routed value aggregation | Learned compression weights, lightning indexer, local token window, exact sparse kernels, production cache layout |
| HCA | Concept model | Multi-resolution sequence summaries and routed aggregation | The paper's heavily compressed non-overlapping cache, dense attention over compressed entries, local window, model-specific projections |

The UI names concept models explicitly where confusion with a production
implementation would be likely. Contributions that improve fidelity must add a
numerical parity test and update this table.

## Core equations

### Scaled dot-product attention

```text
scores = (Q @ K^T) / sqrt(d_head)
P = softmax(causal_mask(scores))
O = P @ V
```

Decode computes only `Q_new`, `K_new`, and `V_new`, appends the new K/V values
to persistent state, and evaluates `Q_new` against the complete cache.

### Multi-query and grouped-query attention

```text
KV_for_query_head[h] = KV_cache[floor(h / group_size)]
group_size = num_query_heads / num_kv_heads
```

The expanded tensor is temporary. Persistent memory retains only the original
KV heads.

### Rotary position embedding

For each adjacent feature pair:

```text
y_even = x_even * cos(theta) - x_odd * sin(theta)
y_odd  = x_even * sin(theta) + x_odd * cos(theta)
theta  = position / base^(2 * pair_index / d_head)
```

### MLA concept model

```text
c_kv = X @ W_down
K = c_kv @ W_k_up
V = c_kv @ W_v_up
```

Only `c_kv` is persisted. This captures the low-rank cache idea but not the full
DeepSeek-V2 MLA formulation.

### KDA concept model

```text
S_decay = decay * S_previous
prediction = k @ S_decay
S_erase = S_decay - write_rate * outer(k, prediction)
S_write = S_erase + write_rate * outer(k, v)
output = q @ S_write
```

AttnLab uses scalar `decay` and `write_rate` values. The Kimi Linear KDA paper
uses learned, finer-grained gating and an efficient chunkwise formulation.

### CSA and HCA concept models

```text
compressed_KV = causal_window_mean(KV)
route_scores = Q @ compressed_K^T / sqrt(d_head)
route_ids = top_k(route_scores)
output = softmax(selected_scores) @ selected_values
```

CSA uses one compression scale. The current HCA teaching path uses two scales
to make resolution changes visible; it must not be interpreted as a faithful
DeepSeek-V4 HCA kernel.

## References

1. Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017.
2. Shazeer, [Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150), 2019.
3. Ainslie et al., [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245), 2023.
4. Su et al., [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864), 2021.
5. DeepSeek-AI, [DeepSeek-V2](https://arxiv.org/abs/2405.04434), 2024.
6. Kimi Team, [Kimi Linear](https://arxiv.org/abs/2510.26692), 2025.
7. DeepSeek-AI, [DeepSeek-V4](https://arxiv.org/abs/2606.19348), 2026.
