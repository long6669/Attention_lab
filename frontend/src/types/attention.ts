export type TensorScalar = number | null;
export type TensorValues = TensorScalar | TensorValues[];

export interface TensorSpec {
  id: string;
  name: string;
  shape: number[];
  dtype: string;
}

export interface TensorData extends TensorSpec {
  numel: number;
  bytes: number;
  values_loaded: boolean;
  values?: TensorValues;
}

export interface GraphNode {
  id: string;
  op: string;
  label: string;
  inputs: string[];
  outputs: string[];
  attrs: Record<string, unknown>;
}

export interface GraphEdge {
  source: string;
  target: string;
  tensor_id: string;
}

export interface AttentionGraph {
  nodes: GraphNode[];
  edges: GraphEdge[];
  tensors: Record<string, TensorSpec>;
}

export interface TraceEvent {
  step: number;
  node_id: string;
  op: string;
  inputs: string[];
  outputs: string[];
  title: string;
}

export interface CacheSummary {
  shape: number[];
  dtype: string;
  elements: number;
  bytes: number;
}

export interface MemoryBlock {
  start: number;
  end: number;
  head_start: number | null;
  head_end: number | null;
  numel: number;
  min: number | null;
  max: number | null;
  mean_abs: number | null;
  l2: number | null;
}

export interface MemoryTensorSpec {
  id: string;
  name: string;
  kind: string;
  role: string;
  shape: number[];
  dtype: string;
  numel: number;
  bytes: number;
  axes: string[];
  growth_axis: number | null;
  growth_axis_name: string | null;
  values_loaded: boolean;
  values?: TensorValues;
  blocks: MemoryBlock[];
}

export interface MemorySpec {
  kind: string;
  tensors: MemoryTensorSpec[];
  total_numel: number;
  total_bytes: number;
  growth_axes: string[];
}

export interface MemorySlice {
  id: string;
  shape: number[];
  dtype: string;
  numel: number;
  bytes: number;
  axes: string[];
  selection: {
    start: number;
    end: number;
    head: number | null;
  };
  values: TensorValues;
}

export interface AttentionRun {
  session_id: string;
  phase: "prefill" | "decode";
  tokens: string[];
  config: {
    architecture: AttentionArchitecture;
    batch_size: number;
    seq_len: number;
    d_model: number;
    num_heads: number;
    num_q_heads: number;
    num_kv_heads: number;
    head_dim: number;
    dtype: string;
    seed: number;
    use_rope: boolean;
    rope_base: number;
    kv_lora_rank: number;
    state_decay: number;
    state_write_rate: number;
    compression_window: number;
    routing_top_k: number;
  };
  graph: AttentionGraph;
  trace: TraceEvent[];
  tensors: Record<string, TensorData>;
  memory: {
    cache_kind: "kv" | "latent" | "recurrent";
    tokens: number;
    k_cache?: CacheSummary;
    v_cache?: CacheSummary;
    latent_cache?: CacheSummary;
    recurrent_state?: CacheSummary;
    total_elements: number;
    total_bytes: number;
    spec: MemorySpec;
  };
  cache_activity: {
    phase: "prefill" | "decode";
    read_tokens: number;
    appended_tokens: number;
    resulting_tokens: number;
    update_kind?: "append" | "state_update";
  };
  warnings: string[];
  decoded_token?: string;
}

export type AttentionArchitecture =
  | "mha"
  | "mqa"
  | "gqa"
  | "rope"
  | "mla"
  | "kda"
  | "csa"
  | "hca";
