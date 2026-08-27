export type TensorScalar = number | null;
export type TensorValues = TensorScalar | TensorValues[];

export interface TensorSpec {
  id: string;
  name: string;
  shape: number[];
  dtype: string;
}

export interface TensorData extends TensorSpec {
  values: TensorValues;
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
  };
  graph: AttentionGraph;
  trace: TraceEvent[];
  tensors: Record<string, TensorData>;
  memory: {
    cache_kind: "kv" | "latent";
    tokens: number;
    k_cache?: CacheSummary;
    v_cache?: CacheSummary;
    latent_cache?: CacheSummary;
    total_elements: number;
    total_bytes: number;
  };
  cache_activity: {
    phase: "prefill" | "decode";
    read_tokens: number;
    appended_tokens: number;
    resulting_tokens: number;
  };
  warnings: string[];
  decoded_token?: string;
}

export type AttentionArchitecture = "mha" | "mqa" | "gqa" | "rope" | "mla";
