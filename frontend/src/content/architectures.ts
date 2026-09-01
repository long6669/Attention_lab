import type { AttentionArchitecture } from "../types/attention";

export interface ArchitectureLesson {
  shortName: string;
  concept: string;
  formula: string;
  execution: string;
  memory: string;
}

export const ARCHITECTURE_LABELS: Record<AttentionArchitecture, string> = {
  mha: "Multi-Head Attention",
  mqa: "Multi-Query Attention",
  gqa: "Grouped-Query Attention",
  rope: "MHA with RoPE",
  mla: "MLA Concept Model",
  kda: "KDA Concept Model",
  csa: "CSA Concept Model",
  hca: "HCA Concept Model",
};

export const ARCHITECTURE_LESSONS: Record<
  AttentionArchitecture,
  ArchitectureLesson
> = {
  mha: {
    shortName: "MHA",
    concept: "Each query head owns a matching key and value head.",
    formula: "softmax(QK^T / sqrt(d))V",
    execution:
      "Project Q, K, and V, split every projection into heads, then run dense causal attention.",
    memory:
      "Stores one K and V vector per token for every query head, so cache size grows linearly.",
  },
  mqa: {
    shortName: "MQA",
    concept:
      "All query heads read one shared key head and one shared value head.",
    formula: "softmax(Q_h K_shared^T / sqrt(d))V_shared",
    execution:
      "K and V are projected once, persisted compactly, and repeated only for the attention operation.",
    memory:
      "Cache growth is linear in sequence length but uses fewer KV heads than MHA.",
  },
  gqa: {
    shortName: "GQA",
    concept:
      "Groups of query heads share a smaller set of key and value heads.",
    formula: "KV(h) = KV[floor(h / group_size)]",
    execution:
      "The runtime keeps grouped KV tensors and expands each group only before score computation.",
    memory:
      "Cache size sits between MHA and MQA while retaining multiple KV groups.",
  },
  rope: {
    shortName: "RoPE",
    concept: "Position is encoded by rotating adjacent Q and K feature pairs.",
    formula: "[x_0, x_1] -> [x_0 cos θ - x_1 sin θ, x_0 sin θ + x_1 cos θ]",
    execution:
      "Apply pairwise rotations at absolute token positions before K enters the persistent cache.",
    memory:
      "Uses the same KV capacity as MHA; cached keys already contain their positional rotation.",
  },
  mla: {
    shortName: "MLA",
    concept:
      "Persist a low-rank latent and reconstruct keys and values when attention runs.",
    formula: "c_KV = XW_down; K = c_KV W_K; V = c_KV W_V",
    execution:
      "Compress each token once, append the latent, then reconstruct complete K and V views.",
    memory:
      "Stores one latent vector per token instead of separate per-head K and V tensors.",
  },
  kda: {
    shortName: "KDA",
    concept:
      "A recurrent matrix is updated by decay, erase, and write operations.",
    formula: "S_t = decay(S_{t-1}) - erase(k_t) + write(k_t, v_t)",
    execution:
      "Scan tokens sequentially during prefill; decode reads and updates only the final recurrent state.",
    memory:
      "State shape is fixed, so persistent memory remains constant as sequence length grows.",
  },
  csa: {
    shortName: "CSA",
    concept:
      "Queries route to a small Top-K set of causal compressed sequence summaries.",
    formula: "softmax(topk(QC_K^T)) C_V",
    execution:
      "Compress causal windows, score summaries, select Top-K routes, and aggregate routed values.",
    memory:
      "The teaching path retains KV cache while reducing the set used by routed attention.",
  },
  hca: {
    shortName: "HCA",
    concept:
      "Multiple compression resolutions expose both local and broader sequence summaries.",
    formula: "C = concat(C_local, C_coarse); softmax(topk(QC_K^T)) C_V",
    execution:
      "Build two causal compression levels and route each query over the combined hierarchy.",
    memory:
      "The concept model retains KV cache; its routing graph is larger than the single-scale CSA path.",
  },
};

export interface ExampleDefinition {
  id: string;
  title: string;
  question: string;
  text: string;
  mode: "workbench" | "compare";
  architecture: AttentionArchitecture;
  compareArchitectures: AttentionArchitecture[];
}

export const EXAMPLES: ExampleDefinition[] = [
  {
    id: "mqa-cache",
    title: "Why is MQA cache smaller?",
    question: "Compare identical dense attention with two KV storage layouts.",
    text: "the model reads a growing context efficiently",
    mode: "compare",
    architecture: "mqa",
    compareArchitectures: ["mha", "mqa", "gqa"],
  },
  {
    id: "mla-latent",
    title: "How does MLA compress KV?",
    question: "Follow low-rank storage and K/V reconstruction.",
    text: "latent vectors preserve compact attention context",
    mode: "compare",
    architecture: "mla",
    compareArchitectures: ["mha", "mla"],
  },
  {
    id: "kda-state",
    title: "Why does KDA memory stay flat?",
    question: "Contrast a growing KV cache with a recurrent matrix.",
    text: "each token updates a persistent recurrent memory",
    mode: "compare",
    architecture: "kda",
    compareArchitectures: ["mha", "kda"],
  },
  {
    id: "csa-routing",
    title: "How does CSA choose routes?",
    question: "Inspect compression, index scores, Top-K, and routing.",
    text: "compressed summaries route attention across the sequence",
    mode: "workbench",
    architecture: "csa",
    compareArchitectures: ["csa", "hca"],
  },
];
