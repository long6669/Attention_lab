import type { Edge, Node } from "@xyflow/react";

import type { AttentionGraph, GraphNode, TensorSpec } from "../types/attention";

export type GraphNodeData = {
  graphNode: GraphNode;
  inputShape?: number[];
  outputShape?: number[];
  isCurrent: boolean;
  isSelected: boolean;
  isDimmed: boolean;
};

const SUPPORTED_OPS = new Set([
  "input",
  "embedding",
  "linear",
  "low_rank_compression",
  "split_heads",
  "repeat_kv",
  "rope",
  "transpose",
  "matmul",
  "scale",
  "causal_mask",
  "softmax",
  "decay",
  "erase",
  "write",
  "scan",
  "sequence_compression",
  "indexer",
  "topk",
  "routing",
  "merge_heads",
  "output",
]);

function firstTensor(
  ids: string[],
  tensors: Record<string, TensorSpec>,
): TensorSpec | undefined {
  const id = ids[0];
  return id ? tensors[id] : undefined;
}

export function graphToFlow(
  graph: AttentionGraph,
  currentNodeId: string | undefined,
  selectedNodeId: string | undefined,
): { nodes: Node<GraphNodeData>[]; edges: Edge[] } {
  const nodes: Node<GraphNodeData>[] = graph.nodes.map((graphNode) => {
    const isCurrent = graphNode.id === currentNodeId;
    const isSelected = graphNode.id === selectedNodeId;
    return {
      id: graphNode.id,
      type:
        graphNode.op === "cache_append" ||
        graphNode.op === "cache_read" ||
        graphNode.op === "state_init" ||
        graphNode.op === "state_read" ||
        graphNode.op === "state_update"
          ? "cache"
          : SUPPORTED_OPS.has(graphNode.op)
            ? "operator"
            : "generic",
      position: { x: 0, y: 0 },
      data: {
        graphNode,
        inputShape: firstTensor(graphNode.inputs, graph.tensors)?.shape,
        outputShape: firstTensor(graphNode.outputs, graph.tensors)?.shape,
        isCurrent,
        isSelected,
        isDimmed: Boolean(currentNodeId && !isCurrent),
      },
    };
  });

  const edges: Edge[] = graph.edges.map((edge, index) => {
    const isActive =
      edge.source === currentNodeId || edge.target === currentNodeId;
    return {
      id: `${edge.source}-${edge.target}-${edge.tensor_id}-${index}`,
      source: edge.source,
      target: edge.target,
      type: "smoothstep",
      animated: isActive,
      style: {
        stroke: isActive ? "#087f8c" : "#b8c0c7",
        strokeWidth: isActive ? 2.5 : 1.25,
        opacity: currentNodeId && !isActive ? 0.32 : 1,
      },
    };
  });

  return { nodes, edges };
}
