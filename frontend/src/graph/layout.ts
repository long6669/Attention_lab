import type { Edge, Node } from "@xyflow/react";

import type { GraphNodeData } from "./adapter";

const elkPromise = import("elkjs/lib/elk.bundled.js").then(
  ({ default: ELK }) => new ELK(),
);
const NODE_WIDTH = 184;
const NODE_HEIGHT = 78;

export async function layoutGraph(
  nodes: Node<GraphNodeData>[],
  edges: Edge[],
): Promise<{ nodes: Node<GraphNodeData>[]; edges: Edge[] }> {
  const elk = await elkPromise;
  const layout = await elk.layout({
    id: "attn-graph",
    layoutOptions: {
      "elk.algorithm": "layered",
      "elk.direction": "DOWN",
      "elk.spacing.nodeNode": "34",
      "elk.layered.spacing.nodeNodeBetweenLayers": "54",
      "elk.layered.nodePlacement.strategy": "NETWORK_SIMPLEX",
    },
    children: nodes.map((node) => ({
      id: node.id,
      width: NODE_WIDTH,
      height: NODE_HEIGHT,
    })),
    edges: edges.map((edge) => ({
      id: edge.id,
      sources: [edge.source],
      targets: [edge.target],
    })),
  });

  const positions = new Map(
    layout.children?.map((node) => [
      node.id,
      { x: node.x ?? 0, y: node.y ?? 0 },
    ]),
  );

  return {
    nodes: nodes.map((node) => ({
      ...node,
      position: positions.get(node.id) ?? node.position,
      width: NODE_WIDTH,
      height: NODE_HEIGHT,
    })),
    edges,
  };
}
