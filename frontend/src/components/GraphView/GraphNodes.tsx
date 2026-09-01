import { Handle, Position, type Node, type NodeProps } from "@xyflow/react";
import { Box, Database, HelpCircle } from "lucide-react";

import type { GraphNodeData } from "../../graph/adapter";

type AttnFlowNode = Node<GraphNodeData>;

function shapeLabel(shape?: number[]): string {
  return shape ? `[${shape.join(", ")}]` : "none";
}

function NodeShell({
  data,
  kind,
}: {
  data: GraphNodeData;
  kind: "operator" | "cache" | "generic";
}) {
  const { graphNode } = data;
  const classes = [
    "graph-node",
    `graph-node--${kind}`,
    data.isCurrent ? "is-current" : "",
    data.isSelected ? "is-selected" : "",
    data.isDimmed ? "is-dimmed" : "",
  ]
    .filter(Boolean)
    .join(" ");

  const Icon =
    kind === "cache" ? Database : kind === "generic" ? HelpCircle : Box;
  const cacheTokens = graphNode.attrs.cached_tokens;
  const previousTokens = graphNode.attrs.previous_tokens;
  const appendedTokens = graphNode.attrs.appended_tokens;
  const cacheDescription =
    graphNode.op === "cache_read" && typeof cacheTokens === "number"
      ? `Read ${cacheTokens} tokens`
      : typeof previousTokens === "number" &&
          typeof appendedTokens === "number" &&
          typeof cacheTokens === "number"
        ? `${previousTokens} + ${appendedTokens} -> ${cacheTokens} tokens`
        : typeof cacheTokens === "number"
          ? `${cacheTokens} cached tokens`
          : undefined;

  return (
    <div className={classes}>
      <Handle type="target" position={Position.Top} />
      <div className="graph-node__header">
        <Icon size={14} aria-hidden="true" />
        <span>{graphNode.label}</span>
      </div>
      {kind === "cache" && cacheDescription ? (
        <div className="graph-node__cache">{cacheDescription}</div>
      ) : (
        <div className="graph-node__shapes">
          <span>{shapeLabel(data.inputShape)}</span>
          <span aria-hidden="true">→</span>
          <span>{shapeLabel(data.outputShape)}</span>
        </div>
      )}
      <div className="graph-node__op">{graphNode.op}</div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  );
}

export function OperatorNode({ data }: NodeProps<AttnFlowNode>) {
  return <NodeShell data={data} kind="operator" />;
}

export function CacheNode({ data }: NodeProps<AttnFlowNode>) {
  return <NodeShell data={data} kind="cache" />;
}

export function GenericNode({ data }: NodeProps<AttnFlowNode>) {
  return <NodeShell data={data} kind="generic" />;
}
