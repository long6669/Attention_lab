import { useEffect, useMemo, useState } from "react";
import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  PanOnScrollMode,
  ReactFlow,
  type Edge,
  type Node,
  type ReactFlowInstance,
} from "@xyflow/react";

import { graphToFlow, type GraphNodeData } from "../../graph/adapter";
import { layoutGraph } from "../../graph/layout";
import type { AttentionGraph } from "../../types/attention";
import { CacheNode, GenericNode, OperatorNode } from "./GraphNodes";

const nodeTypes = {
  operator: OperatorNode,
  cache: CacheNode,
  generic: GenericNode,
};

interface GraphViewProps {
  graph: AttentionGraph;
  currentNodeId?: string;
  selectedNodeId?: string;
  onSelectNode: (nodeId: string) => void;
}

export function GraphView({
  graph,
  currentNodeId,
  selectedNodeId,
  onSelectNode,
}: GraphViewProps) {
  const adapted = useMemo(
    () => graphToFlow(graph, currentNodeId, selectedNodeId),
    [graph, currentNodeId, selectedNodeId],
  );
  const [nodes, setNodes] = useState<Node<GraphNodeData>[]>([]);
  const [edges, setEdges] = useState<Edge[]>([]);
  const [instance, setInstance] =
    useState<ReactFlowInstance<Node<GraphNodeData>, Edge>>();

  useEffect(() => {
    let active = true;
    layoutGraph(adapted.nodes, adapted.edges).then((layout) => {
      if (!active) {
        return;
      }
      setNodes(layout.nodes);
      setEdges(layout.edges);
      requestAnimationFrame(() => {
        const focusNode =
          layout.nodes.find((node) => node.id === currentNodeId) ??
          layout.nodes[0];
        if (focusNode) {
          instance?.setCenter(
            focusNode.position.x + 92,
            focusNode.position.y + 39,
            { zoom: 0.82, duration: 250 },
          );
        }
      });
    });
    return () => {
      active = false;
    };
  }, [adapted, instance]);

  return (
    <div className="graph-canvas" aria-label="Attention computation graph">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        onInit={setInstance}
        onNodeClick={(_, node) => onSelectNode(node.id)}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable
        panOnDrag={false}
        panOnScroll
        panOnScrollMode={PanOnScrollMode.Free}
        zoomOnScroll={false}
        zoomOnPinch
        zoomOnDoubleClick={false}
        minZoom={0.2}
        maxZoom={1.8}
        defaultViewport={{ x: 0, y: 0, zoom: 0.82 }}
        proOptions={{ hideAttribution: true }}
      >
        <Background
          variant={BackgroundVariant.Dots}
          gap={18}
          size={1}
          color="#d9dee2"
        />
        <MiniMap
          pannable
          zoomable
          nodeColor={(node) =>
            node.id === currentNodeId ? "#087f8c" : "#c8d0d5"
          }
          maskColor="rgba(248, 249, 250, 0.72)"
        />
        <Controls showInteractive={false} />
      </ReactFlow>
    </div>
  );
}
