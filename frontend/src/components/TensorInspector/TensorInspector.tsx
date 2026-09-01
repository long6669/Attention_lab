import { useEffect, useMemo, useState } from "react";
import { Braces, MoveRight } from "lucide-react";

import type { GraphNode, TensorData } from "../../types/attention";

interface TensorInspectorProps {
  node?: GraphNode;
  tensors: Record<string, TensorData>;
}

export function TensorInspector({ node, tensors }: TensorInspectorProps) {
  const availableIds = useMemo(
    () => (node ? [...node.inputs, ...node.outputs] : []),
    [node],
  );
  const preferredId = node?.outputs[0] ?? node?.inputs[0];
  const [activeId, setActiveId] = useState<string | undefined>(preferredId);

  useEffect(() => {
    setActiveId(preferredId);
  }, [preferredId]);

  const tensor = activeId ? tensors[activeId] : undefined;

  if (!node || !tensor) {
    return (
      <section className="detail-panel tensor-inspector">
        <div className="panel-heading">
          <div>
            <span className="eyebrow">INSPECTOR</span>
            <h2>Tensor details</h2>
          </div>
          <Braces size={18} aria-hidden="true" />
        </div>
        <div className="empty-state">
          Select a graph node to inspect its tensors.
        </div>
      </section>
    );
  }

  return (
    <section className="detail-panel tensor-inspector">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">INSPECTOR</span>
          <h2>{node.label}</h2>
        </div>
        <span className="op-badge">{node.op}</span>
      </div>

      <div className="tensor-tabs" role="tablist" aria-label="Node tensors">
        {availableIds.map((tensorId) => {
          const item = tensors[tensorId];
          const isOutput = node.outputs.includes(tensorId);
          return (
            <button
              type="button"
              role="tab"
              aria-selected={tensorId === activeId}
              className={tensorId === activeId ? "is-active" : ""}
              key={tensorId}
              onClick={() => setActiveId(tensorId)}
            >
              <span>{isOutput ? "OUT" : "IN"}</span>
              {item?.name ?? tensorId}
            </button>
          );
        })}
      </div>

      <dl className="tensor-meta">
        <div>
          <dt>Name</dt>
          <dd>{tensor.name}</dd>
        </div>
        <MoveRight size={15} aria-hidden="true" />
        <div>
          <dt>Shape</dt>
          <dd>[{tensor.shape.join(", ")}]</dd>
        </div>
        <div>
          <dt>dtype</dt>
          <dd>{tensor.dtype}</dd>
        </div>
      </dl>

      <div className="tensor-values">
        <div className="tensor-values__label">
          <span>Values</span>
          <span>
            {tensor.shape.reduce((total, size) => total * size, 1)} items
          </span>
        </div>
        <pre>
          {JSON.stringify(
            tensor.values,
            (_key, value) => (value === null ? "masked" : value),
            2,
          )}
        </pre>
      </div>
    </section>
  );
}
