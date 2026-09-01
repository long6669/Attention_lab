import { useEffect, useMemo, useState } from "react";
import { Grid3X3 } from "lucide-react";

import type {
  GraphNode,
  TensorData,
  TensorScalar,
  TensorValues,
} from "../../types/attention";

interface AttentionMatrixProps {
  node?: GraphNode;
  tensor?: TensorData;
  tokens: string[];
  phase?: "prefill" | "decode";
}

function extractMatrix(
  values: TensorValues | undefined,
  rank: number,
  head: number,
): TensorScalar[][] | undefined {
  if (values === undefined) {
    return undefined;
  }
  if (rank === 0) {
    return [[typeof values === "number" || values === null ? values : null]];
  }
  if (!Array.isArray(values)) {
    return undefined;
  }

  let selected: TensorValues = values;
  if (rank >= 3) {
    selected = values[0] ?? [];
  }
  if (rank >= 4 && Array.isArray(selected)) {
    selected = selected[head] ?? [];
  }
  return flattenTensorRows(selected);
}

function flattenTensorRows(values: TensorValues): TensorScalar[][] {
  if (!Array.isArray(values)) {
    return [[values]];
  }
  if (
    values.every(
      (value) => typeof value === "number" || value === null,
    )
  ) {
    return [values as TensorScalar[]];
  }
  return values.flatMap((value) => flattenTensorRows(value));
}

function cellColor(value: TensorScalar, min: number, max: number): string {
  if (value === null) {
    return "#edf0f2";
  }
  const range = max - min || 1;
  const normalized = Math.max(0, Math.min(1, (value - min) / range));
  const lightness = 96 - normalized * 48;
  return `hsl(184 77% ${lightness}%)`;
}

export function AttentionMatrix({
  node,
  tensor,
  tokens,
  phase,
}: AttentionMatrixProps) {
  const isAttention = node?.attrs.visualization === "attention_matrix";
  const headCount = tensor?.shape.length === 4 ? tensor.shape[1] : 1;
  const [head, setHead] = useState(0);

  useEffect(() => {
    setHead(0);
  }, [tensor?.id]);

  const matrix = useMemo(
    () =>
      tensor
        ? extractMatrix(tensor.values, tensor.shape.length, head)
        : undefined,
    [head, tensor],
  );
  const finiteValues = matrix
    ?.flat()
    .filter((value): value is number => value !== null) ?? [];
  const min = finiteValues.length ? Math.min(...finiteValues) : 0;
  const max = finiteValues.length ? Math.max(...finiteValues) : 1;
  const queryTokens =
    phase === "decode" && matrix ? tokens.slice(-matrix.length) : tokens;

  return (
    <section className="detail-panel matrix-panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">MATRIX VIEW</span>
          <h2>{node?.label ?? "Tensor representation"}</h2>
        </div>
        <Grid3X3 size={18} aria-hidden="true" />
      </div>

      {!tensor ? (
        <div className="empty-state">
          Select an executed graph node to inspect its output tensor.
        </div>
      ) : !matrix ? (
        <div className="empty-state tensor-values-omitted">
          <div>
            <strong>{tensor.name}</strong>
            <code>[{tensor.shape.join(", ")}]</code>
            <span>
              Values are not loaded for this {tensor.numel}-element tensor.
            </span>
          </div>
        </div>
      ) : (
        <>
          <div className="matrix-toolbar">
            {headCount > 1 ? (
              <div className="segmented-control" aria-label="Tensor head">
                {Array.from({ length: headCount }, (_, index) => (
                  <button
                    type="button"
                    className={index === head ? "is-active" : ""}
                    aria-pressed={index === head}
                    onClick={() => setHead(index)}
                    key={index}
                  >
                    Head {index}
                  </button>
                ))}
              </div>
            ) : (
              <span>{tensor.name}</span>
            )}
            <span>
              [{tensor.shape.join(", ")}] · {tensor.dtype}
            </span>
          </div>

          <div className="matrix-scroll">
            <div
              className="matrix"
              style={{
                gridTemplateColumns: `72px repeat(${matrix[0]?.length ?? 0}, minmax(38px, 1fr))`,
              }}
            >
              <div className="matrix__corner">
                {isAttention ? "Q / K" : "row / feature"}
              </div>
              {(matrix[0] ?? []).map((_, column) => (
                <div className="matrix__axis" key={`column-${column}`}>
                  {isAttention ? tokens[column] ?? column : column}
                </div>
              ))}
              {matrix.map((row, rowIndex) => [
                <div className="matrix__axis matrix__axis--row" key={`row-${rowIndex}`}>
                  {isAttention
                    ? queryTokens[rowIndex] ?? rowIndex
                    : matrix.length === tokens.length
                      ? tokens[rowIndex]
                      : rowIndex}
                </div>,
                ...row.map((value, columnIndex) => (
                  <div
                    className={`matrix__cell ${value === null ? "is-masked" : ""}`}
                    style={{
                      backgroundColor: cellColor(value, min, max),
                      color:
                        value !== null && (value - min) / (max - min || 1) > 0.62
                          ? "#ffffff"
                          : "#243037",
                    }}
                    title={
                      value === null
                        ? "Masked"
                        : `q${rowIndex}, k${columnIndex}: ${value}`
                    }
                    key={`${rowIndex}-${columnIndex}`}
                  >
                    {value === null ? "--" : value.toFixed(2)}
                  </div>
                )),
              ])}
            </div>
          </div>
        </>
      )}
    </section>
  );
}
