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

function extractHead(
  values: TensorValues,
  head: number,
): TensorScalar[][] | undefined {
  if (!Array.isArray(values) || !Array.isArray(values[0])) {
    return undefined;
  }
  const heads = values[0] as TensorValues[];
  const matrix = heads[head];
  if (!Array.isArray(matrix)) {
    return undefined;
  }
  return matrix.map((row) =>
    Array.isArray(row) ? (row as TensorScalar[]) : [],
  );
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
  const headCount = tensor?.shape[1] ?? 0;
  const [head, setHead] = useState(0);

  useEffect(() => {
    setHead(0);
  }, [tensor?.id]);

  const matrix = useMemo(
    () => (tensor ? extractHead(tensor.values, head) : undefined),
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
          <h2>{node?.label ?? "Attention matrix"}</h2>
        </div>
        <Grid3X3 size={18} aria-hidden="true" />
      </div>

      {!matrix || !tensor ? (
        <div className="empty-state">
          Step into QK^T, Scale, Causal Mask, or Softmax.
        </div>
      ) : (
        <>
          <div className="matrix-toolbar">
            <div className="segmented-control" aria-label="Attention head">
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
            <span>
              {matrix.length} x {matrix[0]?.length ?? 0}
            </span>
          </div>

          <div className="matrix-scroll">
            <div
              className="matrix"
              style={{
                gridTemplateColumns: `72px repeat(${matrix[0]?.length ?? 0}, minmax(38px, 1fr))`,
              }}
            >
              <div className="matrix__corner">Q / K</div>
              {(matrix[0] ?? []).map((_, column) => (
                <div className="matrix__axis" key={`column-${column}`}>
                  {tokens[column] ?? column}
                </div>
              ))}
              {matrix.map((row, rowIndex) => [
                <div className="matrix__axis matrix__axis--row" key={`row-${rowIndex}`}>
                  {queryTokens[rowIndex] ?? rowIndex}
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
