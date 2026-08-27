import {
  Braces,
  Calculator,
  ChevronDown,
  ChevronRight,
} from "lucide-react";
import {
  type ComponentType,
  type ReactNode,
  useState,
} from "react";

import type {
  GraphNode,
  TensorData,
  TensorScalar,
  TensorValues,
} from "../../types/attention";

interface NodeCalculationInspectorProps {
  node?: GraphNode;
  tensors: Record<string, TensorData>;
  tokens: string[];
}

interface CalculationProps {
  node: GraphNode;
  tensors: Record<string, TensorData>;
  tokens: string[];
}

interface OpDescription {
  role: string;
  formula: string;
}

const OP_DESCRIPTIONS: Record<string, OpDescription> = {
  input: {
    role: "Introduces token positions into the executable graph.",
    formula: "ids = [0, 1, ..., T - 1]",
  },
  embedding: {
    role: "Maps each token to a deterministic dense vector.",
    formula: "X[t] = Embedding(token[t], seed)",
  },
  linear: {
    role: "Projects every input vector through a learned weight matrix.",
    formula: "Y = X @ W (+ b)",
  },
  low_rank_compression: {
    role: "Compresses token features into a smaller latent KV representation.",
    formula: "C_KV = X @ W_DKV",
  },
  split_heads: {
    role: "Reshapes the model dimension into independent attention heads.",
    formula: "[B, T, D] -> [B, H, T, D/H]",
  },
  transpose: {
    role: "Reorders tensor axes for the following operation.",
    formula: "Y = transpose(X, axes)",
  },
  repeat_kv: {
    role: "Maps each shared KV head to its group of query heads.",
    formula: "KV_for_q[h] = KV_cache[floor(h / group_size)]",
  },
  rope: {
    role: "Encodes absolute token position by rotating pairs of head features.",
    formula: "[x_even, x_odd] @ rotation(position, frequency)",
  },
  matmul: {
    role: "Contracts the last dimension of the left tensor with the right tensor.",
    formula: "C[..., i, j] = sum_k A[..., i, k] * B[..., k, j]",
  },
  scale: {
    role: "Normalizes attention logits by the configured divisor.",
    formula: "Y = X / divisor",
  },
  causal_mask: {
    role: "Blocks attention to key positions later than the current query.",
    formula: "Y[q, k] = X[q, k] if k <= q, otherwise -inf",
  },
  softmax: {
    role: "Converts each score row into normalized attention probabilities.",
    formula: "P_i = exp(x_i - max(x)) / sum_j exp(x_j - max(x))",
  },
  cache_read: {
    role: "Reads historical K or V values from persistent runtime state.",
    formula: "cache_previous = state.cache",
  },
  cache_append: {
    role: "Appends new K or V vectors along the sequence axis.",
    formula: "cache_new = concat(cache_previous, value_new, axis=T)",
  },
  merge_heads: {
    role: "Combines independent head outputs into the model dimension.",
    formula: "[B, H, T, Dh] -> [B, T, H * Dh]",
  },
  output: {
    role: "Publishes the final attention result.",
    formula: "output = merged_context",
  },
};

const CALCULATION_INSPECTORS: Record<
  string,
  ComponentType<CalculationProps>
> = {
  linear: LinearInspector,
  low_rank_compression: LinearInspector,
  matmul: MatMulInspector,
  repeat_kv: RepeatKVInspector,
  rope: RoPEInspector,
  scale: ScaleInspector,
  causal_mask: MaskInspector,
  softmax: SoftmaxInspector,
  cache_append: CacheInspector,
};

export function NodeCalculationInspector({
  node,
  tensors,
  tokens,
}: NodeCalculationInspectorProps) {
  const [showCalculation, setShowCalculation] = useState(false);

  if (!node) {
    return (
      <section className="detail-panel calculation-inspector">
        <div className="panel-heading">
          <div>
            <span className="eyebrow">NODE INSPECTOR</span>
            <h2>Calculation details</h2>
          </div>
          <Braces size={18} aria-hidden="true" />
        </div>
        <div className="empty-state">Select a graph node to inspect its calculation.</div>
      </section>
    );
  }

  const description = OP_DESCRIPTIONS[node.op] ?? {
    role: "Executes an IR operation and records its tensor outputs.",
    formula: `${node.op}(inputs) -> outputs`,
  };
  const parameterIds = readStringArray(node.attrs.parameter_ids);
  const CalculationInspector =
    CALCULATION_INSPECTORS[node.op] ?? GenericInspector;

  return (
    <section className="detail-panel calculation-inspector">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">NODE CALCULATION</span>
          <h2>{node.label}</h2>
        </div>
        <span className="op-badge">{node.op}</span>
      </div>

      <div className="calculation-overview">
        <Definition label="Purpose" value={description.role} />
        <Definition label="Formula" value={description.formula} code />
        <Definition
          label="Shape transformation"
          value={shapeTransformation(node, tensors, parameterIds)}
          code
        />
      </div>

      <div className="tensor-groups">
        <TensorGroup label="Input tensor" ids={node.inputs} tensors={tensors} />
        {parameterIds.length ? (
          <TensorGroup
            label="Parameter tensor"
            ids={parameterIds}
            tensors={tensors}
          />
        ) : null}
        <TensorGroup label="Output tensor" ids={node.outputs} tensors={tensors} />
      </div>

      <button
        className="calculation-toggle"
        type="button"
        aria-expanded={showCalculation}
        onClick={() => setShowCalculation((visible) => !visible)}
      >
        <Calculator size={16} />
        Show Calculation
        {showCalculation ? (
          <ChevronDown size={16} />
        ) : (
          <ChevronRight size={16} />
        )}
      </button>

      {showCalculation ? (
        <div className="calculation-detail">
          <CalculationInspector node={node} tensors={tensors} tokens={tokens} />
        </div>
      ) : null}
    </section>
  );
}

function Definition({
  label,
  value,
  code = false,
}: {
  label: string;
  value: string;
  code?: boolean;
}) {
  return (
    <div className="definition">
      <span>{label}</span>
      {code ? <code>{value}</code> : <p>{value}</p>}
    </div>
  );
}

function TensorGroup({
  label,
  ids,
  tensors,
}: {
  label: string;
  ids: string[];
  tensors: Record<string, TensorData>;
}) {
  if (!ids.length) {
    return (
      <div className="tensor-group">
        <span className="tensor-group__label">{label}</span>
        <p className="tensor-group__empty">None</p>
      </div>
    );
  }

  return (
    <div className="tensor-group">
      <span className="tensor-group__label">{label}</span>
      {ids.map((id) => {
        const tensor = tensors[id];
        if (!tensor) {
          return null;
        }
        return (
          <details className="tensor-record" key={id} open>
            <summary>
              <strong>{tensor.name}</strong>
              <code>[{tensor.shape.join(", ")}]</code>
              <span>{tensor.dtype}</span>
            </summary>
            <pre>{formatTensorValues(tensor.values)}</pre>
          </details>
        );
      })}
    </div>
  );
}

function LinearInspector({ node, tensors, tokens }: CalculationProps) {
  const input = tensors[node.inputs[0]];
  const output = tensors[node.outputs[0]];
  const weightId = readStringArray(node.attrs.parameter_ids)[0];
  const weight = tensors[weightId];
  const tokenCount = input?.shape.at(-2) ?? 0;
  const outputSize = output?.shape.at(-1) ?? 0;
  const [tokenIndex, setTokenIndex] = useState(0);
  const [outputIndex, setOutputIndex] = useState(0);

  if (!input || !output || !weight) {
    return <GenericInspector node={node} tensors={tensors} tokens={tokens} />;
  }

  const inputVector = vectorAt(input.values, [0, tokenIndex]);
  const weightColumn = Array.from(
    { length: weight.shape[0] ?? 0 },
    (_, index) => scalarAt(weight.values, [index, outputIndex]),
  );
  const products = inputVector.map((value, index) =>
    multiply(value, weightColumn[index]),
  );
  const result = scalarAt(output.values, [0, tokenIndex, outputIndex]);
  const tokenLabel =
    tokenCount === 1 ? tokens.at(-1) : tokens[tokenIndex] ?? `token ${tokenIndex}`;

  return (
    <div className="specialized-calculation">
      <CalculationControls>
        <IndexSelect
          label="Token"
          value={tokenIndex}
          count={tokenCount}
          onChange={setTokenIndex}
          format={(index) =>
            tokenCount === 1
              ? `0 / ${tokens.at(-1) ?? "new token"}`
              : `${index} / ${tokens[index] ?? "token"}`
          }
        />
        <IndexSelect
          label="Output channel"
          value={outputIndex}
          count={outputSize}
          onChange={setOutputIndex}
        />
      </CalculationControls>
      <CalculationEquation>
        {output.name}[{tokenIndex}, {outputIndex}] = {input.name}[
        {tokenIndex}, :] @ {weight.name}[:, {outputIndex}]
      </CalculationEquation>
      <p className="calculation-caption">Selected token: {tokenLabel}</p>
      <VectorLine label="Input vector" values={inputVector} />
      <VectorLine label={`${weight.name} column`} values={weightColumn} />
      <VectorLine label="Element products" values={products} />
      <ScalarResult label="Sum / output" value={result} />
    </div>
  );
}

function MatMulInspector({ node, tensors, tokens }: CalculationProps) {
  const left = tensors[node.inputs[0]];
  const right = tensors[node.inputs[1]];
  const output = tensors[node.outputs[0]];
  const heads = output?.shape.length === 4 ? output.shape[1] : 1;
  const rows = output?.shape.at(-2) ?? 0;
  const columns = output?.shape.at(-1) ?? 0;
  const [head, setHead] = useState(0);
  const [row, setRow] = useState(0);
  const [column, setColumn] = useState(0);

  if (!left || !right || !output) {
    return <GenericInspector node={node} tensors={tensors} tokens={tokens} />;
  }

  const leftMatrix = matrixAtHead(left, head);
  const rightMatrix = matrixAtHead(right, head);
  const leftVector = leftMatrix[row] ?? [];
  const rightColumn = rightMatrix.map((values) => values[column] ?? null);
  const products = leftVector.map((value, index) =>
    multiply(value, rightColumn[index]),
  );
  const outputIndices =
    output.shape.length === 4 ? [0, head, row, column] : [row, column];
  const result = scalarAt(output.values, outputIndices);

  return (
    <div className="specialized-calculation">
      <CalculationControls>
        <IndexSelect label="Head" value={head} count={heads} onChange={setHead} />
        <IndexSelect label="Row / query" value={row} count={rows} onChange={setRow} />
        <IndexSelect
          label="Column / key"
          value={column}
          count={columns}
          onChange={setColumn}
        />
      </CalculationControls>
      <CalculationEquation>
        C[{head}, {row}, {column}] = dot(A[{head}, {row}, :], B[
        {head}, :, {column}])
      </CalculationEquation>
      <VectorLine label="Left vector" values={leftVector} />
      <VectorLine label="Right column" values={rightColumn} />
      <VectorLine label="Pairwise products" values={products} />
      <ScalarResult label="Dot product" value={result} />
    </div>
  );
}

function RepeatKVInspector({ node, tensors, tokens }: CalculationProps) {
  const input = tensors[node.inputs[0]];
  const output = tensors[node.outputs[0]];
  const repeats =
    typeof node.attrs.repeats === "number" ? node.attrs.repeats : 1;
  const queryHeads = output?.shape[1] ?? 0;
  const tokenCount = output?.shape.at(-2) ?? 0;
  const [queryHead, setQueryHead] = useState(0);
  const [tokenIndex, setTokenIndex] = useState(0);

  if (!input || !output) {
    return <GenericInspector node={node} tensors={tensors} tokens={tokens} />;
  }

  const kvHead = Math.floor(queryHead / repeats);
  return (
    <div className="specialized-calculation">
      <CalculationControls>
        <IndexSelect
          label="Query head"
          value={queryHead}
          count={queryHeads}
          onChange={setQueryHead}
        />
        <IndexSelect
          label="Token"
          value={tokenIndex}
          count={tokenCount}
          onChange={setTokenIndex}
        />
      </CalculationControls>
      <CalculationEquation>
        query head {queryHead} reads KV head {kvHead} (group size {repeats})
      </CalculationEquation>
      <VectorLine
        label={`KV head ${kvHead}`}
        values={vectorAt(input.values, [0, kvHead, tokenIndex])}
      />
      <VectorLine
        label={`Expanded head ${queryHead}`}
        values={vectorAt(output.values, [0, queryHead, tokenIndex])}
      />
    </div>
  );
}

function RoPEInspector({ node, tensors, tokens }: CalculationProps) {
  const input = tensors[node.inputs[0]];
  const output = tensors[node.outputs[0]];
  const heads = input?.shape[1] ?? 0;
  const tokenCount = input?.shape.at(-2) ?? 0;
  const pairCount = (input?.shape.at(-1) ?? 0) / 2;
  const positions = Array.isArray(node.attrs.positions)
    ? node.attrs.positions.filter(
        (value): value is number => typeof value === "number",
      )
    : [];
  const base = typeof node.attrs.base === "number" ? node.attrs.base : 10000;
  const [head, setHead] = useState(0);
  const [tokenIndex, setTokenIndex] = useState(0);
  const [pair, setPair] = useState(0);

  if (!input || !output) {
    return <GenericInspector node={node} tensors={tensors} tokens={tokens} />;
  }

  const position = positions[tokenIndex] ?? tokenIndex;
  const headDim = input.shape.at(-1) ?? 0;
  const angle = position / base ** ((2 * pair) / headDim);
  const evenIndex = pair * 2;
  const oddIndex = evenIndex + 1;
  const xEven = scalarAt(input.values, [0, head, tokenIndex, evenIndex]);
  const xOdd = scalarAt(input.values, [0, head, tokenIndex, oddIndex]);
  const yEven = scalarAt(output.values, [0, head, tokenIndex, evenIndex]);
  const yOdd = scalarAt(output.values, [0, head, tokenIndex, oddIndex]);

  return (
    <div className="specialized-calculation">
      <CalculationControls>
        <IndexSelect label="Head" value={head} count={heads} onChange={setHead} />
        <IndexSelect
          label="Token"
          value={tokenIndex}
          count={tokenCount}
          onChange={setTokenIndex}
        />
        <IndexSelect
          label="Feature pair"
          value={pair}
          count={pairCount}
          onChange={setPair}
        />
      </CalculationControls>
      <CalculationEquation>
        position {position}, angle = {formatScalar(angle)}, cos ={" "}
        {formatScalar(Math.cos(angle))}, sin = {formatScalar(Math.sin(angle))}
      </CalculationEquation>
      <VectorLine label="Input pair" values={[xEven, xOdd]} />
      <VectorLine label="Rotated pair" values={[yEven, yOdd]} />
    </div>
  );
}

function ScaleInspector({ node, tensors, tokens }: CalculationProps) {
  const input = tensors[node.inputs[0]];
  const output = tensors[node.outputs[0]];
  const divisor =
    typeof node.attrs.divisor === "number" ? node.attrs.divisor : 1;

  if (!input || !output) {
    return <GenericInspector node={node} tensors={tensors} tokens={tokens} />;
  }

  return (
    <ScalarTensorCalculation
      input={input}
      output={output}
      operation={(value) => `${formatScalar(value)} / ${formatScalar(divisor)}`}
      detail={`Every score is divided by sqrt(head_dim) = ${formatScalar(divisor)}.`}
    />
  );
}

function MaskInspector({ node, tensors, tokens }: CalculationProps) {
  const input = tensors[node.inputs[0]];
  const output = tensors[node.outputs[0]];

  if (!input || !output) {
    return <GenericInspector node={node} tensors={tensors} tokens={tokens} />;
  }

  return (
    <ScalarTensorCalculation
      input={input}
      output={output}
      operation={(value, row, column) => {
        const queryOffset = Math.max(
          (input.shape.at(-1) ?? 0) - (input.shape.at(-2) ?? 0),
          0,
        );
        return column > row + queryOffset
          ? `${formatScalar(value)} -> -inf (future key)`
          : `${formatScalar(value)} -> unchanged`;
      }}
      detail="The mask compares absolute query and key positions before Softmax."
    />
  );
}

function SoftmaxInspector({ node, tensors, tokens }: CalculationProps) {
  const input = tensors[node.inputs[0]];
  const output = tensors[node.outputs[0]];
  const heads = input?.shape.length === 4 ? input.shape[1] : 1;
  const rows = input?.shape.at(-2) ?? 0;
  const [head, setHead] = useState(0);
  const [row, setRow] = useState(0);

  if (!input || !output) {
    return <GenericInspector node={node} tensors={tensors} tokens={tokens} />;
  }

  const inputRow = matrixAtHead(input, head)[row] ?? [];
  const outputRow = matrixAtHead(output, head)[row] ?? [];
  const finiteInputs = inputRow.filter(
    (value): value is number => value !== null,
  );
  const max = finiteInputs.length ? Math.max(...finiteInputs) : 0;
  const exponentials = inputRow.map((value) =>
    value === null ? 0 : Math.exp(value - max),
  );
  const sum = exponentials.reduce((total, value) => total + value, 0);

  return (
    <div className="specialized-calculation">
      <CalculationControls>
        <IndexSelect label="Head" value={head} count={heads} onChange={setHead} />
        <IndexSelect label="Query row" value={row} count={rows} onChange={setRow} />
      </CalculationControls>
      <CalculationEquation>
        softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
      </CalculationEquation>
      <VectorLine label="Input" values={inputRow} />
      <ScalarResult label="Row max" value={max} />
      <VectorLine label="exp(input - max)" values={exponentials} />
      <ScalarResult label="Exponential sum" value={sum} />
      <VectorLine label="Output probabilities" values={outputRow} />
    </div>
  );
}

function CacheInspector({ node, tensors, tokens }: CalculationProps) {
  const inputs = node.inputs.map((id) => tensors[id]).filter(Boolean);
  const output = tensors[node.outputs[0]];
  if (!output || !inputs.length) {
    return <GenericInspector node={node} tensors={tensors} tokens={tokens} />;
  }

  const oldCache = inputs.length > 1 ? inputs[0] : undefined;
  const appended = inputs[inputs.length - 1];
  const previousTokens =
    typeof node.attrs.previous_tokens === "number"
      ? node.attrs.previous_tokens
      : 0;
  const appendedTokens =
    typeof node.attrs.appended_tokens === "number"
      ? node.attrs.appended_tokens
      : appended.shape.at(-2) ?? 0;

  return (
    <div className="specialized-calculation">
      <CalculationEquation>
        {oldCache?.name ?? "empty cache"} + {appended.name} -&gt; {output.name}
      </CalculationEquation>
      <div className="cache-calculation">
        <ShapeBox
          label="Old cache"
          shape={oldCache?.shape}
          detail={`${previousTokens} tokens`}
        />
        <span>+</span>
        <ShapeBox
          label="New values"
          shape={appended.shape}
          detail={`${appendedTokens} token${appendedTokens === 1 ? "" : "s"}`}
        />
        <span>=</span>
        <ShapeBox
          label="New cache"
          shape={output.shape}
          detail={`${node.attrs.cached_tokens ?? output.shape.at(-2)} tokens`}
          active
        />
      </div>
      <VectorLine
        label="Appended vector (head 0)"
        values={vectorAt(appended.values, [0, 0, 0])}
      />
    </div>
  );
}

function GenericInspector({ node, tensors }: CalculationProps) {
  const inputShapes = node.inputs
    .map((id) => tensors[id]?.shape)
    .filter(Boolean)
    .map(formatShape)
    .join(", ");
  const outputShapes = node.outputs
    .map((id) => tensors[id]?.shape)
    .filter(Boolean)
    .map(formatShape)
    .join(", ");

  return (
    <div className="generic-calculation">
      <CalculationEquation>
        {node.op}({inputShapes || "no tensor input"}) -&gt;{" "}
        {outputShapes || "no tensor output"}
      </CalculationEquation>
      <p>
        This operation has no specialized arithmetic renderer. Its complete
        input and output values remain available above.
      </p>
    </div>
  );
}

function ScalarTensorCalculation({
  input,
  output,
  operation,
  detail,
}: {
  input: TensorData;
  output: TensorData;
  operation: (value: TensorScalar, row: number, column: number) => string;
  detail: string;
}) {
  const heads = input.shape.length === 4 ? input.shape[1] : 1;
  const rows = input.shape.at(-2) ?? 0;
  const columns = input.shape.at(-1) ?? 0;
  const [head, setHead] = useState(0);
  const [row, setRow] = useState(0);
  const [column, setColumn] = useState(0);
  const inputValue = matrixAtHead(input, head)[row]?.[column] ?? null;
  const outputValue = matrixAtHead(output, head)[row]?.[column] ?? null;

  return (
    <div className="specialized-calculation">
      <CalculationControls>
        <IndexSelect label="Head" value={head} count={heads} onChange={setHead} />
        <IndexSelect label="Query" value={row} count={rows} onChange={setRow} />
        <IndexSelect
          label="Key"
          value={column}
          count={columns}
          onChange={setColumn}
        />
      </CalculationControls>
      <CalculationEquation>
        {operation(inputValue, row, column)} = {formatScalar(outputValue)}
      </CalculationEquation>
      <p className="calculation-caption">{detail}</p>
    </div>
  );
}

function CalculationControls({ children }: { children: ReactNode }) {
  return <div className="calculation-controls">{children}</div>;
}

function IndexSelect({
  label,
  value,
  count,
  onChange,
  format,
}: {
  label: string;
  value: number;
  count: number;
  onChange: (value: number) => void;
  format?: (value: number) => string;
}) {
  return (
    <label>
      <span>{label}</span>
      <select
        value={Math.min(value, Math.max(count - 1, 0))}
        onChange={(event) => onChange(Number(event.target.value))}
      >
        {Array.from({ length: count }, (_, index) => (
          <option value={index} key={index}>
            {format ? format(index) : index}
          </option>
        ))}
      </select>
    </label>
  );
}

function CalculationEquation({ children }: { children: ReactNode }) {
  return <code className="calculation-equation">{children}</code>;
}

function VectorLine({
  label,
  values,
}: {
  label: string;
  values: TensorScalar[];
}) {
  return (
    <div className="vector-line">
      <span>{label}</span>
      <div>
        {values.map((value, index) => (
          <code key={index}>{formatScalar(value)}</code>
        ))}
      </div>
    </div>
  );
}

function ScalarResult({
  label,
  value,
}: {
  label: string;
  value: TensorScalar;
}) {
  return (
    <div className="scalar-result">
      <span>{label}</span>
      <strong>{formatScalar(value)}</strong>
    </div>
  );
}

function ShapeBox({
  label,
  shape,
  detail,
  active = false,
}: {
  label: string;
  shape?: number[];
  detail: string;
  active?: boolean;
}) {
  return (
    <div className={`shape-box ${active ? "is-active" : ""}`}>
      <span>{label}</span>
      <code>{shape ? formatShape(shape) : "[]"}</code>
      <small>{detail}</small>
    </div>
  );
}

function shapeTransformation(
  node: GraphNode,
  tensors: Record<string, TensorData>,
  parameterIds: string[],
): string {
  const inputs = node.inputs
    .map((id) => tensors[id]?.shape)
    .filter((shape): shape is number[] => Boolean(shape));
  const parameters = parameterIds
    .map((id) => tensors[id]?.shape)
    .filter((shape): shape is number[] => Boolean(shape));
  const outputs = node.outputs
    .map((id) => tensors[id]?.shape)
    .filter((shape): shape is number[] => Boolean(shape));
  const separator =
    node.op === "linear" ||
    node.op === "low_rank_compression" ||
    node.op === "matmul"
      ? " @ "
      : node.op === "cache_append"
        ? " + "
        : ", ";
  const left = [...inputs, ...parameters].map(formatShape).join(separator);
  return `${left || "state"} -> ${outputs.map(formatShape).join(", ") || "none"}`;
}

function readStringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string")
    : [];
}

function formatShape(shape: number[]): string {
  return `[${shape.join(", ")}]`;
}

function formatTensorValues(values: TensorValues): string {
  return JSON.stringify(
    values,
    (_key, value) => (value === null ? "masked" : value),
    2,
  );
}

function scalarAt(
  values: TensorValues,
  indices: number[],
): TensorScalar {
  let current: TensorValues | undefined = values;
  for (const index of indices) {
    if (!Array.isArray(current)) {
      return null;
    }
    current = current[index];
  }
  return typeof current === "number" || current === null ? current : null;
}

function vectorAt(
  values: TensorValues,
  indices: number[],
): TensorScalar[] {
  let current: TensorValues | undefined = values;
  for (const index of indices) {
    if (!Array.isArray(current)) {
      return [];
    }
    current = current[index];
  }
  if (!Array.isArray(current)) {
    return [];
  }
  return current.map((value) =>
    typeof value === "number" || value === null ? value : null,
  );
}

function matrixAtHead(
  tensor: TensorData,
  head: number,
): TensorScalar[][] {
  const indices = tensor.shape.length === 4 ? [0, head] : [];
  let current: TensorValues | undefined = tensor.values;
  for (const index of indices) {
    if (!Array.isArray(current)) {
      return [];
    }
    current = current[index];
  }
  if (!Array.isArray(current)) {
    return [];
  }
  return current.map((row) =>
    Array.isArray(row)
      ? row.map((value) =>
          typeof value === "number" || value === null ? value : null,
        )
      : [],
  );
}

function multiply(
  left: TensorScalar,
  right: TensorScalar | undefined,
): TensorScalar {
  return left === null || right === null || right === undefined
    ? null
    : left * right;
}

function formatScalar(value: TensorScalar): string {
  if (value === null || !Number.isFinite(value)) {
    return "-inf";
  }
  if (Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 0.001)) {
    return value.toExponential(3);
  }
  return value.toFixed(4);
}
