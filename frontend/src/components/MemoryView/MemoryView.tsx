import {
  Boxes,
  Database,
  Grid2X2,
  HardDrive,
  Layers3,
  LoaderCircle,
  Plus,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { loadMemorySlice } from "../../api/attention";
import type {
  AttentionRun,
  MemorySlice,
  MemoryTensorSpec,
} from "../../types/attention";

type Memory = AttentionRun["memory"];
type CacheActivity = AttentionRun["cache_activity"];
type ViewMode = "overview" | "blocks" | "slice";

interface MemoryViewProps {
  sessionId?: string;
  memory?: Memory;
  previousMemory?: Memory;
  activity?: CacheActivity;
  decodedToken?: string;
  isDecoding: boolean;
  canDecode: boolean;
  onDecode: () => void;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(2)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function formatMetric(value: number | null): string {
  if (value === null) {
    return "--";
  }
  return Math.abs(value) < 0.001 && value !== 0
    ? value.toExponential(2)
    : value.toFixed(3);
}

export function MemoryView({
  sessionId,
  memory,
  previousMemory,
  activity,
  decodedToken,
  isDecoding,
  canDecode,
  onDecode,
}: MemoryViewProps) {
  const [mode, setMode] = useState<ViewMode>("overview");
  const [selectedId, setSelectedId] = useState("");
  const [head, setHead] = useState(0);
  const [start, setStart] = useState(0);
  const [end, setEnd] = useState(1);
  const [slice, setSlice] = useState<MemorySlice>();
  const [sliceError, setSliceError] = useState<string>();
  const [isLoadingSlice, setIsLoadingSlice] = useState(false);

  const tensors = useMemo(() => memory?.spec.tensors ?? [], [memory]);
  const selected =
    tensors.find((tensor) => tensor.id === selectedId) ?? tensors[0];
  const growthSize =
    selected?.growth_axis === null || selected?.growth_axis === undefined
      ? 1
      : selected.shape[selected.growth_axis];
  const headAxis = selected?.axes.indexOf("head") ?? -1;
  const headCount =
    selected && headAxis >= 0 ? selected.shape[headAxis] : 0;

  useEffect(() => {
    const first = tensors[0];
    if (!first) {
      return;
    }
    setSelectedId(first.id);
    setHead(0);
    const size =
      first.growth_axis === null ? 1 : first.shape[first.growth_axis];
    setStart(Math.max(0, size - 8));
    setEnd(size);
    setSlice(undefined);
    setSliceError(undefined);
  }, [memory?.tokens, memory?.spec.kind, tensors]);

  if (!memory) {
    return null;
  }

  async function requestSlice() {
    if (!sessionId || !selected) {
      return;
    }
    setIsLoadingSlice(true);
    setSliceError(undefined);
    try {
      const payload = await loadMemorySlice(
        sessionId,
        selected.id,
        start,
        end,
        headAxis >= 0 ? head : undefined,
      );
      setSlice(payload);
    } catch (error) {
      setSliceError(
        error instanceof Error ? error.message : "Failed to load slice.",
      );
    } finally {
      setIsLoadingSlice(false);
    }
  }

  return (
    <section className="cache-panel memory-view">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">PERSISTENT RUNTIME STATE</span>
          <h2>Memory View</h2>
        </div>
        <div className="cache-phase">
          <span>{activity?.phase ?? "prefill"}</span>
          <HardDrive size={18} aria-hidden="true" />
        </div>
      </div>

      <div className="memory-view__toolbar">
        <div className="segmented-control" aria-label="Memory view mode">
          <ModeButton
            label="Overview"
            active={mode === "overview"}
            onClick={() => setMode("overview")}
          />
          <ModeButton
            label="Block View"
            active={mode === "blocks"}
            onClick={() => setMode("blocks")}
          />
          <ModeButton
            label="Slice View"
            active={mode === "slice"}
            onClick={() => setMode("slice")}
          />
        </div>
        <span className="memory-kind">{memory.spec.kind}</span>
      </div>

      {mode === "overview" ? (
        <Overview
          tensors={tensors}
          totalNumel={memory.spec.total_numel}
          totalBytes={memory.spec.total_bytes}
          previousBytes={previousMemory?.spec.total_bytes}
        />
      ) : null}

      {mode === "blocks" ? (
        <BlockView
          tensors={tensors}
          selected={selected}
          onSelect={setSelectedId}
        />
      ) : null}

      {mode === "slice" && selected ? (
        <div className="memory-slice-view">
          <div className="memory-slice-controls">
            <label>
              <span>Memory tensor</span>
              <select
                value={selected.id}
                onChange={(event) => {
                  const next = tensors.find(
                    (tensor) => tensor.id === event.target.value,
                  );
                  setSelectedId(event.target.value);
                  setHead(0);
                  const size =
                    next?.growth_axis === null ||
                    next?.growth_axis === undefined
                      ? 1
                      : next.shape[next.growth_axis];
                  setStart(Math.max(0, size - 8));
                  setEnd(size);
                  setSlice(undefined);
                }}
              >
                {tensors.map((tensor) => (
                  <option value={tensor.id} key={tensor.id}>
                    {tensor.name}
                  </option>
                ))}
              </select>
            </label>
            {headAxis >= 0 ? (
              <label>
                <span>Head</span>
                <select
                  value={head}
                  onChange={(event) => {
                    setHead(Number(event.target.value));
                    setSlice(undefined);
                  }}
                >
                  {Array.from({ length: headCount }, (_, index) => (
                    <option value={index} key={index}>
                      {index}
                    </option>
                  ))}
                </select>
              </label>
            ) : null}
            <label>
              <span>{selected.growth_axis_name ?? "Range"} start</span>
              <input
                type="number"
                min={0}
                max={Math.max(growthSize - 1, 0)}
                value={start}
                onChange={(event) => {
                  setStart(Number(event.target.value));
                  setSlice(undefined);
                }}
              />
            </label>
            <label>
              <span>{selected.growth_axis_name ?? "Range"} end</span>
              <input
                type="number"
                min={1}
                max={growthSize}
                value={end}
                onChange={(event) => {
                  setEnd(Number(event.target.value));
                  setSlice(undefined);
                }}
              />
            </label>
            <button
              type="button"
              className="memory-load-button"
              onClick={() => void requestSlice()}
              disabled={isLoadingSlice || end <= start}
            >
              {isLoadingSlice ? (
                <LoaderCircle className="spin" size={15} />
              ) : (
                <Layers3 size={15} />
              )}
              Load slice
            </button>
          </div>
          <div className="memory-slice-result">
            {sliceError ? <p className="memory-error">{sliceError}</p> : null}
            {slice ? (
              <>
                <div className="memory-slice-meta">
                  <strong>{slice.id}</strong>
                  <code>[{slice.shape.join(", ")}]</code>
                  <span>{slice.numel} values</span>
                  <span>{formatBytes(slice.bytes)}</span>
                </div>
                <pre>{JSON.stringify(slice.values, null, 2)}</pre>
              </>
            ) : (
              <div className="memory-empty">
                Select a bounded range, then load only that slice.
              </div>
            )}
          </div>
        </div>
      ) : null}

      <div className="cache-activity">
        <div className={activity?.phase === "decode" ? "is-active" : ""}>
          <Database size={16} aria-hidden="true" />
          <span>Memory Read</span>
          <strong>{activity?.read_tokens ?? 0} steps</strong>
        </div>
        <div className="is-active">
          <Plus size={16} aria-hidden="true" />
          <span>
            {activity?.update_kind === "state_update"
              ? "State Update"
              : "Append"}
          </span>
          <strong>
            {`${activity?.update_kind === "state_update" ? "" : "+"}${
              activity?.appended_tokens ?? memory.tokens
            } ${
              (activity?.appended_tokens ?? memory.tokens) === 1
                ? "step"
                : "steps"
            }`}
          </strong>
        </div>
        <div>
          <HardDrive size={16} aria-hidden="true" />
          <span>Total Size</span>
          <strong>{formatBytes(memory.spec.total_bytes)}</strong>
        </div>
      </div>

      <div className="cache-actions">
        <div className="decode-status">
          {decodedToken
            ? `Persistent state updated with ${decodedToken}`
            : "Persistent state ready"}
        </div>
        <button
          className="decode-button"
          type="button"
          onClick={onDecode}
          disabled={isDecoding || !canDecode}
        >
          {isDecoding ? (
            <LoaderCircle className="spin" size={17} />
          ) : (
            <Plus size={17} />
          )}
          {canDecode ? "Decode One Token" : "Decode limit reached"}
        </button>
      </div>
    </section>
  );
}

function ModeButton({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      className={active ? "is-active" : ""}
      aria-pressed={active}
      onClick={onClick}
    >
      {label}
    </button>
  );
}

function Overview({
  tensors,
  totalNumel,
  totalBytes,
  previousBytes,
}: {
  tensors: MemoryTensorSpec[];
  totalNumel: number;
  totalBytes: number;
  previousBytes?: number;
}) {
  return (
    <div className="memory-overview">
      <div className="memory-overview__total">
        <div>
          <Boxes size={18} />
          <span>Total numel</span>
          <strong>{totalNumel.toLocaleString()}</strong>
        </div>
        <div>
          <HardDrive size={18} />
          <span>Allocated</span>
          <strong>{formatBytes(totalBytes)}</strong>
          {previousBytes !== undefined ? (
            <small>+{formatBytes(totalBytes - previousBytes)}</small>
          ) : null}
        </div>
      </div>
      <div className="memory-spec-list">
        {tensors.map((tensor) => (
          <article className="memory-spec-row" key={tensor.id}>
            <div className="memory-spec-row__identity">
              <Database size={16} />
              <div>
                <strong>{tensor.name}</strong>
                <span>{tensor.role} · {tensor.kind}</span>
              </div>
            </div>
            <code>[{tensor.shape.join(", ")}]</code>
            <span>{tensor.dtype}</span>
            <span>{tensor.numel.toLocaleString()} values</span>
            <span>{formatBytes(tensor.bytes)}</span>
            <span>
              grows on {tensor.growth_axis_name ?? "fixed state"}
            </span>
          </article>
        ))}
      </div>
    </div>
  );
}

function BlockView({
  tensors,
  selected,
  onSelect,
}: {
  tensors: MemoryTensorSpec[];
  selected?: MemoryTensorSpec;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="memory-block-view">
      <div className="memory-block-header">
        <label>
          <span>Memory tensor</span>
          <select
            value={selected?.id ?? ""}
            onChange={(event) => onSelect(event.target.value)}
          >
            {tensors.map((tensor) => (
              <option value={tensor.id} key={tensor.id}>
                {tensor.name}
              </option>
            ))}
          </select>
        </label>
        <span>
          <Grid2X2 size={15} />
          aggregated by head and {selected?.growth_axis_name ?? "state"}
        </span>
      </div>
      <div className="memory-block-grid">
        {selected?.blocks.map((block, index) => (
          <div
            className="memory-block"
            style={{
              opacity: 0.5 + Math.min(block.mean_abs ?? 0, 1) * 0.5,
            }}
            title={`min ${formatMetric(block.min)}, max ${formatMetric(block.max)}, L2 ${formatMetric(block.l2)}`}
            key={`${block.head_start}-${block.start}-${index}`}
          >
            <span>
              {block.head_start !== null
                ? `H${block.head_start}${block.head_end !== block.head_start + 1 ? `-${(block.head_end ?? 1) - 1}` : ""}`
                : selected.growth_axis_name ?? "state"}
            </span>
            <strong>{block.start}:{block.end}</strong>
            <small>|x| {formatMetric(block.mean_abs)}</small>
          </div>
        ))}
        {!selected?.blocks.length ? (
          <div className="memory-empty">
            This state has no append axis; inspect it as a slice.
          </div>
        ) : null}
      </div>
    </div>
  );
}
