import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  AlertCircle,
  Cpu,
  GitCompareArrows,
  LoaderCircle,
  Network,
} from "lucide-react";

import { decodeOneToken, runAttention } from "./api/attention";
import { AttentionMatrix } from "./components/AttentionMatrix/AttentionMatrix";
import { CompareView } from "./components/CompareView/CompareView";
import { ExamplesGallery } from "./components/ExamplesGallery/ExamplesGallery";
import { GraphView } from "./components/GraphView/GraphView";
import { LearningGuide } from "./components/LearningGuide/LearningGuide";
import { MemoryView } from "./components/MemoryView/MemoryView";
import { NodeCalculationInspector } from "./components/NodeCalculationInspector/NodeCalculationInspector";
import { ShareToolbar } from "./components/ShareToolbar/ShareToolbar";
import { Timeline } from "./components/Timeline/Timeline";
import {
  ARCHITECTURE_LABELS,
  type ExampleDefinition,
} from "./content/architectures";
import type {
  AttentionArchitecture,
  AttentionRun,
  GraphNode,
} from "./types/attention";

const DEFAULT_TEXT = "I love learning how attention works today";
const ALL_ARCHITECTURES = Object.keys(
  ARCHITECTURE_LABELS,
) as AttentionArchitecture[];
const DEFAULT_COMPARISON: AttentionArchitecture[] = [
  "mha",
  "mqa",
  "mla",
  "kda",
];
type AppMode = "workbench" | "compare";

const INITIAL_CONFIG = readUrlConfig();

export default function App() {
  const [text, setText] = useState(INITIAL_CONFIG.text);
  const [architecture, setArchitecture] = useState<AttentionArchitecture>(
    INITIAL_CONFIG.architecture,
  );
  const [mode, setMode] = useState<AppMode>(INITIAL_CONFIG.mode);
  const [compareArchitectures, setCompareArchitectures] = useState<
    AttentionArchitecture[]
  >(INITIAL_CONFIG.compareArchitectures);
  const [comparisonResults, setComparisonResults] = useState<AttentionRun[]>(
    [],
  );
  const [result, setResult] = useState<AttentionRun>();
  const [currentStep, setCurrentStep] = useState(0);
  const [selectedNodeId, setSelectedNodeId] = useState<string>();
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [isComparing, setIsComparing] = useState(false);
  const [isDecoding, setIsDecoding] = useState(false);
  const [error, setError] = useState<string>();
  const [previousMemory, setPreviousMemory] =
    useState<AttentionRun["memory"]>();
  const didInitialize = useRef(false);

  const execute = useCallback(
    async (input: string, selectedArchitecture: AttentionArchitecture) => {
      setIsLoading(true);
      setError(undefined);
      setIsPlaying(false);
      try {
        const payload = await runAttention(input, selectedArchitecture);
        setResult(payload);
        setPreviousMemory(undefined);
        setCurrentStep(0);
        setSelectedNodeId(payload.trace[0]?.node_id);
      } catch (runError) {
        setError(
          runError instanceof Error
            ? runError.message
            : "Failed to run attention.",
        );
      } finally {
        setIsLoading(false);
      }
    },
    [],
  );

  const executeComparison = useCallback(
    async (input: string, selectedArchitectures: AttentionArchitecture[]) => {
      setIsComparing(true);
      setError(undefined);
      try {
        const payloads = await Promise.all(
          selectedArchitectures.map((selectedArchitecture) =>
            runAttention(input, selectedArchitecture),
          ),
        );
        setComparisonResults(payloads);
      } catch (runError) {
        setError(
          runError instanceof Error
            ? runError.message
            : "Failed to compare attention architectures.",
        );
      } finally {
        setIsComparing(false);
      }
    },
    [],
  );

  useEffect(() => {
    if (didInitialize.current) {
      return;
    }
    didInitialize.current = true;
    if (INITIAL_CONFIG.mode === "compare") {
      void executeComparison(
        INITIAL_CONFIG.text,
        INITIAL_CONFIG.compareArchitectures,
      );
    } else {
      void execute(INITIAL_CONFIG.text, INITIAL_CONFIG.architecture);
    }
  }, [execute, executeComparison]);

  const changeStep = useCallback(
    (step: number) => {
      if (!result) {
        return;
      }
      const nextStep = Math.max(0, Math.min(step, result.trace.length - 1));
      setCurrentStep(nextStep);
      setSelectedNodeId(result.trace[nextStep]?.node_id);
    },
    [result],
  );

  useEffect(() => {
    if (!isPlaying || !result) {
      return;
    }
    if (currentStep >= result.trace.length - 1) {
      setIsPlaying(false);
      return;
    }
    const timer = window.setTimeout(() => {
      changeStep(currentStep + 1);
    }, 850);
    return () => window.clearTimeout(timer);
  }, [changeStep, currentStep, isPlaying, result]);

  const currentNodeId = result?.trace[currentStep]?.node_id;
  const selectedNode = useMemo(
    () => result?.graph.nodes.find((node) => node.id === selectedNodeId),
    [result, selectedNodeId],
  );
  const matrixTensor = useMemo(
    () => getVisualTensor(selectedNode, result),
    [selectedNode, result],
  );

  function handleSubmit(event: { preventDefault: () => void }) {
    event.preventDefault();
    if (mode === "compare") {
      void executeComparison(text, compareArchitectures);
    } else {
      void execute(text, architecture);
    }
  }

  function toggleComparisonArchitecture(
    selectedArchitecture: AttentionArchitecture,
  ) {
    setCompareArchitectures((current) => {
      if (current.includes(selectedArchitecture)) {
        return current.length > 2
          ? current.filter((item) => item !== selectedArchitecture)
          : current;
      }
      return current.length < 4 ? [...current, selectedArchitecture] : current;
    });
  }

  function openArchitecture(selectedArchitecture: AttentionArchitecture) {
    setArchitecture(selectedArchitecture);
    setMode("workbench");
    void execute(text, selectedArchitecture);
    window.requestAnimationFrame(() => {
      document
        .querySelector(".workspace")
        ?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }

  function openExample(example: ExampleDefinition) {
    setText(example.text);
    setArchitecture(example.architecture);
    setMode(example.mode);
    setCompareArchitectures(example.compareArchitectures);
    if (example.mode === "compare") {
      void executeComparison(example.text, example.compareArchitectures);
    } else {
      void execute(example.text, example.architecture);
    }
  }

  const shareUrl = useMemo(
    () =>
      buildShareUrl({
        text,
        architecture,
        mode,
        compareArchitectures,
      }),
    [architecture, compareArchitectures, mode, text],
  );

  useEffect(() => {
    window.history.replaceState({}, "", shareUrl);
  }, [shareUrl]);

  async function handleDecode() {
    if (!result) {
      return;
    }
    setIsDecoding(true);
    setError(undefined);
    setIsPlaying(false);
    try {
      const payload = await decodeOneToken(result.session_id);
      const memoryReadStep = payload.trace.findIndex(
        (event) => event.op === "cache_read" || event.op === "state_read",
      );
      setPreviousMemory(result.memory);
      setResult(payload);
      setCurrentStep(Math.max(memoryReadStep, 0));
      setSelectedNodeId(payload.trace[Math.max(memoryReadStep, 0)]?.node_id);
    } catch (decodeError) {
      setError(
        decodeError instanceof Error
          ? decodeError.message
          : "Failed to decode one token.",
      );
    } finally {
      setIsDecoding(false);
    }
  }

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="brand">
          <div className="brand__mark" aria-hidden="true">
            <Activity size={20} />
          </div>
          <div>
            <h1>AttnLab</h1>
            <p>Multi-Head Attention / execution workspace</p>
          </div>
        </div>
        <div className="header-actions">
          <nav className="mode-switch" aria-label="Workspace mode">
            <button
              type="button"
              className={mode === "workbench" ? "is-active" : ""}
              onClick={() => setMode("workbench")}
            >
              <Network size={15} />
              Workbench
            </button>
            <button
              type="button"
              className={mode === "compare" ? "is-active" : ""}
              onClick={() => {
                setMode("compare");
                if (comparisonResults.length === 0) {
                  void executeComparison(text, compareArchitectures);
                }
              }}
            >
              <GitCompareArrows size={15} />
              Compare
            </button>
          </nav>
          <div className="runtime-status">
            <span className="status-dot" />
            NumPy Runtime
          </div>
        </div>
      </header>

      <main>
        <ExamplesGallery onSelect={openExample} />

        <form className="run-bar" onSubmit={handleSubmit}>
          <div className="run-field run-field--input">
            <label htmlFor="token-input">Input tokens</label>
            <input
              id="token-input"
              value={text}
              onChange={(event) => setText(event.target.value)}
              placeholder="Enter up to 10 whitespace-separated tokens"
              autoComplete="off"
            />
          </div>
          {mode === "workbench" ? (
            <div className="run-field run-field--architecture">
              <label htmlFor="architecture-select">Architecture</label>
              <select
                id="architecture-select"
                value={architecture}
                onChange={(event) =>
                  setArchitecture(event.target.value as AttentionArchitecture)
                }
              >
                {Object.entries(ARCHITECTURE_LABELS).map(([value, label]) => (
                  <option value={value} key={value}>
                    {label}
                  </option>
                ))}
              </select>
            </div>
          ) : (
            <fieldset className="architecture-picker">
              <legend>Compare 2–4 architectures</legend>
              {ALL_ARCHITECTURES.map((value) => (
                <label key={value}>
                  <input
                    type="checkbox"
                    checked={compareArchitectures.includes(value)}
                    disabled={
                      !compareArchitectures.includes(value) &&
                      compareArchitectures.length >= 4
                    }
                    onChange={() => toggleComparisonArchitecture(value)}
                  />
                  <span>{value.toUpperCase()}</span>
                </label>
              ))}
            </fieldset>
          )}
          <button
            className="run-button"
            type="submit"
            disabled={isLoading || isComparing}
          >
            {isLoading || isComparing ? (
              <LoaderCircle className="spin" size={17} />
            ) : mode === "compare" ? (
              <GitCompareArrows size={17} />
            ) : (
              <Cpu size={17} />
            )}
            {mode === "compare" ? "Compare" : "Run"}
          </button>
        </form>

        {error ? (
          <div className="message message--error" role="alert">
            <AlertCircle size={17} />
            {error}
          </div>
        ) : null}
        {(mode === "compare"
          ? comparisonResults.flatMap((item) => item.warnings)
          : (result?.warnings ?? [])
        ).map((warning) => (
          <div className="message message--warning" role="status" key={warning}>
            <AlertCircle size={17} />
            {warning}
          </div>
        ))}

        {mode === "compare" ? (
          <>
            {isComparing && comparisonResults.length === 0 ? (
              <div className="comparison-loading">
                <LoaderCircle className="spin" size={20} />
                Executing selected architectures
              </div>
            ) : (
              <CompareView
                results={comparisonResults}
                onOpen={openArchitecture}
              />
            )}
            {comparisonResults.length > 0 ? (
              <ShareToolbar
                runs={comparisonResults}
                shareUrl={shareUrl}
                captureSelector="#comparison-view"
              />
            ) : null}
          </>
        ) : (
          <>
            <LearningGuide
              architecture={result?.config.architecture ?? architecture}
            />
            <section className="workspace" id="execution-workspace">
              <div className="workspace__heading">
                <div>
                  <span className="eyebrow">EXECUTION GRAPH</span>
                  <h2>
                    {result
                      ? ARCHITECTURE_LABELS[result.config.architecture]
                      : ARCHITECTURE_LABELS[architecture]}
                  </h2>
                </div>
                {result ? (
                  <div className="config-strip">
                    <span
                      className={`phase-badge phase-badge--${result.phase}`}
                    >
                      {result.phase}
                    </span>
                    <span>B {result.config.batch_size}</span>
                    <span>S {result.config.seq_len}</span>
                    <span>D {result.config.d_model}</span>
                    <span>QH {result.config.num_q_heads}</span>
                    <span>KVH {result.config.num_kv_heads}</span>
                    {result.config.architecture === "mla" ? (
                      <span>R {result.config.kv_lora_rank}</span>
                    ) : null}
                    {result.config.use_rope ? <span>RoPE</span> : null}
                    {result.config.architecture === "kda" ? (
                      <>
                        <span>Decay {result.config.state_decay}</span>
                        <span>Write {result.config.state_write_rate}</span>
                      </>
                    ) : null}
                    {result.config.architecture === "csa" ||
                    result.config.architecture === "hca" ? (
                      <>
                        <span>Window {result.config.compression_window}</span>
                        <span>TopK {result.config.routing_top_k}</span>
                      </>
                    ) : null}
                    <span>{result.config.dtype}</span>
                  </div>
                ) : null}
              </div>

              <div className="token-strip" aria-label="Input token sequence">
                {result?.tokens.map((token, index) => (
                  <span key={`${token}-${index}`}>
                    <small>{index}</small>
                    {token}
                  </span>
                ))}
              </div>

              <div className="graph-workbench">
                <div className="graph-region">
                  {result ? (
                    <GraphView
                      graph={result.graph}
                      currentNodeId={currentNodeId}
                      selectedNodeId={selectedNodeId}
                      onSelectNode={setSelectedNodeId}
                    />
                  ) : (
                    <div className="loading-state">
                      <LoaderCircle className="spin" size={22} />
                      Building attention graph
                    </div>
                  )}
                </div>
                <aside className="graph-inspector-slot">
                  <NodeCalculationInspector
                    key={selectedNode?.id ?? "empty"}
                    node={selectedNode}
                    tensors={result?.tensors ?? {}}
                    tokens={result?.tokens ?? []}
                  />
                </aside>
              </div>

              {result ? (
                <Timeline
                  trace={result.trace}
                  currentStep={currentStep}
                  isPlaying={isPlaying}
                  onStepChange={changeStep}
                  onPlayingChange={setIsPlaying}
                />
              ) : null}
            </section>

            <div className="detail-grid detail-grid--single">
              <AttentionMatrix
                node={selectedNode}
                tensor={matrixTensor}
                tokens={result?.tokens ?? []}
                phase={result?.phase}
              />
            </div>

            <MemoryView
              sessionId={result?.session_id}
              memory={result?.memory}
              previousMemory={previousMemory}
              activity={result?.cache_activity}
              decodedToken={result?.decoded_token}
              isDecoding={isDecoding}
              canDecode={Boolean(result && result.tokens.length < 11)}
              onDecode={() => void handleDecode()}
            />
            {result ? (
              <ShareToolbar
                runs={[result]}
                shareUrl={shareUrl}
                captureSelector="#execution-workspace"
              />
            ) : null}
          </>
        )}
      </main>
    </div>
  );
}

function getVisualTensor(
  node: GraphNode | undefined,
  result: AttentionRun | undefined,
) {
  if (!node) {
    return undefined;
  }
  const tensorId = node.outputs[0] ?? node.inputs[0];
  return tensorId ? result?.tensors[tensorId] : undefined;
}

interface UrlConfig {
  text: string;
  architecture: AttentionArchitecture;
  mode: AppMode;
  compareArchitectures: AttentionArchitecture[];
}

function isArchitecture(value: string | null): value is AttentionArchitecture {
  return (
    value !== null && ALL_ARCHITECTURES.includes(value as AttentionArchitecture)
  );
}

function readUrlConfig(): UrlConfig {
  const params = new URLSearchParams(window.location.search);
  const architectureParam = params.get("architecture");
  const mode = params.get("mode") === "compare" ? "compare" : "workbench";
  const selected = (params.get("compare") ?? "")
    .split(",")
    .filter(isArchitecture)
    .slice(0, 4);

  return {
    text: params.get("text")?.trim() || DEFAULT_TEXT,
    architecture: isArchitecture(architectureParam) ? architectureParam : "mha",
    mode,
    compareArchitectures: selected.length >= 2 ? selected : DEFAULT_COMPARISON,
  };
}

function buildShareUrl(config: UrlConfig): string {
  const url = new URL(window.location.href);
  url.search = "";
  url.searchParams.set("mode", config.mode);
  url.searchParams.set("text", config.text);
  if (config.mode === "compare") {
    url.searchParams.set("compare", config.compareArchitectures.join(","));
  } else {
    url.searchParams.set("architecture", config.architecture);
  }
  return url.toString();
}
