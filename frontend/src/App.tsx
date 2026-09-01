import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Activity, AlertCircle, Cpu, LoaderCircle } from "lucide-react";

import { decodeOneToken, runAttention } from "./api/attention";
import { AttentionMatrix } from "./components/AttentionMatrix/AttentionMatrix";
import { GraphView } from "./components/GraphView/GraphView";
import { MemoryView } from "./components/MemoryView/MemoryView";
import { NodeCalculationInspector } from "./components/NodeCalculationInspector/NodeCalculationInspector";
import { Timeline } from "./components/Timeline/Timeline";
import type {
  AttentionArchitecture,
  AttentionRun,
  GraphNode,
} from "./types/attention";

const DEFAULT_TEXT = "I love learning how attention works today";
const ARCHITECTURE_LABELS: Record<AttentionArchitecture, string> = {
  mha: "Multi-Head Attention",
  mqa: "Multi-Query Attention",
  gqa: "Grouped-Query Attention",
  rope: "MHA with RoPE",
  mla: "MLA Concept Model",
  kda: "KDA Concept Model",
  csa: "CSA Concept Model",
  hca: "HCA Concept Model",
};

export default function App() {
  const [text, setText] = useState(DEFAULT_TEXT);
  const [architecture, setArchitecture] =
    useState<AttentionArchitecture>("mha");
  const [result, setResult] = useState<AttentionRun>();
  const [currentStep, setCurrentStep] = useState(0);
  const [selectedNodeId, setSelectedNodeId] = useState<string>();
  const [isPlaying, setIsPlaying] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
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

  useEffect(() => {
    if (didInitialize.current) {
      return;
    }
    didInitialize.current = true;
    void execute(DEFAULT_TEXT, "mha");
  }, [execute]);

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
    void execute(text, architecture);
  }

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
        <div className="runtime-status">
          <span className="status-dot" />
          NumPy Runtime
        </div>
      </header>

      <main>
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
          <button className="run-button" type="submit" disabled={isLoading}>
            {isLoading ? (
              <LoaderCircle className="spin" size={17} />
            ) : (
              <Cpu size={17} />
            )}
            Run
          </button>
        </form>

        {error ? (
          <div className="message message--error" role="alert">
            <AlertCircle size={17} />
            {error}
          </div>
        ) : null}
        {result?.warnings.map((warning) => (
          <div className="message message--warning" role="status" key={warning}>
            <AlertCircle size={17} />
            {warning}
          </div>
        ))}

        <section className="workspace">
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
                <span className={`phase-badge phase-badge--${result.phase}`}>
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
