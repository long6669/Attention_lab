import { ArrowUpRight, Database, GitCompareArrows, Sigma } from "lucide-react";

import {
  ARCHITECTURE_LABELS,
  ARCHITECTURE_LESSONS,
} from "../../content/architectures";
import type {
  AttentionArchitecture,
  AttentionRun,
} from "../../types/attention";

interface CompareViewProps {
  results: AttentionRun[];
  onOpen: (architecture: AttentionArchitecture) => void;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  return `${(bytes / 1024).toFixed(2)} KB`;
}

function formatCount(value: number): string {
  if (value < 1000) {
    return value.toLocaleString();
  }
  return `${(value / 1000).toFixed(1)}K`;
}

function delta(value: number, baseline: number): string {
  if (value === baseline) {
    return "baseline";
  }
  const percent = baseline === 0 ? 0 : ((value - baseline) / baseline) * 100;
  return `${percent > 0 ? "+" : ""}${percent.toFixed(0)}% vs first`;
}

export function CompareView({ results, onOpen }: CompareViewProps) {
  if (results.length === 0) {
    return null;
  }
  const baseline = results[0];
  const maxMemory = Math.max(
    ...results.map((result) => result.memory.total_bytes),
    1,
  );

  return (
    <section className="compare-view" id="comparison-view">
      <div className="compare-view__heading">
        <div>
          <span className="eyebrow">SIDE-BY-SIDE EXECUTION</span>
          <h2>Architecture Compare</h2>
        </div>
        <span className="estimate-note">
          <Sigma size={14} />
          FLOPs are shape-based estimates
        </span>
      </div>

      <div
        className="compare-grid"
        style={{
          gridTemplateColumns: `repeat(${results.length}, minmax(0, 1fr))`,
        }}
      >
        {results.map((result) => {
          const architecture = result.config.architecture;
          const lesson = ARCHITECTURE_LESSONS[architecture];
          const operations = result.graph.nodes.map((node) => node.op);
          const memoryPercent = (result.memory.total_bytes / maxMemory) * 100;

          return (
            <article className="compare-card" key={architecture}>
              <div className="compare-card__header">
                <div>
                  <span>{lesson.shortName}</span>
                  <h3>{ARCHITECTURE_LABELS[architecture]}</h3>
                </div>
                <button
                  type="button"
                  title={`Inspect ${lesson.shortName}`}
                  onClick={() => onOpen(architecture)}
                >
                  <ArrowUpRight size={16} />
                </button>
              </div>

              <p className="compare-card__concept">{lesson.concept}</p>

              <dl className="compare-metrics">
                <div>
                  <dt>Persistent memory</dt>
                  <dd>{formatBytes(result.memory.total_bytes)}</dd>
                  <small>
                    {delta(
                      result.memory.total_bytes,
                      baseline.memory.total_bytes,
                    )}
                  </small>
                </div>
                <div>
                  <dt>Estimated FLOPs</dt>
                  <dd>{formatCount(result.metrics.estimated_flops)}</dd>
                  <small>
                    {delta(
                      result.metrics.estimated_flops,
                      baseline.metrics.estimated_flops,
                    )}
                  </small>
                </div>
                <div>
                  <dt>Graph</dt>
                  <dd>{result.metrics.graph_nodes} nodes</dd>
                  <small>{result.metrics.trace_steps} trace steps</small>
                </div>
                <div>
                  <dt>Growth</dt>
                  <dd>{result.metrics.memory_growth}</dd>
                  <small>
                    {result.metrics.memory_growth === "constant"
                      ? "fixed state"
                      : `${formatBytes(
                          result.metrics.memory_bytes_per_token,
                        )} / token`}
                  </small>
                </div>
              </dl>

              <div className="memory-comparison">
                <div>
                  <Database size={13} />
                  <span>{result.memory.spec.kind}</span>
                </div>
                <div className="memory-comparison__track">
                  <span style={{ width: `${memoryPercent}%` }} />
                </div>
              </div>

              <div className="compare-graph">
                <div>
                  <GitCompareArrows size={13} />
                  <span>Executed graph</span>
                </div>
                <div className="compare-graph__ops">
                  {operations.map((operation, index) => (
                    <span key={`${operation}-${index}`}>{operation}</span>
                  ))}
                </div>
              </div>

              <div className="compare-card__memory">
                <strong>State difference</strong>
                <p>{lesson.memory}</p>
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}
