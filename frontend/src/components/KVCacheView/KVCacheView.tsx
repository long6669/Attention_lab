import {
  ArrowRight,
  Database,
  HardDrive,
  LoaderCircle,
  Plus,
} from "lucide-react";

import type { AttentionRun, CacheSummary } from "../../types/attention";

type Memory = AttentionRun["memory"];
type CacheActivity = AttentionRun["cache_activity"];

interface KVCacheViewProps {
  memory?: Memory;
  previousMemory?: Memory;
  activity?: CacheActivity;
  decodedToken?: string;
  isDecoding: boolean;
  canDecode: boolean;
  onDecode: () => void;
}

function formatBytes(bytes: number): string {
  return bytes < 1024 ? `${bytes} B` : `${(bytes / 1024).toFixed(2)} KB`;
}

function TransitionValue({
  previous,
  current,
  suffix = "",
}: {
  previous?: number;
  current: number;
  suffix?: string;
}) {
  return (
    <span className={previous !== undefined ? "cache-value is-updated" : "cache-value"}>
      {previous !== undefined ? <del>{previous}{suffix}</del> : null}
      {previous !== undefined ? <span aria-hidden="true">→</span> : null}
      <strong>{current}{suffix}</strong>
    </span>
  );
}

export function KVCacheView({
  memory,
  previousMemory,
  activity,
  decodedToken,
  isDecoding,
  canDecode,
  onDecode,
}: KVCacheViewProps) {
  if (!memory) {
    return null;
  }

  const previousRatio = previousMemory
    ? (previousMemory.tokens / memory.tokens) * 100
    : 100;
  const cacheEntries: Array<[string, CacheSummary]> =
    memory.cache_kind === "latent" && memory.latent_cache
      ? [["Latent", memory.latent_cache]]
      : [
          ["K", memory.k_cache],
          ["V", memory.v_cache],
        ].filter(
          (entry): entry is [string, CacheSummary] =>
            entry[1] !== undefined,
        );
  const cacheTitle =
    memory.cache_kind === "latent" ? "Latent Cache" : "KV Cache";

  return (
    <section className="cache-panel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">INFERENCE MEMORY</span>
          <h2>{cacheTitle}</h2>
        </div>
        <div className="cache-phase">
          <span>{activity?.phase ?? "prefill"}</span>
          <HardDrive size={18} aria-hidden="true" />
        </div>
      </div>

      <div className="cache-summary">
        <div>
          <span>Tokens</span>
          <TransitionValue
            previous={previousMemory?.tokens}
            current={memory.tokens}
          />
        </div>
        <div>
          <span>Total values</span>
          <TransitionValue
            previous={previousMemory?.total_elements}
            current={memory.total_elements}
          />
        </div>
        <div>
          <span>Memory</span>
          <span className={previousMemory ? "cache-value is-updated" : "cache-value"}>
            {previousMemory ? <del>{formatBytes(previousMemory.total_bytes)}</del> : null}
            {previousMemory ? <span aria-hidden="true">→</span> : null}
            <strong>{formatBytes(memory.total_bytes)}</strong>
          </span>
        </div>
      </div>

      <div
        className={`cache-detail ${
          cacheEntries.length === 1 ? "cache-detail--single" : ""
        }`}
      >
        {cacheEntries.map(([label, cache]) => (
          <div className="cache-row" key={label}>
            <Database size={17} aria-hidden="true" />
            <strong>{label} Cache</strong>
            <code>[{cache.shape.join(", ")}]</code>
            <span>{cache.elements} values</span>
            <span>{formatBytes(cache.bytes)}</span>
          </div>
        ))}
      </div>

      <div className="cache-growth" aria-label="KV cache token growth">
        <div
          className="cache-growth__existing"
          style={{ width: `${previousRatio}%` }}
        />
        {previousMemory ? <div className="cache-growth__new" /> : null}
      </div>

      <div className="cache-activity">
        <div className={activity?.phase === "decode" ? "is-active" : ""}>
          <Database size={16} aria-hidden="true" />
          <span>Cache Read</span>
          <strong>{activity?.read_tokens ?? 0} tokens</strong>
        </div>
        <ArrowRight size={17} aria-hidden="true" />
        <div className="is-active">
          <Plus size={16} aria-hidden="true" />
          <span>Cache Append</span>
          <strong>+{activity?.appended_tokens ?? memory.tokens} tokens</strong>
        </div>
        <ArrowRight size={17} aria-hidden="true" />
        <div>
          <HardDrive size={16} aria-hidden="true" />
          <span>Cache Size</span>
          <strong>{formatBytes(memory.total_bytes)}</strong>
        </div>
      </div>

      <div className="cache-actions">
        <div className="decode-status">
          {decodedToken ? (
            <>
              Read historical {cacheTitle}, then appended{" "}
              <code>{decodedToken}</code>
            </>
          ) : (
            `${cacheTitle} prefill ready`
          )}
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
