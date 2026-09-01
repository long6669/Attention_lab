import { Check, Copy, FileJson, ImageDown, ListTree } from "lucide-react";
import { toPng } from "html-to-image";
import { useEffect, useState } from "react";

import type { AttentionRun } from "../../types/attention";

interface ShareToolbarProps {
  runs: AttentionRun[];
  shareUrl: string;
  captureSelector: string;
}

function downloadBlob(filename: string, blob: Blob) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

function downloadJson(filename: string, value: unknown) {
  downloadBlob(
    filename,
    new Blob([JSON.stringify(value, null, 2)], {
      type: "application/json",
    }),
  );
}

export function ShareToolbar({
  runs,
  shareUrl,
  captureSelector,
}: ShareToolbarProps) {
  const [copied, setCopied] = useState(false);
  const [exportingImage, setExportingImage] = useState(false);

  useEffect(() => {
    if (!copied) {
      return;
    }
    const timer = window.setTimeout(() => setCopied(false), 1800);
    return () => window.clearTimeout(timer);
  }, [copied]);

  async function copyLink() {
    await navigator.clipboard.writeText(shareUrl);
    window.history.replaceState({}, "", shareUrl);
    setCopied(true);
  }

  async function exportPng() {
    const target = document.querySelector<HTMLElement>(captureSelector);
    if (!target) {
      return;
    }
    setExportingImage(true);
    try {
      const dataUrl = await toPng(target, {
        backgroundColor: "#eef1f2",
        pixelRatio: 2,
      });
      const anchor = document.createElement("a");
      anchor.href = dataUrl;
      anchor.download = "attnlab-graph.png";
      anchor.click();
    } finally {
      setExportingImage(false);
    }
  }

  const graphPayload =
    runs.length === 1
      ? runs[0]?.graph
      : Object.fromEntries(
          runs.map((run) => [run.config.architecture, run.graph]),
        );
  const tracePayload =
    runs.length === 1
      ? runs[0]?.trace
      : Object.fromEntries(
          runs.map((run) => [run.config.architecture, run.trace]),
        );

  return (
    <div className="share-toolbar" aria-label="Share and export experiment">
      <button type="button" onClick={() => void copyLink()}>
        {copied ? <Check size={15} /> : <Copy size={15} />}
        {copied ? "Copied" : "Copy link"}
      </button>
      <button
        type="button"
        onClick={() => downloadJson("attnlab-graph.json", graphPayload)}
      >
        <FileJson size={15} />
        Graph JSON
      </button>
      <button
        type="button"
        onClick={() => downloadJson("attnlab-trace.json", tracePayload)}
      >
        <ListTree size={15} />
        Trace JSON
      </button>
      <button
        type="button"
        disabled={exportingImage}
        onClick={() => void exportPng()}
      >
        <ImageDown size={15} />
        {exportingImage ? "Rendering" : "Graph PNG"}
      </button>
    </div>
  );
}
