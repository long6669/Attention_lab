import {
  ChevronLeft,
  ChevronRight,
  Pause,
  Play,
  RotateCcw,
} from "lucide-react";

import type { TraceEvent } from "../../types/attention";

interface TimelineProps {
  trace: TraceEvent[];
  currentStep: number;
  isPlaying: boolean;
  onStepChange: (step: number) => void;
  onPlayingChange: (playing: boolean) => void;
}

export function Timeline({
  trace,
  currentStep,
  isPlaying,
  onStepChange,
  onPlayingChange,
}: TimelineProps) {
  const lastStep = Math.max(trace.length - 1, 0);
  const event = trace[currentStep];

  return (
    <section className="timeline" aria-label="Execution timeline">
      <div className="timeline__event">
        <span className="timeline__step">
          STEP {String(currentStep + 1).padStart(2, "0")} /{" "}
          {String(trace.length).padStart(2, "0")}
        </span>
        <strong>{event?.title ?? "No execution"}</strong>
        <span className="timeline__op">{event?.op ?? "idle"}</span>
      </div>

      <input
        className="timeline__range"
        type="range"
        min={0}
        max={lastStep}
        value={Math.min(currentStep, lastStep)}
        onChange={(event) => onStepChange(Number(event.target.value))}
        aria-label="Execution step"
      />

      <div className="timeline__controls">
        <button
          className="icon-button"
          type="button"
          onClick={() => onStepChange(0)}
          disabled={currentStep === 0}
          title="Restart timeline"
          aria-label="Restart timeline"
        >
          <RotateCcw size={17} />
        </button>
        <button
          className="icon-button"
          type="button"
          onClick={() => onStepChange(Math.max(0, currentStep - 1))}
          disabled={currentStep === 0}
          title="Previous step"
          aria-label="Previous step"
        >
          <ChevronLeft size={19} />
        </button>
        <button
          className="icon-button icon-button--primary"
          type="button"
          onClick={() => onPlayingChange(!isPlaying)}
          title={isPlaying ? "Pause timeline" : "Play timeline"}
          aria-label={isPlaying ? "Pause timeline" : "Play timeline"}
        >
          {isPlaying ? <Pause size={18} /> : <Play size={18} />}
        </button>
        <button
          className="icon-button"
          type="button"
          onClick={() => onStepChange(Math.min(lastStep, currentStep + 1))}
          disabled={currentStep >= lastStep}
          title="Next step"
          aria-label="Next step"
        >
          <ChevronRight size={19} />
        </button>
      </div>
    </section>
  );
}
