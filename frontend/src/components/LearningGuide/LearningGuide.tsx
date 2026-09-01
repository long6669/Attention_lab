import { BookOpen, Braces, Database, Workflow } from "lucide-react";
import { useEffect, useState } from "react";

import { ARCHITECTURE_LESSONS } from "../../content/architectures";
import type { AttentionArchitecture } from "../../types/attention";

type LessonStage = "concept" | "formula" | "execution" | "memory";

const STAGES: Array<{
  id: LessonStage;
  label: string;
  icon: typeof BookOpen;
}> = [
  { id: "concept", label: "Concept", icon: BookOpen },
  { id: "formula", label: "Formula", icon: Braces },
  { id: "execution", label: "Execution", icon: Workflow },
  { id: "memory", label: "Memory", icon: Database },
];

interface LearningGuideProps {
  architecture: AttentionArchitecture;
}

export function LearningGuide({ architecture }: LearningGuideProps) {
  const [stage, setStage] = useState<LessonStage>("concept");
  const lesson = ARCHITECTURE_LESSONS[architecture];

  useEffect(() => {
    setStage("concept");
  }, [architecture]);

  return (
    <section className="learning-guide">
      <div className="learning-guide__title">
        <span className="eyebrow">ARCHITECTURE LESSON</span>
        <strong>{lesson.shortName}</strong>
      </div>
      <div className="learning-guide__stages" role="tablist">
        {STAGES.map(({ id, label, icon: Icon }) => (
          <button
            type="button"
            role="tab"
            aria-selected={stage === id}
            className={stage === id ? "is-active" : ""}
            onClick={() => setStage(id)}
            key={id}
          >
            <Icon size={14} />
            {label}
          </button>
        ))}
      </div>
      <div className={`learning-guide__content is-${stage}`}>
        <span>{STAGES.find((item) => item.id === stage)?.label}</span>
        {stage === "formula" ? (
          <code>{lesson[stage]}</code>
        ) : (
          <p>{lesson[stage]}</p>
        )}
      </div>
    </section>
  );
}
