import { ArrowRight, BookMarked } from "lucide-react";

import { EXAMPLES, type ExampleDefinition } from "../../content/architectures";

interface ExamplesGalleryProps {
  onSelect: (example: ExampleDefinition) => void;
}

export function ExamplesGallery({ onSelect }: ExamplesGalleryProps) {
  return (
    <section className="examples-gallery">
      <div className="examples-gallery__heading">
        <BookMarked size={16} />
        <div>
          <span className="eyebrow">EXAMPLES GALLERY</span>
          <strong>Start from a question</strong>
        </div>
      </div>
      <div className="examples-gallery__list">
        {EXAMPLES.map((example) => (
          <button
            type="button"
            onClick={() => onSelect(example)}
            key={example.id}
          >
            <span>
              <strong>{example.title}</strong>
              <small>{example.question}</small>
            </span>
            <ArrowRight size={15} />
          </button>
        ))}
      </div>
    </section>
  );
}
