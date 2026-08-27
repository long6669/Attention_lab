from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class TraceEvent:
    step: int
    node_id: str
    op: str
    inputs: list[str]
    outputs: list[str]
    title: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TraceRecorder:
    events: list[TraceEvent] = field(default_factory=list)

    def record(
        self,
        node_id: str,
        op: str,
        inputs: list[str],
        outputs: list[str],
        title: str,
    ) -> TraceEvent:
        event = TraceEvent(
            step=len(self.events),
            node_id=node_id,
            op=op,
            inputs=list(inputs),
            outputs=list(outputs),
            title=title,
        )
        self.events.append(event)
        return event

    def to_list(self) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self.events]
