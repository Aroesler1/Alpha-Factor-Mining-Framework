"""Long-term reflection memory: the loop's only carried state besides the archive.

ReEvo's contribution over a plain evolutionary loop is that the model is asked,
each round, what it learned -- and that summary is fed back into the next
round's proposals. Their ablation (Table 5) puts the gain from long-term
reflection in the low single digits of percent, so this is included with a small
expectation and a measurement rather than as an act of faith: the feedback
ablation arm in Part C runs the same loop without it.

The memory is capped and FIFO. An unbounded memory would grow past the useful
context and would also let a rule written in round 1, on an archive that no
longer exists, keep steering round 8.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ReflectionMemory:
    """At most `capacity` rules, oldest dropped first."""

    capacity: int = 12
    rules: list[str] = field(default_factory=list)

    def add(self, new_rules: list[str]) -> list[str]:
        """Append, dedupe on exact text, and trim to capacity. Returns what was dropped."""
        dropped: list[str] = []
        for rule in new_rules:
            cleaned = " ".join(str(rule).split())
            if not cleaned or cleaned in self.rules:
                continue
            self.rules.append(cleaned)
        while len(self.rules) > self.capacity:
            dropped.append(self.rules.pop(0))
        return dropped

    def render(self) -> str:
        """The block that goes into EXPLORE and MUTATE prompts."""
        if not self.rules:
            return ("Working memory: empty. This is the first round on this island, "
                    "so there is nothing learned yet to apply.")
        body = "\n".join(f"  {i + 1}. {rule}" for i, rule in enumerate(self.rules))
        return (
            "Working memory: rules written after earlier rounds on this island, "
            "oldest first. Apply them.\n" + body
        )

    def to_dict(self) -> dict:
        return {"capacity": self.capacity, "rules": list(self.rules)}

    @classmethod
    def from_dict(cls, payload: dict) -> "ReflectionMemory":
        return cls(capacity=int(payload.get("capacity", 12)),
                   rules=list(payload.get("rules", [])))

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path, capacity: int = 12) -> "ReflectionMemory":
        path = Path(path)
        if not path.exists():
            return cls(capacity=capacity)
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
