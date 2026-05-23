import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalQuestion:
    question: str
    expected_answer: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass
class EvalDataset:
    game: str
    questions: list[EvalQuestion]

    @classmethod
    def from_file(cls, path: Path) -> "EvalDataset":
        data = json.loads(path.read_text())
        return cls(
            game=data["game"],
            questions=[
                EvalQuestion(
                    question=q["question"],
                    expected_answer=q.get("expected_answer"),
                    tags=q.get("tags", []),
                )
                for q in data["questions"]
            ],
        )
