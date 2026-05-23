from __future__ import annotations

from dataclasses import dataclass, field

from chromadb.api.types import Metadata

from rag_pipeline.eval.settings import EvalSettings


@dataclass
class MetricScores:
    answer_relevancy: float
    contextual_precision: float
    faithfulness: float
    reasons: dict[str, str] = field(default_factory=dict)

    @property
    def average(self) -> float:
        return (
            self.contextual_precision + self.faithfulness + self.answer_relevancy
        ) / 3


@dataclass
class QuestionResult:
    question: str
    actual_answer: str
    chunks: list[tuple[str, Metadata]]
    scores: MetricScores
    expected_answer: str | None = None


@dataclass
class EvalResult:
    config: EvalSettings
    scores: MetricScores
    n_evaluated: int
    question_results: list[QuestionResult] = field(default_factory=list)
