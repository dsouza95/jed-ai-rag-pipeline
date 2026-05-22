import asyncio
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
)
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase

from rag_pipeline.chain import chain as rag_chain
from rag_pipeline.eval.dataset import EvalDataset
from rag_pipeline.eval.settings import EvalSettings
from rag_pipeline.indexer.indexer import index_game
from rag_pipeline.retriever import build_context, retrieve
from rag_pipeline.types import RulebookPage


@dataclass
class MetricScores:
    answer_relevancy: float
    context_relevancy: float
    faithfulness: float

    @property
    def average(self) -> float:
        return (self.context_relevancy + self.faithfulness + self.answer_relevancy) / 3


@dataclass
class EvalResult:
    config: EvalSettings
    scores: MetricScores
    n_evaluated: int


def _score_test_cases(test_cases: list[Any], judge: OllamaModel) -> MetricScores:
    metric_args = {"model": judge, "async_mode": False, "threshold": 0.0}
    metrics = [
        ContextualRelevancyMetric(**metric_args),
        FaithfulnessMetric(**metric_args),
        AnswerRelevancyMetric(**metric_args),
    ]

    result = evaluate(
        test_cases,
        metrics,
        async_config=AsyncConfig(run_async=False),
        display_config=DisplayConfig(print_results=False),
    )

    name_to_key = {
        "Contextual Relevancy": "context_relevancy",
        "Faithfulness": "faithfulness",
        "Answer Relevancy": "answer_relevancy",
    }
    scores: dict[str, list[float]] = {key: [] for key in name_to_key.values()}

    for tr in result.test_results:
        for md in tr.metrics_data or []:
            key = name_to_key.get(md.name)
            if key:
                scores[key].append(md.score or 0.0)

    def avg(lst: list[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    return MetricScores(
        context_relevancy=avg(scores["context_relevancy"]),
        faithfulness=avg(scores["faithfulness"]),
        answer_relevancy=avg(scores["answer_relevancy"]),
    )


async def _build_test_cases(
    dataset: EvalDataset,
    config: EvalSettings,
    nresults: int,
) -> list[Any]:
    col_name = config.collection_name(dataset.game)
    test_cases: list[Any] = []

    for q in dataset.questions:
        chunks = retrieve(col_name, q.question, nresults, cfg=config)
        context_str = build_context(chunks)
        retrieval_context: list[Any] = [
            doc.removeprefix(str(meta.get("context", "")) + "\n\n")
            for doc, meta in chunks
        ]
        answer = await rag_chain.ainvoke(
            {"game": dataset.game, "context": context_str, "input": q.question}
        )
        test_cases.append(
            LLMTestCase(
                input=q.question,
                actual_output=answer,
                expected_output=q.expected_answer,
                retrieval_context=retrieval_context,
            )
        )

    return test_cases


def rulebook_markdown_to_pages(rulebook_markdown: str) -> list[RulebookPage]:
    segments = re.split(r"(?=<page>\d+</page>)", rulebook_markdown)
    pages: list[RulebookPage] = []
    current_page = 1
    for segment in segments:
        m = re.match(r"<page>(\d+)</page>(.*)", segment, re.DOTALL)
        if m:
            current_page = int(m.group(1))
            content = m.group(2).strip()
        else:
            content = segment.strip()
        if content:
            pages.append(RulebookPage(page_number=current_page, markdown=content))
    return pages


async def run_evaluation(
    dataset: EvalDataset,
    configs: list[EvalSettings],
    rulebook_markdown: str,
    nresults: int = 5,
) -> list[EvalResult]:
    results: list[EvalResult] = []

    for config in configs:
        judge = OllamaModel(config.eval_judge_model, timeout=300)
        col_name = config.collection_name(dataset.game)
        print(f"\n→ Config: {config.label()}")

        print(
            f"  Indexing with strategy={config.chunking_strategy}, "
            f"embedding={config.embedding_model} ..."
        )
        n_chunks = await index_game(
            dataset.game,
            rulebook_markdown_to_pages(rulebook_markdown),
            cfg=config,
            collection_name=col_name,
        )
        print(f"  Indexed {n_chunks} chunks → '{col_name}'")

        print(f"  Running {len(dataset.questions)} questions ...")
        test_cases = await _build_test_cases(dataset, config, nresults)

        print("  Scoring with DeepEval ...")
        scores = _score_test_cases(test_cases, judge)

        results.append(
            EvalResult(config=config, scores=scores, n_evaluated=len(test_cases))
        )

        print(
            f"  context_relevancy={scores.context_relevancy:.3f}  "
            f"faithfulness={scores.faithfulness:.3f}  "
            f"answer_relevancy={scores.answer_relevancy:.3f}  "
            f"avg={scores.average:.3f}"
        )

    return results


def print_comparison_table(results: list[EvalResult]) -> None:
    col_w = [30, 22, 19, 13, 17, 7]
    headers = [
        "Strategy",
        "Embedding",
        "Ctx Relevancy",
        "Faithfulness",
        "Ans Relevancy",
        "Avg",
    ]

    def row_str(cells: list[str]) -> str:
        return " | ".join(c.ljust(w) for c, w in zip(cells, col_w))

    sep = "-+-".join("-" * w for w in col_w)
    print()
    print(row_str(headers))
    print(sep)
    for r in results:
        s = r.scores
        print(
            row_str(
                [
                    r.config.chunking_strategy,
                    r.config.embedding_model,
                    f"{s.context_relevancy:.3f}",
                    f"{s.faithfulness:.3f}",
                    f"{s.answer_relevancy:.3f}",
                    f"{s.average:.3f}",
                ]
            )
        )
    print()


if __name__ == "__main__":
    CONFIGS = [
        EvalSettings(chunking_strategy="fixed_size"),
        EvalSettings(chunking_strategy="hierarchical"),
        EvalSettings(chunking_strategy="page"),
    ]

    dataset = EvalDataset.from_file(Path("evaldataset.json"))
    rulebook_markdown = Path("eval_rulebook.md").read_text()

    print(
        f"\nEvaluating '{dataset.game}' — "
        f"{len(CONFIGS)} config(s) × {len(dataset.questions)} question(s)"
    )

    results = asyncio.run(
        run_evaluation(
            dataset=dataset, configs=CONFIGS, rulebook_markdown=rulebook_markdown
        )
    )

    print("\n=== Comparison Table ===")
    print_comparison_table(results)
