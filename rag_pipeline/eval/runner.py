import re

from chromadb.api.types import Metadata
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
from rag_pipeline.eval.types import EvalResult, MetricScores, QuestionResult
from rag_pipeline.indexer.indexer import index_game
from rag_pipeline.retriever import build_context, retrieve
from rag_pipeline.types import RulebookPage


def _score_test_cases(
    test_cases: list[LLMTestCase], judge: OllamaModel
) -> list[MetricScores]:
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

    per_question: list[MetricScores] = []
    for tr in result.test_results:
        q_scores: dict[str, float] = {}
        for md in tr.metrics_data or []:
            key = name_to_key.get(md.name)
            if key:
                q_scores[key] = md.score or 0.0

        per_question.append(
            MetricScores(
                context_relevancy=q_scores.get("context_relevancy", 0.0),
                faithfulness=q_scores.get("faithfulness", 0.0),
                answer_relevancy=q_scores.get("answer_relevancy", 0.0),
            )
        )

    return per_question


async def _build_test_cases(
    dataset: EvalDataset,
    config: EvalSettings,
    nresults: int,
) -> list[tuple[LLMTestCase, list[tuple[str, Metadata]]]]:
    col_name = config.collection_name(dataset.game)
    results: list[tuple[LLMTestCase, list[tuple[str, Metadata]]]] = []

    for q in dataset.questions:
        chunks = retrieve(col_name, q.question, nresults, cfg=config)
        context_str = build_context(chunks)
        answer = await rag_chain.ainvoke(
            {"game": dataset.game, "context": context_str, "input": q.question}
        )
        test_case = LLMTestCase(
            input=q.question,
            actual_output=answer,
            retrieval_context=[doc for doc, _ in chunks],
        )
        results.append((test_case, chunks))

    return results


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
        test_case_pairs = await _build_test_cases(dataset, config, nresults)
        test_cases = [tc for tc, _ in test_case_pairs]

        print("  Scoring with DeepEval ...")
        per_question_scores = _score_test_cases(test_cases, judge)

        def avg(vals: list[float]) -> float:
            return sum(vals) / len(vals) if vals else 0.0

        aggregate = MetricScores(
            context_relevancy=avg([s.context_relevancy for s in per_question_scores]),
            faithfulness=avg([s.faithfulness for s in per_question_scores]),
            answer_relevancy=avg([s.answer_relevancy for s in per_question_scores]),
        )

        question_results = [
            QuestionResult(
                question=q.question,
                actual_answer=tc.actual_output or "",
                chunks=chunks,
                scores=q_scores,
                expected_answer=q.expected_answer,
            )
            for (tc, chunks), q_scores, q in zip(
                test_case_pairs, per_question_scores, dataset.questions
            )
        ]

        results.append(
            EvalResult(
                config=config,
                scores=aggregate,
                n_evaluated=len(test_cases),
                question_results=question_results,
            )
        )

        print(
            f"  context_relevancy={aggregate.context_relevancy:.3f}  "
            f"faithfulness={aggregate.faithfulness:.3f}  "
            f"answer_relevancy={aggregate.answer_relevancy:.3f}  "
            f"avg={aggregate.average:.3f}"
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
