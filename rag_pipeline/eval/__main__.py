import asyncio
import os
from pathlib import Path

os.environ.setdefault("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE", "600")

from rag_pipeline.eval.dataset import EvalDataset
from rag_pipeline.eval.report import write_html_report
from rag_pipeline.eval.runner import print_comparison_table, run_evaluation
from rag_pipeline.eval.settings import EvalSettings

CONFIGS = [
    EvalSettings(chunking_strategy="fixed_size"),
    EvalSettings(chunking_strategy="hierarchical"),
    EvalSettings(chunking_strategy="page"),
]

dataset = EvalDataset.from_file(Path("eval_dataset.json"))
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

report_path = Path("eval_report.html")
write_html_report(results, dataset.game, report_path)
print(f"Report written to {report_path}")
