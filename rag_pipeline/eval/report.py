from __future__ import annotations

import html
from pathlib import Path

from rag_pipeline.eval.types import EvalResult, MetricScores


def _score_badge(score: float) -> str:
    if score >= 0.7:
        color = "#22c55e"
    elif score >= 0.4:
        color = "#f59e0b"
    else:
        color = "#ef4444"
    return (
        f'<span style="background:{color};color:#fff;padding:2px 7px;'
        f'border-radius:4px;font-size:0.8em;font-weight:600">{score:.3f}</span>'
    )


def _scores_row(scores: MetricScores) -> str:
    return (
        f"<td>{_score_badge(scores.contextual_precision)}</td>"
        f"<td>{_score_badge(scores.faithfulness)}</td>"
        f"<td>{_score_badge(scores.answer_relevancy)}</td>"
        f"<td>{_score_badge(scores.average)}</td>"
    )


def _render_chunk(doc: str, meta: dict) -> str:
    page = meta.get("page", "")
    page_label = (
        f'<span style="font-size:0.75em;color:#6b7280">p.{page}</span> ' if page else ""
    )
    escaped = html.escape(doc)
    return (
        f'<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;'
        f"padding:10px 14px;margin-bottom:8px;font-size:0.82em;"
        f'white-space:pre-wrap;font-family:monospace">'
        f"{page_label}{escaped}</div>"
    )


def _render_reasons(scores) -> str:
    labels = {
        "contextual_precision": "Ctx Precision",
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Ans Relevancy",
    }
    if not scores.reasons:
        return ""
    items = "".join(
        f'<div style="margin-bottom:6px">'
        f'<span style="font-weight:600;font-size:0.8em">{labels[k]}:</span> '
        f'<span style="font-size:0.85em;color:#374151">{html.escape(v)}</span>'
        f"</div>"
        for k, v in scores.reasons.items()
        if k in labels
    )
    return (
        f'<div style="background:#fafafa;border:1px solid #e2e8f0;border-radius:6px;'
        f'padding:10px 14px;margin-bottom:16px">{items}</div>'
    )


def _render_question_result(idx: int, result) -> str:
    chunks_html = "".join(_render_chunk(doc, dict(meta)) for doc, meta in result.chunks)
    reasons_html = _render_reasons(result.scores)
    return f"""
    <details style="margin-bottom:12px;
        border:1px solid #e2e8f0;border-radius:8px;overflow:hidden">
      <summary style="padding:12px 16px;cursor:pointer;background:#f1f5f9;
                      display:flex;align-items:center;gap:12px;list-style:none">
        <span style="font-weight:600;flex:1">
            Q{idx}. {html.escape(result.question)}
        </span>
        {_score_badge(result.scores.contextual_precision)}
        {_score_badge(result.scores.faithfulness)}
        {_score_badge(result.scores.answer_relevancy)}
      </summary>
      <div style="padding:16px">
        <div style="display:grid;grid-template-columns:1fr 1fr;
        gap:16px;margin-bottom:16px">
          <div>
            <div style="font-size:0.75em;font-weight:600;color:#6b7280;
                        text-transform:uppercase;margin-bottom:4px">Expected</div>
            <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:6px;
                        padding:10px 14px;font-size:0.9em">
                        {html.escape(result.expected_answer or "—")}
            </div>
          </div>
          <div>
            <div style="font-size:0.75em;font-weight:600;color:#6b7280;
                        text-transform:uppercase;margin-bottom:4px">Actual</div>
            <div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:6px;
                        padding:10px 14px;font-size:0.9em">
                        {html.escape(result.actual_answer)}
            </div>
          </div>
        </div>
        <div style="font-size:0.75em;font-weight:600;color:#6b7280;
                    text-transform:uppercase;margin-bottom:8px">
          Retrieved chunks ({len(result.chunks)})
        </div>
        {reasons_html}
        {chunks_html}
      </div>
    </details>"""


def _render_config_section(result: EvalResult) -> str:
    s = result.scores
    questions_html = "".join(
        _render_question_result(i + 1, qr)
        for i, qr in enumerate(result.question_results)
    )
    return f"""
    <section style="margin-bottom:40px">
      <h2 style="font-size:1.1em;font-weight:700;margin:0 0 12px;
                 padding-bottom:8px;border-bottom:2px solid #e2e8f0">
        {html.escape(result.config.label())}
        <span style="font-weight:400;font-size:0.85em;color:#6b7280;margin-left:8px">
          ({result.n_evaluated} questions · {result.config.chunking_strategy} chunks)
        </span>
      </h2>
      <div style="display:flex;gap:16px;margin-bottom:16px;flex-wrap:wrap">
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
                    padding:12px 20px;text-align:center">
          <div style="font-size:0.7em;color:#6b7280;text-transform:uppercase;
                      font-weight:600;margin-bottom:4px">Ctx Relevancy</div>
          {_score_badge(s.contextual_precision)}
        </div>
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
                    padding:12px 20px;text-align:center">
          <div style="font-size:0.7em;color:#6b7280;text-transform:uppercase;
                      font-weight:600;margin-bottom:4px">Faithfulness</div>
          {_score_badge(s.faithfulness)}
        </div>
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
                    padding:12px 20px;text-align:center">
          <div style="font-size:0.7em;color:#6b7280;text-transform:uppercase;
                      font-weight:600;margin-bottom:4px">Ans Relevancy</div>
          {_score_badge(s.answer_relevancy)}
        </div>
        <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
                    padding:12px 20px;text-align:center">
          <div style="font-size:0.7em;color:#6b7280;text-transform:uppercase;
                      font-weight:600;margin-bottom:4px">Average</div>
          {_score_badge(s.average)}
        </div>
      </div>
      {questions_html}
    </section>"""


def _render_summary_table(results: list[EvalResult]) -> str:
    rows = ""
    for r in results:
        s = r.scores
        rows += (
            f"<tr>"
            f"""<td style='padding:8px 12px;font-weight:600'>
            {html.escape(r.config.chunking_strategy)}
            </td>"""
            f"""<td style='padding:8px 12px;color:#6b7280'>
                {html.escape(r.config.embedding_model)}
            </td>"""
            f"<td style='padding:8px 12px'>{_score_badge(s.contextual_precision)}</td>"
            f"<td style='padding:8px 12px'>{_score_badge(s.faithfulness)}</td>"
            f"<td style='padding:8px 12px'>{_score_badge(s.answer_relevancy)}</td>"
            f"<td style='padding:8px 12px'>{_score_badge(s.average)}</td>"
            f"</tr>"
        )
    return f"""
    <table style="width:100%;border-collapse:collapse;
    margin-bottom:32px;font-size:0.9em">
      <thead>
        <tr style="background:#f1f5f9;text-align:left">
          <th style="padding:10px 12px;font-weight:600">Strategy</th>
          <th style="padding:10px 12px;font-weight:600">Embedding</th>
          <th style="padding:10px 12px;font-weight:600">Ctx Relevancy</th>
          <th style="padding:10px 12px;font-weight:600">Faithfulness</th>
          <th style="padding:10px 12px;font-weight:600">Ans Relevancy</th>
          <th style="padding:10px 12px;font-weight:600">Avg</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>"""


def write_html_report(results: list[EvalResult], game: str, output_path: Path) -> None:
    summary = _render_summary_table(results)
    sections = "".join(_render_config_section(r) for r in results)

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RAG Eval — {html.escape(game)}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box }}
    body {{ font-family: system-ui, sans-serif; margin: 0; padding: 32px;
           max-width: 1100px; margin-inline: auto; color: #1e293b; background: #fff }}
    h1 {{ font-size: 1.5em; margin: 0 0 8px }}
    details > summary::-webkit-details-marker {{ display: none }}
  </style>
</head>
<body>
  <h1>RAG Eval Report — {html.escape(game)}</h1>
  <p style="color:#6b7280;margin:0 0 24px;font-size:0.9em">
    {len(results)} config(s) · {len(results[0].question_results) if results else 0}
    question(s)
  </p>
  <h2 style="font-size:1em;font-weight:700;text-transform:uppercase;
             letter-spacing:.05em;color:#6b7280;margin:0 0 12px">Summary</h2>
  {summary}
  <h2 style="font-size:1em;font-weight:700;text-transform:uppercase;
             letter-spacing:.05em;color:#6b7280;margin:0 0 16px">Results by Config</h2>
  {sections}
</body>
</html>"""

    output_path.write_text(html_doc, encoding="utf-8")
