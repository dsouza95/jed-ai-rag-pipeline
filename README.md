# RAG Pipeline

A from-scratch implementation of a Retrieval-Augmented Generation (RAG) pipeline, built as part of the Factored Jed-AI mission series. The goal is to understand the core mechanics of RAG — document ingestion, chunking, embedding, retrieval, and generation — without relying on high-level abstractions that obscure what's happening under the hood.

## Stack

- **[LangChain](https://python.langchain.com/)** — chain orchestration
- **[Ollama](https://ollama.com/)** — local LLM inference
- **[Chainlit](https://chainlit.io/)** — chat UI

## Getting started

Requires [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com/) running locally.

```bash
uv sync
uv run dev        # starts the Chainlit dev server
```

## Quality checks

[Ruff](https://docs.astral.sh/ruff/) (lint + format) and [basedpyright](https://docs.basedpyright.com/) (type checking) run automatically on every commit via [pre-commit](https://pre-commit.com/).

Install the hooks after cloning:

```bash
uv sync
uv run pre-commit install
```

Run the checks manually at any time:

```bash
uv run ruff check .       # lint
uv run ruff format .      # format
uv run basedpyright       # type check
```
