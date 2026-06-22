# LLM Zoomcamp 2026 — Homework

My homework, notes, and experiments for [**DataTalksClub's LLM Zoomcamp**](https://github.com/DataTalksClub/llm-zoomcamp) — a free, 10-week course on building practical, production-ready LLM applications.

This single repository tracks my work across **all modules** of the course. Each module's solutions live in their own notebook(s) as I progress.

---

## About the course

LLM Zoomcamp teaches you how to build AI systems that answer questions over a knowledge base using modern retrieval and generation techniques — from a simple keyword-search RAG pipeline all the way to a deployed, monitored, end-to-end application.

**Prerequisites:** comfort with Python and the command line, basic Docker familiarity, and a small amount (~$1–5) of LLM API credits. No GPU and no prior ML/LLM experience required.

### Syllabus

| Module | Title | Topics | Status |
|:------:|-------|--------|:------:|
| 1 | **Agentic RAG** | Keyword search pipelines, function calling, the agentic loop | ✅ In progress |
| 2 | **Vector Search** | Semantic search with embeddings, `minsearch`, sqlitesearch, PGVector | ⬜ |
| 3 | **Orchestration** | AI workflow orchestration with Kestra | ⬜ |
| 4 | **Evaluation** | Retrieval & answer-quality metrics, offline/online evaluation | ⬜ |
| 5 | **Monitoring** | User-feedback tracking, system-health dashboards | ⬜ |
| 6 | **Best Practices** | LangChain, hybrid search, result reranking | ⬜ |
| 7 | **End-to-End Project** | Complete fitness-assistant pipeline | ⬜ |
| — | **Capstone** | Self-designed RAG application, from concept to deployment | ⬜ |

---

## Repository contents

| File | Description |
|------|-------------|
| `homework.ipynb` | **Module 1 homework** — builds a keyword-search RAG over the course's lesson Markdown: ingests pages from the `llm-zoomcamp` repo with `gitsource`, indexes them with `minsearch`, counts prompt tokens with `tiktoken`, and generates answers with OpenAI. |
| `agentic.ipynb` | **Agentic RAG experiment** — a function-calling agent (Claude via the Anthropic SDK) that runs a tool-use loop, repeatedly calling a `search` tool over a chunked course FAQ until it has enough context to answer. |
| `OpenAI Check.ipynb` | Quick scratch notebook to verify the OpenAI client and API key are wired up. |
| `main.py` | Project entry point stub. |
| `pyproject.toml` / `uv.lock` | Project metadata and locked dependencies (managed with [uv](https://github.com/astral-sh/uv)). |
| `.python-version` | Pins Python to 3.13. |

---

## Setup

This project uses [**uv**](https://github.com/astral-sh/uv) for dependency and environment management.

```bash
# 1. Clone
git clone https://github.com/nadeemramli/llm-zoomcamp-june-2026-homework.git
cd llm-zoomcamp-june-2026-homework

# 2. Install dependencies into a virtual environment (uv reads pyproject.toml + uv.lock)
uv sync

# 3. Launch Jupyter
uv run jupyter lab
```

> Requires Python **3.13**. `uv` will install it automatically if it isn't present.

### Environment variables

The notebooks load secrets from a `.env` file (via `python-dotenv`). Create one in the repo root:

```dotenv
# Used by homework.ipynb and "OpenAI Check.ipynb"
OPENAI_API_KEY=sk-...

# Used by agentic.ipynb (Anthropic / Claude)
CLAUDE_API_KEY=sk-ant-...
```

`.env` is git-ignored, so your keys stay local.

---

## Key dependencies

- [`openai`](https://pypi.org/project/openai/) — OpenAI chat & embeddings client
- [`anthropic`](https://pypi.org/project/anthropic/) — Claude client used for the agentic / tool-use loop
- [`minsearch`](https://pypi.org/project/minsearch/) — lightweight in-memory search index used throughout the course
- [`gitsource`](https://pypi.org/project/gitsource/) — pulls lesson Markdown straight from a GitHub repo/commit
- [`tiktoken`](https://pypi.org/project/tiktoken/) — token counting
- [`toyaikit`](https://pypi.org/project/toyaikit/) — course helper toolkit
- [`jupyter`](https://jupyter.org/), [`requests`](https://pypi.org/project/requests/), [`python-dotenv`](https://pypi.org/project/python-dotenv/)

---

## Links

- 📚 [LLM Zoomcamp course repo](https://github.com/DataTalksClub/llm-zoomcamp)
- 🌐 [DataTalksClub](https://datatalks.club/)

---

*This is a personal learning repository. Solutions reflect my own work for the course homework and may differ from the official answers.*
