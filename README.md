# LLM API Router

A production-style router for choosing between local, Azure OpenAI, and OpenAI-compatible LLM providers based on cost, latency, and task requirements.

## Features

- Provider interface with deterministic local fallback
- Azure OpenAI adapter using environment variables
- Cost estimation
- Routing traces as JSON and CSV
- CI-safe demo that does not need cloud credentials

## Quickstart

```bash
python scripts/run_demo.py
python -m unittest discover -s tests
```

## Azure deployments

Set `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, and `AZURE_OPENAI_CHAT_DEPLOYMENT`. Example deployment names that fit this router: `gpt-4o`, `gpt-4o-mini`, `gpt-5-mini`, or `o4-mini`.
