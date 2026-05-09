from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderResponse:
    provider: str
    model: str
    text: str
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float


class BaseProvider:
    name: str
    model: str
    input_cost_per_million: float
    output_cost_per_million: float

    def complete(self, prompt: str) -> ProviderResponse:
        raise NotImplementedError

    def estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return round(
            (prompt_tokens / 1_000_000) * self.input_cost_per_million
            + (completion_tokens / 1_000_000) * self.output_cost_per_million,
            6,
        )


class LocalDeterministicProvider(BaseProvider):
    def __init__(self, name: str, model: str, latency_ms: int, input_cost: float, output_cost: float) -> None:
        self.name = name
        self.model = model
        self.latency_ms = latency_ms
        self.input_cost_per_million = input_cost
        self.output_cost_per_million = output_cost

    def complete(self, prompt: str) -> ProviderResponse:
        start = time.perf_counter()
        words = prompt.split()
        text = " ".join(words[:42])
        if len(words) > 42:
            text += " ..."
        text = f"[{self.model}] {text}"
        elapsed = int((time.perf_counter() - start) * 1000) + self.latency_ms
        prompt_tokens = max(1, len(words))
        completion_tokens = max(8, len(text.split()))
        return ProviderResponse(
            self.name,
            self.model,
            text,
            elapsed,
            prompt_tokens,
            completion_tokens,
            self.estimate_cost(prompt_tokens, completion_tokens),
        )


class AzureOpenAIProvider(BaseProvider):
    def __init__(self, deployment: str | None = None) -> None:
        self.name = "azure-openai"
        self.model = deployment or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4o-mini")
        self.input_cost_per_million = 0.15
        self.output_cost_per_million = 0.60
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

    @property
    def available(self) -> bool:
        return bool(self.endpoint and self.api_key)

    def complete(self, prompt: str) -> ProviderResponse:
        if not self.available:
            raise RuntimeError("Azure provider is not configured. Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY.")
        start = time.perf_counter()
        url = f"{self.endpoint}/openai/deployments/{self.model}/chat/completions?api-version={self.api_version}"
        payload = json.dumps({"messages": [{"role": "user", "content": prompt}], "temperature": 0.2}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, method="POST")
        req.add_header("api-key", self.api_key)
        req.add_header("content-type", "application/json")
        with urllib.request.urlopen(req, timeout=45) as response:
            data = json.loads(response.read().decode("utf-8"))
        usage = data.get("usage", {})
        text = data["choices"][0]["message"]["content"]
        prompt_tokens = int(usage.get("prompt_tokens", len(prompt.split())))
        completion_tokens = int(usage.get("completion_tokens", len(text.split())))
        return ProviderResponse(
            self.name,
            self.model,
            text,
            int((time.perf_counter() - start) * 1000),
            prompt_tokens,
            completion_tokens,
            self.estimate_cost(prompt_tokens, completion_tokens),
        )
