from __future__ import annotations

from dataclasses import dataclass

from .providers import BaseProvider, ProviderResponse


@dataclass(frozen=True)
class RouteRequest:
    prompt: str
    priority: str = "balanced"
    max_latency_ms: int = 1800
    max_cost_usd: float = 0.002
    needs_reasoning: bool = False


class LLMRouter:
    def __init__(self, providers: list[BaseProvider]) -> None:
        if not providers:
            raise ValueError("At least one provider is required")
        self.providers = providers

    def choose(self, request: RouteRequest) -> BaseProvider:
        candidates = sorted(
            self.providers,
            key=lambda provider: (
                self._score(provider, request),
                provider.input_cost_per_million + provider.output_cost_per_million,
            ),
        )
        return candidates[0]

    def complete(self, request: RouteRequest) -> tuple[ProviderResponse, dict]:
        provider = self.choose(request)
        response = provider.complete(request.prompt)
        trace = {
            "selected_provider": provider.name,
            "selected_model": provider.model,
            "priority": request.priority,
            "latency_ms": response.latency_ms,
            "estimated_cost_usd": response.estimated_cost_usd,
        }
        return response, trace

    @staticmethod
    def _score(provider: BaseProvider, request: RouteRequest) -> float:
        cost = provider.input_cost_per_million + provider.output_cost_per_million
        name = f"{provider.name} {provider.model}".lower()
        if request.priority == "cost":
            return cost
        if request.priority == "latency":
            return getattr(provider, "latency_ms", 1000)
        if request.needs_reasoning and ("o3" in name or "reason" in name):
            return -1
        if "mini" in name:
            return 1
        return 2
