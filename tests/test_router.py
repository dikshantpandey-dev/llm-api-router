import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_api_router.providers import LocalDeterministicProvider
from llm_api_router.router import LLMRouter, RouteRequest


class RouterTest(unittest.TestCase):
    def test_cost_priority_selects_cheapest_provider(self):
        router = LLMRouter([
            LocalDeterministicProvider("expensive", "large", 200, 1.0, 3.0),
            LocalDeterministicProvider("cheap", "mini", 100, 0.01, 0.02),
        ])
        provider = router.choose(RouteRequest("hello", priority="cost"))
        self.assertEqual(provider.name, "cheap")


if __name__ == "__main__":
    unittest.main()
