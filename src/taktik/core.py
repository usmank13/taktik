"""
Core Taktik orchestrator.

Usage:
    from taktik import Taktik
    
    tk = Taktik(llm=my_llm_fn)
    result = tk.run("What is 15% of 240?")
    print(result.answer, result.technique_used, result.confidence)
"""

from typing import Optional

from taktik.types import TechniqueResult, LLMFunction
from taktik.techniques.base import Technique
from taktik.router.base import Router


class Taktik:
    """
    Main orchestrator. Takes a query, routes to the best technique, returns result.
    
    Args:
        llm: A callable that takes a prompt string and returns a completion.
        router: Strategy router (defaults to rule-based).
        techniques: Dict of technique name → Technique instance.
            If None, uses the default technique library.
    """

    def __init__(
        self,
        llm: LLMFunction,
        router: Optional[Router] = None,
        techniques: Optional[dict[str, Technique]] = None,
    ):
        self.llm = llm
        self.router = router or _default_router()
        self.techniques = techniques or _default_techniques()

    def run(
        self,
        query: str,
        technique: Optional[str] = None,
        **kwargs,
    ) -> TechniqueResult:
        """
        Run a query through the selected technique.
        
        Args:
            query: The user's question/task.
            technique: Force a specific technique (bypasses router).
            **kwargs: Passed to the technique's run method.
        
        Returns:
            TechniqueResult with answer, technique used, confidence, etc.
        """
        if technique:
            tech_name = technique
        else:
            tech_name = self.router.select(query, list(self.techniques.keys()))

        tech = self.techniques.get(tech_name)
        if tech is None:
            raise ValueError(f"Unknown technique: {tech_name}. Available: {list(self.techniques.keys())}")

        return tech.run(query, self.llm, **kwargs)

    def run_compare(
        self,
        query: str,
        techniques: Optional[list[str]] = None,
    ) -> list[TechniqueResult]:
        """
        Run multiple techniques on the same query for comparison.
        
        Args:
            query: The user's question.
            techniques: List of technique names to try. If None, uses all.
        
        Returns:
            List of TechniqueResult, one per technique.
        """
        names = techniques or list(self.techniques.keys())
        results = []
        for name in names:
            try:
                result = self.run(query, technique=name)
                results.append(result)
            except Exception as e:
                results.append(TechniqueResult(
                    answer="",
                    technique_used=name,
                    confidence=0.0,
                    metadata={"error": str(e)},
                ))
        return results

    def list_techniques(self) -> list[str]:
        """List available technique names."""
        return list(self.techniques.keys())


def _default_router() -> Router:
    from taktik.router.rule_based import RuleBasedRouter
    return RuleBasedRouter()


def _default_techniques() -> dict[str, Technique]:
    from taktik.techniques.direct import Direct
    from taktik.techniques.cot import ChainOfThought
    from taktik.techniques.self_consistency import SelfConsistency
    from taktik.techniques.pot import ProgramOfThought
    from taktik.techniques.self_refine import SelfRefine
    from taktik.techniques.decompose import Decompose
    return {
        "direct": Direct(),
        "cot": ChainOfThought(),
        "self_consistency": SelfConsistency(),
        "pot": ProgramOfThought(),
        "self_refine": SelfRefine(),
        "decompose": Decompose(),
    }
