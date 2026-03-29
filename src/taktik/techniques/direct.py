"""Direct prompting — zero-shot baseline."""

from taktik.techniques.base import Technique
from taktik.types import TechniqueResult, LLMFunction


class Direct(Technique):
    """Zero-shot direct prompting. Just ask the question."""
    
    name = "direct"
    description = "Direct zero-shot prompting (baseline)"
    typical_llm_calls = 1

    def run(self, query: str, llm: LLMFunction, **kwargs) -> TechniqueResult:
        answer = llm(query)
        return TechniqueResult(
            answer=answer.strip(),
            technique_used=self.name,
            confidence=0.5,  # no signal on confidence
            llm_calls=1,
        )
