"""Self-Consistency — sample multiple CoT paths, majority vote."""

from collections import Counter
from taktik.techniques.base import Technique
from taktik.techniques.cot import ChainOfThought, _extract_final_answer
from taktik.types import TechniqueResult, LLMFunction


class SelfConsistency(Technique):
    """
    Self-Consistency (Wang et al. 2023).
    
    Samples k CoT reasoning paths and takes majority vote on final answer.
    """
    
    name = "self_consistency"
    description = "Self-Consistency: sample multiple reasoning paths, majority vote"
    typical_llm_calls = 5
    supports_open_ended = False  # only works for discrete answers

    def __init__(self, k: int = 5, cot_prompt: str = "Let's think step by step."):
        self.k = k
        self.cot_prompt = cot_prompt

    def run(self, query: str, llm: LLMFunction, **kwargs) -> TechniqueResult:
        k = kwargs.get("k", self.k)
        prompt = f"{query}\n\n{self.cot_prompt}"
        
        answers = []
        reasonings = []
        for _ in range(k):
            response = llm(prompt)
            reasonings.append(response.strip())
            answer = _extract_final_answer(response)
            answers.append(_normalize_answer(answer))
        
        # Majority vote
        counter = Counter(answers)
        best_answer, best_count = counter.most_common(1)[0]
        confidence = best_count / k
        
        return TechniqueResult(
            answer=best_answer,
            technique_used=self.name,
            confidence=confidence,
            reasoning=f"Sampled {k} paths. Votes: {dict(counter)}",
            llm_calls=k,
            metadata={
                "all_answers": answers,
                "vote_distribution": dict(counter),
                "k": k,
            },
        )


def _normalize_answer(answer: str) -> str:
    """Normalize answer for comparison (lowercase, strip punctuation)."""
    return answer.lower().strip().rstrip(".")
