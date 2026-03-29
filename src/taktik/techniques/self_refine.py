"""
Self-Refine — generate, critique, and improve iteratively.

Madaan et al. 2023: The model generates an initial answer, then
critiques its own output, then refines based on the critique.
Repeats for N rounds or until the model is satisfied.

Effective for: code generation, writing, analysis — any task where
quality improves with iteration.
"""

from taktik.techniques.base import Technique
from taktik.types import TechniqueResult, LLMFunction

_CRITIQUE_PROMPT = """Here is a question and an answer. Critique the answer — identify any errors, 
weaknesses, missing information, or areas for improvement. Be specific and constructive.

Question: {query}

Answer: {answer}

Critique:"""

_REFINE_PROMPT = """Here is a question, an initial answer, and a critique of that answer.
Produce an improved answer that addresses the critique.

Question: {query}

Initial answer: {answer}

Critique: {critique}

Improved answer:"""


class SelfRefine(Technique):
    """
    Self-Refine: generate → critique → refine, iteratively.
    
    Each round: 1 generation + 1 critique + 1 refinement = 3 LLM calls.
    Default: 1 round (initial + critique + refine = 3 calls total).
    """

    name = "self_refine"
    description = "Self-Refine: generate, critique, and iteratively improve the answer"
    typical_llm_calls = 3
    supports_open_ended = True

    def __init__(self, max_rounds: int = 1):
        self.max_rounds = max_rounds

    def run(self, query: str, llm: LLMFunction, **kwargs) -> TechniqueResult:
        max_rounds = kwargs.get("max_rounds", self.max_rounds)
        total_calls = 0

        # Initial generation
        answer = llm(query).strip()
        total_calls += 1

        rounds_log = [f"Initial: {answer[:200]}"]

        for i in range(max_rounds):
            # Critique
            critique_prompt = _CRITIQUE_PROMPT.format(query=query, answer=answer)
            critique = llm(critique_prompt).strip()
            total_calls += 1

            # Check if critique indicates the answer is already good
            if _is_satisfied(critique):
                rounds_log.append(f"Round {i+1} critique: satisfied, stopping")
                break

            # Refine
            refine_prompt = _REFINE_PROMPT.format(
                query=query, answer=answer, critique=critique
            )
            answer = llm(refine_prompt).strip()
            total_calls += 1

            rounds_log.append(f"Round {i+1}: critique={critique[:100]}... → refined")

        return TechniqueResult(
            answer=answer,
            technique_used=self.name,
            confidence=min(0.5 + 0.15 * total_calls, 0.9),  # more rounds → higher confidence
            reasoning="\n".join(rounds_log),
            llm_calls=total_calls,
            metadata={"rounds": len(rounds_log) - 1},
        )


def _is_satisfied(critique: str) -> bool:
    """Check if the critique indicates the answer is already good."""
    satisfaction_signals = [
        "no major issues",
        "answer is correct",
        "well-structured",
        "no significant errors",
        "looks good",
        "nothing to improve",
        "comprehensive and accurate",
    ]
    critique_lower = critique.lower()
    return any(signal in critique_lower for signal in satisfaction_signals)
