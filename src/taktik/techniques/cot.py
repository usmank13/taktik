"""Chain-of-Thought prompting."""

import re
from taktik.techniques.base import Technique
from taktik.types import TechniqueResult, LLMFunction


class ChainOfThought(Technique):
    """
    Zero-shot Chain-of-Thought.
    
    Appends "Let's think step by step." to the query,
    then extracts the final answer.
    """
    
    name = "cot"
    description = "Chain-of-Thought: step-by-step reasoning before answering"
    typical_llm_calls = 1

    def __init__(self, cot_prompt: str = "Let's think step by step."):
        self.cot_prompt = cot_prompt

    def run(self, query: str, llm: LLMFunction, **kwargs) -> TechniqueResult:
        prompt = f"{query}\n\n{self.cot_prompt}"
        response = llm(prompt)
        
        # Try to extract a final answer after reasoning
        answer = _extract_final_answer(response)
        
        return TechniqueResult(
            answer=answer,
            technique_used=self.name,
            reasoning=response.strip(),
            confidence=0.6,  # slightly higher than direct — reasoning present
            llm_calls=1,
        )


def _extract_final_answer(response: str) -> str:
    """
    Extract the final answer from a CoT response.
    
    Looks for common patterns like "The answer is X", "Therefore, X",
    or falls back to the last sentence.
    """
    # Common answer patterns
    patterns = [
        r"(?:the answer is|therefore,?\s*the answer is)\s*[:\s]*(.+?)\.?\s*$",
        r"(?:therefore|thus|so|hence),?\s*(.+?)\.?\s*$",
        r"(?:in conclusion|to summarize),?\s*(.+?)\.?\s*$",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()
    
    # Fallback: last non-empty line
    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
    return lines[-1] if lines else response.strip()
