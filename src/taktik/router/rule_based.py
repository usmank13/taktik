"""Rule-based router (v1) — hand-crafted heuristics."""

import re
from taktik.router.base import Router


# Simple keyword/pattern signals for task classification
_MATH_PATTERNS = [
    r"\d+\s*[\+\-\*/\%]",       # arithmetic operators
    r"how (?:much|many)",
    r"calculate|compute|solve",
    r"percent|ratio|fraction",
    r"equation|formula",
]

_LOGIC_PATTERNS = [
    r"if .+ then",
    r"true or false",
    r"which of the following",
    r"is it possible",
    r"does .+ imply",
]

_CODE_PATTERNS = [
    r"write (?:a |the )?(?:code|function|program|script)",
    r"debug|fix (?:this|the) (?:code|bug|error)",
    r"implement|refactor",
    r"```",
]

_SIMPLE_PATTERNS = [
    r"^what is .{1,30}\??$",     # short factual
    r"^who is",
    r"^when (?:was|did|is)",
    r"^where (?:is|was|are)",
    r"^define ",
]


def _matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)


class RuleBasedRouter(Router):
    """
    V1 router: simple rule-based routing.
    
    Rules (in priority order):
    1. Simple factual → direct
    2. Math/computation → cot (or pot if available)
    3. Logic/reasoning → cot
    4. Code → direct (models are good at code zero-shot)
    5. Complex/long → self_consistency (if available) or cot
    6. Default → cot
    """

    def select(self, query: str, available_techniques: list[str]) -> str:
        q = query.strip()
        
        # Simple factual questions → direct
        if _matches_any(q, _SIMPLE_PATTERNS) and len(q) < 100:
            return "direct" if "direct" in available_techniques else available_techniques[0]
        
        # Math → prefer PoT, fall back to CoT
        if _matches_any(q, _MATH_PATTERNS):
            if "pot" in available_techniques:
                return "pot"
            return "cot" if "cot" in available_techniques else available_techniques[0]
        
        # Logic/reasoning → CoT
        if _matches_any(q, _LOGIC_PATTERNS):
            return "cot" if "cot" in available_techniques else available_techniques[0]
        
        # Code → direct (LLMs are good at code without CoT scaffolding)
        if _matches_any(q, _CODE_PATTERNS):
            return "direct" if "direct" in available_techniques else available_techniques[0]
        
        # Long/complex queries → self-consistency if available
        if len(q) > 200:
            if "self_consistency" in available_techniques:
                return "self_consistency"
            return "cot" if "cot" in available_techniques else available_techniques[0]
        
        # Default → CoT (safest general-purpose choice)
        return "cot" if "cot" in available_techniques else available_techniques[0]
