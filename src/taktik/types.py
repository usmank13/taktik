"""Shared types — no internal imports to avoid circular dependencies."""

from dataclasses import dataclass, field
from typing import Callable

LLMFunction = Callable[[str], str]
"""Type alias: function that takes a prompt string, returns completion string."""


@dataclass
class TechniqueResult:
    """Result from running a technique."""
    answer: str
    technique_used: str
    confidence: float = 0.0
    reasoning: str = ""
    metadata: dict = field(default_factory=dict)
    llm_calls: int = 1
    total_tokens: int = 0
