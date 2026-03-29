"""Base class for prompting techniques."""

from abc import ABC, abstractmethod
from typing import Callable

from taktik.types import TechniqueResult, LLMFunction


class Technique(ABC):
    """
    Abstract base for a prompting technique.
    
    Each technique takes a query + LLM callable, applies its strategy,
    and returns a TechniqueResult.
    """

    name: str = "base"
    description: str = ""
    
    # Cost characteristics (for router decisions)
    typical_llm_calls: int = 1
    supports_open_ended: bool = True

    @abstractmethod
    def run(self, query: str, llm: LLMFunction, **kwargs) -> TechniqueResult:
        """Execute this technique on the given query."""
        ...

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.name}>"
