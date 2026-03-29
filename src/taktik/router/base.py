"""Base router interface."""

from abc import ABC, abstractmethod


class Router(ABC):
    """Selects the best prompting technique for a given query."""

    @abstractmethod
    def select(self, query: str, available_techniques: list[str]) -> str:
        """
        Choose a technique for this query.
        
        Args:
            query: The user's question/task.
            available_techniques: Names of registered techniques.
        
        Returns:
            Name of the selected technique.
        """
        ...
