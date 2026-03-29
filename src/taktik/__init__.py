"""
Taktik — Intelligent prompt strategy orchestration for AI agents.

Routes to the right prompting technique (CoT, SC, ConDil, PoT, etc.)
based on task characteristics. Composable technique pipelines.
"""

__version__ = "0.1.0"

from taktik.core import Taktik
from taktik.types import TechniqueResult
from taktik.techniques.base import Technique
from taktik.router.base import Router

__all__ = ["Taktik", "Technique", "TechniqueResult", "Router"]
