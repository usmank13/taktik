"""
LLM-based router — uses a protocol markdown file to guide technique selection.

The routing protocol is a plain markdown file that defines classification
signals, routing rules, and output format. The LLM reads the protocol +
the user's query and outputs a technique selection.

This makes routing logic editable text rather than brittle code.
"""

import json
import logging
from pathlib import Path
from typing import Optional

from taktik.router.base import Router
from taktik.types import LLMFunction

logger = logging.getLogger(__name__)

_DEFAULT_PROTOCOL_PATH = Path(__file__).parent.parent / "routing_protocol.md"


class LLMRouter(Router):
    """
    Routes queries using an LLM guided by a markdown protocol.

    Args:
        llm: LLM function for the routing call (can be a cheap/fast model).
        protocol_path: Path to the routing protocol markdown file.
            Defaults to the bundled routing_protocol.md.
        protocol_text: Raw protocol text (overrides protocol_path if provided).
    """

    def __init__(
        self,
        llm: LLMFunction,
        protocol_path: Optional[str | Path] = None,
        protocol_text: Optional[str] = None,
    ):
        self.llm = llm
        if protocol_text:
            self._protocol = protocol_text
        else:
            path = Path(protocol_path) if protocol_path else _DEFAULT_PROTOCOL_PATH
            self._protocol = path.read_text()

    def select(self, query: str, available_techniques: list[str]) -> str:
        """
        Use the LLM to classify the query and select a technique.
        """
        # Build the prompt: protocol + available techniques + query
        techniques_desc = ", ".join(available_techniques)
        protocol = self._protocol.replace("{available_techniques}", techniques_desc)

        prompt = f"{protocol}\n\n---\n\nUser query: {query}"

        try:
            response = self.llm(prompt)
            selection = _parse_response(response, available_techniques)
            logger.info(
                "LLM router: '%s' → %s (%s, %s)",
                query[:60],
                selection["technique"],
                selection.get("task_type", "?"),
                selection.get("reasoning", ""),
            )
            return selection["technique"]
        except Exception as e:
            logger.warning("LLM router failed (%s), falling back to 'cot'", e)
            return "cot" if "cot" in available_techniques else available_techniques[0]

    def select_with_reasoning(
        self, query: str, available_techniques: list[str]
    ) -> dict:
        """
        Like select(), but returns the full classification dict.

        Returns:
            {
                "technique": "cot",
                "task_type": "MATH",
                "complexity": "MODERATE",
                "answer_type": "DISCRETE",
                "reasoning": "Math word problem requiring multi-step calculation"
            }
        """
        techniques_desc = ", ".join(available_techniques)
        protocol = self._protocol.replace("{available_techniques}", techniques_desc)
        prompt = f"{protocol}\n\n---\n\nUser query: {query}"

        try:
            response = self.llm(prompt)
            return _parse_response(response, available_techniques)
        except Exception as e:
            return {
                "technique": "cot" if "cot" in available_techniques else available_techniques[0],
                "task_type": "UNKNOWN",
                "complexity": "MODERATE",
                "answer_type": "OPEN_ENDED",
                "reasoning": f"Router error: {e}",
            }


def _parse_response(response: str, available_techniques: list[str]) -> dict:
    """
    Parse the LLM's JSON response, with fallback extraction.
    """
    # Try direct JSON parse
    text = response.strip()

    # Extract JSON from markdown code block if present
    if "```" in text:
        for block in text.split("```"):
            block = block.strip()
            if block.startswith("json"):
                block = block[4:].strip()
            if block.startswith("{"):
                text = block
                break

    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                result = json.loads(text[start:end])
            except json.JSONDecodeError:
                raise ValueError(f"Could not parse router response as JSON: {text[:200]}")
        else:
            raise ValueError(f"No JSON object found in router response: {text[:200]}")

    # Validate technique is available
    technique = result.get("technique", "")
    if technique not in available_techniques:
        logger.warning(
            "Router selected unavailable technique '%s', falling back", technique
        )
        result["technique"] = "cot" if "cot" in available_techniques else available_techniques[0]

    return result
