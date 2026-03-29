"""Tests for LLM-based router."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import json
import pytest
from taktik.router.llm_router import LLMRouter, _parse_response


TECHNIQUES = ["direct", "cot", "self_consistency"]


def mock_router_llm(prompt: str) -> str:
    """Mock LLM that returns routing decisions based on keywords in the query."""
    query = prompt.split("User query: ")[-1].strip() if "User query:" in prompt else prompt

    if any(w in query.lower() for w in ["calculate", "math", "solve", "%"]):
        return json.dumps({
            "technique": "cot",
            "task_type": "MATH",
            "complexity": "MODERATE",
            "answer_type": "DISCRETE",
            "reasoning": "Math problem requiring step-by-step calculation",
        })
    elif any(w in query.lower() for w in ["what is", "define", "who is"]):
        return json.dumps({
            "technique": "direct",
            "task_type": "FACTUAL",
            "complexity": "SIMPLE",
            "answer_type": "DISCRETE",
            "reasoning": "Simple factual question",
        })
    else:
        return json.dumps({
            "technique": "cot",
            "task_type": "ANALYSIS",
            "complexity": "MODERATE",
            "answer_type": "OPEN_ENDED",
            "reasoning": "Default to CoT for analysis",
        })


class TestLLMRouter:

    def test_math_routes_to_cot(self):
        router = LLMRouter(llm=mock_router_llm)
        result = router.select("Calculate 15% of 240", TECHNIQUES)
        assert result == "cot"

    def test_factual_routes_to_direct(self):
        router = LLMRouter(llm=mock_router_llm)
        result = router.select("What is Python?", TECHNIQUES)
        assert result == "direct"

    def test_select_with_reasoning(self):
        router = LLMRouter(llm=mock_router_llm)
        result = router.select_with_reasoning("Calculate 15% of 240", TECHNIQUES)
        assert result["technique"] == "cot"
        assert result["task_type"] == "MATH"
        assert "reasoning" in result

    def test_custom_protocol(self):
        protocol = """You are a router. Always select "direct" no matter what.
Output: {"technique": "direct", "task_type": "ANY", "complexity": "SIMPLE", "answer_type": "DISCRETE", "reasoning": "always direct"}

User query: {available_techniques}"""
        
        def always_direct(prompt):
            return '{"technique": "direct", "task_type": "ANY", "complexity": "SIMPLE", "answer_type": "DISCRETE", "reasoning": "custom protocol"}'
        
        router = LLMRouter(llm=always_direct, protocol_text=protocol)
        assert router.select("Complex analysis task", TECHNIQUES) == "direct"

    def test_fallback_on_error(self):
        def broken_llm(prompt):
            raise Exception("LLM error")

        router = LLMRouter(llm=broken_llm)
        result = router.select("test query", TECHNIQUES)
        assert result == "cot"  # falls back to cot

    def test_fallback_on_bad_json(self):
        def bad_json_llm(prompt):
            return "I think you should use cot because reasons"

        router = LLMRouter(llm=bad_json_llm)
        result = router.select("test", TECHNIQUES)
        assert result == "cot"  # falls back


class TestParseResponse:

    def test_clean_json(self):
        result = _parse_response(
            '{"technique": "cot", "task_type": "MATH"}',
            TECHNIQUES,
        )
        assert result["technique"] == "cot"

    def test_json_in_code_block(self):
        result = _parse_response(
            '```json\n{"technique": "direct", "task_type": "FACTUAL"}\n```',
            TECHNIQUES,
        )
        assert result["technique"] == "direct"

    def test_json_with_surrounding_text(self):
        result = _parse_response(
            'Based on my analysis, here is the result:\n{"technique": "cot", "task_type": "LOGIC"}\nThat is my recommendation.',
            TECHNIQUES,
        )
        assert result["technique"] == "cot"

    def test_unavailable_technique_falls_back(self):
        result = _parse_response(
            '{"technique": "tree_of_thought", "task_type": "COMPLEX"}',
            TECHNIQUES,
        )
        assert result["technique"] == "cot"  # fallback

    def test_no_json_raises(self):
        with pytest.raises(ValueError):
            _parse_response("No JSON here at all", TECHNIQUES)
