"""Tests for Taktik core orchestrator."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from taktik import Taktik, TechniqueResult


def mock_llm(prompt: str) -> str:
    """Simple mock LLM that echoes a canned answer."""
    if "step by step" in prompt.lower():
        return "Let me think... The answer is 42."
    return "42"


class TestTaktik:

    def test_basic_run(self):
        tk = Taktik(llm=mock_llm)
        result = tk.run("What is 6 * 7?")
        assert isinstance(result, TechniqueResult)
        assert result.answer != ""
        assert result.technique_used in tk.list_techniques()

    def test_force_technique(self):
        tk = Taktik(llm=mock_llm)
        result = tk.run("What is 6 * 7?", technique="direct")
        assert result.technique_used == "direct"
        assert result.answer == "42"

    def test_cot_technique(self):
        tk = Taktik(llm=mock_llm)
        result = tk.run("What is 6 * 7?", technique="cot")
        assert result.technique_used == "cot"
        assert result.reasoning != ""

    def test_self_consistency(self):
        tk = Taktik(llm=mock_llm)
        result = tk.run("What is 6 * 7?", technique="self_consistency", k=3)
        assert result.technique_used == "self_consistency"
        assert result.llm_calls == 3
        assert result.confidence > 0

    def test_unknown_technique_raises(self):
        tk = Taktik(llm=mock_llm)
        with pytest.raises(ValueError, match="Unknown technique"):
            tk.run("test", technique="nonexistent")

    def test_list_techniques(self):
        tk = Taktik(llm=mock_llm)
        techs = tk.list_techniques()
        assert "direct" in techs
        assert "cot" in techs
        assert "self_consistency" in techs

    def test_run_compare(self):
        tk = Taktik(llm=mock_llm)
        results = tk.run_compare("What is 6 * 7?", techniques=["direct", "cot"])
        assert len(results) == 2
        assert results[0].technique_used == "direct"
        assert results[1].technique_used == "cot"


class TestRouter:

    def test_math_routes_to_cot(self):
        tk = Taktik(llm=mock_llm)
        result = tk.run("Calculate 15% of 240")
        assert result.technique_used == "cot"

    def test_simple_routes_to_direct(self):
        tk = Taktik(llm=mock_llm)
        result = tk.run("What is Python?")
        assert result.technique_used == "direct"

    def test_code_routes_to_direct(self):
        tk = Taktik(llm=mock_llm)
        result = tk.run("Write a function to sort a list")
        assert result.technique_used == "direct"
