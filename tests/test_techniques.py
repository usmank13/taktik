"""Tests for individual techniques."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
from taktik.types import TechniqueResult


# ── Mock LLMs ────────────────────────────────────────────────────────────────

def mock_llm(prompt: str) -> str:
    """General mock that returns sensible responses based on prompt content."""
    if "Python program" in prompt or "Python code" in prompt:
        return "```python\nx = 15 / 100 * 240\nanswer = x\n```"
    if "Critique" in prompt:
        return "The answer could be more precise about the units."
    if "Improved answer" in prompt:
        return "The answer is 36.0 dollars."
    if "Break this problem" in prompt or "Sub-problems" in prompt:
        return "1. Calculate 15% as a decimal\n2. Multiply by 240\n3. State the final answer"
    if "sub-problem" in prompt.lower() and "Current" in prompt:
        if "decimal" in prompt.lower():
            return "15% as a decimal is 0.15"
        if "multiply" in prompt.lower() or "240" in prompt:
            return "0.15 × 240 = 36"
        return "The answer is 36"
    if "Final answer" in prompt:
        return "15% of 240 is 36."
    if "step by step" in prompt.lower():
        return "Step 1: 15% = 0.15\nStep 2: 0.15 × 240 = 36\nTherefore, the answer is 36."
    return "42"


# ── PoT Tests ────────────────────────────────────────────────────────────────

class TestProgramOfThought:

    def test_basic_execution(self):
        from taktik.techniques.pot import ProgramOfThought
        pot = ProgramOfThought()
        result = pot.run("What is 15% of 240?", mock_llm)
        assert result.technique_used == "pot"
        assert "36" in result.answer
        assert result.confidence > 0.5
        assert result.metadata.get("code")

    def test_execution_error_handled(self):
        from taktik.techniques.pot import ProgramOfThought

        def bad_code_llm(prompt):
            return "```python\nanswer = 1/0\n```"

        pot = ProgramOfThought()
        result = pot.run("Divide by zero", bad_code_llm)
        assert "error" in result.answer.lower() or result.confidence < 0.5

    def test_blocked_imports(self):
        from taktik.techniques.pot import ProgramOfThought

        def dangerous_llm(prompt):
            return "```python\nimport os\nanswer = os.listdir('/')\n```"

        pot = ProgramOfThought(allow_imports=False)
        result = pot.run("List files", dangerous_llm)
        assert result.confidence < 0.5  # should fail

    def test_code_extraction_no_block(self):
        from taktik.techniques.pot import _extract_code
        code = _extract_code("x = 5\nanswer = x * 2")
        assert "answer" in code


# ── Self-Refine Tests ────────────────────────────────────────────────────────

class TestSelfRefine:

    def test_basic_refine(self):
        from taktik.techniques.self_refine import SelfRefine
        sr = SelfRefine(max_rounds=1)
        result = sr.run("What is 15% of 240?", mock_llm)
        assert result.technique_used == "self_refine"
        assert result.llm_calls == 3  # initial + critique + refine
        assert result.answer != ""

    def test_early_stop_on_satisfaction(self):
        from taktik.techniques.self_refine import SelfRefine

        call_count = 0
        def satisfied_llm(prompt):
            nonlocal call_count
            call_count += 1
            if "Critique" in prompt:
                return "The answer is correct. No major issues found."
            return "42"

        sr = SelfRefine(max_rounds=3)
        result = sr.run("Simple question", satisfied_llm)
        assert call_count == 2  # initial + critique (no refine needed)
        assert result.llm_calls == 2

    def test_multiple_rounds(self):
        from taktik.techniques.self_refine import SelfRefine
        sr = SelfRefine(max_rounds=2)
        result = sr.run("Complex question", mock_llm)
        assert result.llm_calls >= 3


# ── Decompose Tests ──────────────────────────────────────────────────────────

class TestDecompose:

    def test_basic_decompose(self):
        from taktik.techniques.decompose import Decompose
        d = Decompose()
        result = d.run("What is 15% of 240?", mock_llm)
        assert result.technique_used == "decompose"
        assert result.llm_calls >= 3  # decompose + sub-problems + synthesize
        assert result.metadata.get("subproblems")

    def test_subproblem_parsing(self):
        from taktik.techniques.decompose import _parse_subproblems
        text = "1. First thing\n2. Second thing\n3. Third thing"
        problems = _parse_subproblems(text, max_count=5)
        assert len(problems) == 3
        assert problems[0] == "First thing"

    def test_subproblem_parsing_dashes(self):
        from taktik.techniques.decompose import _parse_subproblems
        text = "- Calculate the percentage\n- Multiply\n- Report"
        problems = _parse_subproblems(text, max_count=5)
        assert len(problems) == 3

    def test_max_subproblems_limit(self):
        from taktik.techniques.decompose import _parse_subproblems
        text = "\n".join(f"{i}. Problem {i}" for i in range(1, 20))
        problems = _parse_subproblems(text, max_count=3)
        assert len(problems) == 3

    def test_fallback_on_empty_decomposition(self):
        from taktik.techniques.decompose import Decompose

        def no_decompose_llm(prompt):
            if "Break this" in prompt:
                return ""  # empty decomposition
            return "Direct answer: 42"

        d = Decompose()
        result = d.run("Simple thing", no_decompose_llm)
        assert result.answer != ""


# ── Integration: all techniques via Taktik ───────────────────────────────────

class TestAllTechniques:

    def test_all_registered(self):
        from taktik import Taktik
        tk = Taktik(llm=mock_llm)
        techs = tk.list_techniques()
        assert "direct" in techs
        assert "cot" in techs
        assert "self_consistency" in techs
        assert "pot" in techs
        assert "self_refine" in techs
        assert "decompose" in techs

    def test_force_each_technique(self):
        from taktik import Taktik
        tk = Taktik(llm=mock_llm)
        for name in tk.list_techniques():
            result = tk.run("What is 15% of 240?", technique=name)
            assert result.technique_used == name
            assert result.answer != ""
