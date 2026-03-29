"""
Program-of-Thought (PoT) — generate code to solve the problem, then execute it.

Chen et al. 2023: ~12% average gain over CoT across math datasets.
Separates reasoning from computation — the LLM reasons in code,
Python handles the arithmetic exactly.
"""

import logging
import re
from taktik.techniques.base import Technique
from taktik.types import TechniqueResult, LLMFunction

logger = logging.getLogger(__name__)

_POT_PROMPT = """Solve this problem by writing a Python program. 
Write ONLY the Python code that computes the answer.
Store the final answer in a variable called `answer`.
Do not include any explanation — just the code.

Problem: {query}"""


class ProgramOfThought(Technique):
    """
    Program-of-Thought: generate Python code, execute it, return the result.
    
    Eliminates arithmetic errors by offloading computation to Python.
    Best for: math word problems, financial calculations, data analysis.
    """

    name = "pot"
    description = "Program-of-Thought: generate and execute Python code to solve the problem"
    typical_llm_calls = 1
    supports_open_ended = False  # only works for computable answers

    def __init__(self, allow_imports: bool = False, timeout: int = 5):
        self.allow_imports = allow_imports
        self.timeout = timeout

    def run(self, query: str, llm: LLMFunction, **kwargs) -> TechniqueResult:
        prompt = _POT_PROMPT.format(query=query)
        response = llm(prompt)

        # Extract code from response
        code = _extract_code(response)

        # Execute the code
        try:
            answer, output = _safe_exec(code, self.timeout, self.allow_imports)
        except Exception as e:
            logger.warning("PoT execution failed: %s", e)
            return TechniqueResult(
                answer=f"Execution error: {e}",
                technique_used=self.name,
                confidence=0.1,
                reasoning=f"Generated code:\n{code}\n\nError: {e}",
                llm_calls=1,
                metadata={"code": code, "error": str(e)},
            )

        return TechniqueResult(
            answer=str(answer),
            technique_used=self.name,
            confidence=0.85,  # high confidence — computed exactly
            reasoning=f"Generated code:\n{code}\n\nOutput: {output}",
            llm_calls=1,
            metadata={"code": code, "output": output},
        )


def _extract_code(response: str) -> str:
    """Extract Python code from LLM response."""
    # Try to find code in markdown code block
    match = re.search(r"```(?:python)?\s*\n(.*?)```", response, re.DOTALL)
    if match:
        return match.group(1).strip()

    # If no code block, assume the whole response is code
    # Strip any leading explanation lines
    lines = response.strip().split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        # Heuristic: code lines start with valid Python
        stripped = line.strip()
        if stripped and (
            stripped[0] in "abcdefghijklmnopqrstuvwxyz_#"
            or stripped.startswith("import ")
            or stripped.startswith("from ")
            or stripped.startswith("def ")
            or stripped.startswith("for ")
            or stripped.startswith("if ")
            or stripped.startswith("print")
            or stripped.startswith("answer")
            or stripped[0].isdigit()
        ):
            in_code = True
        if in_code:
            code_lines.append(line)

    return "\n".join(code_lines) if code_lines else response.strip()


def _safe_exec(code: str, timeout: int = 5, allow_imports: bool = False) -> tuple[str, str]:
    """
    Execute Python code in a restricted environment.

    Returns (answer, stdout_output).
    """
    import io
    import contextlib
    import signal

    if not allow_imports:
        # Block dangerous imports
        blocked = ["os", "sys", "subprocess", "shutil", "pathlib", "socket", "http", "urllib"]
        for mod in blocked:
            if f"import {mod}" in code or f"from {mod}" in code:
                raise ValueError(f"Import of '{mod}' is not allowed")

    # Set up restricted globals
    restricted_globals = {"__builtins__": {"__import__": __builtins__.__import__} if allow_imports else {
        "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
        "len": len, "range": range, "int": int, "float": float, "str": str,
        "list": list, "dict": dict, "tuple": tuple, "set": set,
        "sorted": sorted, "enumerate": enumerate, "zip": zip,
        "map": map, "filter": filter, "print": print,
        "True": True, "False": False, "None": None,
        "pow": pow, "divmod": divmod,
    }}
    local_vars = {}

    # Capture stdout
    stdout_capture = io.StringIO()

    # Timeout handler
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Code execution timed out after {timeout}s")

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, restricted_globals, local_vars)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    output = stdout_capture.getvalue().strip()
    answer = local_vars.get("answer", output or "No answer variable found")

    return str(answer), output
