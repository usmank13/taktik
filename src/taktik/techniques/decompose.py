"""
Least-to-Most Decomposition — break complex problems into sub-problems.

Zhou et al. 2023: Decompose → solve sub-problems sequentially, 
each building on previous answers. Particularly effective for 
compositional generalization — problems harder than the examples.

Best for: multi-step reasoning, planning, complex analysis.
"""

from taktik.techniques.base import Technique
from taktik.types import TechniqueResult, LLMFunction

_DECOMPOSE_PROMPT = """Break this problem into smaller, simpler sub-problems that can be solved 
one at a time. List each sub-problem on its own line, in order.
Only list the sub-problems — do not solve them yet.

Problem: {query}

Sub-problems:"""

_SOLVE_PROMPT = """Solve this sub-problem. You have the context from previous sub-problems below.

Original problem: {query}

Previous context:
{context}

Current sub-problem: {subproblem}

Answer:"""

_SYNTHESIZE_PROMPT = """Given the original problem and the answers to all sub-problems, 
provide the final answer.

Original problem: {query}

Sub-problem answers:
{answers}

Final answer:"""


class Decompose(Technique):
    """
    Least-to-Most Decomposition: break down → solve parts → synthesize.
    
    LLM calls: 1 (decompose) + N (solve sub-problems) + 1 (synthesize).
    """

    name = "decompose"
    description = "Decomposition: break into sub-problems, solve sequentially, synthesize"
    typical_llm_calls = 5  # rough estimate
    supports_open_ended = True

    def __init__(self, max_subproblems: int = 5):
        self.max_subproblems = max_subproblems

    def run(self, query: str, llm: LLMFunction, **kwargs) -> TechniqueResult:
        max_sub = kwargs.get("max_subproblems", self.max_subproblems)
        total_calls = 0

        # Step 1: Decompose
        decompose_prompt = _DECOMPOSE_PROMPT.format(query=query)
        decomposition = llm(decompose_prompt).strip()
        total_calls += 1

        subproblems = _parse_subproblems(decomposition, max_sub)

        if not subproblems:
            # Couldn't decompose — fall back to direct answer
            answer = llm(query).strip()
            return TechniqueResult(
                answer=answer,
                technique_used=self.name,
                confidence=0.5,
                reasoning="Could not decompose; fell back to direct.",
                llm_calls=2,
            )

        # Step 2: Solve each sub-problem sequentially
        context_parts = []
        for i, sub in enumerate(subproblems):
            context = "\n".join(context_parts) if context_parts else "(none yet)"
            solve_prompt = _SOLVE_PROMPT.format(
                query=query, context=context, subproblem=sub
            )
            sub_answer = llm(solve_prompt).strip()
            total_calls += 1
            context_parts.append(f"{i+1}. {sub}\n   → {sub_answer}")

        # Step 3: Synthesize
        answers_text = "\n".join(context_parts)
        synth_prompt = _SYNTHESIZE_PROMPT.format(query=query, answers=answers_text)
        final_answer = llm(synth_prompt).strip()
        total_calls += 1

        return TechniqueResult(
            answer=final_answer,
            technique_used=self.name,
            confidence=0.75,
            reasoning=f"Decomposed into {len(subproblems)} sub-problems:\n{answers_text}",
            llm_calls=total_calls,
            metadata={
                "subproblems": subproblems,
                "sub_answers": context_parts,
            },
        )


def _parse_subproblems(text: str, max_count: int) -> list[str]:
    """Parse sub-problems from LLM output."""
    lines = text.strip().split("\n")
    problems = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Strip numbering (1., 1), -, *)
        for prefix in ["- ", "* ", ") "]:
            if prefix in line[:5]:
                line = line[line.index(prefix) + len(prefix):]
                break
        # Strip leading numbers
        if line and line[0].isdigit():
            i = 0
            while i < len(line) and (line[i].isdigit() or line[i] in ".)"):
                i += 1
            line = line[i:].strip()
        if line:
            problems.append(line)
        if len(problems) >= max_count:
            break
    return problems
