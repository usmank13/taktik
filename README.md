# Taktik

**Intelligent prompt strategy orchestration for AI agents.**

Taktik sits between your agent and its LLM, automatically selecting and applying the most effective prompting technique for each task. Instead of using chain-of-thought for everything, Taktik routes to the right strategy — CoT for math, direct for simple recall, self-consistency for high-stakes answers, program-of-thought for computation.

Think model routing (like OpenRouter), but for *reasoning strategies* instead of models.

## Quick Start

```python
from taktik import Taktik

tk = Taktik(llm=your_llm_function)

# Auto-routes to the best technique
result = tk.run("What is 15% of 240?")
print(result.answer)          # "36"
print(result.technique_used)  # "pot" (Program-of-Thought)
print(result.confidence)      # 0.85

# Force a specific technique
result = tk.run("Explain quantum computing", technique="self_refine")

# Compare techniques head-to-head
results = tk.run_compare("Is a hot dog a sandwich?", 
                          techniques=["direct", "cot", "self_consistency"])
```

## Techniques

| Technique | LLM Calls | Best For |
|-----------|-----------|----------|
| `direct` | 1 | Simple factual, code generation |
| `cot` | 1 | Math, logic, complex reasoning |
| `self_consistency` | k (5-40) | High-stakes discrete answers |
| `pot` | 1 + exec | Math/computation (exact arithmetic) |
| `self_refine` | 3+ | Writing, analysis, iterative quality |
| `decompose` | 2+N | Multi-step problems, planning |

## Routing

Taktik includes two routers:

### Rule-Based (default, zero overhead)
Fast heuristic routing — pattern matching on query characteristics. No extra LLM call.

### LLM-Based (editable protocol)
A markdown file (`routing_protocol.md`) defines classification signals and routing rules. An LLM reads the protocol + your query and outputs a structured technique selection. Change routing logic by editing the markdown — no code changes needed.

```python
from taktik.router.llm_router import LLMRouter

router = LLMRouter(llm=your_llm_function)
tk = Taktik(llm=your_llm_function, router=router)

# Returns full classification
info = router.select_with_reasoning("Calculate 15% of 240", tk.list_techniques())
# {"technique": "pot", "task_type": "MATH", "complexity": "MODERATE", ...}
```

## Claude Code Integration

Taktik can run as an MCP tool server for the Claude Agent SDK:

```python
from taktik.integrations.claude_code import create_taktik_mcp_server
from claude_agent_sdk import query, ClaudeAgentOptions

server = create_taktik_mcp_server()

async for msg in query(
    prompt="Solve this problem...",
    options=ClaudeAgentOptions(mcp_servers=[server]),
):
    print(msg)
```

Exposes three tools:
- `taktik_query` — route + execute the best technique
- `taktik_classify` — classify without executing (inspect the routing decision)
- `taktik_compare` — run multiple techniques on the same query

## LLM Function Interface

Taktik works with any LLM — just provide a callable that takes a prompt string and returns a completion string:

```python
# OpenAI
def my_llm(prompt):
    response = openai.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Anthropic
def my_llm(prompt):
    response = anthropic.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=1024,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

# Local (ollama, vllm, etc.)
def my_llm(prompt):
    return requests.post("http://localhost:11434/api/generate", 
                         json={"model": "llama3", "prompt": prompt}).json()["response"]

tk = Taktik(llm=my_llm)
```

## Project Structure

```
taktik/
├── src/taktik/
│   ├── core.py                 # Taktik orchestrator
│   ├── types.py                # Shared types
│   ├── routing_protocol.md     # Editable LLM routing protocol
│   ├── techniques/
│   │   ├── direct.py           # Zero-shot baseline
│   │   ├── cot.py              # Chain-of-Thought
│   │   ├── self_consistency.py # Sample + majority vote
│   │   ├── pot.py              # Program-of-Thought (code execution)
│   │   ├── self_refine.py      # Generate → critique → refine
│   │   └── decompose.py        # Break down → solve → synthesize
│   ├── router/
│   │   ├── rule_based.py       # Heuristic router
│   │   └── llm_router.py       # LLM-based protocol router
│   └── integrations/
│       └── claude_code.py      # Claude Agent SDK MCP server
└── tests/                      # 35 tests
```

## License

MIT
