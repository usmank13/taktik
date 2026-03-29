"""
Claude Code / Claude Agent SDK integration.

Exposes Taktik as an MCP tool that Claude Code can call, and provides
a hook for intercepting queries to apply strategy routing automatically.

## Usage as MCP Tool

```python
from taktik.integrations.claude_code import create_taktik_mcp_server

server = create_taktik_mcp_server(api_key="sk-...")

# Pass to Claude Agent SDK
from claude_agent_sdk import query, ClaudeAgentOptions

async for msg in query(
    prompt="Solve this step by step...",
    options=ClaudeAgentOptions(mcp_servers=[server]),
):
    print(msg)
```

## Usage as Hook

```python
from taktik.integrations.claude_code import taktik_pre_tool_hook

options = ClaudeAgentOptions(
    hooks={"PreToolUse": [taktik_pre_tool_hook]}
)
```
"""

from typing import Any, Optional


def create_taktik_tools():
    """
    Create Taktik tool definitions compatible with Claude Agent SDK's @tool decorator.

    Returns a list of tool definition dicts that can be registered with
    create_sdk_mcp_server.
    """
    return [
        {
            "name": "taktik_query",
            "description": (
                "Route a question through the optimal prompting strategy. "
                "Taktik analyzes the query and applies the best technique "
                "(Chain-of-Thought, Self-Consistency, Program-of-Thought, etc.) "
                "to get a higher-quality answer. Use this for questions where "
                "accuracy matters — especially math, logic, and complex reasoning."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question or task to solve",
                    },
                    "technique": {
                        "type": "string",
                        "description": (
                            "Force a specific technique. If omitted, Taktik auto-selects. "
                            "Options: direct, cot, self_consistency, pot"
                        ),
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "taktik_classify",
            "description": (
                "Classify a query without executing a technique. Returns the "
                "recommended technique, task type, complexity, and reasoning. "
                "Use this to understand what strategy Taktik would choose."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question to classify",
                    },
                },
                "required": ["query"],
            },
        },
        {
            "name": "taktik_compare",
            "description": (
                "Run multiple techniques on the same query and compare results. "
                "Use this for benchmarking or when you want to see how different "
                "strategies handle the same question."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The question to solve with multiple techniques",
                    },
                    "techniques": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of technique names to compare. If omitted, uses all.",
                    },
                },
                "required": ["query"],
            },
        },
    ]


def create_taktik_mcp_server(
    api_key: Optional[str] = None,
    model: str = "claude-3-5-haiku-20241022",
    router_model: Optional[str] = None,
):
    """
    Create an MCP server exposing Taktik tools for Claude Agent SDK.

    Args:
        api_key: Anthropic API key. If None, reads from ANTHROPIC_API_KEY env var.
        model: Model to use for technique execution.
        router_model: Model for routing decisions (defaults to same as model).

    Returns:
        MCP server object compatible with ClaudeAgentOptions.mcp_servers.

    Example:
        ```python
        from claude_agent_sdk import query, ClaudeAgentOptions
        from taktik.integrations.claude_code import create_taktik_mcp_server

        server = create_taktik_mcp_server()

        async for msg in query(
            prompt="What is 15% of 340?",
            options=ClaudeAgentOptions(mcp_servers=[server]),
        ):
            print(msg)
        ```
    """
    try:
        from claude_agent_sdk import tool, create_sdk_mcp_server
    except ImportError:
        raise ImportError(
            "claude-agent-sdk is required for Claude Code integration. "
            "Install with: pip install claude-agent-sdk"
        )

    # Build the LLM function
    llm_fn = _make_anthropic_llm(api_key, model)
    router_llm = _make_anthropic_llm(api_key, router_model or model) if router_model else llm_fn

    # Initialize Taktik
    from taktik.core import Taktik
    from taktik.router.llm_router import LLMRouter

    tk = Taktik(llm=llm_fn, router=LLMRouter(llm=router_llm))

    # Define tools
    @tool("taktik_query", "Route a query through the optimal prompting strategy", {"query": str, "technique": str})
    async def taktik_query(args: dict[str, Any]) -> dict:
        result = tk.run(args["query"], technique=args.get("technique"))
        return {
            "content": [{
                "type": "text",
                "text": (
                    f"**Technique:** {result.technique_used}\n"
                    f"**Confidence:** {result.confidence:.0%}\n"
                    f"**Answer:** {result.answer}\n"
                    f"**Reasoning:** {result.reasoning}"
                ),
            }]
        }

    @tool("taktik_classify", "Classify a query to see what technique Taktik would choose", {"query": str})
    async def taktik_classify(args: dict[str, Any]) -> dict:
        if isinstance(tk.router, LLMRouter):
            classification = tk.router.select_with_reasoning(
                args["query"], tk.list_techniques()
            )
        else:
            technique = tk.router.select(args["query"], tk.list_techniques())
            classification = {"technique": technique}
        return {
            "content": [{"type": "text", "text": str(classification)}]
        }

    @tool("taktik_compare", "Compare multiple techniques on the same query", {"query": str})
    async def taktik_compare(args: dict[str, Any]) -> dict:
        results = tk.run_compare(args["query"], args.get("techniques"))
        lines = []
        for r in results:
            lines.append(
                f"**{r.technique_used}** (confidence={r.confidence:.0%}, "
                f"calls={r.llm_calls}): {r.answer}"
            )
        return {
            "content": [{"type": "text", "text": "\n\n".join(lines)}]
        }

    return create_sdk_mcp_server(
        name="taktik",
        version="0.1.0",
        tools=[taktik_query, taktik_classify, taktik_compare],
    )


def _make_anthropic_llm(api_key: Optional[str], model: str):
    """Create an LLM function using the Anthropic API."""
    import os

    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("No API key provided. Set ANTHROPIC_API_KEY or pass api_key.")

    def llm(prompt: str) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    return llm
