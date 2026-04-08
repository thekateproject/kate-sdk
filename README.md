![PyPI](https://img.shields.io/pypi/v/projectkate)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![Python](https://img.shields.io/badge/python-3.11+-blue)

# KATE SDK

Auto-eval, observability, and knowledge marketplace for AI agents. Trace every LLM call, run evaluations, discover and use marketplace tools, and catch regressions before they ship.

## Install

```bash
pip install projectkate
```

### Optional instrumentation extras

```bash
pip install projectkate[openai]                # OpenAI
pip install projectkate[anthropic-instrument]  # Anthropic
pip install projectkate[langchain]             # LangChain / LangGraph
pip install projectkate[mistral]               # Mistral
pip install projectkate[vertexai]              # Vertex AI
pip install projectkate[google-genai]          # Google GenAI
pip install projectkate[crewai]                # CrewAI
pip install projectkate[all]                   # All supported providers
```

## Quick Start

### Trace mode — instrument your agent

```python
import projectkate
from openai import AsyncOpenAI

# Initialize — reads KATE_API_URL and KATE_API_KEY from env
projectkate.init(
    agent_name="News Summarizer",
    agent_objective="Summarize news articles concisely",
    agent_domain="content",
)

client = AsyncOpenAI()

@projectkate.trace("summarize")
async def summarize(text: str) -> str:
    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"Summarize: {text}"}],
    )
    return response.choices[0].message.content

async with projectkate.run():
    result = await summarize("Today's top news stories...")
    print(result)
```

### Management client — programmatic platform access

```python
from projectkate import KateClient

async with KateClient(api_key="kate_...") as kate:
    # List your agents
    agents = await kate.agents.list()

    # Check eval results for a run
    evals = await kate.evals.get_run_evals(run_id="...")

    # Publish an artifact
    await kate.artifacts.publish(artifact_id="...")

    # Check wallet balance
    balance = await kate.wallet.get_balance()
```

## Tools — discover and use marketplace tools

KATE agents can discover and execute tools from the marketplace. The SDK provides a tool loop that handles the LLM ↔ tool-call cycle automatically.

### Agentic tool loop

Wire up your LLM client and let the SDK handle tool discovery, execution, and chaining:

```python
import projectkate
from openai import AsyncOpenAI

projectkate.init()
llm = AsyncOpenAI()

messages = [
    {"role": "system", "content": "You are a helpful assistant with access to tools."},
    {"role": "user", "content": "Find SEO keywords for 'AI observability'"},
]

result = await projectkate.tool_loop(
    llm,
    model="gpt-4o",
    messages=messages,
    max_rounds=10,
)

print(result.content)          # Final LLM response
print(result.tool_calls_made)  # Number of tool calls executed
```

Works with both OpenAI and Anthropic clients — the SDK detects the provider automatically.

### Local tools

You can register your own tools alongside marketplace tools. The SDK merges them and routes calls to the right handler:

```python
from projectkate import LocalTool

def get_current_date() -> str:
    from datetime import date
    return date.today().isoformat()

result = await projectkate.tool_loop(
    llm,
    model="gpt-4o",
    messages=messages,
    local_tools=[
        LocalTool(
            name="get_current_date",
            description="Returns today's date in ISO format",
            parameters={"type": "object", "properties": {}},
            fn=get_current_date,
        ),
    ],
)
```

Local tools run in-process. Async functions are supported.

### Direct tool management

Use the management client for lower-level control:

```python
async with KateClient(api_key="kate_...") as kate:
    # List tools available to your agent
    tools = await kate.tools.list(agent_id="...")

    # Execute a specific tool
    result = await kate.tools.execute(
        agent_id="...",
        tool_name="seo_keyword_research",
        input_data={"query": "AI observability"},
    )
    print(result.output)

    # Check credential status for subscribed tools
    statuses = await kate.tools.status(agent_id="...")
```

## Local Eval (no server needed)

Run evaluations locally against your agent with zero infrastructure:

```python
from projectkate.local import LocalEvalRunner

runner = LocalEvalRunner(agent_fn=my_agent)
results = await runner.run(test_cases=[
    {"input": "Summarize the news", "expected": "A concise summary..."},
])
runner.print_results(results)
```

## Documentation

- [Docs](https://docs.projectkate.com) — guides, API reference, and examples

## License

Apache 2.0 — see [LICENSE](LICENSE).
