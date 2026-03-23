# KATE SDK

Auto-eval and observability for AI agents. Trace every LLM call, run evaluations locally or against a KATE server, and catch regressions before they ship.

## Install

```bash
pip install projectkate
```

### Optional instrumentation extras

```bash
pip install projectkate[openai]                # Auto-instrument OpenAI SDK
pip install projectkate[anthropic-instrument]  # Auto-instrument Anthropic SDK
pip install projectkate[langchain]             # Auto-instrument LangChain / LangGraph
pip install projectkate[all]                   # All supported providers
```

## Quick Start

```python
import kate_sdk

# Initialize — reads KATE_API_URL, KATE_API_KEY, KATE_AGENT_ID from env
kate_sdk.init()

# Trace any function that calls an LLM
@kate_sdk.trace("summarize")
def summarize(text: str) -> str:
    return client.messages.create(
        model="claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": f"Summarize: {text}"}],
    ).content[0].text

# Run context: creates a run, captures traces, triggers eval on exit
async with kate_sdk.run() as ctx:
    result = summarize("Today's top news stories...")
    ctx.output(result)
```

## Local Eval (no server needed)

Run evaluations locally against your agent with zero infrastructure:

```python
from kate_sdk.local import LocalRunner

runner = LocalRunner(agent_fn=my_agent)
results = await runner.run(test_cases=[
    {"input": "Summarize the news", "expected": "A concise summary..."},
])
runner.print_results(results)
```

## Documentation

- [Getting Started](docs/getting_started.md) — full integration guide
- [Examples](examples/) — runnable example agents

## License

Apache 2.0 — see [LICENSE](LICENSE).
