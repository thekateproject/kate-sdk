![PyPI](https://img.shields.io/pypi/v/projectkate)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![Python](https://img.shields.io/badge/python-3.9+-blue)

# KATE SDK

Auto-eval and observability for AI agents. Trace every LLM call, run evaluations, and catch regressions before they ship.

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

# Initialize — reads KATE_API_URL, KATE_API_KEY, KATE_AGENT_ID from env
projectkate.init()

@projectkate.trace("summarize")
def summarize(text: str) -> str:
    return client.messages.create(
        model="claude-sonnet-4-20250514",
        messages=[{"role": "user", "content": f"Summarize: {text}"}],
    ).content[0].text

async with projectkate.run() as ctx:
    result = summarize("Today's top news stories...")
    ctx.output(result)
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

## How KATE Compares

| Feature | KATE | LangSmith | Arize Phoenix |
| --- | --- | --- | --- |
| Built for agentic loops | Yes | Partial (RAG-focused) | Partial |
| Auto-instrumentation | Yes, zero-config | Manual setup | Manual setup |
| Hallucination detection | Built-in | Separate tool needed | Built-in |
| Open source | Yes | No | Yes |
| Framework agnostic | Yes | LangChain-first | Yes |
| Self-hostable | Yes | No | Yes |

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
- [Examples](examples/) — runnable example agents

## License

Apache 2.0 — see [LICENSE](LICENSE).
