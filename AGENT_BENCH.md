# Agent benchmark: tool by tool vs one Forge call

Forge's core claim: an agent spends fewer tokens when it writes a whole job as one
program instead of making one tool call per step, because every round trip re-sends
the whole conversation. `forge agent-bench` measures that directly.

## What it runs

Five tasks, each solved twice by the **same model**:

| Mode | The model gets | How it works |
|------|----------------|--------------|
| `tools` | every Forge tool as an ordinary function (`sales_data`, `sum`, `write_file`, …) | the usual agent loop: call a tool, read the result, decide the next call |
| `forge` | one tool, `forge_run(source)`, with the language spec in its description | writes a Forge program; ideally one call does the whole job |

| Task | What makes it interesting |
|------|---------------------------|
| `big-deals` | fetch, filter, sum: a short chain |
| `four-regions` | the same work for 4 regions: many round trips in `tools` mode, one loop in Forge |
| `research` | search, filter by relevance, count, pick the top title |
| `report-file` | fetch, write a file, read it back |
| `math-chain` | three dependent calculations |

The tools are offline and deterministic (no network), so every answer has a known
correct value. A run counts as solved only if the final answer contains it, which
means a cheap wrong answer can't win.

Tokens are the provider's own `usage` numbers, summed over **every turn**, including
the tool definitions each mode sends. If a provider returns no usage, tokens are
estimated from text length and the summary says so.

## Run it

```bash
# free and local (tool calling needs a 3B+ model)
./start_ollama.sh && ollama pull llama3.2
./forge agent-bench

# any API key: Groq, Gemini, xAI Grok, OpenRouter, OpenAI, DeepSeek
export GROQ_API_KEY=...        # or XAI_API_KEY=... for Grok
./forge agent-bench --backend openai

# options
./forge agent-bench --tasks four-regions research --runs 3
./forge agent-bench --price-in 0.20 --price-out 0.60   # $ per 1M tokens for the cost column
```

Results print as a table and are saved to `agent_bench_results.json`.

## Reading the results

- **`token savings on tasks both modes solved`** is the fair number: it compares only
  tasks each mode got right.
- `turns` and `calls` show where the savings come from: fewer round trips means the
  conversation is re-sent fewer times.
- Forge's tool description (the spec) is larger than a single tool schema, so on very
  short jobs Forge can cost *more*. The win should grow with the number of steps;
  `four-regions` is the task to watch.
- Small models may fail either mode. Use `--runs 3` or more for a steadier picture.

## Results

No results have been published yet. Run it with your model and share
`agent_bench_results.json`.
