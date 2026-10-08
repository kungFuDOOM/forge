# Forge — JavaScript for AI

An **AI-native agent language**. Write a short program. Run a whole multi-step job.
Built for how AIs generate code — not 1990s human IDE habits.

**Repo:** https://github.com/kungFuDOOM/forge  
**Landing:** [docs/index.html](docs/index.html)

No dependencies — Python 3.10+ standard library only.

---

## 60-second start

```bash
git clone https://github.com/kungFuDOOM/forge.git
cd forge
./forge quickstart
```

That runs a demo and creates `my-first-agent.forge`. Then:

```bash
./forge run my-first-agent.forge
./forge check my-first-agent.forge
./forge tools
```

Or install the `forge` command anywhere:

```bash
pip install git+https://github.com/kungFuDOOM/forge.git
forge quickstart
```

No API key required for demos (`--llm mock` is the default).

---

## Give your AI agent Forge (MCP)

This is where Forge saves credits. An agent that does *fetch → check → filter → summarize → save*
with ordinary tools makes one round trip per step, and every round trip re-sends the whole
conversation. With Forge it writes one short program and runs it in **one** tool call.

```bash
# Claude Code
claude mcp add forge -- python3 /path/to/forge/forge_cli.py mcp
# or, after pip install
claude mcp add forge -- forge mcp
```

Any MCP client works the same way: the server command is `forge mcp` (stdio).
It exposes two tools:

| Tool | What it does |
|------|--------------|
| `forge_run` | Runs a program, returns the `RETURN` value as JSON (or an error naming what to fix) |
| `forge_check` | Validates without running: syntax, undefined `$vars`, unknown tools |

The language spec ships inside the tool description, so the agent can use Forge with no other docs.
Add your own tools with `forge mcp --tools my_tools.py`.

---

## Example

```
AGENT "url-fetcher"

MEMORY {
  url: "https://example.com"
}

STEP fetch TOOL http_get INPUT { url: $url } OUTPUT page

VERIFY $page.status == 200

REASON "In one sentence, what is this page about?" ON $page.body OUTPUT summary

RETURN { url: $url, status: $page.status, summary: $summary }
```

- **Field access:** `$page.body`, `$hits.0.title` read one field of a tool result, so REASON only sends the LLM what it needs.
- **Lists:** `MEMORY { deals: [{ name: "Acme", amount: 15000 }], tags: ["a", "b"] }`
- **VERIFY anywhere:** use as many as you like. A failed VERIFY stops the program *before* the next tool or LLM call.
- **Pre-run check:** undefined `$vars` and unknown tools are caught before anything executes, all reported at once.

Full reference (about 400 tokens, written for an AI's context window): `./forge spec`

---

## Your own tools

A tool is a Python function that takes one dict of inputs. The first line of its docstring
documents it in `forge tools` and `forge spec`.

```python
# my_tools.py
def word_count(inputs):
    """INPUT { text } → number of words"""
    return len(inputs["text"].split())
```

```bash
./forge run agent.forge --tools my_tools.py
```

Or define `TOOLS = {"name": fn, ...}` in the file to choose exactly what gets exported.

Built-in tools: `http_get`, `read_file`, `write_file` (real; files are limited to the current
directory), plus the demo tools `web_search` (offline mock), `sales_data`, `arithmetic_add` and `get_value`.

---

## English → Forge → run

```bash
./forge ask "Fetch https://example.com and summarize it"
```

If the generated program fails to parse, check or run, the exact error goes back to the model
for a fix (`--retries 2` by default). Forge errors name what is defined and available, so one
round usually fixes it. `--live-reason` uses the real LLM for REASON steps too.

LLM backends (all plain HTTP, no SDKs):

| Backend | Setup |
|---------|-------|
| Ollama (free, local) | `./start_ollama.sh && ollama pull llama3.2` |
| Groq / Gemini / OpenRouter (free tiers) | `export GROQ_API_KEY=…` / `GEMINI_API_KEY` / `OPENROUTER_API_KEY` |
| OpenAI / DeepSeek | `export OPENAI_API_KEY=…` / `DEEPSEEK_API_KEY` |

---

## Commands

```
./forge                  # welcome
./forge quickstart       # guided first run
./forge run FILE.forge   # execute (--llm mock|auto|ollama|openai, --tools FILE.py)
./forge check FILE.forge # validate + pre-run check
./forge ask "…"          # English → Forge → run, with error-feedback retries
./forge spec             # compact language reference for AI context
./forge mcp              # MCP server for AI agents
./forge init NAME        # new agent (--template basic|http|file)
./forge examples         # list/run demos
./forge tools            # tool reference
./forge tokens FILE      # rough token vs Python/LangChain
./forge doctor           # setup check
./forge repl             # interactive
./forge credit-test      # Forge vs Python credits (see CREDIT_TEST.md)
./forge bench            # LLM generation benchmark (see BENCH.md)
```

Same via `python3 forge_cli.py …`.

---

## Design

- Primitives: `AGENT` `MEMORY` `STEP` `TOOL` `FILTER` `REASON` `VERIFY` `RETURN`
- Dual path: text syntax **or** JSON AST
- Pipeline: emit → repair → validate → check → run
- See [VISION.md](VISION.md) · [BENCH.md](BENCH.md) · [CREDIT_TEST.md](CREDIT_TEST.md)

## Tests

```bash
python3 -m unittest discover -s tests -v
```

## Reliability (local `llama3.2:1b`, measured before v0.2)

| Suite | Result |
|-------|--------|
| 5 tasks (text) | 100% |
| 18 tasks (text) | 77.8% |
| 18 tasks (JSON AST) | 0% on 1B (use 3B+ for JSON) |

These numbers predate error-feedback retries and list support; re-run `./forge bench` to update them.
