# Forge — JavaScript for AI

**An AI-native programming language for agents and LLMs.**  
Made by AI, for AI — explicit, unambiguous, dual text + JSON AST.

**Repo:** https://github.com/kungFuDOOM/forge

---

## Why Forge exists

Python and JavaScript were built for humans decades ago. Agents forced into those languages fight syntax, precedence, and framework glue.

**Forge is the agent loop as a language:**

```
AGENT → MEMORY → STEP/TOOL → REASON → VERIFY → RETURN
```

LLMs can emit **surface syntax** or **JSON AST** (skip parsing entirely). Same semantics.

---

## New user — 60 seconds

```bash
git clone https://github.com/kungFuDOOM/forge.git
cd forge

# 1) See it work (no API key)
python3 forge_cli.py examples basic-calculator

# 2) Run a file
python3 forge_cli.py run examples/basic_calculator.forge

# 3) Scaffold your own agent
python3 forge_cli.py init my-researcher
python3 forge_cli.py run my-researcher.forge

# 4) Check setup
python3 forge_cli.py doctor
```

### Optional: free local LLM (Ollama)

```bash
./start_ollama.sh
ollama pull llama3.2:1b   # or llama3.2 for stronger generation

# Ask an AI to write Forge, then run it
python3 forge_generate.py --backend ollama "Add 3 and 9, verify total > 10, return total"

# Measure generation reliability
python3 forge_cli.py bench --backend ollama
```

---

## CLI reference

| Command | What it does |
|---------|----------------|
| `python forge_cli.py examples` | List / run built-in demos |
| `python forge_cli.py run FILE.forge` | Compile + execute |
| `python forge_cli.py check FILE.forge` | Parse + validate (+ `--json` AST) |
| `python forge_cli.py init NAME` | Scaffold a starter agent |
| `python forge_cli.py repl` | Interactive REPL |
| `python forge_cli.py doctor` | Local setup check |
| `python forge_cli.py bench` | LLM generation benchmark |
| `python forge_generate.py "…"` | LLM writes Forge → repair → run |

---

## Example program

```
AGENT "basic-calculator"

MEMORY {
  a: 15
  b: 7
}

STEP add TOOL arithmetic_add INPUT { x: $a, y: $b } OUTPUT sum

REASON "Explain what the sum represents in a short sentence" ON $sum OUTPUT explanation

VERIFY $sum > 0

RETURN { result: $sum, note: $explanation }
```

---

## Dual representation

| Path | Input | Best for |
|------|--------|----------|
| Surface syntax | Human-readable text | Humans, demos |
| JSON AST | Strict AST object | LLM generation (zero syntax errors) |

Both compile to identical AST nodes. JSON AST is the source of truth.

---

## Design (locked)

- Agent-first primitives: `AGENT`, `MEMORY`, `STEP`, `TOOL`, `FILTER`, `REASON`, `VERIFY`, `RETURN`
- `$variables`, uppercase keywords, no precedence surprises
- Generation pipeline: **emit → repair → validate → run**
- v0.1: linear workflows (loops/functions later)
- See [VISION.md](VISION.md)

---

## Reliability metrics

| Target | Metric |
|--------|--------|
| ≥ 97% | LLM generation success (parse + run) |
| High | Token efficiency vs Python agent glue |
| ≥ 95% | Execution correctness |

| Suite | Model | Result |
|-------|-------|--------|
| 5 tasks (text) | `llama3.2:1b` | **100%** |
| **18 tasks (text)** | `llama3.2:1b` | **77.8%** |
| 18 tasks (JSON AST) | `llama3.2:1b` | **0%** (too small to emit valid AST JSON) |

See [BENCH.md](BENCH.md). Next: stronger local model (`llama3.2` 3B) + keep repair climbing toward **≥97% on 18+ tasks**.

---

## Project layout

| File | Role |
|------|------|
| `forge_core.py` | AST, lexer, parser, JSON AST, validator |
| `forge_runtime.py` | Evaluator, tools, LLM clients, REPL |
| `forge_repair.py` | LLM output normalize / repair |
| `forge_cli.py` | User-facing CLI |
| `forge_generate.py` | Natural language → Forge → run |
| `forge_benchmark.py` | Generation reliability harness |
| `examples/*.forge` | Sample agents |

---

## Built-in mock tools

`arithmetic_add` · `web_search` · `get_value` · `sales_data`

Register your own:

```python
from forge_runtime import ToolRegistry
reg = ToolRegistry()
reg.register("my_tool", lambda inputs: {"ok": True})
```
