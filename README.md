# Forge — JavaScript for AI

**Repo:** https://github.com/kungFuDOOM/forge  
**Tagline:** JavaScript for AI

**Agent-first programming language** optimized for LLMs: explicit, unambiguous, token-efficient.

## Dual representation

| Path | Input | Use case |
|------|--------|----------|
| Surface syntax | Human-readable text | Humans, demos |
| JSON AST | Strict AST object | LLM generation (skip parsing) |

Both compile to identical AST nodes.

## Quick start

```bash
cd forge
python forge_core.py              # parser tests
python forge_runtime.py           # runtime tests (mock LLM)
python forge_offline_bench.py --mutations   # full offline metrics (no API)
```

### Free LLM generation benchmark (no paid credits)

**Option A — Ollama (fully free, local)**

```bash
# Install from https://ollama.com then:
ollama pull llama3.2
python forge_benchmark.py --backend ollama
```

**Option B — Free cloud tiers** (signup; usually no credit card)

```bash
export GROQ_API_KEY=...          # https://console.groq.com  (fast free tier)
# or
export GEMINI_API_KEY=...        # https://aistudio.google.com/apikey
# or
export OPENROUTER_API_KEY=...    # free models end with :free

python forge_benchmark.py --backend openai
python forge_benchmark.py --mode json
```

**Option C — Offline only** (language correctness + mutation stress)

```bash
python forge_offline_bench.py --mutations
```

DeepSeek/OpenAI still work if you have balance; they are optional.

REPL:

```bash
python forge_runtime.py repl
```

## Example

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

## Files

| File | Role |
|------|------|
| `forge_core.py` | AST, lexer, parser, `dict_to_ast`, validator |
| `forge_runtime.py` | Evaluator, tools, LLM clients, REPL |
| `forge_benchmark.py` | Few-shot LLM generation harness |

## Design (locked)

- **AI-native language** — built for how models generate code, not 1990s human IDE habits  
- Uppercase keywords, `$variables`, no operator precedence surprises  
- Primitives: `AGENT`, `MEMORY`, `STEP`, `TOOL`, `FILTER`, `REASON`, `VERIFY`, `RETURN`  
- Dual path: surface text **or** JSON AST  
- v0.1: linear workflows only (no loops/functions yet)  
- See [VISION.md](VISION.md)

## Metrics

- Primary: LLM generation success ≥ 97%
- Secondary: token efficiency vs Python
- Tertiary: execution correctness ≥ 95%

### Latest local bench (Ollama `llama3.2:1b`, free)

| Run | Full success |
|-----|----------------|
| First pass (strict syntax) | **20%** |
| After AI-friendly parse fixes | **80%** |

Still below 97% — next: better few-shots, larger local model, JSON AST mode.
