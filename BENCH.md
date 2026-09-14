# Forge generation reliability

Primary metric: **LLM can emit Forge that parses and runs** after few-shot prompting.

Model under test: local **Ollama `llama3.2:1b`** (free, small — stress test, not ceiling).

## Results

| Suite | Mode | Tasks | Full success | Notes |
|-------|------|-------|--------------|-------|
| Original | text | 5 | **100%** | After AI-friendly syntax + repair |
| Expanded | text | 18 | **77.8%** (14/18) | Parse 88.9% |
| Expanded | json AST | 18 | **0%** | 1B model truncates / invalid JSON |

Target: ≥97% on expanded suite (needs stronger model and/or more repair, or JSON with 7B+).

## What works on tiny models

- Surface **text** Forge + `forge_repair` pipeline  
- Short linear agents (add / search / filter / return)

## What does not (yet) on `llama3.2:1b`

- Full **JSON AST** emission (objects too large; JSON breaks mid-stream)  
- Inventing multi-step logic when the model drops a STEP (VERIFY then correctly fails)  
- Hallucinated MEMORY arrays (`[{...}]`) — not in v0.1

## How to re-run

```bash
./start_ollama.sh
python3 forge_cli.py bench --backend ollama --mode text --out benchmark_results_text18.json
python3 forge_cli.py bench --backend ollama --mode json --out benchmark_results_json18.json
```

Stronger local model (recommended next):

```bash
ollama pull llama3.2   # 3B
OLLAMA_MODEL=llama3.2 python3 forge_cli.py bench --backend ollama --mode both
```

## Design takeaway

Forge’s dual representation is still right: **JSON AST is the reliability path for capable models**; **text + repair** is the path for small local models. The language stays AI-native either way.
