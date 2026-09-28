# Forge vs Python — credit / token test

## Can I run this without you?

Yes:

```bash
./forge credit-test --backend ollama --tasks 5
# or with a paid key:
export GROQ_API_KEY=...   # or GEMINI / OPENAI / OPENROUTER
./forge credit-test --backend openai --tasks 5
```

## Latest live run (local Ollama `llama3.2:1b`, 5 tasks)

| Metric | Forge | Python |
|--------|-------|--------|
| First-try success | **4/5 (80%)** | 3/5 (60%) |
| Total tokens | 3252 | **2409** |
| Est. USD @ $0.14/1M in, $0.28/1M out | $0.000592 | **$0.000412** |
| Token “savings” (Forge vs Python) | **-35%** (Forge used **more**) | |

### What this means

1. **On this small model, Forge did not use fewer tokens** for a single generation pass. The Forge few-shot prompt is larger (teaching a new language), and some completions hit the 350-token cap.
2. **Forge won on first-try success** (4/5 vs 3/5). Retries cost real money — that is where Forge can still win end-to-end.
3. **Local Ollama billed $0.** Token counts are real; dollar figures are “if you paid these rates.”

### Fair next tests

- Stronger model (`ollama pull llama3.2` or Groq/Gemini free tier)
- More tasks (`--tasks 18`)
- Count retries until success (effective credit cost)

```bash
OLLAMA_MODEL=llama3.2 ./forge credit-test --backend ollama --tasks 10
```
