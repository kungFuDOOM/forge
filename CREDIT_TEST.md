# Forge vs Python — credit / token test

## Can I run this without you?

Yes:

```bash
./forge credit-test --backend ollama --tasks 5 --retries 3
# lean = steady-state (minimal teach prompt):
./forge credit-test --backend ollama --tasks 5 --lean --retries 3
# paid key:
export GROQ_API_KEY=...   # or GEMINI / OPENAI / OPENROUTER
./forge credit-test --backend openai --tasks 5 --retries 3
```

## Why the first run looked worse for Forge

The first live run (before harness fixes) showed Forge using **more** tokens (−35%). That was a measurement bug around the thesis, not proof the language is denser-expensive.

| Root cause | What happened | Fix |
|------------|---------------|-----|
| **Runaway completions** | 2/5 Forge gens hit `num_predict=350` (babble past `RETURN`) → 700 wasted output tokens | Cap Forge at 180 + stop sequences |
| **Teach-prompt tax** | `FORGE_SPEC_BRIEF` was ~50% larger than the Python spec (~+90 input tok/task) | Slim credit-test spec to Python parity |
| **First-try-only accounting** | Forge won reliability (4/5 vs 3/5) but retries weren’t charged — hidden Python cost | `--retries N` charges every attempt |
| **Tiny 1B model** | `llama3.2:1b` often echoes few-shots / opens ` ``` ` fences | Stronger “start with AGENT” instruction; still fragile on 1B |

**Language density is about completion size and retries.** Cold-start teach tokens amortize once a model knows Forge (`--lean` approximates that).

## Latest live run (local Ollama `llama3.2:1b`, 5 tasks, `--retries 3`)

| Metric | Forge | Python |
|--------|-------|--------|
| Success (within 3 tries) | **4/5 (80%)** | 2/5 (40%) |
| Total tokens (all attempts) | **3112** | 5290 |
| Est. USD @ $0.14/1M in, $0.28/1M out | **$0.000512** | $0.000927 |
| Token savings (Forge vs Python) | **+41.2%** | |
| USD savings | **+44.8%** | |
| Avg extracted program size | ~58 tok | ~79 tok |

### What this means

1. **With fair retry accounting + stop caps, Forge used fewer credits** on this suite — mainly because Python burned 3 attempts on several tasks while Forge often succeeded first try with shorter programs.
2. **Completion tokens favored Forge** (545 vs 1331) — that is the “efficient language” claim.
3. **Local Ollama billed $0.** Token counts are real; dollar figures are “if you paid these rates.”
4. **1B is a harsh teacher.** Re-run on `llama3.2` / Groq / Gemini for a cleaner generation signal.

### Flags that change the story

| Flag | Effect |
|------|--------|
| *(default)* | Cold few-shot teach — includes language-teach tax |
| `--lean` | Minimal prompts — steady-state density |
| `--retries 3` | Charges every failed generation (real credit cost) |
| `--tasks 10` | Broader sample |

```bash
OLLAMA_MODEL=llama3.2 ./forge credit-test --backend ollama --tasks 10 --retries 3
```
