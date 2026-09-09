# Forge Vision — JavaScript for AI

## The point

Forge is not "another scripting language."  
It is a **programming language designed for AI agents and LLMs** — by AI, for AI.

Humans built Python, JavaScript, and Java for human readability, historical accidents, and decades of library ecosystem. Those languages force models to:

- invent syntax they half-remember  
- fight operator precedence and implicit coercion  
- wrap agent logic in frameworks (LangChain, etc.) that are still just Python  
- burn tokens on boilerplate that means nothing to the agent loop  

**Forge inverts that.** The language is the agent loop.

## Why this is more efficient for AI

| Legacy (Python/JS) | Forge |
|--------------------|--------|
| Human-centric syntax | Agent-first primitives: `AGENT`, `STEP`, `TOOL`, `REASON`, `VERIFY`, `RETURN` |
| Ambiguous grammar | Strict, one valid interpretation |
| Parse → fail → retry | Dual path: text **or** JSON AST (skip parse entirely) |
| Libraries for "agents" | Agent workflow **is** the program |
| Tokens wasted on ceremony | Minimal surface; `$vars`, uppercase keywords, linear flow |

Efficiency here means:

1. **Higher generation success** — models emit correct programs more often after few examples  
2. **Fewer tokens** — shorter programs, optional JSON AST mode  
3. **Deterministic execution** — VERIFY fail-fast, no hidden coercion  
4. **Native mental model** — STEP / TOOL / REASON matches how agents actually run  

## Dual representation (the killer feature)

```
Human or LLM text  ──►  Lexer/Parser  ──►  AST  ──►  Runtime
LLM JSON AST       ─────────────────────►  AST  ──►  Runtime
```

Same nodes. Same semantics. LLMs can skip syntax entirely when that is more reliable.

## Locked identity

- **Name:** Forge  
- **Tagline:** JavaScript for AI  
- **Role split:** Human = vision; AI = implementation, syntax, tests, iteration  
- **v0.1:** linear agent workflows (no loops/functions yet)  

## Success metric

Primary: **LLM generation success ≥ 97%** on held-out agent tasks  
(with free local models via Ollama, or cloud free tiers — not locked to paid APIs)

## What we are not

- Not "better Python"  
- Not a LangChain clone  
- Not a chat prompt template  

Forge is **SQL for AI agents**: a declarative, purpose-built language for one critical job — defining and running agent workflows reliably.
