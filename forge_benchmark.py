"""
Forge v0.1 — LLM Generation Benchmark
======================================
Few-shot prompt LLMs to write Forge, measure parse + runtime success.

Free backends (no paid credits):
  # Local Ollama (recommended free path)
  ollama pull llama3.2
  python forge_benchmark.py --backend ollama

  # Free cloud tiers (signup, usually no card)
  export GROQ_API_KEY=...
  python forge_benchmark.py --backend openai

  export GEMINI_API_KEY=...
  python forge_benchmark.py --backend openai

  export OPENROUTER_API_KEY=...   # has :free models
  python forge_benchmark.py --backend openai

Paid / existing:
  export DEEPSEEK_API_KEY=... or OPENAI_API_KEY=...

Offline (no LLM generation):
  python forge_offline_bench.py --mutations
  python forge_benchmark.py --dry-run
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

from forge_core import (
    EXAMPLES,
    compile_auto,
    compile_forge,
    ForgeError,
    ast_to_json,
    _ast_to_dict,
)
from forge_runtime import (
    ToolRegistry,
    MockLLMClient,
    OllamaLLMClient,
    OpenAILLMClient,
    Evaluator,
    ForgeRuntimeError,
    ForgeVerifyError,
    ollama_available,
    make_llm_client,
)


# =============================================================================
# PROMPTS
# =============================================================================

FORGE_SPEC_BRIEF = """
Forge is an agent programming language. Rules:
- Keywords are UPPERCASE: AGENT, MEMORY, STEP, TOOL, FILTER, INPUT, OUTPUT, ON, REASON, VERIFY, RETURN
- Variables use $prefix: $var
- Program order: AGENT → MEMORY (optional) → STEP+ → REASON (optional) → VERIFY (optional) → RETURN
- STEP forms:
  STEP name TOOL tool_name INPUT { k: $v } OUTPUT result_var
  STEP name FILTER condition ON $var OUTPUT result_var
- Conditions: $a > 0, field > 0.8, $x != null  (ops: > < >= <= == !=)
- RETURN ends the program. Objects: { key: $var }
- No loops, no functions, no imports in v0.1.
- Emit ONLY Forge code, no markdown fences, no explanation.
""".strip()

FEW_SHOT: list[tuple[str, str]] = [
    (
        "Build a calculator agent that adds two numbers a=15 and b=7, explains the sum, verifies it is positive, and returns result + note.",
        EXAMPLES["basic-calculator"].strip(),
    ),
    (
        "Build a web researcher that searches for 'latest AI safety papers' (5 results), filters to relevance > 0.8, summarizes, verifies non-null, returns papers + summary.",
        EXAMPLES["web-researcher"].strip(),
    ),
    (
        "Build a tweet creator for topic 'AI safety' that researches 3 articles and writes a tweet, verifies tweet is non-empty, returns tweet and source_count 3.",
        EXAMPLES["tweet-creator"].strip(),
    ),
]

BENCHMARK_TASKS: list[dict[str, str]] = [
    {
        "id": "temp-converter",
        "description": (
            "Create an agent 'temp-converter' with MEMORY celsius: 100. "
            "Call TOOL arithmetic_add INPUT { x: $celsius, y: 0 } OUTPUT raw "
            "(placeholder for conversion). REASON a short note about the temperature ON $raw. "
            "VERIFY $raw > 0. RETURN { celsius: $celsius, note: $raw }."
        ),
    },
    {
        "id": "deal-finder",
        "description": (
            "Create agent 'deal-finder'. MEMORY region: \"west\", quarter: \"Q1\". "
            "STEP get TOOL sales_data INPUT { region: $region, period: $quarter } OUTPUT sales. "
            "STEP big FILTER amount > 10000 ON $sales OUTPUT big_deals. "
            "REASON 'What is the biggest opportunity?' ON $big_deals OUTPUT insight. "
            "VERIFY $big_deals != null. RETURN { deals: $big_deals, insight: $insight }."
        ),
    },
    {
        "id": "news-digest",
        "description": (
            "Create agent 'news-digest'. MEMORY query: \"quantum computing news\", max_results: 4. "
            "STEP search TOOL web_search INPUT { q: $query, n: $max_results } OUTPUT hits. "
            "STEP top FILTER relevance > 0.7 ON $hits OUTPUT best. "
            "REASON 'Write a 2-sentence digest' ON $best OUTPUT digest. "
            "VERIFY $best != null. RETURN { digest: $digest, count: $max_results }."
        ),
    },
    {
        "id": "threshold-gate",
        "description": (
            "Create agent 'threshold-gate'. MEMORY threshold: 40. "
            "STEP read TOOL get_value INPUT { } OUTPUT value. "
            "STEP gate FILTER $value > $threshold ON $value OUTPUT passed. "
            "VERIFY $passed != null. RETURN { value: $value, passed: $passed }."
        ),
    },
    {
        "id": "mini-add",
        "description": (
            "Create agent 'mini-add'. MEMORY x: 3, y: 9. "
            "STEP sum TOOL arithmetic_add INPUT { x: $x, y: $y } OUTPUT total. "
            "VERIFY $total > 10. RETURN { total: $total }."
        ),
    },
]


def build_prompt(task_description: str, mode: str = "text", shots: Optional[int] = None) -> str:
    """
    Few-shot prompt for Forge generation.

    Forge exists so AIs emit agent programs more reliably than in Python/JS.
    Keep prompts short: fewer tokens in → fewer tokens out → higher success.
    """
    if shots is None:
        shots = int(os.environ.get("FORGE_BENCH_SHOTS", "2"))
    examples = FEW_SHOT[: max(1, shots)]

    parts = [
        "Forge = AI-native agent language (not Python). Uppercase keywords. Variables use $.",
        FORGE_SPEC_BRIEF,
        "",
        "Examples:",
        "",
    ]
    for i, (desc, code) in enumerate(examples, 1):
        parts.append(f"### Example {i}")
        parts.append(f"Task: {desc}")
        if mode == "json":
            try:
                prog = compile_forge(code)
                parts.append("Forge JSON AST:")
                parts.append(ast_to_json(prog))
            except Exception:
                parts.append("Forge:")
                parts.append(code)
        else:
            parts.append("Forge:")
            parts.append(code)
        parts.append("")
    parts.append("### Your task")
    parts.append(f"Task: {task_description}")
    if mode == "json":
        parts.append("Write a complete Forge program as a single JSON AST object (node_type: program). No markdown.")
    else:
        parts.append("Write a complete Forge program in surface syntax. Stop after RETURN. No markdown, no explanation.")
    return "\n".join(parts)


# =============================================================================
# LLM CALL
# =============================================================================

def call_llm(
    prompt: str,
    model: Optional[str] = None,
    backend: str = "auto",
) -> tuple[str, dict]:
    """Returns (text, usage_meta). Supports ollama (free/local) and OpenAI-compatible APIs."""
    backend = (backend or "auto").lower()
    t0 = time.time()

    if backend == "auto":
        if ollama_available():
            backend = "ollama"
        elif any(
            os.environ.get(k)
            for k in (
                "GROQ_API_KEY",
                "GEMINI_API_KEY",
                "GOOGLE_API_KEY",
                "OPENROUTER_API_KEY",
                "OPENAI_API_KEY",
                "DEEPSEEK_API_KEY",
            )
        ):
            backend = "openai"
        else:
            raise RuntimeError(
                "No free LLM backend found.\n"
                "  • Install Ollama (free): https://ollama.com  then: ollama pull llama3.2\n"
                "  • Or set GROQ_API_KEY / GEMINI_API_KEY / OPENROUTER_API_KEY (free tiers)\n"
                "  • Or run offline: python forge_offline_bench.py --mutations"
            )

    if backend == "ollama":
        client = OllamaLLMClient(model=model)
        # Generation prompt for writing Forge (not REASON helper)
        text = _ollama_generate(client, prompt)
        elapsed = time.time() - t0
        return text, {
            "backend": "ollama",
            "model": client.model,
            "elapsed_s": round(elapsed, 3),
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }

    # OpenAI-compatible (Groq, Gemini, OpenRouter, OpenAI, DeepSeek)
    if model:
        os.environ.setdefault("FORGE_BENCH_MODEL", model)
    try:
        from openai import OpenAI
    except ImportError as e:
        raise RuntimeError("pip install openai") from e

    api_key, base_url, resolved_model = __import__(
        "forge_runtime", fromlist=["_resolve_openai_compat"]
    )._resolve_openai_compat(model=model)
    if not api_key:
        raise RuntimeError(
            "No API key for openai backend. Set GROQ_API_KEY, GEMINI_API_KEY, "
            "OPENROUTER_API_KEY, OPENAI_API_KEY, or DEEPSEEK_API_KEY."
        )

    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    client = OpenAI(**kwargs)
    resp = client.chat.completions.create(
        model=resolved_model,
        messages=[
            {"role": "system", "content": "You generate only valid Forge programs."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )
    elapsed = time.time() - t0
    text = resp.choices[0].message.content or ""
    usage = {
        "backend": "openai",
        "model": resolved_model,
        "base_url": base_url,
        "elapsed_s": round(elapsed, 3),
        "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
        "completion_tokens": getattr(resp.usage, "completion_tokens", None),
        "total_tokens": getattr(resp.usage, "total_tokens", None),
    }
    return text, usage


def _ollama_generate(client: OllamaLLMClient, prompt: str) -> str:
    """Use Ollama chat with a codegen system prompt. Hard cap tokens so small models finish."""
    import json as _json
    import urllib.request

    num_predict = int(os.environ.get("FORGE_OLLAMA_NUM_PREDICT", "400"))
    body = _json.dumps({
        "model": client.model,
        "stream": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You write Forge, an AI-native agent language. "
                    "Output ONLY a complete Forge program. "
                    "No markdown fences, no explanation, no Python."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "options": {
            "temperature": 0.1,
            "num_predict": num_predict,
        },
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{client.host}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = _json.loads(resp.read().decode("utf-8"))
    msg = data.get("message") or {}
    return msg.get("content") or data.get("response") or ""


def estimate_tokens(text: str) -> int:
    # Rough: ~4 chars per token
    return max(1, len(text) // 4)


# =============================================================================
# EXTRACTION
# =============================================================================

def extract_code(response: str, prefer_json: bool = False) -> str:
    """Pull Forge source or JSON out of an LLM response."""
    text = response.strip()

    # Fenced blocks
    fence = re.search(r"```(?:forge|json|javascript|text)?\s*([\s\S]*?)```", text, re.I)
    if fence:
        text = fence.group(1).strip()

    if prefer_json or text.lstrip().startswith("{"):
        # Extract outermost JSON object
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]

    # Surface syntax: from AGENT to end
    m = re.search(r"(AGENT\s+[\s\S]*)", text)
    if m:
        return m.group(1).strip()

    return text


# =============================================================================
# RESULT
# =============================================================================

@dataclass
class RunResult:
    task_id: str
    mode: str
    parse_ok: bool
    runtime_ok: bool
    verify_ok: bool
    raw_response: str
    extracted: str
    error: Optional[str] = None
    result: Any = None
    usage: dict = field(default_factory=dict)
    est_prompt_tokens: int = 0
    est_completion_tokens: int = 0

    def to_dict(self) -> dict:
        d = asdict(self)
        # result may not be JSON-serializable cleanly
        try:
            json.dumps(d["result"], default=str)
        except Exception:
            d["result"] = str(d["result"])
        return d


def evaluate_generation(task_id: str, response: str, mode: str = "text") -> RunResult:
    extracted = extract_code(response, prefer_json=(mode == "json"))
    est_c = estimate_tokens(extracted)
    rr = RunResult(
        task_id=task_id,
        mode=mode,
        parse_ok=False,
        runtime_ok=False,
        verify_ok=True,
        raw_response=response,
        extracted=extracted,
        est_completion_tokens=est_c,
    )
    try:
        program = compile_auto(extracted)
        rr.parse_ok = True
    except Exception as e:
        rr.error = f"parse: {e}"
        return rr

    try:
        tools = ToolRegistry()
        llm = MockLLMClient()  # reason blocks use mock during bench runtime
        result = Evaluator(tools=tools, llm=llm).run(program)
        rr.runtime_ok = True
        rr.result = result
    except ForgeVerifyError as e:
        rr.verify_ok = False
        rr.error = f"verify: {e}"
    except Exception as e:
        rr.error = f"runtime: {e}"

    return rr


# =============================================================================
# SUITE
# =============================================================================

def run_benchmark(
    mode: str = "text",
    tasks: Optional[list[dict]] = None,
    dry_run: bool = False,
    backend: str = "auto",
) -> list[RunResult]:
    tasks = tasks or BENCHMARK_TASKS
    results: list[RunResult] = []

    for task in tasks:
        tid = task["id"]
        desc = task["description"]
        prompt = build_prompt(desc, mode=mode)
        est_p = estimate_tokens(prompt)
        print(f"\n--- Task: {tid} (mode={mode}, backend={backend}) ---")

        if dry_run:
            # Prefer golden offline programs when available
            try:
                from forge_offline_bench import GOLDEN
                fake = GOLDEN.get(tid, EXAMPLES.get(tid, EXAMPLES["basic-calculator"]))
            except Exception:
                fake = EXAMPLES.get(tid, EXAMPLES["basic-calculator"])
            rr = evaluate_generation(tid, fake, mode="text")
            rr.est_prompt_tokens = est_p
            rr.usage = {"dry_run": True, "backend": "dry-run"}
            results.append(rr)
            print(f"  dry-run parse={rr.parse_ok} runtime={rr.runtime_ok}")
            continue

        try:
            response, usage = call_llm(prompt, backend=backend)
        except Exception as e:
            rr = RunResult(
                task_id=tid,
                mode=mode,
                parse_ok=False,
                runtime_ok=False,
                verify_ok=False,
                raw_response="",
                extracted="",
                error=f"llm: {e}",
                est_prompt_tokens=est_p,
            )
            results.append(rr)
            print(f"  LLM error: {e}")
            continue

        rr = evaluate_generation(tid, response, mode=mode)
        rr.usage = usage
        rr.est_prompt_tokens = est_p
        results.append(rr)
        status = "OK" if rr.parse_ok and rr.runtime_ok else "FAIL"
        print(f"  {status} parse={rr.parse_ok} runtime={rr.runtime_ok} err={rr.error}")
        if rr.extracted:
            preview = rr.extracted[:200].replace("\n", " ")
            print(f"  extracted: {preview}...")

    return results


def summarize(results: list[RunResult]) -> dict:
    n = len(results) or 1
    parse_n = sum(1 for r in results if r.parse_ok)
    run_n = sum(1 for r in results if r.runtime_ok)
    both = sum(1 for r in results if r.parse_ok and r.runtime_ok)
    summary = {
        "total": len(results),
        "parse_success": parse_n,
        "parse_rate": round(100.0 * parse_n / n, 1),
        "runtime_success": run_n,
        "runtime_rate": round(100.0 * run_n / n, 1),
        "full_success": both,
        "full_rate": round(100.0 * both / n, 1),
        "target_rate": 97.0,
        "meets_target": (100.0 * both / n) >= 97.0,
    }
    return summary


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Forge LLM generation benchmark")
    p.add_argument("--mode", choices=["text", "json"], default="text")
    p.add_argument(
        "--backend",
        default="auto",
        help="auto | ollama | openai  (ollama is free/local; openai also covers Groq/Gemini/OpenRouter)",
    )
    p.add_argument("--dry-run", action="store_true", help="Skip LLM; use golden programs")
    p.add_argument("--out", default="benchmark_results.json")
    args = p.parse_args()

    print("=" * 60)
    print("Forge Benchmark v0.1")
    print("=" * 60)
    print(f"backend={args.backend}  ollama_available={ollama_available()}")

    if not args.dry_run and args.backend == "auto":
        has_key = any(
            os.environ.get(k)
            for k in (
                "GROQ_API_KEY",
                "GEMINI_API_KEY",
                "GOOGLE_API_KEY",
                "OPENROUTER_API_KEY",
                "OPENAI_API_KEY",
                "DEEPSEEK_API_KEY",
            )
        )
        if not ollama_available() and not has_key:
            print("No free backend found (no Ollama, no API keys).")
            print("Falling back to golden dry-run. For real LLM scores:")
            print("  ollama pull llama3.2 && python forge_benchmark.py --backend ollama")
            print("  or: export GROQ_API_KEY=... (free tier)")
            print("  or: python forge_offline_bench.py --mutations")
            args.dry_run = True

    results = run_benchmark(mode=args.mode, dry_run=args.dry_run, backend=args.backend)
    summary = summarize(results)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v}")

    out = {
        "summary": summary,
        "results": [r.to_dict() for r in results],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {args.out}")

    if summary["full_rate"] < 97 and not args.dry_run:
        print("\nBelow 97% target — iterate syntax / few-shot examples.")


if __name__ == "__main__":
    main()
