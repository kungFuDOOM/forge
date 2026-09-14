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
Forge is an AI-native agent language (not Python). Emit ONLY Forge.

Skeleton (always this order):
  AGENT "name"
  MEMORY { key: value }
  STEP name TOOL tool_name INPUT { k: $var } OUTPUT out_var
  STEP name FILTER field > 0.8 ON $list OUTPUT filtered
  REASON "prompt" ON $var OUTPUT out_var
  VERIFY $var != null
  RETURN { key: $var }

Rules:
- Keywords UPPERCASE only.
- Variables always $name.
- Strings use double quotes.
- MEMORY uses braces: MEMORY { a: 1 b: 2 }
- At least one STEP. RETURN is last.
- Tools available: arithmetic_add, web_search, get_value, sales_data.
- No markdown. No explanation. Stop after RETURN.
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
            "Call TOOL arithmetic_add INPUT { x: $celsius, y: 0 } OUTPUT raw. "
            "REASON \"short note about the temperature\" ON $raw OUTPUT note. "
            "VERIFY $raw > 0. RETURN { celsius: $celsius, note: $note }."
        ),
    },
    {
        "id": "deal-finder",
        "description": (
            "Create agent 'deal-finder'. MEMORY region: \"west\", quarter: \"Q1\". "
            "STEP get TOOL sales_data INPUT { region: $region, period: $quarter } OUTPUT sales. "
            "STEP big FILTER amount > 10000 ON $sales OUTPUT big_deals. "
            "REASON \"What is the biggest opportunity?\" ON $big_deals OUTPUT insight. "
            "VERIFY $big_deals != null. RETURN { deals: $big_deals, insight: $insight }."
        ),
    },
    {
        "id": "news-digest",
        "description": (
            "Create agent 'news-digest'. MEMORY query: \"quantum computing news\", max_results: 4. "
            "STEP search TOOL web_search INPUT { q: $query, n: $max_results } OUTPUT hits. "
            "STEP top FILTER relevance > 0.7 ON $hits OUTPUT best. "
            "REASON \"Write a 2-sentence digest\" ON $best OUTPUT digest. "
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
    {
        "id": "double-add",
        "description": (
            "Agent 'double-add'. MEMORY a: 5, b: 5. "
            "STEP s1 TOOL arithmetic_add INPUT { x: $a, y: $b } OUTPUT mid. "
            "STEP s2 TOOL arithmetic_add INPUT { x: $mid, y: $b } OUTPUT total. "
            "VERIFY $total > 10. RETURN { mid: $mid, total: $total }."
        ),
    },
    {
        "id": "east-sales",
        "description": (
            "Agent 'east-sales'. MEMORY region: \"east\", quarter: \"Q2\". "
            "STEP pull TOOL sales_data INPUT { region: $region, period: $quarter } OUTPUT raw. "
            "STEP huge FILTER amount > 12000 ON $raw OUTPUT big. "
            "VERIFY $big != null. RETURN { region: $region, deals: $big }."
        ),
    },
    {
        "id": "safety-tweet",
        "description": (
            "Agent 'safety-tweet'. MEMORY topic: \"AI alignment\", max_chars: 280. "
            "STEP research TOOL web_search INPUT { q: $topic, n: 3 } OUTPUT articles. "
            "REASON \"Write one engaging tweet\" ON $articles OUTPUT tweet. "
            "VERIFY $tweet != \"\". RETURN { tweet: $tweet }."
        ),
    },
    {
        "id": "value-check",
        "description": (
            "Agent 'value-check'. MEMORY threshold: 10. "
            "STEP read TOOL get_value INPUT { } OUTPUT value. "
            "VERIFY $value > $threshold. RETURN { value: $value }."
        ),
    },
    {
        "id": "paper-skim",
        "description": (
            "Agent 'paper-skim'. MEMORY query: \"LLM evaluation benchmarks\", max_results: 5. "
            "STEP search TOOL web_search INPUT { q: $query, n: $max_results } OUTPUT results. "
            "STEP keep FILTER relevance > 0.75 ON $results OUTPUT top. "
            "REASON \"List top themes in 3 bullets\" ON $top OUTPUT themes. "
            "VERIFY $top != null. RETURN { themes: $themes, papers: $top }."
        ),
    },
    {
        "id": "sum-only",
        "description": (
            "Agent 'sum-only'. MEMORY x: 100, y: 1. "
            "STEP add TOOL arithmetic_add INPUT { x: $x, y: $y } OUTPUT sum. "
            "RETURN { sum: $sum }."
        ),
    },
    {
        "id": "north-q4",
        "description": (
            "Agent 'north-q4'. MEMORY region: \"north\", quarter: \"Q4\". "
            "STEP get TOOL sales_data INPUT { region: $region, period: $quarter } OUTPUT sales. "
            "REASON \"One-sentence sales outlook\" ON $sales OUTPUT outlook. "
            "VERIFY $sales != null. RETURN { outlook: $outlook }."
        ),
    },
    {
        "id": "filter-gate",
        "description": (
            "Agent 'filter-gate'. MEMORY threshold: 30. "
            "STEP read TOOL get_value INPUT { } OUTPUT raw. "
            "STEP gate FILTER $raw > $threshold ON $raw OUTPUT passed. "
            "VERIFY $passed != null. RETURN { raw: $raw, passed: $passed }."
        ),
    },
    {
        "id": "multi-search",
        "description": (
            "Agent 'multi-search'. MEMORY query: \"rust async runtime\", n: 2. "
            "STEP s TOOL web_search INPUT { q: $query, n: $n } OUTPUT hits. "
            "STEP f FILTER relevance > 0.5 ON $hits OUTPUT kept. "
            "VERIFY $kept != null. RETURN { kept: $kept, n: $n }."
        ),
    },
    {
        "id": "add-explain",
        "description": (
            "Agent 'add-explain'. MEMORY x: 2, y: 2. "
            "STEP add TOOL arithmetic_add INPUT { x: $x, y: $y } OUTPUT sum. "
            "REASON \"Explain the sum in five words\" ON $sum OUTPUT note. "
            "VERIFY $sum > 0. RETURN { sum: $sum, note: $note }."
        ),
    },
    {
        "id": "west-filter",
        "description": (
            "Agent 'west-filter'. MEMORY region: \"west\", quarter: \"Q3\". "
            "STEP get TOOL sales_data INPUT { region: $region, period: $quarter } OUTPUT sales. "
            "STEP f FILTER amount > 9000 ON $sales OUTPUT deals. "
            "REASON \"Name the strongest deal pattern\" ON $deals OUTPUT pattern. "
            "VERIFY $deals != null. RETURN { deals: $deals, pattern: $pattern }."
        ),
    },
    {
        "id": "hello-search",
        "description": (
            "Agent 'hello-search'. MEMORY topic: \"hello world programming\". "
            "STEP research TOOL web_search INPUT { q: $topic, n: 3 } OUTPUT hits. "
            "VERIFY $hits != null. RETURN { topic: $topic, hits: $hits }."
        ),
    },
    {
        "id": "triple-sum",
        "description": (
            "Agent 'triple-sum'. MEMORY a: 1, b: 2. "
            "STEP s1 TOOL arithmetic_add INPUT { x: $a, y: $b } OUTPUT t1. "
            "STEP s2 TOOL arithmetic_add INPUT { x: $t1, y: $a } OUTPUT t2. "
            "VERIFY $t2 >= 4. RETURN { t1: $t1, t2: $t2 }."
        ),
    },
]

JSON_AST_BRIEF = """
Emit ONLY a JSON object (Forge AST). Root node_type must be "program".
Required keys: agent, memory (or null), steps (>=1), reason (or null), verify (or null), return.
agent: {"node_type":"agent_decl","name":"..."}
memory: {"node_type":"memory_block","entries":[{"node_type":"memory_entry","key":"k","value":{"node_type":"literal","value":1}}]}
tool step action: {"node_type":"tool_call","tool_name":"arithmetic_add","inputs":{"x":{"node_type":"variable","name":"a"},"y":{"node_type":"variable","name":"b"}},"output_var":"sum"}
filter action: {"node_type":"filter","condition":{"node_type":"comparison","left":"amount","operator":">","right":{"node_type":"literal","value":10000},"left_kind":"field"},"input_var":"sales","output_var":"big"}
reason: {"node_type":"reason","prompt":"...","input_var":"sum","output_var":"note"}
verify: {"node_type":"verify","condition":{"node_type":"comparison","left":{"node_type":"variable","name":"sum"},"operator":">","right":{"node_type":"literal","value":0},"left_kind":"expr"}}
return: {"node_type":"return","value":{"node_type":"object_literal","properties":{"total":{"node_type":"variable","name":"sum"}}}}
No markdown. JSON only.
""".strip()


def build_prompt(task_description: str, mode: str = "text", shots: Optional[int] = None) -> str:
    """
    Few-shot prompt for Forge generation.

    mode=text  → surface syntax
    mode=json  → JSON AST (preferred for LLM reliability / fewer syntax failures)
    """
    if shots is None:
        # JSON AST examples are large — default to 1 shot for small local models
        default_shots = "1" if mode == "json" else "2"
        shots = int(os.environ.get("FORGE_BENCH_SHOTS", default_shots))
    examples = FEW_SHOT[: max(1, shots)]

    if mode == "json":
        # Prefer shortest examples — JSON AST few-shots are token-heavy
        mini = (
            "Agent 'mini-add'. MEMORY x: 3, y: 9. "
            "STEP sum TOOL arithmetic_add INPUT { x: $x, y: $y } OUTPUT total. "
            "VERIFY $total > 10. RETURN { total: $total }."
        )
        mini_code = '''
AGENT "mini-add"
MEMORY { x: 3 y: 9 }
STEP sum TOOL arithmetic_add INPUT { x: $x, y: $y } OUTPUT total
VERIFY $total > 10
RETURN { total: $total }
'''.strip()
        json_examples = [(mini, mini_code)] + [
            (d, c) for d, c in examples if "mini-add" not in d.lower()
        ]
        json_examples = json_examples[: max(1, shots)]

        parts = [
            "Forge JSON AST mode: emit a program as JSON (skip text syntax).",
            JSON_AST_BRIEF,
            "",
            "Examples:",
            "",
        ]
        for i, (desc, code) in enumerate(json_examples, 1):
            parts.append(f"### Example {i}")
            parts.append(f"Task: {desc}")
            try:
                prog = compile_forge(code)
                parts.append("JSON AST:")
                parts.append(json.dumps(_ast_to_dict(prog), separators=(",", ":")))
            except Exception:
                parts.append("(example omitted)")
            parts.append("")
        parts.append("### Your task")
        parts.append(f"Task: {task_description}")
        parts.append("Return ONLY the JSON AST object for this task.")
        return "\n".join(parts)

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
        parts.append("Forge:")
        parts.append(code)
        parts.append("")
    parts.append("### Your task")
    parts.append(f"Task: {task_description}")
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
    """Pull Forge source or JSON out of an LLM response, then repair near-misses."""
    from forge_repair import normalize_llm_output

    return normalize_llm_output(response, prefer_json=prefer_json)


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
    from forge_repair import try_compile_repaired

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
        program, used, notes = try_compile_repaired(response if mode != "json" else extracted)
        rr.extracted = used
        rr.parse_ok = True
        if notes:
            rr.usage = {**(rr.usage or {}), "repair": notes}
    except Exception as e:
        # fallback: direct compile of extracted
        try:
            program = compile_auto(extracted)
            rr.parse_ok = True
        except Exception as e2:
            rr.error = f"parse: {e2}"
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
        rr.usage = {**(rr.usage or {}), **usage}
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
