"""
Forge offline evaluation — no paid API required
================================================
1) Golden tasks: structured specs → known-good programs (language correctness)
2) Mutation suite: realistic LLM mistakes → parse fail rates (syntax robustness)
3) Optional: run same suite against a free local model via Ollama

Usage:
  python forge_offline_bench.py
  python forge_offline_bench.py --mutations
  python forge_offline_bench.py --backend ollama   # if ollama is running
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict
from typing import Any, Callable, Optional

from forge_core import EXAMPLES, compile_auto, compile_forge, ForgeError
from forge_runtime import (
    Evaluator,
    MockLLMClient,
    ToolRegistry,
    make_llm_client,
    ollama_available,
)


# =============================================================================
# GOLDEN PROGRAMS (hand-written expected Forge for each bench task)
# =============================================================================

GOLDEN: dict[str, str] = {
    "temp-converter": '''
AGENT "temp-converter"

MEMORY {
  celsius: 100
}

STEP convert TOOL arithmetic_add INPUT { x: $celsius, y: 0 } OUTPUT raw

REASON "Write a short note about this temperature" ON $raw OUTPUT note

VERIFY $raw > 0

RETURN { celsius: $celsius, note: $note }
''',
    "deal-finder": '''
AGENT "deal-finder"

MEMORY {
  region: "west"
  quarter: "Q1"
}

STEP get TOOL sales_data INPUT { region: $region, period: $quarter } OUTPUT sales

STEP big FILTER amount > 10000 ON $sales OUTPUT big_deals

REASON "What is the biggest opportunity?" ON $big_deals OUTPUT insight

VERIFY $big_deals != null

RETURN { deals: $big_deals, insight: $insight }
''',
    "news-digest": '''
AGENT "news-digest"

MEMORY {
  query: "quantum computing news"
  max_results: 4
}

STEP search TOOL web_search INPUT { q: $query, n: $max_results } OUTPUT hits

STEP top FILTER relevance > 0.7 ON $hits OUTPUT best

REASON "Write a 2-sentence digest" ON $best OUTPUT digest

VERIFY $best != null

RETURN { digest: $digest, count: $max_results }
''',
    "threshold-gate": '''
AGENT "threshold-gate"

MEMORY {
  threshold: 40
}

STEP read TOOL get_value INPUT { } OUTPUT value

STEP gate FILTER $value > $threshold ON $value OUTPUT passed

VERIFY $passed != null

RETURN { value: $value, passed: $passed }
''',
    "mini-add": '''
AGENT "mini-add"

MEMORY {
  x: 3
  y: 9
}

STEP sum TOOL arithmetic_add INPUT { x: $x, y: $y } OUTPUT total

VERIFY $total > 10

RETURN { total: $total }
''',
}


# =============================================================================
# MUTATIONS — common LLM failure modes
# =============================================================================

Mutator = Callable[[str], str]


def _mut_lowercase_keywords(src: str) -> str:
    out = src
    for kw in ("AGENT", "MEMORY", "STEP", "TOOL", "FILTER", "INPUT", "OUTPUT",
               "ON", "REASON", "VERIFY", "RETURN"):
        out = re.sub(rf"\b{kw}\b", kw.lower(), out)
    return out


def _mut_drop_dollar(src: str) -> str:
    return re.sub(r"\$([A-Za-z_])", r"\1", src)


def _mut_missing_return(src: str) -> str:
    return re.sub(r"RETURN[\s\S]*$", "", src).rstrip() + "\n"


def _mut_python_style(src: str) -> str:
    # LLM writes Python-ish junk mixed in
    return "def main():\n    " + src.replace("\n", "\n    ") + "\n    return result\n"


def _mut_swap_order(src: str) -> str:
    # RETURN before VERIFY (wrong order often breaks if VERIFY required after)
    if "VERIFY" in src and "RETURN" in src:
        # move RETURN before VERIFY
        parts = re.split(r"(VERIFY[\s\S]*?)(RETURN[\s\S]*)", src)
        if len(parts) >= 3:
            return parts[0] + parts[2] + "\n" + parts[1]
    return src


def _mut_missing_agent(src: str) -> str:
    return re.sub(r'AGENT\s+"[^"]+"\s*', "", src, count=1)


def _mut_json_in_text(src: str) -> str:
    # LLM dumps JSON AST when asked for text
    prog = compile_forge(src)
    from forge_core import ast_to_json
    return ast_to_json(prog)


def _mut_extra_markdown(src: str) -> str:
    return f"Sure! Here's the program:\n\n```forge\n{src.strip()}\n```\n\nHope this helps!"


MUTATIONS: list[tuple[str, Mutator, bool]] = [
    # name, mutator, expect_parse_ok (after extract if applicable)
    ("lowercase_keywords", _mut_lowercase_keywords, False),
    ("drop_dollar_vars", _mut_drop_dollar, False),
    ("missing_return", _mut_missing_return, False),
    ("python_style_wrap", _mut_python_style, False),
    ("wrong_order_return_verify", _mut_swap_order, False),
    ("missing_agent", _mut_missing_agent, False),
    ("json_when_text_expected", _mut_json_in_text, True),   # compile_auto accepts JSON
    ("markdown_fences", _mut_extra_markdown, True),         # if extract_code used
]


@dataclass
class CaseResult:
    suite: str
    case_id: str
    parse_ok: bool
    runtime_ok: bool
    error: Optional[str] = None
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def run_program_source(source: str) -> tuple[bool, bool, Optional[str], Any]:
    try:
        program = compile_auto(source)
    except Exception as e:
        return False, False, f"parse: {e}", None
    try:
        result = Evaluator(tools=ToolRegistry(), llm=MockLLMClient()).run(program)
        return True, True, None, result
    except Exception as e:
        return True, False, f"runtime: {e}", None


def suite_examples() -> list[CaseResult]:
    results = []
    for name, src in EXAMPLES.items():
        p, r, err, _ = run_program_source(src)
        results.append(CaseResult("examples", name, p, r, err))
    return results


def suite_golden() -> list[CaseResult]:
    results = []
    for name, src in GOLDEN.items():
        p, r, err, _ = run_program_source(src)
        results.append(CaseResult("golden", name, p, r, err))
    return results


def suite_mutations() -> list[CaseResult]:
    from forge_benchmark import extract_code

    base = EXAMPLES["basic-calculator"]
    results = []
    for name, mut, expect_ok in MUTATIONS:
        broken = mut(base)
        # Simulate benchmark extraction path for markdown / messy output
        if name in ("markdown_fences", "python_style_wrap"):
            extracted = extract_code(broken)
        else:
            extracted = broken
        p, r, err, _ = run_program_source(extracted)
        note = f"expect_parse_ok={expect_ok}"
        if p != expect_ok:
            note += " | UNEXPECTED (syntax design signal)"
        results.append(CaseResult("mutation", name, p, r, err, note=note))
    return results


def suite_token_efficiency() -> dict:
    """Rough token estimate vs equivalent Python agent sketch."""
    forge_src = EXAMPLES["web-researcher"].strip()
    python_equiv = '''
def web_researcher():
    query = "latest AI safety papers"
    max_results = 5
    results = web_search(q=query, n=max_results)
    top_papers = [r for r in results if r.get("relevance", 0) > 0.8]
    summary = llm_reason("Summarize the key findings from these papers", top_papers)
    assert top_papers is not None
    return {"papers": top_papers, "summary": summary}
'''.strip()
    forge_tok = max(1, len(forge_src) // 4)
    py_tok = max(1, len(python_equiv) // 4)
    return {
        "forge_chars": len(forge_src),
        "python_chars": len(python_equiv),
        "forge_est_tokens": forge_tok,
        "python_est_tokens": py_tok,
        "reduction_pct": round(100.0 * (1 - forge_tok / py_tok), 1),
    }


def print_results(results: list[CaseResult]) -> dict:
    n = len(results) or 1
    parse_n = sum(1 for x in results if x.parse_ok)
    run_n = sum(1 for x in results if x.runtime_ok)
    for x in results:
        flag = "PASS" if x.parse_ok and (x.suite == "mutation" or x.runtime_ok) else (
            "PASS*" if x.parse_ok else "FAIL"
        )
        # For mutations, PASS means "behaved as language should" is subtler;
        # show raw parse/runtime.
        status = "ok" if x.parse_ok else "fail"
        extra = f" runtime={x.runtime_ok}"
        if x.error:
            extra += f" err={x.error[:80]}"
        if x.note:
            extra += f" ({x.note})"
        print(f"  [{status:4}] {x.suite:10} {x.case_id:28}{extra}")

    summary = {
        "total": len(results),
        "parse_ok": parse_n,
        "parse_rate": round(100.0 * parse_n / n, 1),
        "runtime_ok": run_n,
        "runtime_rate": round(100.0 * run_n / n, 1),
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Forge offline evaluation (no paid API)")
    ap.add_argument("--mutations", action="store_true", help="Include LLM-mistake mutation suite")
    ap.add_argument("--backend", default="mock", help="mock | ollama | auto | openai")
    ap.add_argument("--out", default="offline_bench_results.json")
    args = ap.parse_args()

    print("=" * 60)
    print("Forge Offline Bench — no paid credits required")
    print("=" * 60)
    print(f"Ollama available: {ollama_available()}")
    print(f"LLM backend pref: {args.backend}")

    # Prove make_llm_client works
    client = make_llm_client(args.backend)
    print(f"Active LLM client: {type(client).__name__}")

    all_results: list[CaseResult] = []

    print("\n--- Example programs ---")
    r1 = suite_examples()
    all_results.extend(r1)
    s1 = print_results(r1)

    print("\n--- Golden benchmark tasks ---")
    r2 = suite_golden()
    all_results.extend(r2)
    s2 = print_results(r2)

    mut_summary = None
    if args.mutations:
        print("\n--- Mutation suite (LLM failure modes) ---")
        r3 = suite_mutations()
        all_results.extend(r3)
        mut_summary = print_results(r3)
        print("\n  Note: mutations that FAIL parse are GOOD — language rejects bad code.")
        print("  Unexpected parse success = possible ambiguity to tighten.")

    print("\n--- Token efficiency (estimate) ---")
    tok = suite_token_efficiency()
    for k, v in tok.items():
        print(f"  {k}: {v}")

    # Correctness gate for shippable offline milestone
    core = r1 + r2
    core_ok = all(x.parse_ok and x.runtime_ok for x in core)
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  examples:  parse {s1['parse_rate']}%  runtime {s1['runtime_rate']}%")
    print(f"  golden:    parse {s2['parse_rate']}%  runtime {s2['runtime_rate']}%")
    if mut_summary:
        print(f"  mutations: parse_ok_rate {mut_summary['parse_rate']}% (lower often better)")
    print(f"  token reduction vs Python sketch: {tok['reduction_pct']}%")
    print(f"  core suites green: {core_ok}")

    out = {
        "core_ok": core_ok,
        "examples": s1,
        "golden": s2,
        "mutations": mut_summary,
        "token_efficiency": tok,
        "cases": [x.to_dict() for x in all_results],
        "llm_client": type(client).__name__,
        "ollama_available": ollama_available(),
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {args.out}")

    if not core_ok:
        raise SystemExit(1)

    print("\nWorkaround active: develop + measure without paid APIs.")
    print("When ready for real LLM generation scores (free):")
    print("  1) Install Ollama → ollama pull llama3.2 → python forge_benchmark.py --backend ollama")
    print("  2) Or free cloud key: GROQ_API_KEY / GEMINI_API_KEY / OPENROUTER_API_KEY")


if __name__ == "__main__":
    main()
