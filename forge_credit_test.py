#!/usr/bin/env python3
"""
Forge vs Python — live credit / token comparison
================================================

Same tasks, same model. Measure:
  - prompt tokens
  - completion tokens
  - first-try success (Forge: parse+run / Python: parse+exec with mocks)
  - estimated $ using a configurable $/1M token rate

Usage:
  ./forge credit-test                 # Ollama local (no paid credits)
  ./forge credit-test --backend openai
  ./forge credit-test --tasks 5

You do NOT need a paid key to get token economics via Ollama.
Paid backends report the same metrics against real usage fields.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from forge_benchmark import FORGE_SPEC_BRIEF, FEW_SHOT, BENCHMARK_TASKS, extract_code
from forge_repair import try_compile_repaired
from forge_runtime import Evaluator, MockLLMClient, ToolRegistry, ollama_available


# =============================================================================
# Pricing (editable) — used only to translate tokens → estimated USD
# =============================================================================

# Default: DeepSeek-ish cheap chat rate as a reference; override with flags.
DEFAULT_INPUT_PER_M = float(os.environ.get("FORGE_PRICE_IN_PER_M", "0.14"))
DEFAULT_OUTPUT_PER_M = float(os.environ.get("FORGE_PRICE_OUT_PER_M", "0.28"))


@dataclass
class LegResult:
    task_id: str
    lang: str  # forge | python
    success: bool
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    elapsed_s: float
    error: Optional[str] = None
    preview: str = ""


@dataclass
class TaskCompare:
    task_id: str
    description: str
    forge: LegResult
    python: LegResult

    @property
    def forge_cheaper_tokens(self) -> bool:
        return self.forge.total_tokens < self.python.total_tokens


@dataclass
class SuiteSummary:
    tasks: int
    forge_success: int
    python_success: int
    forge_tokens: int
    python_tokens: int
    forge_prompt: int
    python_prompt: int
    forge_completion: int
    python_completion: int
    forge_est_usd: float
    python_est_usd: float
    token_savings_pct: float
    usd_savings_pct: float
    notes: list[str] = field(default_factory=list)


def est_usd(prompt_tok: int, completion_tok: int, pin: float, pout: float) -> float:
    return (prompt_tok / 1_000_000.0) * pin + (completion_tok / 1_000_000.0) * pout


# =============================================================================
# Prompts
# =============================================================================

PYTHON_SPEC = """
You write a single Python function for an agent task.
Rules:
- Define exactly one function: def run(tools, llm) -> dict
- Use only these tools via tools.call(name, inputs_dict):
  arithmetic_add({x,y}) -> number
  web_search({q,n}) -> list
  get_value({}) -> number
  sales_data({region,period}) -> list
  http_get({url}) -> dict
  read_file({path}) -> dict
  write_file({path,body}) -> dict
- For reasoning use: llm.complete(prompt, context=data) -> str
- No imports. No classes. No markdown. Return a dict.
""".strip()

PYTHON_FEW_SHOT = '''
Task: Add a=15 and b=7, explain the sum briefly, verify sum > 0, return result and note.

def run(tools, llm):
    a, b = 15, 7
    total = tools.call("arithmetic_add", {"x": a, "y": b})
    note = llm.complete("Explain what the sum represents in a short sentence", context=total)
    assert total > 0
    return {"result": total, "note": note}
'''.strip()


def forge_prompt(task: str) -> str:
    shots = FEW_SHOT[:1]
    parts = [
        "Forge = AI-native agent language. Emit ONLY Forge code.",
        FORGE_SPEC_BRIEF,
        "",
        "Example:",
        f"Task: {shots[0][0]}",
        shots[0][1],
        "",
        "Your task:",
        task,
        "Write a complete Forge program. Stop after RETURN. No markdown.",
    ]
    return "\n".join(parts)


def python_prompt(task: str) -> str:
    return "\n".join([
        PYTHON_SPEC,
        "",
        "Example:",
        PYTHON_FEW_SHOT,
        "",
        "Your task:",
        task,
        "Write only the def run(tools, llm): function. No markdown.",
    ])


# =============================================================================
# LLM call with usage
# =============================================================================

def call_ollama(prompt: str, model: str, num_predict: int = 350) -> tuple[str, dict]:
    host = (os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
    body = json.dumps({
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": "You generate only code. No explanation."},
            {"role": "user", "content": prompt},
        ],
        "options": {"temperature": 0.1, "num_predict": num_predict},
    }).encode()
    req = urllib.request.Request(
        f"{host}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode())
    elapsed = time.time() - t0
    text = ((data.get("message") or {}).get("content")) or data.get("response") or ""
    # Ollama usage fields
    prompt_tok = int(data.get("prompt_eval_count") or max(1, len(prompt) // 4))
    completion_tok = int(data.get("eval_count") or max(1, len(text) // 4))
    return text, {
        "backend": "ollama",
        "model": model,
        "prompt_tokens": prompt_tok,
        "completion_tokens": completion_tok,
        "total_tokens": prompt_tok + completion_tok,
        "elapsed_s": round(elapsed, 3),
    }


def call_openai_compat(prompt: str, model: Optional[str] = None) -> tuple[str, dict]:
    from forge_runtime import _resolve_openai_compat
    from openai import OpenAI

    api_key, base_url, resolved = _resolve_openai_compat(model=model)
    if not api_key:
        raise RuntimeError("No API key for openai backend")
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    client = OpenAI(**kwargs)
    t0 = time.time()
    resp = client.chat.completions.create(
        model=resolved,
        messages=[
            {"role": "system", "content": "You generate only code. No explanation."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
    )
    elapsed = time.time() - t0
    text = resp.choices[0].message.content or ""
    usage = resp.usage
    prompt_tok = int(getattr(usage, "prompt_tokens", None) or max(1, len(prompt) // 4))
    completion_tok = int(getattr(usage, "completion_tokens", None) or max(1, len(text) // 4))
    return text, {
        "backend": "openai",
        "model": resolved,
        "base_url": base_url,
        "prompt_tokens": prompt_tok,
        "completion_tokens": completion_tok,
        "total_tokens": prompt_tok + completion_tok,
        "elapsed_s": round(elapsed, 3),
    }


def call_llm(prompt: str, backend: str, model: Optional[str] = None) -> tuple[str, dict]:
    backend = backend.lower()
    if backend == "auto":
        backend = "ollama" if ollama_available() else "openai"
    if backend == "ollama":
        model = model or os.environ.get("OLLAMA_MODEL") or "llama3.2:1b"
        return call_ollama(prompt, model=model)
    return call_openai_compat(prompt, model=model)


# =============================================================================
# Evaluate generated code
# =============================================================================

def eval_forge(text: str) -> tuple[bool, Optional[str], str]:
    try:
        program, used, _notes = try_compile_repaired(text)
        Evaluator(tools=ToolRegistry(), llm=MockLLMClient()).run(program)
        return True, None, used
    except Exception as e:
        return False, f"{type(e).__name__}: {e}", extract_code(text)


def _extract_python(text: str) -> str:
    raw = text.strip()
    fence = re.search(r"```(?:python)?\s*([\s\S]*?)```", raw, re.I)
    if fence:
        raw = fence.group(1).strip()
    m = re.search(r"(def\s+run\s*\([\s\S]*)", raw)
    if m:
        return m.group(1).strip()
    return raw


def eval_python(text: str) -> tuple[bool, Optional[str], str]:
    code = _extract_python(text)
    try:
        tree = ast.parse(code)
    except Exception as e:
        return False, f"SyntaxError: {e}", code

    # Must define run
    has_run = any(
        isinstance(n, ast.FunctionDef) and n.name == "run" for n in tree.body
    )
    if not has_run:
        return False, "Missing def run(tools, llm)", code

    # Ban imports for safety
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            return False, "Imports not allowed in credit-test Python", code

    tools = ToolRegistry()
    llm = MockLLMClient()
    ns: dict[str, Any] = {}
    try:
        exec(compile(tree, "<forge_credit_python>", "exec"), ns, ns)
        fn = ns.get("run")
        if not callable(fn):
            return False, "run is not callable", code
        result = fn(tools, llm)
        if not isinstance(result, dict):
            return False, f"run() must return dict, got {type(result).__name__}", code
        return True, None, code
    except Exception as e:
        return False, f"{type(e).__name__}: {e}", code


# =============================================================================
# Suite
# =============================================================================

def run_credit_test(
    backend: str = "auto",
    tasks: Optional[list[dict]] = None,
    limit: Optional[int] = None,
    model: Optional[str] = None,
    price_in: float = DEFAULT_INPUT_PER_M,
    price_out: float = DEFAULT_OUTPUT_PER_M,
) -> tuple[list[TaskCompare], SuiteSummary]:
    tasks = list(tasks or BENCHMARK_TASKS)
    if limit:
        tasks = tasks[:limit]

    compares: list[TaskCompare] = []

    for task in tasks:
        tid = task["id"]
        desc = task["description"]
        print(f"\n--- {tid} ---")

        # Forge leg
        fp = forge_prompt(desc)
        try:
            f_text, f_usage = call_llm(fp, backend=backend, model=model)
            f_ok, f_err, f_code = eval_forge(f_text)
        except Exception as e:
            f_text, f_usage = "", {
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "elapsed_s": 0,
            }
            f_ok, f_err, f_code = False, str(e), ""
        forge_leg = LegResult(
            task_id=tid,
            lang="forge",
            success=f_ok,
            prompt_tokens=int(f_usage.get("prompt_tokens", 0)),
            completion_tokens=int(f_usage.get("completion_tokens", 0)),
            total_tokens=int(f_usage.get("total_tokens", 0)),
            elapsed_s=float(f_usage.get("elapsed_s", 0)),
            error=f_err,
            preview=(f_code or f_text)[:180].replace("\n", " "),
        )
        print(
            f"  FORGE   ok={f_ok}  tokens={forge_leg.total_tokens} "
            f"(in={forge_leg.prompt_tokens} out={forge_leg.completion_tokens}) "
            f"err={f_err}"
        )

        # Python leg
        pp = python_prompt(desc)
        try:
            p_text, p_usage = call_llm(pp, backend=backend, model=model)
            p_ok, p_err, p_code = eval_python(p_text)
        except Exception as e:
            p_text, p_usage = "", {
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "elapsed_s": 0,
            }
            p_ok, p_err, p_code = False, str(e), ""
        py_leg = LegResult(
            task_id=tid,
            lang="python",
            success=p_ok,
            prompt_tokens=int(p_usage.get("prompt_tokens", 0)),
            completion_tokens=int(p_usage.get("completion_tokens", 0)),
            total_tokens=int(p_usage.get("total_tokens", 0)),
            elapsed_s=float(p_usage.get("elapsed_s", 0)),
            error=p_err,
            preview=(p_code or p_text)[:180].replace("\n", " "),
        )
        print(
            f"  PYTHON  ok={p_ok}  tokens={py_leg.total_tokens} "
            f"(in={py_leg.prompt_tokens} out={py_leg.completion_tokens}) "
            f"err={p_err}"
        )

        compares.append(TaskCompare(task_id=tid, description=desc, forge=forge_leg, python=py_leg))

    f_tok = sum(c.forge.total_tokens for c in compares)
    p_tok = sum(c.python.total_tokens for c in compares)
    f_in = sum(c.forge.prompt_tokens for c in compares)
    p_in = sum(c.python.prompt_tokens for c in compares)
    f_out = sum(c.forge.completion_tokens for c in compares)
    p_out = sum(c.python.completion_tokens for c in compares)
    f_usd = est_usd(f_in, f_out, price_in, price_out)
    p_usd = est_usd(p_in, p_out, price_in, price_out)

    summary = SuiteSummary(
        tasks=len(compares),
        forge_success=sum(1 for c in compares if c.forge.success),
        python_success=sum(1 for c in compares if c.python.success),
        forge_tokens=f_tok,
        python_tokens=p_tok,
        forge_prompt=f_in,
        python_prompt=p_in,
        forge_completion=f_out,
        python_completion=p_out,
        forge_est_usd=round(f_usd, 6),
        python_est_usd=round(p_usd, 6),
        token_savings_pct=round(100.0 * (1 - f_tok / p_tok), 1) if p_tok else 0.0,
        usd_savings_pct=round(100.0 * (1 - f_usd / p_usd), 1) if p_usd else 0.0,
        notes=[
            f"Price assumption: ${price_in}/1M input, ${price_out}/1M output tokens.",
            "Ollama reports real token counts but $0 billed locally.",
            "Success = first-try runnable with mock tools (no retries charged).",
        ],
    )
    return compares, summary


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Forge vs Python live credit/token test")
    ap.add_argument("--backend", default="auto", help="auto|ollama|openai")
    ap.add_argument("--model", default=None)
    ap.add_argument("--tasks", type=int, default=5, help="How many benchmark tasks (default 5, max 18)")
    ap.add_argument("--price-in", type=float, default=DEFAULT_INPUT_PER_M)
    ap.add_argument("--price-out", type=float, default=DEFAULT_OUTPUT_PER_M)
    ap.add_argument("--out", default="credit_test_results.json")
    args = ap.parse_args(argv)

    backend = args.backend
    if backend == "auto":
        if ollama_available():
            backend = "ollama"
        else:
            backend = "openai"

    print("=" * 60)
    print("Forge vs Python — credit / token test")
    print("=" * 60)
    print(f"backend={backend}  model={args.model or '(default)'}  tasks={args.tasks}")
    print(f"price_in=${args.price_in}/1M  price_out=${args.price_out}/1M")
    if backend == "ollama":
        print("Note: Ollama is local — $ billed = $0; tokens still real for credit math.")

    compares, summary = run_credit_test(
        backend=backend,
        limit=args.tasks,
        model=args.model,
        price_in=args.price_in,
        price_out=args.price_out,
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  tasks:              {summary.tasks}")
    print(f"  forge success:      {summary.forge_success}/{summary.tasks}")
    print(f"  python success:     {summary.python_success}/{summary.tasks}")
    print(f"  forge tokens:       {summary.forge_tokens}  (in={summary.forge_prompt} out={summary.forge_completion})")
    print(f"  python tokens:      {summary.python_tokens}  (in={summary.python_prompt} out={summary.python_completion})")
    print(f"  token savings:      {summary.token_savings_pct}%  (positive = Forge used fewer)")
    print(f"  forge est. USD:     ${summary.forge_est_usd}")
    print(f"  python est. USD:    ${summary.python_est_usd}")
    print(f"  USD savings:        {summary.usd_savings_pct}%  (at assumed rates)")
    for n in summary.notes:
        print(f"  note: {n}")

    payload = {
        "summary": asdict(summary),
        "tasks": [
            {
                "task_id": c.task_id,
                "description": c.description,
                "forge": asdict(c.forge),
                "python": asdict(c.python),
            }
            for c in compares
        ],
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
