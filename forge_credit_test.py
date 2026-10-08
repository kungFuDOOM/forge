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
import builtins
import json
import os
import re
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from forge_benchmark import BENCHMARK_TASKS, extract_code
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
    attempts: int = 1
    hit_cap: bool = False
    code_tokens: int = 0  # extracted program density (chars//4)


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
    forge_code_tokens: int = 0
    python_code_tokens: int = 0
    code_density_savings_pct: float = 0.0
    forge_hit_cap: int = 0
    python_hit_cap: int = 0
    lean: bool = False
    max_attempts: int = 1
    diagnosis: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def est_usd(prompt_tok: int, completion_tok: int, pin: float, pout: float) -> float:
    return (prompt_tok / 1_000_000.0) * pin + (completion_tok / 1_000_000.0) * pout


# =============================================================================
# Prompts
# =============================================================================

# Credit-test Forge spec — kept at ~Python teach cost (the old FORGE_SPEC_BRIEF
# was ~50% larger and alone erased language-density wins).
FORGE_CREDIT_SPEC = """
Forge agent language. Emit ONLY Forge. Keywords UPPERCASE. Vars $name.

AGENT "name"
MEMORY { key: value }
STEP n TOOL tool INPUT { k: $v } OUTPUT out
STEP n FILTER field > 0.8 ON $list OUTPUT filtered
REASON "prompt" ON $var OUTPUT out
VERIFY $var != null
RETURN { key: $var }

Tools: arithmetic_add, web_search, get_value, sales_data.
Stop immediately after the RETURN line. No markdown. No second AGENT.
""".strip()

PYTHON_SPEC = """
You write a single Python function for an agent task.
Rules:
- Define exactly one function: def run(tools, llm) -> dict
- Use only these tools via tools.call(name, inputs_dict):
  arithmetic_add({x,y}) -> number
  web_search({q,n}) -> list
  get_value({}) -> number
  sales_data({region,period}) -> list
- For reasoning use: llm.complete(prompt, context=data) -> str
- No imports. No classes. No markdown. Return a dict.
- Solve THIS task (do not copy the example's numbers).
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

FORGE_FEW_SHOT_CODE = '''
AGENT "basic-calculator"
MEMORY { a: 15 b: 7 }
STEP add TOOL arithmetic_add INPUT { x: $a, y: $b } OUTPUT sum
REASON "Explain what the sum represents in a short sentence" ON $sum OUTPUT explanation
VERIFY $sum > 0
RETURN { result: $sum, note: $explanation }
'''.strip()


def forge_prompt(task: str, *, lean: bool = False) -> str:
    """lean=True: minimal teach (steady-state / model already knows Forge)."""
    if lean:
        return "\n".join([
            "Emit ONLY a Forge program for this task.",
            "First character must be A of AGENT. Stop after RETURN. No markdown fences.",
            task,
        ])
    return "\n".join([
        FORGE_CREDIT_SPEC,
        "",
        "Example (copy this shape, change the values for the task):",
        "Task: Add a=15 and b=7, explain briefly, verify > 0, return result + note.",
        FORGE_FEW_SHOT_CODE,
        "",
        "Your task:",
        task,
        "Reply with Forge only. First line must be: AGENT \"...\"",
        "Stop after RETURN. Do not wrap in ``` fences.",
    ])


def python_prompt(task: str, *, lean: bool = False) -> str:
    if lean:
        return "\n".join([
            "Emit ONLY def run(tools, llm) -> dict for this task. No markdown.",
            task,
        ])
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

# Valid Forge programs are ~40–120 tokens. Cap + stop kill runaway babble that
# previously burned 350 completion tokens and hid the density win.
FORGE_NUM_PREDICT = 180
PYTHON_NUM_PREDICT = 220
# Stop only on clear "done / started over" markers. Avoid bare ``` — tiny models
# often open a fence and an over-eager stop yields 4-token garbage "savings".
FORGE_STOP = ["\nAGENT \"", "\n\n\n"]
PYTHON_STOP = ["\n\ndef run", "\n\n\n"]


def call_ollama(
    prompt: str,
    model: str,
    num_predict: int = 220,
    stop: Optional[list[str]] = None,
) -> tuple[str, dict]:
    host = (os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
    options: dict[str, Any] = {"temperature": 0.1, "num_predict": num_predict}
    if stop:
        options["stop"] = stop
    body = json.dumps({
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": "You generate only code. No explanation."},
            {"role": "user", "content": prompt},
        ],
        "options": options,
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
        "hit_cap": completion_tok >= num_predict,
    }


def call_openai_compat(
    prompt: str,
    model: Optional[str] = None,
    num_predict: int = 220,
    stop: Optional[list[str]] = None,
) -> tuple[str, dict]:
    from forge_runtime import chat_completion

    t0 = time.time()
    text, usage = chat_completion(
        [
            {"role": "system", "content": "You generate only code. No explanation."},
            {"role": "user", "content": prompt},
        ],
        model=model,
        temperature=0.1,
        max_tokens=num_predict,
        stop=stop,
    )
    elapsed = time.time() - t0
    prompt_tok = int(usage.get("prompt_tokens") or max(1, len(prompt) // 4))
    completion_tok = int(usage.get("completion_tokens") or max(1, len(text) // 4))
    return text, {
        **usage,
        "prompt_tokens": prompt_tok,
        "completion_tokens": completion_tok,
        "total_tokens": prompt_tok + completion_tok,
        "elapsed_s": round(elapsed, 3),
        "hit_cap": completion_tok >= num_predict,
    }


def call_llm(
    prompt: str,
    backend: str,
    model: Optional[str] = None,
    *,
    lang: str = "forge",
) -> tuple[str, dict]:
    backend = backend.lower()
    if backend == "auto":
        backend = "ollama" if ollama_available() else "openai"
    if lang == "forge":
        num_predict, stop = FORGE_NUM_PREDICT, FORGE_STOP
    else:
        num_predict, stop = PYTHON_NUM_PREDICT, PYTHON_STOP
    if backend == "ollama":
        model = model or os.environ.get("OLLAMA_MODEL") or "llama3.2:1b"
        return call_ollama(prompt, model=model, num_predict=num_predict, stop=stop)
    return call_openai_compat(prompt, model=model, num_predict=num_predict, stop=stop)


def code_token_estimate(text: str) -> int:
    """Rough token estimate for extracted program body (language density)."""
    return max(1, len(text) // 4)


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


# Builtins visible to generated Python: enough for agent glue, nothing that
# touches the filesystem, network, or interpreter (open, __import__, eval, ...).
_SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in (
        "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
        "int", "isinstance", "len", "list", "map", "max", "min", "print",
        "range", "reversed", "round", "set", "sorted", "str", "sum", "tuple",
        "zip", "Exception", "ValueError", "KeyError", "TypeError",
    )
}


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

    # Ban imports and dunder access (the usual sandbox escapes)
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            return False, "Imports not allowed in credit-test Python", code
        if isinstance(n, ast.Attribute) and n.attr.startswith("__"):
            return False, "Dunder attribute access not allowed in credit-test Python", code

    tools = ToolRegistry()
    llm = MockLLMClient()
    ns: dict[str, Any] = {"__builtins__": _SAFE_BUILTINS}
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

def _run_leg(
    *,
    tid: str,
    desc: str,
    lang: str,
    backend: str,
    model: Optional[str],
    lean: bool,
    max_attempts: int,
) -> LegResult:
    """Generate + eval one language leg, optionally retrying until success."""
    prompt_fn = forge_prompt if lang == "forge" else python_prompt
    eval_fn = eval_forge if lang == "forge" else eval_python
    prompt = prompt_fn(desc, lean=lean)

    total_in = total_out = 0
    elapsed = 0.0
    hit_cap = False
    ok = False
    err: Optional[str] = "no attempt"
    code = ""
    attempts = 0

    for attempt in range(1, max_attempts + 1):
        attempts = attempt
        try:
            text, usage = call_llm(prompt, backend=backend, model=model, lang=lang)
            ok, err, code = eval_fn(text)
        except Exception as e:
            text, usage = "", {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "elapsed_s": 0,
                "hit_cap": False,
            }
            ok, err, code = False, str(e), ""
        total_in += int(usage.get("prompt_tokens", 0))
        total_out += int(usage.get("completion_tokens", 0))
        elapsed += float(usage.get("elapsed_s", 0))
        hit_cap = hit_cap or bool(usage.get("hit_cap"))
        if ok:
            break

    return LegResult(
        task_id=tid,
        lang=lang,
        success=ok,
        prompt_tokens=total_in,
        completion_tokens=total_out,
        total_tokens=total_in + total_out,
        elapsed_s=round(elapsed, 3),
        error=err,
        preview=(code or "")[:180].replace("\n", " "),
        attempts=attempts,
        hit_cap=hit_cap,
        code_tokens=code_token_estimate(code) if code else 0,
    )


def run_credit_test(
    backend: str = "auto",
    tasks: Optional[list[dict]] = None,
    limit: Optional[int] = None,
    model: Optional[str] = None,
    price_in: float = DEFAULT_INPUT_PER_M,
    price_out: float = DEFAULT_OUTPUT_PER_M,
    lean: bool = False,
    max_attempts: int = 1,
) -> tuple[list[TaskCompare], SuiteSummary]:
    tasks = list(tasks or BENCHMARK_TASKS)
    if limit:
        tasks = tasks[:limit]

    compares: list[TaskCompare] = []

    for task in tasks:
        tid = task["id"]
        desc = task["description"]
        print(f"\n--- {tid} ---")

        forge_leg = _run_leg(
            tid=tid, desc=desc, lang="forge", backend=backend, model=model,
            lean=lean, max_attempts=max_attempts,
        )
        print(
            f"  FORGE   ok={forge_leg.success}  tokens={forge_leg.total_tokens} "
            f"(in={forge_leg.prompt_tokens} out={forge_leg.completion_tokens}) "
            f"code~{forge_leg.code_tokens}t  attempts={forge_leg.attempts}"
            f"{'  HIT_CAP' if forge_leg.hit_cap else ''}  err={forge_leg.error}"
        )

        py_leg = _run_leg(
            tid=tid, desc=desc, lang="python", backend=backend, model=model,
            lean=lean, max_attempts=max_attempts,
        )
        print(
            f"  PYTHON  ok={py_leg.success}  tokens={py_leg.total_tokens} "
            f"(in={py_leg.prompt_tokens} out={py_leg.completion_tokens}) "
            f"code~{py_leg.code_tokens}t  attempts={py_leg.attempts}"
            f"{'  HIT_CAP' if py_leg.hit_cap else ''}  err={py_leg.error}"
        )

        compares.append(TaskCompare(task_id=tid, description=desc, forge=forge_leg, python=py_leg))

    f_tok = sum(c.forge.total_tokens for c in compares)
    p_tok = sum(c.python.total_tokens for c in compares)
    f_in = sum(c.forge.prompt_tokens for c in compares)
    p_in = sum(c.python.prompt_tokens for c in compares)
    f_out = sum(c.forge.completion_tokens for c in compares)
    p_out = sum(c.python.completion_tokens for c in compares)
    f_ok_n = sum(1 for c in compares if c.forge.success)
    p_ok_n = sum(1 for c in compares if c.python.success)
    f_code = sum(c.forge.code_tokens for c in compares if c.forge.success)
    p_code = sum(c.python.code_tokens for c in compares if c.python.success)
    f_code_avg = round(f_code / f_ok_n) if f_ok_n else 0
    p_code_avg = round(p_code / p_ok_n) if p_ok_n else 0
    f_usd = est_usd(f_in, f_out, price_in, price_out)
    p_usd = est_usd(p_in, p_out, price_in, price_out)

    diagnosis: list[str] = []
    if f_in > p_in:
        diagnosis.append(
            f"Prompt teach tax: Forge prompts used {f_in - p_in} more input tokens "
            f"({100 * (f_in / p_in - 1):.0f}% heavier). Amortizes once the model knows Forge."
        )
    elif p_in > f_in:
        diagnosis.append(
            f"Prompt: Forge prompts used {p_in - f_in} fewer input tokens than Python."
        )
    if f_out > p_out:
        diagnosis.append(
            f"Completion: Forge emitted {f_out - p_out} more output tokens. "
            "Check HIT_CAP rows — runaway babble past RETURN hides density wins."
        )
    elif p_out > f_out:
        diagnosis.append(
            f"Completion win: Forge used {p_out - f_out} fewer output tokens "
            "(the language-density thesis)."
        )
    if f_code_avg and p_code_avg:
        dens = 100.0 * (1 - f_code_avg / p_code_avg)
        diagnosis.append(
            f"Avg extracted-program size: Forge ~{f_code_avg}t vs Python ~{p_code_avg}t "
            f"({dens:+.0f}% — positive means Forge programs are smaller)."
        )
    f_ok, p_ok = f_ok_n, p_ok_n
    if f_ok != p_ok:
        diagnosis.append(
            f"Reliability: Forge {f_ok}/{len(compares)} vs Python {p_ok}/{len(compares)} "
            f"(max {max_attempts} attempt(s) each). Failed tries are included in token totals."
        )

    summary = SuiteSummary(
        tasks=len(compares),
        forge_success=f_ok,
        python_success=p_ok,
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
        forge_code_tokens=f_code_avg,
        python_code_tokens=p_code_avg,
        code_density_savings_pct=(
            round(100.0 * (1 - f_code_avg / p_code_avg), 1) if p_code_avg else 0.0
        ),
        forge_hit_cap=sum(1 for c in compares if c.forge.hit_cap),
        python_hit_cap=sum(1 for c in compares if c.python.hit_cap),
        lean=lean,
        max_attempts=max_attempts,
        diagnosis=diagnosis,
        notes=[
            f"Price assumption: ${price_in}/1M input, ${price_out}/1M output tokens.",
            "Ollama reports real token counts but $0 billed locally.",
            (
                f"Success = runnable with mock tools; attempts charged up to {max_attempts}."
                if max_attempts > 1
                else "Success = first-try runnable with mock tools (use --retries N to charge retries)."
            ),
            (
                "Mode=lean: minimal prompts (steady-state language density)."
                if lean
                else "Mode=cold: full few-shot teach (includes language-teach tax)."
            ),
            f"Forge num_predict={FORGE_NUM_PREDICT} with stop sequences; Python={PYTHON_NUM_PREDICT}.",
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
    ap.add_argument(
        "--lean",
        action="store_true",
        help="Minimal prompts (steady-state). Tests language density without teach tax.",
    )
    ap.add_argument(
        "--retries",
        type=int,
        default=1,
        metavar="N",
        help="Max generation attempts per leg (charges all tries). Default 1 = first-try only.",
    )
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
    print(f"mode={'lean' if args.lean else 'cold'}  max_attempts={args.retries}")
    print(f"price_in=${args.price_in}/1M  price_out=${args.price_out}/1M")
    if backend == "ollama":
        print("Note: Ollama is local — $ billed = $0; tokens still real for credit math.")

    compares, summary = run_credit_test(
        backend=backend,
        limit=args.tasks,
        model=args.model,
        price_in=args.price_in,
        price_out=args.price_out,
        lean=args.lean,
        max_attempts=max(1, args.retries),
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
    print(f"  code density:       forge~{summary.forge_code_tokens}t  python~{summary.python_code_tokens}t  "
          f"({summary.code_density_savings_pct}% smaller Forge programs)")
    print(f"  hit output cap:     forge={summary.forge_hit_cap}  python={summary.python_hit_cap}")
    print(f"  forge est. USD:     ${summary.forge_est_usd}")
    print(f"  python est. USD:    ${summary.python_est_usd}")
    print(f"  USD savings:        {summary.usd_savings_pct}%  (at assumed rates)")
    if summary.diagnosis:
        print("  --- why ---")
        for d in summary.diagnosis:
            print(f"  · {d}")
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
