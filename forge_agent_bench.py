#!/usr/bin/env python3
"""
Forge agent benchmark: tool-by-tool vs one forge_run call
=========================================================

The claim behind Forge is that an agent spends fewer tokens when it writes
the whole job as one program instead of making a tool call per step (every
round trip re-sends the conversation). This measures it on the same tasks,
same model, same tools:

  tools mode   the model gets every Forge tool as an ordinary function and
               works step by step (the usual agent loop)
  python mode  "code mode": one tool, run_python(code), whose script calls the
               same tools (the approach of Anthropic's programmatic tool
               calling, Cloudflare Code Mode and smolagents' CodeAgent)
  forge mode   the model gets one tool, forge_run(source), and writes a program

Every turn's prompt + completion tokens are summed from the provider's usage
fields. Answers are checked against known results (the tools are offline and
deterministic), so a cheap wrong answer doesn't count as a win.

  ./forge agent-bench                        # auto: Ollama, else an API key
  ./forge agent-bench --backend openai       # GROQ / GEMINI / XAI / OPENAI ... key
  ./forge agent-bench --tasks four-regions --runs 3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Optional

from forge_runtime import (
    ForgeError,
    MockLLMClient,
    ToolRegistry,
    chat_message,
    has_api_key,
    language_spec,
    ollama_available,
    run_forge,
)

# Network tools are left out so every run sees the same deterministic data.
EXCLUDED_TOOLS = {"http_get", "http_post"}
MAX_TURNS = 12
TOOL_RESULT_CHARS = 4000

SYSTEM_PROMPT = (
    "You are a careful agent. Use the provided tools to get facts; never guess numbers. "
    "When you have the answer, reply with one short sentence that includes it."
)
MODES = ("tools", "python", "forge")
PYTHON_TIMEOUT_S = 30
PYTHON_HINT = (
    "Prefer doing the whole job in a single run_python call: loops, filters, math "
    "and file writes all fit in one script."
)
FORGE_HINT = (
    "Prefer doing the whole job in a single forge_run call: loops, filters, math "
    "and file writes all fit in one program."
)

AGENT_TASKS: list[dict] = [
    {
        "id": "big-deals",
        "prompt": "Get the Q4 sales data for the north region. What is the total amount of the deals larger than 10000?",
        "expect": ["37000"],
    },
    {
        "id": "four-regions",
        "prompt": (
            "Get the Q1 sales data for each of the regions north, south, east and west. "
            "Across all four regions, what is the total amount of the deals larger than 10000?"
        ),
        "expect": ["148000"],
    },
    {
        "id": "research",
        "prompt": (
            "Search the web for 'agent languages' with 5 results. How many results have relevance "
            "above 0.7, and what is the title of the most relevant one?"
        ),
        "expect": ["3", "Result 1 for 'agent languages'"],
    },
    {
        "id": "report-file",
        "prompt": (
            "Write the names of all Q4 north-region deals to the file out/deals.txt, one name per line, "
            "then read the file back. How many characters does it contain?"
        ),
        "expect_any": ["31", "32"],  # with or without a trailing newline
    },
    {
        "id": "math-chain",
        "prompt": "Using the calc tool, compute ((17 * 23) + 19) / 2.",
        "expect": ["205"],
    },
]

DEFAULT_PRICE_IN = float(os.environ.get("FORGE_PRICE_IN_PER_M", "0.14"))
DEFAULT_PRICE_OUT = float(os.environ.get("FORGE_PRICE_OUT_PER_M", "0.28"))


# =============================================================================
# Tool definitions
# =============================================================================

def _bench_tools() -> ToolRegistry:
    tools = ToolRegistry()
    docs = tools.docs()

    def offline(name: str):
        def tool(inputs: dict) -> Any:
            raise ForgeError(f"{name} is disabled in the benchmark (offline, deterministic tools only)")
        return tool

    for name in EXCLUDED_TOOLS:
        tools.register(name, offline(name), docs[name])
    return tools


_NUMBER_INPUTS = {"x", "y", "n", "limit", "max_bytes", "max_chars"}
_TYPED_INPUTS = {"list": "array", "desc": "boolean", "json": "object", "headers": "object"}


def tool_schema(name: str, doc: str) -> dict:
    """OpenAI function schema from a Forge tool doc line ("INPUT { a, b? } → ...")."""
    props: dict[str, dict] = {}
    required: list[str] = []
    extra = False
    m = re.match(r"INPUT \{([^}]*)\}", doc)
    for raw in (m.group(1) if m else "").split(","):
        for part in raw.split("|"):
            key = part.strip()
            if not key:
                continue
            if key.startswith("..."):
                extra = True
                continue
            optional = key.endswith("?")
            key = key.rstrip("?")
            kind = "number" if key in _NUMBER_INPUTS else _TYPED_INPUTS.get(key, "string")
            schema: dict[str, Any] = {"type": kind}
            if kind == "array":
                schema["items"] = {}
            props[key] = schema
            if not optional and "|" not in raw:
                required.append(key)
    params: dict[str, Any] = {"type": "object", "properties": props}
    if required:
        params["required"] = required
    if extra:
        params["additionalProperties"] = True
    return {"type": "function", "function": {"name": name, "description": doc, "parameters": params}}


def tool_defs(mode: str, tools: ToolRegistry) -> list[dict]:
    if mode == "tools":
        return [tool_schema(n, d) for n, d in tools.docs().items() if n not in EXCLUDED_TOOLS]
    if mode == "python":
        listing = "\n".join(f"  {n}: {d}" for n, d in tools.docs().items() if n not in EXCLUDED_TOOLS)
        return [{
            "type": "function",
            "function": {
                "name": "run_python",
                "description": (
                    "Run a Python script in a sandbox and get back what it prints. Call tools with "
                    "tools.call(name, inputs_dict), e.g. tools.call('sales_data', {'region': 'north', "
                    "'period': 'Q1'}). No imports; json and math are preloaded; basic builtins only. "
                    "print() the result you need.\n\nTools:\n" + listing
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"code": {"type": "string", "description": "Python source"}},
                    "required": ["code"],
                },
            },
        }]
    spec = language_spec(tools)
    return [{
        "type": "function",
        "function": {
            "name": "forge_run",
            "description": (
                "Run a Forge agent program and get its RETURN value as JSON, or an error naming "
                "what to fix. REASON uses a mock LLM here, so compute answers with tools.\n\n" + spec
            ),
            "parameters": {
                "type": "object",
                "properties": {"source": {"type": "string", "description": "Forge program source"}},
                "required": ["source"],
            },
        },
    }]


def run_python_sandboxed(code: str, timeout: float = PYTHON_TIMEOUT_S) -> str:
    """Run model-written Python in a child process (killable on timeout) that
    exposes only the benchmark tools, no imports and a small builtin set."""
    try:
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--sandbox-child"],
            input=json.dumps({"code": code}), capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"Error: script timed out after {timeout:g}s"
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])["output"]
    except (IndexError, ValueError, KeyError):
        return f"Error: sandbox failed: {(proc.stderr or proc.stdout)[-500:]}"


def _sandbox_child() -> None:
    """Entry point of the run_python child process (reads {"code"} on stdin)."""
    import ast
    import io
    import math

    from forge_credit_test import _FRAME_ATTRS, _SAFE_BUILTINS

    code = json.loads(sys.stdin.read())["code"]
    out = io.StringIO()

    def respond(text: str) -> None:
        sys.__stdout__.write(json.dumps({"output": text}) + "\n")

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return respond(f"Error: SyntaxError: {e}")
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            return respond("Error: imports are not allowed (json and math are preloaded)")
        if isinstance(n, ast.Attribute) and (n.attr.startswith("_") or n.attr in _FRAME_ATTRS):
            return respond(f"Error: attribute {n.attr!r} is not allowed")

    tools = _bench_tools()

    class Tools:
        def call(self, name: str, inputs: Optional[dict] = None) -> Any:
            return tools.call(name, dict(inputs or {}))

    builtins = {**_SAFE_BUILTINS, "print": lambda *a, **k: print(*a, **{**k, "file": out})}
    ns: dict[str, Any] = {"__builtins__": builtins, "tools": Tools(), "json": json, "math": math}
    try:
        exec(compile(tree, "<run_python>", "exec"), ns, ns)
        text = out.getvalue() or "(no output: print() the result you need)"
    except Exception as e:
        text = out.getvalue() + f"Error: {type(e).__name__}: {e}"
    respond(text)


def execute_tool(mode: str, name: str, args: dict, tools: ToolRegistry) -> str:
    try:
        if mode == "python":
            if name != "run_python":
                return f"Unknown tool {name!r}; the only tool is run_python"
            text = run_python_sandboxed(str(args.get("code", "")))
            if len(text) > TOOL_RESULT_CHARS:
                text = text[:TOOL_RESULT_CHARS] + f"…[truncated {len(text) - TOOL_RESULT_CHARS} chars]"
            return text
        if mode == "forge":
            if name != "forge_run":
                return f"Unknown tool {name!r}; the only tool is forge_run"
            result = run_forge(str(args.get("source", "")), tools=tools, llm=MockLLMClient(),
                               max_steps=10_000, max_seconds=60)
        else:
            result = tools.call(name, args)
        text = json.dumps(result, default=str)
    except Exception as e:
        text = f"Error: {type(e).__name__}: {e}"
    if len(text) > TOOL_RESULT_CHARS:
        text = text[:TOOL_RESULT_CHARS] + f"…[truncated {len(text) - TOOL_RESULT_CHARS} chars]"
    return text


# =============================================================================
# Agent loop
# =============================================================================

ModelCall = Callable[[list, list], tuple[dict, dict]]


@dataclass
class AgentRun:
    task_id: str
    mode: str
    ok: bool
    turns: int
    tool_calls: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    usd: float
    estimated: bool  # provider gave no usage; tokens estimated from text length
    elapsed_s: float
    answer: str = ""
    error: Optional[str] = None


def answer_ok(task: dict, answer: str) -> bool:
    text = answer.replace(",", "")
    if "expect" in task and not all(e.replace(",", "") in text for e in task["expect"]):
        return False
    if "expect_any" in task and not any(e in text for e in task["expect_any"]):
        return False
    return True


def run_agent(task: dict, mode: str, call: ModelCall, price_in: float, price_out: float) -> AgentRun:
    tools = _bench_tools()
    defs = tool_defs(mode, tools)
    system = SYSTEM_PROMPT + {"forge": " " + FORGE_HINT, "python": " " + PYTHON_HINT}.get(mode, "")
    messages: list[dict] = [{"role": "system", "content": system}, {"role": "user", "content": task["prompt"]}]
    prompt_tok = completion_tok = tool_calls = turns = 0
    estimated = False
    answer, error = "", None
    t0 = time.time()
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)  # file tools write into a scratch folder
        try:
            while True:
                if turns >= MAX_TURNS:
                    error = f"no final answer after {MAX_TURNS} turns"
                    break
                turns += 1
                message, usage = call(messages, defs)
                if usage.get("prompt_tokens") is None or usage.get("completion_tokens") is None:
                    estimated = True
                    prompt_tok += len(json.dumps(messages)) // 4 + len(json.dumps(defs)) // 4
                    completion_tok += max(1, len(json.dumps(message)) // 4)
                else:
                    prompt_tok += int(usage["prompt_tokens"])
                    completion_tok += int(usage["completion_tokens"])
                calls = message.get("tool_calls") or []
                messages.append({"role": "assistant", "content": message.get("content") or "",
                                 **({"tool_calls": calls} if calls else {})})
                if not calls:
                    answer = message.get("content") or ""
                    break
                for tc in calls:
                    tool_calls += 1
                    fn = tc.get("function") or {}
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except ValueError:
                        args = {}
                    result = execute_tool(mode, fn.get("name", ""), args if isinstance(args, dict) else {}, tools)
                    messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": result})
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
        finally:
            os.chdir(cwd)
    ok = error is None and answer_ok(task, answer)
    if error is None and not ok:
        error = "wrong or incomplete answer"
    usd = prompt_tok / 1e6 * price_in + completion_tok / 1e6 * price_out
    return AgentRun(task["id"], mode, ok, turns, tool_calls, prompt_tok, completion_tok,
                    prompt_tok + completion_tok, round(usd, 8), estimated, round(time.time() - t0, 2),
                    answer[:300], error)


def _savings(runs: list[AgentRun], base: str, mode: str) -> tuple[Optional[float], list[str]]:
    """Token savings of `mode` vs `base` on tasks both solved (the fair view)."""
    ok_base = [r for r in runs if r.mode == base and r.ok]
    ok_mode = [r for r in runs if r.mode == mode and r.ok]
    shared = {r.task_id for r in ok_base} & {r.task_id for r in ok_mode}
    bt = sum(r.total_tokens for r in ok_base if r.task_id in shared)
    mt = sum(r.total_tokens for r in ok_mode if r.task_id in shared)
    return (round(100 * (1 - mt / bt), 1) if bt else None), sorted(shared)


def summarize(runs: list[AgentRun]) -> dict:
    out: dict[str, Any] = {}
    modes = [m for m in MODES if any(r.mode == m for r in runs)]
    for mode in modes:
        rs = [r for r in runs if r.mode == mode]
        out[mode] = {
            "runs": len(rs),
            "success": sum(r.ok for r in rs),
            "total_tokens": sum(r.total_tokens for r in rs),
            "usd": round(sum(r.usd for r in rs), 6),
            "turns": sum(r.turns for r in rs),
        }
    if "tools" in out and "forge" in out:
        t, f = out["tools"]["total_tokens"], out["forge"]["total_tokens"]
        out["token_savings_pct"] = round(100 * (1 - f / t), 1) if t else None
        out["savings_on_tasks_both_solved_pct"], out["tasks_both_solved"] = _savings(runs, "tools", "forge")
    # Every pair, e.g. "forge_vs_python": savings of forge relative to python
    out["pairwise"] = {}
    for i, base in enumerate(modes):
        for mode in modes[i + 1:]:
            pct, shared = _savings(runs, base, mode)
            out["pairwise"][f"{mode}_vs_{base}"] = {"savings_pct": pct, "tasks": shared}
    out["estimated_tokens"] = any(r.estimated for r in runs)
    return out


def run_agent_bench(
    call: ModelCall,
    task_ids: Optional[list[str]] = None,
    runs: int = 1,
    price_in: float = DEFAULT_PRICE_IN,
    price_out: float = DEFAULT_PRICE_OUT,
    on_result: Optional[Callable[[AgentRun], None]] = None,
    modes: tuple = MODES,
) -> tuple[list[AgentRun], dict]:
    tasks = [t for t in AGENT_TASKS if not task_ids or t["id"] in task_ids]
    if task_ids and len(tasks) != len(set(task_ids)):
        known = ", ".join(t["id"] for t in AGENT_TASKS)
        raise ValueError(f"Unknown task id in {task_ids}; known: {known}")
    results: list[AgentRun] = []
    for task in tasks:
        for _ in range(runs):
            for mode in modes:
                r = run_agent(task, mode, call, price_in, price_out)
                results.append(r)
                if on_result:
                    on_result(r)
    return results, summarize(results)


# =============================================================================
# Backends
# =============================================================================

def make_call(backend: str, model: Optional[str]) -> tuple[ModelCall, str]:
    backend = backend.lower()
    if backend == "auto":
        backend = "ollama" if ollama_available() else ("openai" if has_api_key() else "")
    if backend == "ollama":
        host = (os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
        model = model or os.environ.get("OLLAMA_MODEL") or "llama3.2"  # tool calling needs 3B+
        def call(messages: list, tools: list) -> tuple[dict, dict]:
            return chat_message(messages, tools=tools, api_key="ollama", base_url=f"{host}/v1",
                                model=model, temperature=0.0)
        return call, f"ollama/{model}"
    if backend == "openai" and has_api_key():
        def call(messages: list, tools: list) -> tuple[dict, dict]:
            return chat_message(messages, tools=tools, model=model, temperature=0.0)
        return call, "api"
    raise RuntimeError(
        "No LLM backend: start Ollama (./start_ollama.sh, then ollama pull llama3.2) or set an API key "
        "(GROQ_API_KEY, GEMINI_API_KEY, XAI_API_KEY, OPENROUTER_API_KEY, OPENAI_API_KEY, DEEPSEEK_API_KEY)."
    )


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="forge agent-bench",
                                 description="Agent benchmark: tool-by-tool vs Python code mode vs one forge_run call")
    ap.add_argument("--backend", default="auto", help="auto | ollama | openai")
    ap.add_argument("--model", default=None)
    ap.add_argument("--tasks", nargs="*", help=f"task ids (default all): {', '.join(t['id'] for t in AGENT_TASKS)}")
    ap.add_argument("--runs", type=int, default=1, help="repeat each task (all modes) N times")
    ap.add_argument("--modes", nargs="*", default=list(MODES), choices=list(MODES),
                    help="which contestants to run (default: tools python forge)")
    ap.add_argument("--price-in", type=float, default=DEFAULT_PRICE_IN, help="$ per 1M input tokens")
    ap.add_argument("--price-out", type=float, default=DEFAULT_PRICE_OUT, help="$ per 1M output tokens")
    ap.add_argument("--out", default="agent_bench_results.json")
    args = ap.parse_args(argv)

    try:
        call, label = make_call(args.backend, args.model)
    except RuntimeError as e:
        print(e)
        return 1

    print(f"Agent benchmark  model={label}  runs={args.runs}\n")
    print(f"{'task':14} {'mode':6} {'ok':3} {'turns':>5} {'calls':>5} {'tokens':>8} {'usd':>10}  note")

    def show(r: AgentRun) -> None:
        note = r.error or ""
        print(f"{r.task_id:14} {r.mode:6} {'✓' if r.ok else '✗':3} {r.turns:5} {r.tool_calls:5} "
              f"{r.total_tokens:8} {r.usd:10.6f}  {note[:60]}")

    try:
        results, summary = run_agent_bench(call, args.tasks, args.runs, args.price_in, args.price_out, show,
                                           tuple(m for m in MODES if m in args.modes))
    except ValueError as e:
        print(e)
        return 1

    print("\nSUMMARY")
    for mode in MODES:
        if mode in summary:
            m = summary[mode]
            print(f"  {mode:6} solved {m['success']}/{m['runs']}  tokens {m['total_tokens']}  ${m['usd']:.6f}  turns {m['turns']}")
    print("\n  token savings on tasks both contestants solved (positive = first one is cheaper):")
    for pair, v in summary["pairwise"].items():
        a, b = pair.split("_vs_")
        print(f"    {a} vs {b}: {v['savings_pct']}%  ({', '.join(v['tasks']) or 'no shared solved tasks'})")
    if summary["estimated_tokens"]:
        print("  note: the provider returned no usage for some calls; those tokens are estimated")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"model": label, "summary": summary, "runs": [asdict(r) for r in results]}, f, indent=2)
    print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--sandbox-child"]:
        _sandbox_child()
        raise SystemExit(0)
    raise SystemExit(main())
