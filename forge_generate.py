#!/usr/bin/env python3
"""
Forge generate — AI writes Forge, then we repair/validate/run
=============================================================

  python forge_generate.py "Add 10 and 20 and return the sum"
  python forge_generate.py --backend ollama "Search AI news and summarize"
  python forge_generate.py --json "..."     # ask for JSON AST

This is the core UX of Forge: models emit agent programs in an AI-native
language, not Python spaghetti.
"""

from __future__ import annotations

import argparse
import json
import sys

from forge_benchmark import build_prompt, call_llm, extract_code
from forge_repair import try_compile_repaired
from forge_runtime import Evaluator, MockLLMClient, ToolRegistry, make_llm_client


FIX_PROMPT = """{prompt}

Your previous program:
{program}

It failed with:
{error}

Reply with the complete corrected Forge program only."""


def generate_and_run(
    task: str,
    backend: str = "auto",
    mode: str = "text",
    execute: bool = True,
    retries: int = 2,
    tools: ToolRegistry | None = None,
    llm_call=None,
) -> dict:
    """
    Ask an LLM for a Forge program, then repair → check → run it.
    On failure the exact error is sent back to the model (up to `retries`
    extra attempts); Forge errors name what is defined/available, so one
    round usually fixes it. Usage from every attempt is summed.
    """
    llm_call = llm_call or (lambda prompt: call_llm(prompt, backend=backend))
    tools = tools or ToolRegistry()
    base_prompt = build_prompt(task, mode=mode)
    prompt = base_prompt
    usage_total: dict = {}
    out: dict = {}

    for attempt in range(1, retries + 2):
        response, usage = llm_call(prompt)
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if usage.get(k) is not None:
                usage_total[k] = usage_total.get(k, 0) + usage[k]
        out = _attempt(task, mode, response, execute, tools, backend)
        out["attempts"] = attempt
        out["usage"] = {**usage, **usage_total}
        if out["parse_ok"] and (not execute or out["runtime_ok"]):
            return out
        prompt = FIX_PROMPT.format(
            prompt=base_prompt,
            program=out.get("extracted") or response,
            error=out.get("error", "unknown error"),
        )
    return out


def _attempt(task: str, mode: str, response: str, execute: bool, tools: ToolRegistry, backend: str) -> dict:
    out: dict = {
        "task": task,
        "mode": mode,
        "raw_response": response,
        "extracted": extract_code(response, prefer_json=(mode == "json")),
        "parse_ok": False,
        "runtime_ok": False,
    }
    try:
        program, used, notes = try_compile_repaired(response if mode == "text" else out["extracted"])
        out["extracted"] = used
        out["parse_ok"] = True
        out["repair"] = notes
        out["agent"] = program.agent.name
    except Exception as e:
        out["error"] = f"parse: {e}"
        return out

    if not execute:
        return out

    try:
        # REASON uses the mock unless live reasoning was requested
        live = getattr(generate_and_run, "_live_reason", False)
        llm = make_llm_client(backend) if live and backend not in ("mock", "dry") else MockLLMClient()
        out["result"] = Evaluator(tools=tools, llm=llm).run(program)
        out["runtime_ok"] = True
    except Exception as e:
        out["error"] = f"runtime: {e}"
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Ask an LLM to write Forge, then run it")
    p.add_argument("task", help="Natural language description of the agent")
    p.add_argument("--backend", default="auto", help="auto|ollama|openai")
    p.add_argument(
        "--mode",
        choices=["text", "json"],
        default="text",
        help="text surface syntax (default; best for small local models) or json AST",
    )
    p.add_argument("--no-run", action="store_true", help="Only generate + parse")
    p.add_argument("--retries", type=int, default=2, help="Fix attempts after a failure (error fed back)")
    p.add_argument("--live-reason", action="store_true", help="Use real LLM for REASON blocks")
    p.add_argument("--show-source", action="store_true", help="Print generated Forge")
    args = p.parse_args()

    generate_and_run._live_reason = args.live_reason  # type: ignore

    try:
        out = generate_and_run(
            args.task,
            backend=args.backend,
            mode=args.mode,
            execute=not args.no_run,
            retries=args.retries,
        )
    except Exception as e:
        print(f"Generation failed: {e}", file=sys.stderr)
        print("Tip: start Ollama (`./start_ollama.sh`) or set GROQ_API_KEY / GEMINI_API_KEY / XAI_API_KEY (Grok)", file=sys.stderr)
        return 1

    if args.show_source or not out.get("runtime_ok"):
        print("=== Generated Forge ===")
        print(out.get("extracted") or out.get("raw_response", "")[:2000])
        print()

    print("=== Status ===")
    print(f"parse_ok:   {out.get('parse_ok')}")
    print(f"runtime_ok: {out.get('runtime_ok')}")
    if out.get("repair"):
        print(f"repair:     {out['repair']}")
    if out.get("error"):
        print(f"error:      {out['error']}")
    if out.get("result") is not None:
        print("\n=== Result ===")
        print(json.dumps(out["result"], indent=2, default=str))
    return 0 if out.get("parse_ok") and (args.no_run or out.get("runtime_ok")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
