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


def generate_and_run(
    task: str,
    backend: str = "auto",
    mode: str = "text",
    execute: bool = True,
) -> dict:
    prompt = build_prompt(task, mode=mode)
    response, usage = call_llm(prompt, backend=backend)
    extracted = extract_code(response, prefer_json=(mode == "json"))

    out: dict = {
        "task": task,
        "mode": mode,
        "usage": usage,
        "raw_response": response,
        "extracted": extracted,
        "parse_ok": False,
        "runtime_ok": False,
    }

    try:
        program, used, notes = try_compile_repaired(response if mode == "text" else extracted)
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
        # Use real LLM for REASON if backend is live; else mock
        llm = make_llm_client("mock" if backend in ("mock", "dry") else backend)
        # Prefer mock for deterministic demo unless --live-reason
        if not getattr(generate_and_run, "_live_reason", False):
            llm = MockLLMClient()
        result = Evaluator(tools=ToolRegistry(), llm=llm).run(program)
        out["runtime_ok"] = True
        out["result"] = result
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
        )
    except Exception as e:
        print(f"Generation failed: {e}", file=sys.stderr)
        print("Tip: start Ollama (`./start_ollama.sh`) or set GROQ_API_KEY / GEMINI_API_KEY", file=sys.stderr)
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
