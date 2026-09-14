#!/usr/bin/env python3
"""
Forge CLI — how a new user (or AI) uses the language
====================================================

  python forge_cli.py run examples/basic_calculator.forge
  python forge_cli.py check path/to/agent.forge
  python forge_cli.py examples
  python forge_cli.py init my-agent
  python forge_cli.py repl
  python forge_cli.py doctor
  python forge_cli.py bench --backend ollama

Forge is JavaScript for AI: agent programs AIs can emit and run reliably.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def cmd_run(args: argparse.Namespace) -> int:
    from forge_core import compile_auto, ast_to_json
    from forge_repair import normalize_llm_output, try_compile_repaired
    from forge_runtime import run_forge, run_program, make_llm_client, ToolRegistry

    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 1

    source = path.read_text(encoding="utf-8")
    llm = make_llm_client(args.llm)
    tools = ToolRegistry()

    try:
        if args.repair:
            program, used, notes = try_compile_repaired(source)
            if notes:
                print(f"[repair] applied: {', '.join(notes)}", file=sys.stderr)
            result = run_program(program, tools=tools, llm=llm)
        else:
            result = run_forge(source, tools=tools, llm=llm)
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    from forge_core import compile_auto, ast_to_json, validate_ast
    from forge_repair import try_compile_repaired

    path = Path(args.file)
    source = path.read_text(encoding="utf-8")
    try:
        if args.repair:
            program, used, notes = try_compile_repaired(source)
        else:
            program = compile_auto(source)
            notes = []
    except Exception as e:
        print(f"FAIL  {path}: {e}")
        return 1

    errs = validate_ast(program)
    if errs:
        print(f"FAIL  {path}")
        for e in errs:
            print(f"  - {e}")
        return 1

    print(f"OK    {path}  agent={program.agent.name!r} steps={len(program.steps)}")
    if args.json:
        print(ast_to_json(program))
    if notes:
        print(f"      (repaired: {', '.join(notes)})")
    return 0


def cmd_examples(args: argparse.Namespace) -> int:
    from forge_core import EXAMPLES
    from forge_runtime import run_forge, MockLLMClient, ToolRegistry

    examples_dir = ROOT / "examples"
    names = sorted(EXAMPLES.keys())
    if args.list or not args.name:
        print("Built-in examples:")
        for n in names:
            print(f"  {n}")
        if examples_dir.exists():
            print("\nFile examples:")
            for p in sorted(examples_dir.glob("*.forge")):
                print(f"  {p.name}")
        print("\nRun one:  python forge_cli.py examples basic-calculator")
        print("Or file:  python forge_cli.py run examples/basic_calculator.forge")
        return 0

    name = args.name
    if name not in EXAMPLES:
        # try file
        candidate = examples_dir / name
        if not candidate.exists():
            candidate = examples_dir / f"{name}.forge"
        if candidate.exists():
            source = candidate.read_text(encoding="utf-8")
        else:
            print(f"Unknown example: {name}", file=sys.stderr)
            return 1
    else:
        source = EXAMPLES[name]

    if args.show:
        print(source.strip())
        return 0

    print(f"=== Running {name} ===\n")
    print(source.strip())
    print("\n=== Result ===")
    result = run_forge(source, tools=ToolRegistry(), llm=MockLLMClient())
    print(json.dumps(result, indent=2, default=str))
    return 0


STARTER = '''AGENT "{name}"

MEMORY {{
  topic: "hello forge"
}}

STEP research TOOL web_search INPUT {{ q: $topic, n: 3 }} OUTPUT hits

REASON "Summarize these results in one sentence" ON $hits OUTPUT summary

VERIFY $hits != null

RETURN {{ topic: $topic, summary: $summary }}
'''


def cmd_init(args: argparse.Namespace) -> int:
    name = args.name.strip().replace(" ", "-")
    out = Path(args.out or f"{name}.forge")
    if out.exists() and not args.force:
        print(f"Refusing to overwrite {out} (use --force)", file=sys.stderr)
        return 1
    out.write_text(STARTER.format(name=name), encoding="utf-8")
    print(f"Created {out}")
    print(f"Run it:  python forge_cli.py run {out}")
    print(f"Check:   python forge_cli.py check {out}")
    return 0


def cmd_repl(args: argparse.Namespace) -> int:
    from forge_runtime import ForgeREPL, make_llm_client, ToolRegistry

    ForgeREPL(tools=ToolRegistry(), llm=make_llm_client(args.llm)).run()
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    from forge_runtime import ollama_available, make_llm_client

    print("Forge doctor")
    print(f"  python:     {sys.version.split()[0]}")
    print(f"  forge root: {ROOT}")
    print(f"  ollama:     {'up' if ollama_available() else 'down (optional — free local LLM)'}")
    client = make_llm_client("auto")
    print(f"  llm client: {type(client).__name__}")
    keys = [
        "GROQ_API_KEY",
        "GEMINI_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENAI_API_KEY",
        "DEEPSEEK_API_KEY",
    ]
    for k in keys:
        print(f"  {k}: {'set' if os.environ.get(k) else '-'}")
    print("\nQuick test:")
    print("  python forge_cli.py examples basic-calculator")
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from forge_benchmark import run_benchmark, summarize

    modes = ["text", "json"] if args.mode == "both" else [args.mode]
    combined = {"modes": {}}
    worst_rate = 100.0

    for mode in modes:
        print(f"\n######## MODE: {mode} ########")
        out_path = Path(args.out)
        if args.mode == "both":
            out_path = out_path.with_name(out_path.stem + f"_{mode}" + out_path.suffix)

        results = run_benchmark(
            mode=mode,
            dry_run=args.dry_run,
            backend=args.backend,
        )
        summary = summarize(results)
        print("\nSUMMARY")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        payload = {"summary": summary, "results": [r.to_dict() for r in results]}
        out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        print(f"Wrote {out_path}")
        combined["modes"][mode] = summary
        worst_rate = min(worst_rate, float(summary.get("full_rate", 0)))

    if args.mode == "both":
        Path(args.out).write_text(json.dumps(combined, indent=2), encoding="utf-8")
        print(f"\nCombined summary → {args.out}")
        print(f"Worst full_rate across modes: {worst_rate}%")

    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="forge",
        description="Forge — JavaScript for AI. Agent language made for LLMs.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Compile and execute a .forge program")
    run_p.add_argument("file")
    run_p.add_argument("--llm", default="mock", help="mock|auto|ollama|openai (mock = instant demos)")
    run_p.add_argument("--repair", action="store_true", help="Normalize LLM-ish source before run")
    run_p.set_defaults(func=cmd_run)

    check_p = sub.add_parser("check", help="Parse + validate without executing tools/LLM")
    check_p.add_argument("file")
    check_p.add_argument("--json", action="store_true", help="Print JSON AST")
    check_p.add_argument("--repair", action="store_true")
    check_p.set_defaults(func=cmd_check)

    ex_p = sub.add_parser("examples", help="List or run built-in examples")
    ex_p.add_argument("name", nargs="?", help="Example id to run")
    ex_p.add_argument("--list", action="store_true")
    ex_p.add_argument("--show", action="store_true", help="Print source only")
    ex_p.set_defaults(func=cmd_examples)

    init_p = sub.add_parser("init", help="Scaffold a starter .forge agent")
    init_p.add_argument("name", help="Agent name")
    init_p.add_argument("-o", "--out", help="Output path")
    init_p.add_argument("--force", action="store_true")
    init_p.set_defaults(func=cmd_init)

    repl_p = sub.add_parser("repl", help="Interactive Forge REPL")
    repl_p.add_argument("--llm", default="auto")
    repl_p.set_defaults(func=cmd_repl)

    doc_p = sub.add_parser("doctor", help="Check local setup")
    doc_p.set_defaults(func=cmd_doctor)

    bench_p = sub.add_parser("bench", help="LLM generation reliability benchmark")
    bench_p.add_argument("--backend", default="auto")
    bench_p.add_argument(
        "--mode",
        choices=["text", "json", "both"],
        default="text",
        help="text recommended for small local models; json AST for stronger LLMs",
    )
    bench_p.add_argument("--dry-run", action="store_true")
    bench_p.add_argument("--out", default="benchmark_results.json")
    bench_p.set_defaults(func=cmd_bench)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
