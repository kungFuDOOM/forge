#!/usr/bin/env python3
"""
Forge CLI — usable first, useful second
=======================================

  ./forge                  # welcome + next steps
  ./forge quickstart       # 60-second guided demo
  ./forge run FILE.forge
  ./forge init my-agent
  ./forge examples
  ./forge tools
  ./forge tokens FILE.forge
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


BANNER = """
Forge — JavaScript for AI
Agent programs AIs (and humans) can write and run.

Try this now:
  ./forge quickstart
  ./forge run examples/basic_calculator.forge
  ./forge init my-agent && ./forge run my-agent.forge
""".strip()


def _friendly_error(exc: Exception) -> str:
    name = type(exc).__name__
    msg = str(exc)
    hints = []
    if "Undefined variable" in msg:
        hints.append("Check MEMORY / OUTPUT names — every $var must be set before use.")
    if "Unknown tool" in msg:
        hints.append("Run: ./forge tools")
    if "Expected" in msg or "parse" in name.lower() or "Parse" in name:
        hints.append("Run: ./forge check FILE.forge")
        hints.append("Or:  ./forge check FILE.forge --repair")
    if "http_get" in msg or "URL" in msg:
        hints.append("http_get needs: INPUT { url: \"https://...\" }")
    if hints:
        return f"{name}: {msg}\n  tip: " + "\n  tip: ".join(hints)
    return f"{name}: {msg}"


def cmd_run(args: argparse.Namespace) -> int:
    from forge_repair import try_compile_repaired
    from forge_runtime import run_forge, run_program, make_llm_client, ToolRegistry

    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        print("  tip: ./forge examples", file=sys.stderr)
        return 1

    source = path.read_text(encoding="utf-8")
    llm = make_llm_client(args.llm)
    tools = ToolRegistry()

    try:
        if args.repair:
            program, used, notes = try_compile_repaired(source)
            if notes and not args.quiet:
                print(f"[repair] {', '.join(notes)}", file=sys.stderr)
            result = run_program(program, tools=tools, llm=llm)
        else:
            result = run_forge(source, tools=tools, llm=llm)
    except Exception as e:
        print(_friendly_error(e), file=sys.stderr)
        return 1

    if not args.quiet:
        print(f"# ran {path}  (llm={type(llm).__name__})")
    print(json.dumps(result, indent=2, default=str))
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    from forge_core import compile_auto, ast_to_json, validate_ast
    from forge_repair import try_compile_repaired

    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}")
        return 1
    source = path.read_text(encoding="utf-8")
    try:
        if args.repair:
            program, used, notes = try_compile_repaired(source)
        else:
            program = compile_auto(source)
            notes = []
    except Exception as e:
        print(f"FAIL  {path}")
        print(f"  {_friendly_error(e)}")
        return 1

    errs = validate_ast(program)
    if errs:
        print(f"FAIL  {path}")
        for e in errs:
            print(f"  - {e}")
        return 1

    print(f"OK    {path}")
    print(f"      agent={program.agent.name!r}  steps={len(program.steps)}")
    if notes:
        print(f"      repaired: {', '.join(notes)}")
    if args.json:
        print(ast_to_json(program))
    return 0


def cmd_examples(args: argparse.Namespace) -> int:
    from forge_core import EXAMPLES
    from forge_runtime import run_forge, MockLLMClient, ToolRegistry

    examples_dir = ROOT / "examples"
    names = sorted(EXAMPLES.keys())
    files = sorted(examples_dir.glob("*.forge")) if examples_dir.exists() else []

    if args.list or not args.name:
        print("Built-in demos (always work offline):")
        for n in names:
            print(f"  ./forge examples {n}")
        if files:
            print("\nExample files:")
            for p in files:
                print(f"  ./forge run examples/{p.name}")
        print("\nShow source:  ./forge examples basic-calculator --show")
        return 0

    name = args.name
    source = None
    if name in EXAMPLES:
        source = EXAMPLES[name]
    else:
        for candidate in (
            examples_dir / name,
            examples_dir / f"{name}.forge",
            Path(name),
        ):
            if candidate.exists():
                source = candidate.read_text(encoding="utf-8")
                break
    if source is None:
        print(f"Unknown example: {name}", file=sys.stderr)
        print("  tip: ./forge examples", file=sys.stderr)
        return 1

    if args.show:
        print(source.strip())
        return 0

    print(f"=== {name} ===\n")
    print(source.strip())
    print("\n=== result ===")
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
    if args.template == "http":
        template = (ROOT / "examples" / "url_fetcher.forge").read_text(encoding="utf-8")
        template = template.replace('AGENT "url-fetcher"', f'AGENT "{name}"', 1)
    elif args.template == "file":
        template = (ROOT / "examples" / "file_summarizer.forge").read_text(encoding="utf-8")
        template = template.replace('AGENT "file-summarizer"', f'AGENT "{name}"', 1)
    else:
        template = STARTER.format(name=name)
    out.write_text(template, encoding="utf-8")
    print(f"Created {out}")
    print(f"  check:  ./forge check {out}")
    print(f"  run:    ./forge run {out}")
    return 0


def cmd_repl(args: argparse.Namespace) -> int:
    from forge_runtime import ForgeREPL, make_llm_client, ToolRegistry

    print("Forge REPL — paste a program, blank line to run. :quit to exit.")
    ForgeREPL(tools=ToolRegistry(), llm=make_llm_client(args.llm)).run()
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    from forge_runtime import ollama_available, make_llm_client, ToolRegistry

    print("Forge doctor\n")
    print(f"  python:     {sys.version.split()[0]}")
    print(f"  forge root: {ROOT}")
    print(f"  ollama:     {'up ✓' if ollama_available() else 'down (optional)'}")
    client = make_llm_client("mock")
    print(f"  llm (mock): {type(client).__name__} ✓")
    print(f"  tools:      {', '.join(ToolRegistry().names())}")
    print("\nYou're ready. Run:")
    print("  ./forge quickstart")
    return 0


def cmd_tools(args: argparse.Namespace) -> int:
    from forge_runtime import ToolRegistry

    tools = ToolRegistry()
    docs = {
        "arithmetic_add": "INPUT { x, y } → number (x+y)",
        "web_search": "INPUT { q, n } → list of mock search hits (offline demo)",
        "get_value": "INPUT { } → 42",
        "sales_data": "INPUT { region, period } → list of deals",
        "http_get": "INPUT { url } → { status, body, ... }  (real HTTP)",
        "read_file": "INPUT { path } → { body, chars }  (under cwd only)",
        "write_file": "INPUT { path, body } → { ok, path }  (under cwd only)",
    }
    print("Available tools\n")
    for name in tools.names():
        print(f"  {name}")
        print(f"    {docs.get(name, '')}")
    print("\nExample:")
    print('  STEP fetch TOOL http_get INPUT { url: "https://example.com" } OUTPUT page')
    return 0


def cmd_quickstart(args: argparse.Namespace) -> int:
    from forge_runtime import run_forge, MockLLMClient, ToolRegistry
    from forge_core import EXAMPLES

    print("=" * 56)
    print("Forge quickstart (60 seconds)")
    print("=" * 56)
    print(
        "\nForge is a tiny language for AI agents.\n"
        "You write AGENT → MEMORY → STEP → REASON → VERIFY → RETURN.\n"
        "Then you run it. No Python framework required.\n"
    )

    print("① Running a built-in demo…\n")
    src = EXAMPLES["basic-calculator"]
    print(src.strip())
    print("\n→ result:")
    result = run_forge(src, tools=ToolRegistry(), llm=MockLLMClient())
    print(json.dumps(result, indent=2, default=str))

    out = Path("my-first-agent.forge")
    if not out.exists() or args.force:
        out.write_text(STARTER.format(name="my-first-agent"), encoding="utf-8")
        print(f"\n② Created {out}")
    else:
        print(f"\n② {out} already exists (use --force to overwrite)")

    print(
        f"\n③ Your turn:\n"
        f"  ./forge run {out}\n"
        f"  ./forge check {out}\n"
        f"  ./forge run examples/url_fetcher.forge      # real HTTP\n"
        f"  ./forge run examples/file_summarizer.forge  # read a file\n"
        f"  ./forge tools\n"
    )
    print("That's it. Edit the .forge file — that's the whole program.")
    return 0


def cmd_tokens(args: argparse.Namespace) -> int:
    """Show rough token estimate vs Python / LangChain-style glue."""
    path = Path(args.file)
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 1
    forge_src = path.read_text(encoding="utf-8").strip()

    python_equiv = f'''
def agent():
    # Rough equivalent of {path.name} as handwritten Python
    memory = {{}}  # variables
    # tool calls + filters + llm.reason(...) + asserts
    result = {{}}
    return result
# plus imports, tool wrappers, and agent loop boilerplate
'''.strip()

    langchainish = f'''
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
# plus tool defs, prompt templates, executor wiring for {path.name}
prompt = ChatPromptTemplate.from_messages([("system", "..."), ("human", "{{input}}")])
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools)
result = executor.invoke({{"input": "..."}})
'''.strip()

    def est(s: str) -> int:
        return max(1, len(s) // 4)

    rows = [
        ("Forge program", forge_src),
        ("Minimal Python sketch", python_equiv),
        ("LangChain-style glue", langchainish),
    ]
    print(f"Token estimate for {path} (chars/4 ≈ tokens)\n")
    forge_tok = est(forge_src)
    for label, src in rows:
        t = est(src)
        delta = ""
        if label != "Forge program":
            pct = 100.0 * (1 - forge_tok / t)
            delta = f"  (Forge is ~{pct:.0f}% fewer tokens than this)"
        print(f"  {label:24}  chars={len(src):5}  ~tokens={t:4}{delta}")
    print(
        "\nNote: biggest savings often come from fewer failed generations/retries,\n"
        "not only shorter source. See VISION.md."
    )
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
        results = run_benchmark(mode=mode, dry_run=args.dry_run, backend=args.backend)
        summary = summarize(results)
        print("\nSUMMARY")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        out_path.write_text(
            json.dumps({"summary": summary, "results": [r.to_dict() for r in results]}, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"Wrote {out_path}")
        combined["modes"][mode] = summary
        worst_rate = min(worst_rate, float(summary.get("full_rate", 0)))

    if args.mode == "both":
        Path(args.out).write_text(json.dumps(combined, indent=2), encoding="utf-8")
        print(f"\nCombined → {args.out}  (worst full_rate={worst_rate}%)")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(BANNER)
        print("\nCommands: quickstart · run · check · init · examples · tools · tokens · doctor · repl")
        return 0

    p = argparse.ArgumentParser(
        prog="forge",
        description="Forge — JavaScript for AI. Write agent programs. Run them.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    qs = sub.add_parser("quickstart", help="60-second guided demo")
    qs.add_argument("--force", action="store_true")
    qs.set_defaults(func=cmd_quickstart)

    run_p = sub.add_parser("run", help="Run a .forge program")
    run_p.add_argument("file")
    run_p.add_argument("--llm", default="mock", help="mock|auto|ollama|openai")
    run_p.add_argument("--repair", action="store_true")
    run_p.add_argument("-q", "--quiet", action="store_true")
    run_p.set_defaults(func=cmd_run)

    check_p = sub.add_parser("check", help="Validate a .forge program")
    check_p.add_argument("file")
    check_p.add_argument("--json", action="store_true")
    check_p.add_argument("--repair", action="store_true")
    check_p.set_defaults(func=cmd_check)

    ex_p = sub.add_parser("examples", help="List or run demos")
    ex_p.add_argument("name", nargs="?")
    ex_p.add_argument("--list", action="store_true")
    ex_p.add_argument("--show", action="store_true")
    ex_p.set_defaults(func=cmd_examples)

    init_p = sub.add_parser("init", help="Create a new .forge agent")
    init_p.add_argument("name")
    init_p.add_argument("-o", "--out")
    init_p.add_argument("--force", action="store_true")
    init_p.add_argument(
        "--template",
        choices=["basic", "http", "file"],
        default="basic",
        help="basic | http (fetch URL) | file (summarize file)",
    )
    init_p.set_defaults(func=cmd_init)

    tools_p = sub.add_parser("tools", help="List built-in tools")
    tools_p.set_defaults(func=cmd_tools)

    tok_p = sub.add_parser("tokens", help="Estimate tokens vs Python/LangChain")
    tok_p.add_argument("file")
    tok_p.set_defaults(func=cmd_tokens)

    repl_p = sub.add_parser("repl", help="Interactive REPL")
    repl_p.add_argument("--llm", default="mock")
    repl_p.set_defaults(func=cmd_repl)

    doc_p = sub.add_parser("doctor", help="Check setup")
    doc_p.set_defaults(func=cmd_doctor)

    bench_p = sub.add_parser("bench", help="LLM generation benchmark")
    bench_p.add_argument("--backend", default="auto")
    bench_p.add_argument("--mode", choices=["text", "json", "both"], default="text")
    bench_p.add_argument("--dry-run", action="store_true")
    bench_p.add_argument("--out", default="benchmark_results.json")
    bench_p.set_defaults(func=cmd_bench)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
