"""
Forge MCP server
================
Lets an AI agent (Claude Code, Cursor, any MCP client) run a whole
multi-step Forge program in ONE tool call instead of one round trip per
step. Every round trip re-sends the conversation, so batching the
fetch → filter → check → write chain into a program is where the credit
savings come from.

  forge mcp                         # stdio server
  forge mcp --tools my_tools.py     # plus your own tools
  claude mcp add forge -- forge mcp # register with Claude Code

Stdlib only: newline-delimited JSON-RPC 2.0 over stdin/stdout.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Callable, Optional, TextIO

from forge_core import check_program, compile_auto, validate_ast
from forge_runtime import ToolRegistry, language_spec, make_llm_client, run_program

SERVER_INFO = {"name": "forge", "version": "0.2.0"}
DEFAULT_PROTOCOL = "2024-11-05"


def _tool_defs(tools: ToolRegistry) -> list[dict]:
    spec = language_spec(tools)
    source = {"type": "string", "description": "Forge program source (text syntax or JSON AST)"}
    return [
        {
            "name": "forge_run",
            "description": (
                "Run a Forge agent program: chain tool calls, filters, checks and LLM steps "
                "in ONE call instead of a round trip per step. Returns the RETURN value as JSON, "
                "or an error that names what to fix. REASON steps use the server's LLM "
                "(a mock if none is configured), so when you are the reasoner, RETURN the data "
                "and reason over it yourself.\n\n" + spec
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source": source,
                    "llm": {
                        "type": "string",
                        "enum": ["mock", "auto", "ollama", "openai"],
                        "description": "LLM backend for REASON steps (default: server setting)",
                    },
                },
                "required": ["source"],
            },
        },
        {
            "name": "forge_check",
            "description": "Validate a Forge program without running anything: syntax, undefined $vars, unknown tools.",
            "inputSchema": {
                "type": "object",
                "properties": {"source": source},
                "required": ["source"],
            },
        },
    ]


class ForgeMCPServer:
    def __init__(self, tools: Optional[ToolRegistry] = None, llm: str = "auto") -> None:
        self.tools = tools or ToolRegistry()
        self.llm = llm
        self.methods: dict[str, Callable[[dict], Any]] = {
            "initialize": self._initialize,
            "ping": lambda _p: {},
            "tools/list": lambda _p: {"tools": _tool_defs(self.tools)},
            "tools/call": self._call_tool,
        }

    def handle(self, msg: dict) -> Optional[dict]:
        """Handle one JSON-RPC message; returns the response (None for notifications)."""
        method = msg.get("method")
        msg_id = msg.get("id")
        if msg_id is None:
            return None  # notification (e.g. notifications/initialized)
        fn = self.methods.get(method or "")
        if fn is None:
            return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}
        try:
            return {"jsonrpc": "2.0", "id": msg_id, "result": fn(msg.get("params") or {})}
        except Exception as e:
            return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32603, "message": f"{type(e).__name__}: {e}"}}

    def _initialize(self, params: dict) -> dict:
        return {
            "protocolVersion": params.get("protocolVersion") or DEFAULT_PROTOCOL,
            "capabilities": {"tools": {}},
            "serverInfo": SERVER_INFO,
            "instructions": "Use forge_run to batch multi-step tool work into one call.",
        }

    def _call_tool(self, params: dict) -> dict:
        name = params.get("name")
        args = params.get("arguments") or {}
        source = args.get("source")
        if not isinstance(source, str) or not source.strip():
            return _text("Missing required argument: source", error=True)
        try:
            program = compile_auto(source)
            if name == "forge_check":
                problems = validate_ast(program) or check_program(program, self.tools.names())
                if problems:
                    return _text("FAIL\n" + "\n".join(problems), error=True)
                return _text(f"OK  agent={program.agent.name!r}  steps={len(program.steps)}")
            if name == "forge_run":
                llm = make_llm_client(args.get("llm") or self.llm)
                result = run_program(program, tools=self.tools, llm=llm)
                return _text(json.dumps(result, indent=2, default=str))
        except Exception as e:
            return _text(f"{type(e).__name__}: {e}", error=True)
        return _text(f"Unknown tool: {name}", error=True)


def _text(text: str, error: bool = False) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": error}


def serve(server: ForgeMCPServer, stdin: TextIO = sys.stdin, stdout: TextIO = sys.stdout) -> None:
    # stdout carries protocol messages only; stray print() from tools goes to stderr
    real_stdout, sys.stdout = sys.stdout, sys.stderr
    try:
        _serve(server, stdin, stdout)
    finally:
        sys.stdout = real_stdout


def _serve(server: ForgeMCPServer, stdin: TextIO, stdout: TextIO) -> None:
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            resp: Optional[dict] = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": f"Parse error: {e}"}}
        else:
            resp = server.handle(msg) if isinstance(msg, dict) else None
        if resp is not None:
            stdout.write(json.dumps(resp, default=str) + "\n")
            stdout.flush()
