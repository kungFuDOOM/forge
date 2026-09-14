"""
Forge v0.1 — Runtime
====================
Evaluator, tool registry, LLM clients, REPL, run_forge pipeline.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from forge_core import (
    ASTNode,
    Comparison,
    Filter,
    ForgeError,
    Literal,
    MemoryBlock,
    ObjectLiteral,
    Program,
    Reason,
    Return,
    Step,
    ToolCall,
    Variable,
    Verify,
    compile_auto,
    compile_forge,
    EXAMPLES,
)


# =============================================================================
# EXCEPTIONS
# =============================================================================

class ForgeRuntimeError(ForgeError):
    """Runtime evaluation error."""


class ForgeVerifyError(ForgeRuntimeError):
    """VERIFY assertion failed."""


# =============================================================================
# LLM CLIENTS
# =============================================================================

class LLMClient(ABC):
    @abstractmethod
    def complete(self, prompt: str, context: Any = None) -> str:
        ...


class MockLLMClient(LLMClient):
    """Deterministic mock for offline testing."""

    def complete(self, prompt: str, context: Any = None) -> str:
        ctx_preview = ""
        if context is not None:
            try:
                ctx_preview = json.dumps(context, default=str)[:120]
            except Exception:
                ctx_preview = str(context)[:120]
        return f"[mock-reason] {prompt[:80]} | data={ctx_preview}"


class OpenAILLMClient(LLMClient):
    """OpenAI-compatible client (OpenAI, DeepSeek, Groq, Gemini, local proxies)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        self.api_key, self.base_url, self.model = _resolve_openai_compat(
            api_key=api_key, base_url=base_url, model=model
        )
        if not self.api_key:
            raise ForgeRuntimeError(
                "No API key. Set GROQ_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY, "
                "or DEEPSEEK_API_KEY — or use OllamaLLMClient (free/local)."
            )

    def complete(self, prompt: str, context: Any = None) -> str:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise ForgeRuntimeError("Install openai: pip install openai") from e

        kwargs: dict[str, Any] = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url

        client = OpenAI(**kwargs)
        user_content = prompt
        if context is not None:
            user_content = f"{prompt}\n\nContext data:\n{json.dumps(context, default=str, indent=2)}"

        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a concise reasoning assistant for an agent runtime."},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
        )
        return resp.choices[0].message.content or ""


class OllamaLLMClient(LLMClient):
    """Free local LLM via Ollama (https://ollama.com). No API key required."""

    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
    ):
        self.model = model or os.environ.get("OLLAMA_MODEL") or "llama3.2:1b"
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")

    def complete(self, prompt: str, context: Any = None) -> str:
        import urllib.error
        import urllib.request

        user_content = prompt
        if context is not None:
            user_content = f"{prompt}\n\nContext data:\n{json.dumps(context, default=str, indent=2)}"

        body = json.dumps({
            "model": self.model,
            "stream": False,
            "messages": [
                {"role": "system", "content": "You are a concise reasoning assistant for an agent runtime."},
                {"role": "user", "content": user_content},
            ],
            "options": {"temperature": 0.2},
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise ForgeRuntimeError(
                f"Ollama not reachable at {self.host}. "
                f"Install from https://ollama.com then: ollama pull {self.model}\n"
                f"Original error: {e}"
            ) from e

        msg = data.get("message") or {}
        return msg.get("content") or data.get("response") or ""


def _resolve_openai_compat(
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> tuple[str, Optional[str], str]:
    """Pick free-friendly provider from env when explicit args omitted."""
    model = model or os.environ.get("FORGE_LLM_MODEL") or os.environ.get("FORGE_BENCH_MODEL")

    # Explicit key wins with matching base URL
    if api_key:
        return api_key, base_url, model or "gpt-4o-mini"

    # Prefer free-tier providers first
    if os.environ.get("GROQ_API_KEY"):
        return (
            os.environ["GROQ_API_KEY"],
            base_url or "https://api.groq.com/openai/v1",
            model or "llama-3.1-8b-instant",
        )
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        return (
            key or "",
            base_url or "https://generativelanguage.googleapis.com/v1beta/openai/",
            model or "gemini-2.0-flash",
        )
    if os.environ.get("OPENROUTER_API_KEY"):
        return (
            os.environ["OPENROUTER_API_KEY"],
            base_url or "https://openrouter.ai/api/v1",
            model or "meta-llama/llama-3.2-3b-instruct:free",
        )
    if os.environ.get("OPENAI_API_KEY"):
        return (
            os.environ["OPENAI_API_KEY"],
            base_url or os.environ.get("OPENAI_BASE_URL"),
            model or "gpt-4o-mini",
        )
    if os.environ.get("DEEPSEEK_API_KEY"):
        return (
            os.environ["DEEPSEEK_API_KEY"],
            base_url or os.environ.get("DEEPSEEK_BASE_URL") or "https://api.deepseek.com",
            model or "deepseek-chat",
        )
    return "", base_url, model or "gpt-4o-mini"


def ollama_available(host: Optional[str] = None) -> bool:
    import urllib.error
    import urllib.request

    host = (host or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def make_llm_client(prefer: Optional[str] = None) -> LLMClient:
    """
    Build best available LLM client.

    prefer: "mock" | "ollama" | "openai" | None (auto)
    Free path priority: Ollama local → Groq/Gemini/OpenRouter keys → paid keys → mock
    """
    prefer = (prefer or os.environ.get("FORGE_LLM_BACKEND") or "auto").lower()

    if prefer == "mock":
        return MockLLMClient()
    if prefer == "ollama":
        return OllamaLLMClient()
    if prefer in ("openai", "api"):
        return OpenAILLMClient()

    # auto
    if ollama_available():
        return OllamaLLMClient()
    try:
        return OpenAILLMClient()
    except ForgeRuntimeError:
        return MockLLMClient()


# =============================================================================
# TOOL REGISTRY
# =============================================================================

ToolFn = Callable[[dict], Any]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolFn] = {}
        self._register_builtins()

    def register(self, name: str, fn: ToolFn) -> None:
        self._tools[name] = fn

    def call(self, name: str, inputs: dict) -> Any:
        if name not in self._tools:
            raise ForgeRuntimeError(f"Unknown tool: {name}")
        return self._tools[name](inputs)

    def names(self) -> list[str]:
        return sorted(self._tools.keys())

    def _register_builtins(self) -> None:
        def arithmetic_add(inputs: dict) -> Any:
            x = inputs.get("x", 0)
            y = inputs.get("y", 0)
            # Return numeric sum so VERIFY $sum > 0 works
            return x + y

        def web_search(inputs: dict) -> Any:
            q = inputs.get("q", "")
            n = int(inputs.get("n", 5) or 5)
            results = []
            for i in range(n):
                results.append({
                    "title": f"Result {i + 1} for '{q}'",
                    "url": f"https://example.com/r/{i + 1}",
                    "relevance": round(0.95 - i * 0.1, 2),
                    "snippet": f"Snippet about {q} (#{i + 1})",
                })
            return results

        def get_value(inputs: dict) -> Any:
            return 42

        def sales_data(inputs: dict) -> Any:
            region = inputs.get("region", "unknown")
            period = inputs.get("period", "Q1")
            return [
                {"deal": "Acme", "amount": 15000, "region": region, "period": period},
                {"deal": "BetaCo", "amount": 8000, "region": region, "period": period},
                {"deal": "Gamma Inc", "amount": 22000, "region": region, "period": period},
                {"deal": "Delta LLC", "amount": 5000, "region": region, "period": period},
            ]

        def http_get(inputs: dict) -> Any:
            """Fetch a URL. inputs: url (required), max_bytes (optional)."""
            import urllib.error
            import urllib.request

            url = inputs.get("url") or inputs.get("u")
            if not url or not isinstance(url, str):
                raise ForgeRuntimeError("http_get requires string input: url")
            if not url.startswith(("http://", "https://")):
                raise ForgeRuntimeError("http_get only allows http:// or https:// URLs")
            max_bytes = int(inputs.get("max_bytes", 100_000) or 100_000)
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "ForgeAgent/0.1"},
                method="GET",
            )
            try:
                with urllib.request.urlopen(req, timeout=20) as resp:
                    raw = resp.read(max_bytes + 1)
                    truncated = len(raw) > max_bytes
                    body = raw[:max_bytes].decode("utf-8", errors="replace")
                    return {
                        "status": getattr(resp, "status", 200),
                        "url": url,
                        "content_type": resp.headers.get("Content-Type", ""),
                        "body": body,
                        "truncated": truncated,
                        "chars": len(body),
                    }
            except urllib.error.HTTPError as e:
                body = e.read(max_bytes).decode("utf-8", errors="replace") if e.fp else ""
                return {
                    "status": e.code,
                    "url": url,
                    "content_type": e.headers.get("Content-Type", "") if e.headers else "",
                    "body": body,
                    "truncated": False,
                    "chars": len(body),
                    "error": str(e),
                }
            except Exception as e:
                raise ForgeRuntimeError(f"http_get failed: {e}") from e

        def read_file(inputs: dict) -> Any:
            """Read a local text file (sandboxed to cwd). inputs: path."""
            from pathlib import Path

            path = inputs.get("path") or inputs.get("file")
            if not path or not isinstance(path, str):
                raise ForgeRuntimeError("read_file requires string input: path")
            p = Path(path).expanduser()
            if not p.is_absolute():
                p = Path.cwd() / p
            p = p.resolve()
            cwd = Path.cwd().resolve()
            try:
                rel = p.relative_to(cwd)
            except ValueError as e:
                raise ForgeRuntimeError(
                    f"read_file path must be under current directory: {cwd}"
                ) from e
            if not p.exists() or not p.is_file():
                raise ForgeRuntimeError(f"File not found: {path}")
            max_bytes = int(inputs.get("max_bytes", 200_000) or 200_000)
            data = p.read_bytes()[: max_bytes + 1]
            truncated = len(data) > max_bytes
            text = data[:max_bytes].decode("utf-8", errors="replace")
            return {
                "path": str(rel),
                "body": text,
                "chars": len(text),
                "truncated": truncated,
            }

        def write_file(inputs: dict) -> Any:
            """Write text to a file under cwd. inputs: path, body."""
            from pathlib import Path

            path = inputs.get("path") or inputs.get("file")
            body = inputs.get("body")
            if not path or not isinstance(path, str):
                raise ForgeRuntimeError("write_file requires string input: path")
            if body is None:
                raise ForgeRuntimeError("write_file requires input: body")
            p = Path(path).expanduser()
            if not p.is_absolute():
                p = Path.cwd() / p
            p = p.resolve()
            cwd = Path.cwd().resolve()
            try:
                p.relative_to(cwd)
            except ValueError as e:
                raise ForgeRuntimeError(
                    f"write_file path must be under current directory: {cwd}"
                ) from e
            p.parent.mkdir(parents=True, exist_ok=True)
            text = body if isinstance(body, str) else json.dumps(body, indent=2, default=str)
            p.write_text(text, encoding="utf-8")
            return {"path": str(p.relative_to(cwd)), "chars": len(text), "ok": True}

        self.register("arithmetic_add", arithmetic_add)
        self.register("web_search", web_search)
        self.register("get_value", get_value)
        self.register("sales_data", sales_data)
        self.register("http_get", http_get)
        self.register("read_file", read_file)
        self.register("write_file", write_file)


# =============================================================================
# EVALUATOR
# =============================================================================

class Evaluator:
    def __init__(
        self,
        tools: Optional[ToolRegistry] = None,
        llm: Optional[LLMClient] = None,
    ):
        self.tools = tools or ToolRegistry()
        self.llm = llm or MockLLMClient()
        self.memory: dict[str, Any] = {}
        self.agent_name: str = ""

    def run(self, program: Program) -> Any:
        self.memory = {}
        self.agent_name = program.agent.name

        # 1. AGENT — identity only
        # 2. MEMORY
        if program.memory:
            self._eval_memory(program.memory)

        # 3. STEPS
        for step in program.steps:
            self._eval_step(step)

        # 4. REASON
        if program.reason:
            self._eval_reason(program.reason)

        # 5. VERIFY
        if program.verify:
            self._eval_verify(program.verify)

        # 6. RETURN
        return self._eval_return(program.return_stmt)

    def _eval_memory(self, block: MemoryBlock) -> None:
        for entry in block.entries:
            self.memory[entry.key] = self._eval_value(entry.value)

    def _eval_step(self, step: Step) -> None:
        action = step.action
        if isinstance(action, ToolCall):
            inputs = {k: self._eval_value(v) for k, v in action.inputs.items()}
            result = self.tools.call(action.tool_name, inputs)
            self.memory[action.output_var] = result
        elif isinstance(action, Filter):
            source = self._resolve_var(action.input_var)
            result = self._eval_filter(action.condition, source)
            self.memory[action.output_var] = result
        else:
            raise ForgeRuntimeError(f"Unknown step action: {type(action)}")

    def _eval_filter(self, condition: Comparison, source: Any) -> Any:
        # List: filter items
        if isinstance(source, list):
            out = []
            for item in source:
                if self._eval_comparison(condition, item_context=item):
                    out.append(item)
            return out

        # Scalar / object: pass through if condition true, else null
        if self._eval_comparison(condition, item_context=None):
            return source
        return None

    def _eval_reason(self, reason: Reason) -> None:
        data = self._resolve_var(reason.input_var)
        text = self.llm.complete(reason.prompt, context=data)
        self.memory[reason.output_var] = text

    def _eval_verify(self, verify: Verify) -> None:
        ok = self._eval_comparison(verify.condition)
        if not ok:
            cond = verify.condition
            raise ForgeVerifyError(
                f"VERIFY failed: {self._fmt_comp(cond)} "
                f"(memory keys: {list(self.memory.keys())})"
            )

    def _eval_return(self, ret: Return) -> Any:
        return self._eval_value(ret.value)

    def _eval_value(self, node: ASTNode) -> Any:
        if isinstance(node, Literal):
            return node.value
        if isinstance(node, Variable):
            return self._resolve_var(node.name)
        if isinstance(node, ObjectLiteral):
            return {k: self._eval_value(v) for k, v in node.properties.items()}
        raise ForgeRuntimeError(f"Cannot evaluate value node: {type(node)}")

    def _resolve_var(self, name: str) -> Any:
        # name without $
        if name not in self.memory:
            raise ForgeRuntimeError(f"Undefined variable: ${name}")
        return self.memory[name]

    def _eval_comparison(
        self,
        cond: Comparison,
        item_context: Any = None,
    ) -> bool:
        left = self._eval_comp_side(cond.left, cond.left_kind, item_context)
        right = self._eval_value(cond.right) if isinstance(cond.right, ASTNode) else cond.right
        op = cond.operator

        # Equality ops work on any types
        if op == "==":
            return left == right
        if op == "!=":
            return left != right

        # Ordering ops — allow None-safe false
        if left is None or right is None:
            return False
        try:
            if op == ">":
                return left > right
            if op == "<":
                return left < right
            if op == ">=":
                return left >= right
            if op == "<=":
                return left <= right
        except TypeError as e:
            raise ForgeRuntimeError(
                f"Cannot compare {left!r} {op} {right!r}: {e}"
            ) from e

        raise ForgeRuntimeError(f"Unknown operator: {op}")

    def _eval_comp_side(self, left: Any, left_kind: str, item_context: Any) -> Any:
        if left_kind == "field":
            # Field access on list item, or fallback to memory
            if item_context is not None and isinstance(item_context, dict):
                if left in item_context:
                    return item_context[left]
            # Also allow field-as-var if no item context
            if isinstance(left, str) and left in self.memory:
                return self.memory[left]
            if item_context is not None and isinstance(item_context, dict):
                return item_context.get(left)
            raise ForgeRuntimeError(f"Field not found: {left}")
        if isinstance(left, ASTNode):
            return self._eval_value(left)
        return left

    def _fmt_comp(self, cond: Comparison) -> str:
        if cond.left_kind == "field":
            l = str(cond.left)
        elif isinstance(cond.left, Variable):
            l = f"${cond.left.name}"
        elif isinstance(cond.left, Literal):
            l = repr(cond.left.value)
        else:
            l = str(cond.left)
        if isinstance(cond.right, Variable):
            r = f"${cond.right.name}"
        elif isinstance(cond.right, Literal):
            r = repr(cond.right.value)
        else:
            r = str(cond.right)
        return f"{l} {cond.operator} {r}"


# =============================================================================
# PUBLIC API
# =============================================================================

def run_forge(
    source: str,
    tools: Optional[ToolRegistry] = None,
    llm: Optional[LLMClient] = None,
) -> Any:
    """Compile and execute Forge source (text syntax or JSON AST string)."""
    program = compile_auto(source)
    return Evaluator(tools=tools, llm=llm).run(program)


def run_program(
    program: Program,
    tools: Optional[ToolRegistry] = None,
    llm: Optional[LLMClient] = None,
) -> Any:
    return Evaluator(tools=tools, llm=llm).run(program)


# =============================================================================
# REPL
# =============================================================================

class ForgeREPL:
    def __init__(self, tools: Optional[ToolRegistry] = None, llm: Optional[LLMClient] = None):
        self.tools = tools or ToolRegistry()
        self.llm = llm or MockLLMClient()

    def run(self) -> None:
        print("Forge REPL v0.1 — type Forge program, end with blank line. :quit to exit.")
        print(f"Tools: {', '.join(self.tools.names())}")
        while True:
            try:
                print("\nforge> (paste program, blank line to run)")
                lines: list[str] = []
                while True:
                    line = input()
                    if line.strip() == ":quit":
                        print("bye")
                        return
                    if line.strip() == "" and lines:
                        break
                    lines.append(line)
                source = "\n".join(lines)
                result = run_forge(source, tools=self.tools, llm=self.llm)
                print("=>", json.dumps(result, default=str, indent=2))
            except EOFError:
                print("\nbye")
                return
            except Exception as e:
                print(f"Error: {type(e).__name__}: {e}")


# =============================================================================
# TESTS
# =============================================================================

def run_runtime_tests() -> None:
    print("=" * 60)
    print("Forge Runtime — Example Execution Tests")
    print("=" * 60)
    llm = MockLLMClient()
    tools = ToolRegistry()
    passed = 0
    failed = 0

    for name, source in EXAMPLES.items():
        try:
            result = run_forge(source, tools=tools, llm=llm)
            print(f"  PASS  {name}")
            print(f"        => {json.dumps(result, default=str)[:120]}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
            failed += 1

    # JSON AST direct mode
    try:
        from forge_core import compile_forge, _ast_to_dict, compile_forge_json
        prog = compile_forge(EXAMPLES["basic-calculator"])
        d = _ast_to_dict(prog)
        result = run_program(compile_forge_json(d), tools=tools, llm=llm)
        assert result["result"] == 22
        print("  PASS  json-ast-direct-mode (sum=22)")
        passed += 1
    except Exception as e:
        print(f"  FAIL  json-ast-direct-mode: {e}")
        failed += 1

    # VERIFY fail-fast
    try:
        bad = '''
AGENT "fail-verify"
MEMORY { a: 1 }
STEP add TOOL arithmetic_add INPUT { x: $a, y: $a } OUTPUT sum
VERIFY $sum < 0
RETURN { result: $sum }
'''
        run_forge(bad, tools=tools, llm=llm)
        print("  FAIL  verify-fail-fast (should have raised)")
        failed += 1
    except ForgeVerifyError:
        print("  PASS  verify-fail-fast")
        passed += 1
    except Exception as e:
        print(f"  FAIL  verify-fail-fast: unexpected {e}")
        failed += 1

    print("-" * 60)
    print(f"Results: {passed} passed, {failed} failed / {passed + failed} total")
    if failed:
        raise SystemExit(1)
    print("All runtime tests passed.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "repl":
        ForgeREPL().run()
    else:
        run_runtime_tests()
