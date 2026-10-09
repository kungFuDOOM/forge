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
    ArrayLiteral,
    ASTNode,
    Comparison,
    Filter,
    ForEach,
    ForgeError,
    If,
    Logical,
    Literal,
    MemoryBlock,
    ObjectLiteral,
    Program,
    Reason,
    Return,
    RunProgram,
    Try,
    Step,
    ToolCall,
    Variable,
    Verify,
    Yield,
    check_program,
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


class ForgeCheckError(ForgeRuntimeError):
    """Static pre-run check failed (nothing was executed)."""


class ForgeBudgetError(ForgeRuntimeError):
    """Step or time budget exhausted. Never caught by TRY or RETRY."""


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
        model: Optional[str] = None,  # None = the provider's default for the key found
    ):
        self.api_key, self.base_url, self.model = _resolve_openai_compat(
            api_key=api_key, base_url=base_url, model=model
        )
        if not self.api_key:
            raise ForgeRuntimeError(
                f"No API key. Set one of {', '.join(API_KEY_ENVS)} "
                "— or use OllamaLLMClient (free/local)."
            )

    def complete(self, prompt: str, context: Any = None) -> str:
        user_content = prompt
        if context is not None:
            user_content = f"{prompt}\n\nContext data:\n{json.dumps(context, default=str, indent=2)}"
        text, _usage = chat_completion(
            [
                {"role": "system", "content": "You are a concise reasoning assistant for an agent runtime."},
                {"role": "user", "content": user_content},
            ],
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model,
            temperature=0.2,
        )
        return text


def chat_completion(
    messages: list,
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    stop: Optional[list] = None,
    timeout: float = 120,
) -> tuple[str, dict]:
    """
    OpenAI-compatible /chat/completions over stdlib HTTP (no openai package).
    Works with Groq, Gemini, OpenRouter, xAI, OpenAI, DeepSeek. Returns (text, usage).
    """
    message, usage = chat_message(
        messages, api_key=api_key, base_url=base_url, model=model, temperature=temperature,
        max_tokens=max_tokens, stop=stop, timeout=timeout,
    )
    return message.get("content") or "", usage


def chat_message(
    messages: list,
    *,
    tools: Optional[list] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    stop: Optional[list] = None,
    timeout: float = 120,
) -> tuple[dict, dict]:
    """Like chat_completion, but supports tool calling and returns the whole
    assistant message (content + tool_calls) and usage."""
    import urllib.error
    import urllib.request

    if not api_key:
        api_key, base_url, model = _resolve_openai_compat(base_url=base_url, model=model)
    if not api_key:
        raise ForgeRuntimeError(
            f"No API key. Set one of {', '.join(API_KEY_ENVS)} — or use Ollama (free/local)."
        )
    url = (base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
    body: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
    if max_tokens:
        body["max_tokens"] = max_tokens
    if stop:
        body["stop"] = stop
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "ForgeAgent/0.4",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read(2000).decode("utf-8", errors="replace") if e.fp else ""
        raise ForgeRuntimeError(f"LLM API error {e.code} from {url}: {detail}") from e
    except urllib.error.URLError as e:
        raise ForgeRuntimeError(f"LLM API not reachable at {url}: {e}") from e

    choices = data.get("choices") or [{}]
    message = choices[0].get("message") or {}
    usage = data.get("usage") or {}
    return message, {
        "backend": "openai",
        "model": model,
        "base_url": base_url,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


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


# Env vars that select an OpenAI-compatible provider, in priority order
API_KEY_ENVS = (
    "GROQ_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY",
    "XAI_API_KEY", "GROK_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY",
)


def has_api_key() -> bool:
    return any(os.environ.get(k) for k in API_KEY_ENVS)


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
    xai_key = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
    if xai_key:
        # xAI Grok; grok-latest always points at the newest Grok model
        return (
            xai_key,
            base_url or "https://api.x.ai/v1",
            model or "grok-latest",
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

MAX_FETCH_BYTES = 10_000_000  # largest response http_get / http_post will read
MAX_FILE_BYTES = 10_000_000  # largest file read_file / write_file will handle

# Cloud instance-metadata endpoints hand out credentials; never reachable from a program.
_METADATA_HOSTS = {"metadata.google.internal", "metadata.goog", "metadata", "instance-data"}
_METADATA_IPS = {"169.254.169.254", "169.254.170.2", "fd00:ec2::254", "100.100.100.200"}


def check_url(url: str) -> None:
    """
    Refuse URLs a program must not reach: non-http(s) schemes, and cloud
    metadata / link-local addresses (the classic SSRF credential theft).
    With FORGE_HTTP_BLOCK_PRIVATE=1, also loopback and private networks.
    """
    import ipaddress
    import socket
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ForgeRuntimeError(f"Only http:// and https:// URLs are allowed: {url}")
    host = parts.hostname.lower().rstrip(".")
    if host in _METADATA_HOSTS:
        raise ForgeRuntimeError(f"Blocked: {host} is a cloud metadata endpoint")
    strict = os.environ.get("FORGE_HTTP_BLOCK_PRIVATE") == "1"
    try:
        addrs = {ipaddress.ip_address(host)}
    except ValueError:
        try:
            port = parts.port or (443 if parts.scheme == "https" else 80)
            addrs = {ipaddress.ip_address(info[4][0].split("%")[0])
                     for info in socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)}
        except (socket.gaierror, UnicodeError, ValueError):
            return  # can't resolve locally (e.g. behind a proxy); let the request report it
    for addr in addrs:
        mapped = getattr(addr, "ipv4_mapped", None)
        for a in (addr, mapped) if mapped else (addr,):
            if str(a) in _METADATA_IPS or a.is_link_local:
                raise ForgeRuntimeError(f"Blocked: {host} resolves to a link-local/metadata address ({a})")
            if strict and (a.is_private or a.is_loopback or a.is_reserved or a.is_unspecified or a.is_multicast):
                raise ForgeRuntimeError(
                    f"Blocked: {host} resolves to a private address ({a}) and FORGE_HTTP_BLOCK_PRIVATE=1"
                )


def _safe_urlopen(req: Any, timeout: float = 20) -> Any:
    """urlopen that re-checks every redirect target with check_url."""
    import urllib.request

    class _Redirects(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            check_url(newurl)
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    check_url(req.full_url)
    return urllib.request.build_opener(_Redirects).open(req, timeout=timeout)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolFn] = {}
        self._docs: dict[str, str] = {}
        self._register_builtins()

    def register(self, name: str, fn: ToolFn, doc: Optional[str] = None) -> None:
        self._tools[name] = fn
        if doc is None:
            doc = (fn.__doc__ or "").strip().splitlines()[0] if fn.__doc__ else ""
        self._docs[name] = doc

    def docs(self) -> dict[str, str]:
        return {n: self._docs.get(n, "") for n in self.names()}

    def load_file(self, path: str) -> list[str]:
        """
        Register tools from a Python file. Uses its TOOLS dict if present,
        otherwise every public top-level function defined in the file.
        Each tool takes one dict of inputs; its docstring's first line is its doc.
        """
        import importlib.util
        from pathlib import Path

        file = Path(path).expanduser().resolve()
        if not file.is_file():
            raise ForgeRuntimeError(f"Tools file not found: {path}")
        spec = importlib.util.spec_from_file_location(f"forge_user_tools_{file.stem}", file)
        module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        tools = getattr(module, "TOOLS", None)
        if tools is None:
            tools = {
                name: fn
                for name, fn in vars(module).items()
                if callable(fn) and not name.startswith("_")
                and getattr(fn, "__module__", None) == module.__name__
            }
        for name, fn in tools.items():
            self.register(name, fn)
        return sorted(tools)

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
            if os.environ.get("BRAVE_SEARCH_API_KEY"):
                return _brave_search(q, n)
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
            max_bytes = min(int(inputs.get("max_bytes", 100_000) or 100_000), MAX_FETCH_BYTES)
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "ForgeAgent/0.4"},
                method="GET",
            )
            try:
                with _safe_urlopen(req, timeout=20) as resp:
                    raw = resp.read(max_bytes + 1)
                    truncated = len(raw) > max_bytes
                    body = raw[:max_bytes].decode("utf-8", errors="replace")
                    ctype = resp.headers.get("Content-Type", "")
                    out = {
                        "status": getattr(resp, "status", 200),
                        "url": url,
                        "content_type": ctype,
                        "body": body,
                        "truncated": truncated,
                        "chars": len(body),
                    }
                    # Readable text of HTML pages: usually ~5-20x fewer tokens than body
                    out["text"] = html_to_text(body) if "html" in ctype.lower() else body
                    return out
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
            except ForgeRuntimeError:
                raise
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
            max_bytes = min(int(inputs.get("max_bytes", 200_000) or 200_000), MAX_FILE_BYTES)
            with p.open("rb") as fh:  # never load more than needed
                data = fh.read(max_bytes + 1)
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
            if len(text) > MAX_FILE_BYTES:
                raise ForgeRuntimeError(f"write_file body is over the {MAX_FILE_BYTES:,}-character limit")
            p.write_text(text, encoding="utf-8")
            return {"path": str(p.relative_to(cwd)), "chars": len(text), "ok": True}

        _register_data_tools(self)
        self.register("arithmetic_add", arithmetic_add, "INPUT { x, y } → number x+y")
        self.register("web_search", web_search, "INPUT { q, n } → list of { title, url, relevance, snippet } (real if BRAVE_SEARCH_API_KEY set, else offline mock)")
        self.register("get_value", get_value, "INPUT { } → 42 (demo)")
        self.register("sales_data", sales_data, "INPUT { region, period } → list of { deal, amount, region, period } (demo)")
        self.register("http_get", http_get, "INPUT { url, max_bytes? } → { status, url, content_type, body, text, truncated, chars } (real HTTP; text = readable page text)")
        self.register("read_file", read_file, "INPUT { path, max_bytes? } → { path, body, chars, truncated } (under cwd only)")
        self.register("write_file", write_file, "INPUT { path, body } → { path, chars, ok } (under cwd only)")


def html_to_text(html: str) -> str:
    """Visible text of an HTML document (drops scripts, styles, markup)."""
    from html.parser import HTMLParser

    skip_tags = {"script", "style", "noscript", "svg", "template", "head"}
    block_tags = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
                  "section", "article", "header", "footer", "ul", "ol", "table", "title"}

    class _Text(HTMLParser):
        def __init__(self) -> None:
            super().__init__(convert_charrefs=True)
            self.parts: list[str] = []
            self.skip = 0
            self.title = ""
            self._in_title = False

        def handle_starttag(self, tag, attrs):
            if tag == "title":
                self._in_title = True
            if tag in skip_tags:
                self.skip += 1
            elif tag in block_tags:
                self.parts.append("\n")

        def handle_endtag(self, tag):
            if tag == "title":
                self._in_title = False
            if tag in skip_tags and self.skip:
                self.skip -= 1
            elif tag in block_tags:
                self.parts.append("\n")

        def handle_data(self, data):
            if self._in_title:
                self.title += data
            elif not self.skip:
                self.parts.append(data)

    parser = _Text()
    parser.feed(html)
    parser.close()
    lines = (" ".join(line.split()) for line in "".join(parser.parts).splitlines())
    text = "\n".join(line for line in lines if line)
    title = " ".join(parser.title.split())
    return f"{title}\n{text}" if title and not text.startswith(title) else text


def _brave_search(q: str, n: int) -> list:
    """Real web search via the Brave Search API (BRAVE_SEARCH_API_KEY)."""
    import urllib.error
    import urllib.parse
    import urllib.request

    base = os.environ.get("FORGE_BRAVE_URL", "https://api.search.brave.com/res/v1/web/search")
    url = f"{base}?{urllib.parse.urlencode({'q': q, 'count': max(1, min(n, 20))})}"
    req = urllib.request.Request(url, headers={
        "Accept": "application/json",
        "X-Subscription-Token": os.environ["BRAVE_SEARCH_API_KEY"],
        "User-Agent": "ForgeAgent/0.4",
    })
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise ForgeRuntimeError(f"web_search failed: {e}") from e
    hits = ((data.get("web") or {}).get("results")) or []
    return [
        {
            "title": h.get("title", ""),
            "url": h.get("url", ""),
            "relevance": round(1.0 - i * 0.05, 2),  # rank-based, 1.0 = top hit
            "snippet": html_to_text(h.get("description", "")),
        }
        for i, h in enumerate(hits[:n])
    ]


def _register_data_tools(reg: "ToolRegistry") -> None:
    """General-purpose tools: Forge has no expressions, so data work is tools."""
    import re as _re

    def need(inputs: dict, key: str) -> Any:
        if key not in inputs:
            raise ForgeRuntimeError(f"missing input: {key}")
        return inputs[key]

    def as_list(value: Any, tool: str) -> list:
        if not isinstance(value, list):
            raise ForgeRuntimeError(f"{tool}: list must be a list, got {type(value).__name__}")
        return value

    def field_of(item: Any, path: str) -> Any:
        for part in str(path).split("."):
            if isinstance(item, dict):
                item = item.get(part)
            elif isinstance(item, list) and part.isdigit() and int(part) < len(item):
                item = item[int(part)]
            else:
                return None
        return item

    def count(i: dict) -> Any:
        return len(need(i, "list"))

    def pick(i: dict) -> Any:
        return [field_of(x, need(i, "field")) for x in as_list(need(i, "list"), "pick")]

    def sort(i: dict) -> Any:
        items = as_list(need(i, "list"), "sort")
        by = i.get("by")
        key = (lambda x: field_of(x, by)) if by else (lambda x: x)
        present = [x for x in items if key(x) is not None]
        missing = [x for x in items if key(x) is None]  # always last
        try:
            out = sorted(present, key=key, reverse=bool(i.get("desc"))) + missing
        except TypeError as e:
            raise ForgeRuntimeError(f"sort: values are not comparable: {e}") from e
        limit = i.get("limit")
        return out[: int(limit)] if limit else out

    def join(i: dict) -> Any:
        return str(i.get("sep", "\n")).join(
            x if isinstance(x, str) else json.dumps(x, default=str)
            for x in as_list(need(i, "list"), "join")
        )

    def fmt(i: dict) -> Any:
        # Only {name} placeholders ({{ and }} for literal braces). Unlike
        # str.format there is no {x.attr} / {x[0]} traversal of objects.
        values = {k: v for k, v in i.items() if k != "template"}

        def sub(m: "_re.Match") -> str:
            if m.group(0) in ("{{", "}}"):
                return m.group(0)[0]
            key = m.group(1)
            if key not in values:
                raise ForgeRuntimeError(f"format: template needs '{key}' (give it as an INPUT key)")
            v = values[key]
            return v if isinstance(v, str) else json.dumps(v, default=str)

        return _re.sub(r"\{\{|\}\}|\{([A-Za-z_][A-Za-z0-9_]*)\}", sub, str(need(i, "template")))

    def calc(i: dict) -> Any:
        op, x, y = need(i, "op"), need(i, "x"), i.get("y", 0)
        ops = {
            "add": lambda: x + y, "sub": lambda: x - y, "mul": lambda: x * y,
            "div": lambda: x / y, "min": lambda: min(x, y), "max": lambda: max(x, y),
            "round": lambda: round(x, int(y)),
        }
        if op not in ops:
            raise ForgeRuntimeError(f"calc: op must be one of {', '.join(ops)}")
        try:
            return ops[op]()
        except (TypeError, ZeroDivisionError) as e:
            raise ForgeRuntimeError(f"calc {op}: {e}") from e

    def total(i: dict) -> Any:
        items = as_list(need(i, "list"), "sum")
        vals = [field_of(x, i["field"]) for x in items] if i.get("field") else items
        try:
            return sum(v for v in vals if v is not None)
        except TypeError as e:
            raise ForgeRuntimeError(f"sum: non-numeric values: {e}") from e

    def regex_find(i: dict) -> Any:
        pattern, text = str(need(i, "pattern")), str(need(i, "text"))
        if len(pattern) > 1000 or len(text) > 1_000_000:
            raise ForgeRuntimeError("regex_find: pattern max 1000 chars, text max 1,000,000 chars")
        try:
            return _re.findall(pattern, text)[: int(i.get("limit", 100))]
        except _re.error as e:
            raise ForgeRuntimeError(f"regex_find: bad pattern: {e}") from e

    def json_parse(i: dict) -> Any:
        try:
            return json.loads(need(i, "text"))
        except (TypeError, ValueError) as e:
            raise ForgeRuntimeError(f"json_parse: {e}") from e

    def extract_text(i: dict) -> Any:
        text = html_to_text(str(need(i, "html")))
        limit = i.get("max_chars")
        return text[: int(limit)] if limit else text

    def now(i: dict) -> Any:
        from datetime import datetime, timezone

        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    def http_post(i: dict) -> Any:
        import urllib.error
        import urllib.request

        url = need(i, "url")
        if not isinstance(url, str) or not url.startswith(("http://", "https://")):
            raise ForgeRuntimeError("http_post only allows http:// or https:// URLs")
        check_url(url)
        headers = {"User-Agent": "ForgeAgent/0.4", **(i.get("headers") or {})}
        if "json" in i:
            data = json.dumps(i["json"], default=str).encode("utf-8")
            headers.setdefault("Content-Type", "application/json")
        else:
            data = str(i.get("body", "")).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with _safe_urlopen(req, timeout=20) as resp:
                status, ctype, raw = resp.status, resp.headers.get("Content-Type", ""), resp.read(100_000)
        except urllib.error.HTTPError as e:
            status, ctype, raw = e.code, (e.headers.get("Content-Type", "") if e.headers else ""), (e.read(100_000) if e.fp else b"")
        except ForgeRuntimeError:
            raise
        except Exception as e:
            raise ForgeRuntimeError(f"http_post failed: {e}") from e
        body = raw.decode("utf-8", errors="replace")
        out: dict = {"status": status, "url": url, "content_type": ctype, "body": body}
        if "json" in ctype:
            try:
                out["json"] = json.loads(body)
            except ValueError:
                pass
        return out

    for name, fn, doc in (
        ("count", count, "INPUT { list } → number of items (or characters of a string)"),
        ("pick", pick, "INPUT { list, field } → list of that field from each item (field may be a.b)"),
        ("sort", sort, "INPUT { list, by?, desc?, limit? } → sorted list (by = field name)"),
        ("sum", total, "INPUT { list, field? } → numeric total"),
        ("join", join, "INPUT { list, sep? } → one string (default sep newline)"),
        ("format", fmt, "INPUT { template, ...values } → string; \"Hi {name}\" with name: $n"),
        ("calc", calc, "INPUT { op, x, y } → number; op = add|sub|mul|div|min|max|round"),
        ("regex_find", regex_find, "INPUT { text, pattern, limit? } → list of matches"),
        ("json_parse", json_parse, "INPUT { text } → parsed JSON value"),
        ("extract_text", extract_text, "INPUT { html, max_chars? } → readable text"),
        ("now", now, "INPUT { } → current UTC time, ISO 8601"),
        ("http_post", http_post, "INPUT { url, json? | body?, headers? } → { status, body, json? } (real HTTP)"),
    ):
        reg.register(name, fn, doc)


# =============================================================================
# LANGUAGE SPEC (for AIs: paste into context, or read via `forge spec`)
# =============================================================================

SPEC_HEADER = """
Forge: agent programs. One program runs a whole multi-step job in one go.
Emit only Forge. Keywords UPPERCASE. Lines run top to bottom.

AGENT "name"                                  required, first
MEMORY { k: "v", n: 3, xs: [1, 2], o: { a: 1 } }  optional inputs
STEP s TOOL tool INPUT { k: $v } OUTPUT x     call a tool, result in $x
STEP s FILTER field > 0.8 ON $list OUTPUT y   keep list items whose field matches
REASON "prompt" ON $x OUTPUT z                LLM call on $x, text in $z
VERIFY $x.status == 200                       stop the run if false
FOR EACH item IN $list OUTPUT out ... END     loop; out = list of YIELDed values
  YIELD { k: $item.f }                        (no YIELD: out = last OUTPUT per item)
IF cond ... ELSE ... END                      branch
PARALLEL 8 FOR EACH ... END                   same loop, up to 8 items at once
TRY ... ON ERROR ... END                      on failure run the handler; $error = message
STEP s TOOL t INPUT {..} OUTPUT x RETRY 2     retry a failing TOOL / REASON / RUN step
STEP s RUN "other.forge" INPUT { k: $v } OUTPUT x   run another program; INPUT sets its MEMORY
RETURN { k: $x }                              required, last

Values: "text" 12 -3.5 true false null [a, b] { k: v } $var $var.field $var.0.field
Conditions: == != > < >= <= CONTAINS, joined with AND / OR
Loop bodies are scoped: only the FOR's OUTPUT is visible after END.
VERIFY early to stop before costly steps. Pass REASON only the field it needs
($page.text, not $page). Errors say what is defined/available; fix and resend.
""".strip()


def language_spec(tools: Optional["ToolRegistry"] = None) -> str:
    """Compact language reference including the available tools."""
    docs = (tools or ToolRegistry()).docs()
    width = max((len(n) for n in docs), default=0)
    lines = [SPEC_HEADER, "", "Tools:"]
    lines += [f"  {name.ljust(width)}  {doc}".rstrip() for name, doc in docs.items()]
    return "\n".join(lines)


# =============================================================================
# EVALUATOR
# =============================================================================

DEFAULT_MAX_STEPS = 10_000
REASON_CONTEXT_CHARS = int(os.environ.get("FORGE_REASON_MAX_CHARS", "40000"))  # ~10k tokens


MAX_RUN_DEPTH = 8  # nested RUN "other.forge" calls


class _Budget:
    """Step + time budget shared by a run, its parallel iterations and RUN children."""

    def __init__(self, max_steps: Optional[int], max_seconds: Optional[float]) -> None:
        import threading
        import time

        self.max_steps = max_steps
        self.max_seconds = max_seconds
        self.deadline = time.monotonic() + max_seconds if max_seconds else None
        self.steps = 0
        self._lock = threading.Lock()

    def tick(self, where: str) -> None:
        with self._lock:
            self.steps += 1
            steps = self.steps
        if self.max_steps is not None and steps > self.max_steps:
            raise ForgeBudgetError(f"Step budget exceeded ({self.max_steps} steps) at {where}")
        self.check_time(where)

    def check_time(self, where: str) -> None:
        import time

        if self.deadline is not None and time.monotonic() > self.deadline:
            raise ForgeBudgetError(f"Time budget exceeded ({self.max_seconds}s) at {where}")


class Evaluator:
    def __init__(
        self,
        tools: Optional[ToolRegistry] = None,
        llm: Optional[LLMClient] = None,
        max_steps: Optional[int] = DEFAULT_MAX_STEPS,
        max_seconds: Optional[float] = None,
        base_dir: Optional[str] = None,
    ):
        self.tools = tools or ToolRegistry()
        self.llm = llm or MockLLMClient()
        self.memory: dict[str, Any] = {}
        self.agent_name: str = ""
        self.max_steps = max_steps
        self.max_seconds = max_seconds
        # Directory RUN paths resolve against (the program file's folder); default cwd
        self.base_dir = base_dir
        self._budget = _Budget(max_steps, max_seconds)
        self._collector: Optional[list] = None  # YIELD target inside a loop body
        self._depth = 0

    @property
    def steps_run(self) -> int:
        return self._budget.steps

    def run(self, program: Program) -> Any:
        self._budget = _Budget(self.max_steps, self.max_seconds)
        return self._run(program)

    def _run(self, program: Program, overrides: Optional[dict] = None) -> Any:
        self.memory = {}
        self.agent_name = program.agent.name
        self._collector = None

        # 0. Static check — fail before any tool or LLM call is spent
        problems = check_program(program, self.tools.names())
        if problems:
            raise ForgeCheckError("Check failed:\n  " + "\n  ".join(problems))

        # 1. AGENT — identity only
        # 2. MEMORY (a RUN caller's INPUT overrides it)
        if program.memory:
            self._eval_memory(program.memory)
        self.memory.update(overrides or {})

        # 3. STEPS (incl. REASON / VERIFY / FOR / IF / TRY in program order)
        self._exec_block(program.steps)

        # 4. REASON
        if program.reason:
            self._eval_reason(program.reason)

        # 5. VERIFY
        if program.verify:
            self._eval_verify(program.verify)

        # 6. RETURN
        return self._eval_return(program.return_stmt)

    def _child(self, memory: dict) -> "Evaluator":
        """Evaluator for a loop iteration: own memory, shared tools/LLM/budget."""
        import copy

        child = copy.copy(self)
        child.memory = memory
        child._collector = []
        return child

    def _eval_memory(self, block: MemoryBlock) -> None:
        for entry in block.entries:
            self.memory[entry.key] = self._eval_value(entry.value)

    def _exec_block(self, steps: list) -> None:
        for step in steps:
            self._eval_step(step)

    def _with_retries(self, retries: int, where: str, fn: Callable[[], Any]) -> Any:
        """Run fn, retrying failures `retries` times with short backoff."""
        import time

        for attempt in range(retries + 1):
            try:
                return fn()
            except ForgeBudgetError:
                raise
            except Exception:
                if attempt == retries:
                    raise
                time.sleep(min(0.2 * 2 ** attempt, 2.0))
                self._budget.tick(f"{where} (retry {attempt + 1})")

    def _eval_step(self, step: Step) -> None:
        where = f"STEP {step.step_name}"
        self._budget.tick(where)
        action = step.action
        if isinstance(action, ForEach):
            self._eval_for(action)
        elif isinstance(action, If):
            branch = action.then if self._eval_comparison(action.condition) else action.else_
            self._exec_block(branch)
        elif isinstance(action, Yield):
            self._collector.append(self._eval_value(action.value))
        elif isinstance(action, Try):
            try:
                self._exec_block(action.body)
            except ForgeBudgetError:
                raise
            except Exception as e:
                self.memory[action.error_var] = f"{type(e).__name__}: {e}"
                self._exec_block(action.handler)
        elif isinstance(action, ToolCall):
            inputs = {k: self._eval_value(v) for k, v in action.inputs.items()}
            self.memory[action.output_var] = self._with_retries(
                action.retries, where, lambda: self.tools.call(action.tool_name, inputs)
            )
        elif isinstance(action, RunProgram):
            inputs = {k: self._eval_value(v) for k, v in action.inputs.items()}
            self.memory[action.output_var] = self._with_retries(
                action.retries, where, lambda: self._eval_run(action, inputs)
            )
        elif isinstance(action, Filter):
            source = self._resolve_var(action.input_var)
            result = self._eval_filter(action.condition, source)
            self.memory[action.output_var] = result
        elif isinstance(action, Reason):
            self._with_retries(action.retries, where, lambda: self._eval_reason(action))
        elif isinstance(action, Verify):
            self._eval_verify(action)
        else:
            raise ForgeRuntimeError(f"Unknown step action: {type(action)}")

    def _eval_for(self, loop: ForEach) -> None:
        items = self._eval_value(loop.source)
        if isinstance(items, dict):
            items = [{"key": k, "value": v} for k, v in items.items()]
        if not isinstance(items, list):
            raise ForgeRuntimeError(
                f"FOR EACH {loop.var} needs a list, got {type(items).__name__}"
            )
        # Without YIELD, collect the body's last output each iteration
        implicit = None if _has_yield(loop.body) else _last_output(loop.body)

        def iteration(item: Any) -> list:
            child = self._child({**self.memory, loop.var: item})  # body is its own scope
            child._exec_block(loop.body)
            return [child.memory.get(implicit)] if implicit else child._collector

        if loop.parallel and len(items) > 1 and _threads_available():
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=min(loop.parallel, len(items))) as pool:
                futures = [pool.submit(iteration, item) for item in items]
            chunks, first_error = [], None
            for f in futures:  # keep input order; report the earliest failure
                try:
                    chunks.append(f.result())
                except Exception as e:
                    first_error = first_error or e
            if first_error:
                raise first_error
        else:
            chunks = [iteration(item) for item in items]
        if loop.output_var:
            self.memory[loop.output_var] = [x for chunk in chunks for x in chunk]

    def _eval_run(self, action: RunProgram, inputs: dict) -> Any:
        """RUN another .forge program: its MEMORY is overridden by INPUT."""
        from pathlib import Path

        if self._depth >= MAX_RUN_DEPTH:
            raise ForgeRuntimeError(f"RUN nested deeper than {MAX_RUN_DEPTH} programs")
        base = Path(self.base_dir or Path.cwd()).resolve()
        path = (base / action.path).resolve()
        allowed = [Path.cwd().resolve(), base]
        if not any(path == root or root in path.parents for root in allowed):
            raise ForgeRuntimeError(
                f"RUN path must be inside {allowed[0]} or the calling program's folder: {action.path}"
            )
        if path.suffix not in (".forge", ".json"):
            raise ForgeRuntimeError(f"RUN needs a .forge or .json program file: {action.path}")
        if not path.is_file():
            raise ForgeRuntimeError(f"RUN program not found: {action.path}")
        program = compile_auto(path.read_text(encoding="utf-8"))
        keys = [e.key for e in program.memory.entries] if program.memory else []
        unknown = sorted(set(inputs) - set(keys))
        if unknown:
            raise ForgeRuntimeError(
                f"{action.path} has no MEMORY key {', '.join(unknown)} "
                f"(its MEMORY keys: {', '.join(keys) or 'none'})"
            )
        child = Evaluator(tools=self.tools, llm=self.llm, base_dir=str(path.parent))
        child._budget = self._budget  # shared step/time budget
        child._depth = self._depth + 1
        return child._run(program, overrides=inputs)

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
        # Cap what one REASON can send so a huge page can't blow the budget
        raw = data if isinstance(data, str) else json.dumps(data, default=str)
        if len(raw) > REASON_CONTEXT_CHARS:
            data = raw[:REASON_CONTEXT_CHARS] + f"\n…[truncated {len(raw) - REASON_CONTEXT_CHARS} chars]"
        text = self.llm.complete(reason.prompt, context=data)
        self.memory[reason.output_var] = text

    def _eval_verify(self, verify: Verify) -> None:
        ok = self._eval_comparison(verify.condition)
        if not ok:
            cond = verify.condition
            got = ""
            if isinstance(cond, Comparison) and cond.left_kind != "field":
                try:
                    got = f" — left side was {self._preview(self._eval_comp_side(cond.left, cond.left_kind, None))}"
                except ForgeRuntimeError:
                    pass
            raise ForgeVerifyError(
                f"VERIFY failed: {self._fmt_comp(cond)}{got} "
                f"(memory keys: {list(self.memory.keys())})"
            )

    @staticmethod
    def _preview(value: Any, limit: int = 80) -> str:
        text = json.dumps(value, default=str)
        return text if len(text) <= limit else text[:limit] + "…"

    def _eval_return(self, ret: Return) -> Any:
        return self._eval_value(ret.value)

    def _eval_value(self, node: ASTNode) -> Any:
        if isinstance(node, Literal):
            return node.value
        if isinstance(node, Variable):
            return self._resolve_var(node.name)
        if isinstance(node, ObjectLiteral):
            return {k: self._eval_value(v) for k, v in node.properties.items()}
        if isinstance(node, ArrayLiteral):
            return [self._eval_value(v) for v in node.items]
        raise ForgeRuntimeError(f"Cannot evaluate value node: {type(node)}")

    def _resolve_var(self, name: str) -> Any:
        # name without $, optionally a dotted path: doc.body, hits.0.title
        base, *path = name.split(".")
        if base not in self.memory:
            raise ForgeRuntimeError(f"Undefined variable: ${base}")
        value = self.memory[base]
        walked = base
        for part in path:
            if isinstance(value, dict):
                if part not in value:
                    raise ForgeRuntimeError(
                        f"${walked} has no field {part!r} "
                        f"(fields: {', '.join(map(str, value.keys())) or 'none'})"
                    )
                value = value[part]
            elif isinstance(value, list) and part.isdigit():
                idx = int(part)
                if idx >= len(value):
                    raise ForgeRuntimeError(
                        f"${walked} index {idx} out of range (length {len(value)})"
                    )
                value = value[idx]
            else:
                raise ForgeRuntimeError(
                    f"Cannot read {part!r} from ${walked} ({type(value).__name__})"
                )
            walked += "." + part
        return value

    def _eval_comparison(
        self,
        cond: Any,
        item_context: Any = None,
    ) -> bool:
        if isinstance(cond, Logical):
            first = self._eval_comparison(cond.left, item_context)
            if cond.operator == "AND":
                return first and self._eval_comparison(cond.right, item_context)
            return first or self._eval_comparison(cond.right, item_context)
        left = self._eval_comp_side(cond.left, cond.left_kind, item_context)
        right = self._eval_value(cond.right) if isinstance(cond.right, ASTNode) else cond.right
        op = cond.operator

        # Equality ops work on any types
        if op == "==":
            return left == right
        if op == "!=":
            return left != right
        if op == "CONTAINS":
            # substring (case-insensitive), list membership, or dict key
            if isinstance(left, str) and isinstance(right, str):
                return right.lower() in left.lower()
            if isinstance(left, (list, dict)):
                return right in left
            return False

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

    def _fmt_comp(self, cond: Any) -> str:
        if isinstance(cond, Logical):
            return f"{self._fmt_comp(cond.left)} {cond.operator} {self._fmt_comp(cond.right)}"
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


def _threads_available() -> bool:
    """False in WebAssembly builds (the browser playground), where PARALLEL runs sequentially."""
    import sys

    return sys.platform not in ("emscripten", "wasi")


def _has_yield(steps: list) -> bool:
    """YIELD anywhere in this loop body (not counting nested loops)."""
    for step in steps:
        a = step.action
        if isinstance(a, Yield):
            return True
        if isinstance(a, If) and (_has_yield(a.then) or _has_yield(a.else_)):
            return True
    return False


def _last_output(steps: list) -> Optional[str]:
    for step in reversed(steps):
        name = getattr(step.action, "output_var", None)
        if name:
            return name
    return None


# =============================================================================
# PUBLIC API
# =============================================================================

def run_forge(
    source: str,
    tools: Optional[ToolRegistry] = None,
    llm: Optional[LLMClient] = None,
    **limits: Any,
) -> Any:
    """Compile and execute Forge source (text syntax or JSON AST string).
    limits: max_steps, max_seconds, base_dir (see Evaluator)."""
    program = compile_auto(source)
    return Evaluator(tools=tools, llm=llm, **limits).run(program)


def run_program(
    program: Program,
    tools: Optional[ToolRegistry] = None,
    llm: Optional[LLMClient] = None,
    **limits: Any,
) -> Any:
    return Evaluator(tools=tools, llm=llm, **limits).run(program)


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
