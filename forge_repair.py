"""
Forge LLM output repair
=======================
AIs generate Forge more reliably when we normalize near-misses before parse.
This is intentional: the language is AI-native — the pipeline includes
generation → repair → validate → run.
"""

from __future__ import annotations

import re
from typing import Optional


KEYWORDS = (
    "AGENT", "MEMORY", "STEP", "TOOL", "FILTER", "INPUT", "OUTPUT",
    "ON", "REASON", "VERIFY", "RETURN",
    "FOR", "EACH", "IN", "IF", "ELSE", "END", "YIELD", "AND", "OR", "CONTAINS",
)
_KEYWORD_SET = set(KEYWORDS)
# Words right after these are names, never keywords (STEP end, OUTPUT input)
_NAME_AFTER = {"STEP", "OUTPUT", "TOOL", "EACH", "FOR"}
_WORD_RE = re.compile(r"(?<![$.\w])[A-Za-z_]\w*")


def _uppercase_keywords(code: str) -> str:
    """Uppercase keywords in a code segment, leaving $vars, field paths,
    object keys (`in: 1`) and name positions (`OUTPUT end`) alone."""
    out: list[str] = []
    last = 0
    prev = ""
    for m in _WORD_RE.finditer(code):
        word = m.group(0)
        up = word.upper()
        is_key = re.match(r"\s*:", code[m.end():]) is not None
        name_slot = prev in _NAME_AFTER and not (prev == "FOR" and up == "EACH")
        if up in _KEYWORD_SET and word != up and not name_slot and not is_key:
            out.append(code[last : m.start()] + up)
            last = m.end()
            word = up
        prev = word.upper() if word.upper() in _KEYWORD_SET else word
    out.append(code[last:])
    return "".join(out)

_STRING_RE = re.compile(r'"(?:[^"\\\n]|\\.)*"|\'(?:[^\'\\\n]|\\.)*\'')


def _sub_code(pattern: str, repl, s: str, flags: int = 0) -> str:
    """re.sub applied only outside string literals, so prompts stay verbatim."""
    out: list[str] = []
    last = 0
    for m in _STRING_RE.finditer(s):
        out.append(re.sub(pattern, repl, s[last : m.start()], flags=flags))
        out.append(m.group(0))
        last = m.end()
    out.append(re.sub(pattern, repl, s[last:], flags=flags))
    return "".join(out)


def _mask_strings(s: str) -> str:
    """Same-length copy with string-literal contents blanked, for searching code."""
    return _STRING_RE.sub(lambda m: m.group(0)[0] + " " * (len(m.group(0)) - 2) + m.group(0)[-1], s)


def extract_forge(text: str, prefer_json: bool = False) -> str:
    """Pull Forge source or JSON AST out of messy LLM output."""
    raw = text.strip()
    if not raw:
        return raw

    fence = re.search(r"```(?:forge|json|javascript|text)?\s*([\s\S]*?)```", raw, re.I)
    if fence:
        raw = fence.group(1).strip()

    if prefer_json or raw.lstrip().startswith("{"):
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end > start:
            return raw[start : end + 1]

    # Include MEMORY that appears before AGENT (common LLM slip)
    m = re.search(r"((?:\bMEMORY\b[\s\S]*?)?\bAGENT\b[\s\S]*)", raw, re.I)
    if m:
        return m.group(1).strip()

    return raw


def repair_forge(source: str) -> str:
    """
    Deterministic repairs for common LLM slips.
    Does not invent program logic — only normalizes surface form.
    """
    s = source.strip()
    if not s:
        return s

    # JSON AST path — leave structure alone
    if s.lstrip().startswith("{"):
        return s

    # Drop leading prose before AGENT
    m = re.search(r"\bAGENT\b", s, re.I)
    if m and m.start() > 0:
        s = s[m.start() :]

    # Normalize smart quotes first so string literals are detected correctly
    s = s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")

    # Uppercase reserved keywords (outside strings, never $vars or names)
    s = _sub_code(r"[\s\S]+", lambda m: _uppercase_keywords(m.group(0)), s)

    # AGENT name without quotes: AGENT foo-bar / AGENT deal_finder → AGENT "..."
    s = re.sub(
        r'\bAGENT\s+(?!")([A-Za-z_][A-Za-z0-9_\-]*)\b',
        r'AGENT "\1"',
        s,
    )

    # Mistaken "STEP x TOOL y FILTER ..." → "STEP x FILTER ..."
    s = re.sub(
        r"\bSTEP\s+([A-Za-z_][A-Za-z0-9_]*)\s+TOOL\s+[A-Za-z_][A-Za-z0-9_]*\s+(FILTER\b)",
        r"STEP \1 \2",
        s,
    )

    # Strip junk lines that appear inside MEMORY before STEPs (e.g. "TOP PAPER 1")
    # while keeping list / object values, including multi-line ones.
    _VALUE = r'(-?\d+\.?\d*|true|false|null|"[^"]*"|\'[^\']*\'|\$[A-Za-z_][\w.]*|\[.*|\{.*)'
    _ENTRY = re.compile(rf"^[A-Za-z_][A-Za-z0-9_]*\s*[:=]\s*{_VALUE}\s*,?\s*$", re.I)

    def _clean_memory_section(text: str) -> str:
        out: list[str] = []
        in_memory = False
        depth = 0  # [ / { nesting inside MEMORY (outer brace included)
        for line in text.splitlines():
            stripped = line.strip()
            code = _mask_strings(stripped)
            if not in_memory and re.match(r"^MEMORY\b", stripped):
                depth = code.count("{") + code.count("[") - code.count("}") - code.count("]")
                # MEMORY { ... } closed on one line is done; flat form continues
                in_memory = not ("{" in code and depth <= 0)
                out.append(line)
                continue
            if not in_memory:
                out.append(line)
                continue
            if re.match(r"^(STEP|REASON|VERIFY|RETURN|AGENT|FOR|IF)\b", stripped):
                in_memory = False
                out.append(line)
                continue
            inside_value = depth > 1  # within a list/object value, keep verbatim
            if (
                inside_value
                or not stripped
                or _ENTRY.match(stripped)
                or stripped[0] in "{}[],"
                or re.match(r"^MEMORY\b", stripped)
            ):
                out.append(line)
                depth += code.count("{") + code.count("[") - code.count("}") - code.count("]")
            # else: drop junk (prose / fake nested fields)
            if depth <= 0 and "{" in "".join(out[-3:]):
                in_memory = False
        return "\n".join(out)

    s = _clean_memory_section(s)

    # Hoist MEMORY that LLMs emit before AGENT
    m_agent = re.search(r"\bAGENT\b[^\n]*", s)
    m_mem = re.search(r"\bMEMORY\b[\s\S]*?(?=\bAGENT\b)", s)
    if m_agent and m_mem and m_mem.start() < m_agent.start():
        mem = m_mem.group(0).strip()
        s = s[: m_mem.start()] + s[m_mem.end() :]
        m_agent = re.search(r"\bAGENT\b[^\n]*", s)
        if m_agent:
            at = m_agent.end()
            s = s[:at] + "\n" + mem + "\n" + s[at:]

    # Strip markdown fences / backticks leftovers
    s = s.replace("```", "\n")
    s = re.sub(r"`+", "", s)

    # KEEP FILTER / keep FILTER → STEP keep FILTER
    s = re.sub(r"(?im)^(\s*)KEEP\s+FILTER\b", r"\1STEP keep FILTER", s)

    # Cut trailing junk after RETURN (extra STEP/AGENT/prose/fences)
    ret = re.search(r"\bRETURN\b", _mask_strings(s))
    if ret:
        head = s[: ret.start()]
        tail = s[ret.start() :]
        lines = tail.splitlines()
        kept: list[str] = []
        brace_depth = 0
        started = False
        for line in lines:
            stripped = line.strip()
            if not started:
                kept.append(line)
                started = True
                brace_depth = stripped.count("{") - stripped.count("}")
                if brace_depth <= 0 and ("{" not in stripped or stripped.endswith("}")):
                    break
                continue
            # stop if a new program section starts after RETURN closed
            if brace_depth <= 0 and re.match(
                r"^(AGENT|MEMORY|STEP|REASON|VERIFY|RETURN)\b", stripped
            ):
                break
            if brace_depth <= 0 and stripped.startswith("```"):
                break
            brace_depth += stripped.count("{") - stripped.count("}")
            kept.append(line)
            if brace_depth <= 0 and stripped.endswith("}"):
                break
        s = head + "\n".join(kept)

    # key=value → key: value (MEMORY / INPUT / objects); never touches ==
    s = _sub_code(
        r"([A-Za-z_][A-Za-z0-9_]*)\s*=(?!=)\s*",
        r"\1: ",
        s,
    )

    # Fix: OUTPUT varname then dangling colon garbage
    s = re.sub(r"(OUTPUT\s+[A-Za-z_][A-Za-z0-9_]*)\s*:[^\n]*", r"\1", s)

    # Missing OUTPUT after TOOL ... INPUT { ... } before next keyword
    _TOOL_OUTPUT_DEFAULTS = {
        "get_value": "value",
        "arithmetic_add": "sum",
        "web_search": "results",
        "sales_data": "sales",
    }

    def _pick_out_var(tool: str, step_name: str, after: str) -> str:
        for candidate in ("raw", "value", "results", "sales", "sum", "total", step_name):
            if re.search(rf"\${candidate}\b", after):
                return candidate
        return _TOOL_OUTPUT_DEFAULTS.get(tool, step_name)

    def _inject_output(match: re.Match) -> str:
        step_name = match.group(1)
        tool_chunk = match.group(0)
        tm = re.search(r"\bTOOL\s+([A-Za-z_][A-Za-z0-9_]*)", tool_chunk)
        tool = tm.group(1) if tm else ""
        after = s[match.end() :]
        out_var = _pick_out_var(tool, step_name, after)
        # Already has OUTPUT name — keep
        if re.search(r"\bOUTPUT\s+[A-Za-z_]", tool_chunk):
            return tool_chunk
        # Bare OUTPUT with no name — strip and re-add
        tool_chunk = re.sub(r"\bOUTPUT\s*$", "", tool_chunk.rstrip())
        return f"{tool_chunk.rstrip()} OUTPUT {out_var}"

    s = re.sub(
        r"STEP\s+([A-Za-z_][A-Za-z0-9_]*)\s+TOOL\b[\s\S]*?\bINPUT\s*\{[^{}]*\}(?=\s*(?:STEP|REASON|VERIFY|RETURN|FOR|IF|ELSE|END|YIELD|$))",
        _inject_output,
        s,
    )

    # Global fix: OUTPUT immediately followed by next keyword → insert a name
    def _fix_bare_output(match: re.Match) -> str:
        return f"OUTPUT out\n{match.group(1)}"

    s = re.sub(
        r"\bOUTPUT\s*(?=\n\s*(STEP|REASON|VERIFY|RETURN|FOR|IF|ELSE|END|YIELD)\b)",
        "OUTPUT out\n",
        s,
    )

    # After inject: if body uses $raw but OUTPUT value, rename
    if re.search(r"\$raw\b", s) and re.search(r"\bOUTPUT\s+value\b", s):
        s = re.sub(r"\bOUTPUT\s+value\b", "OUTPUT raw", s, count=1)

    # Close MEMORY { ... that is missing } before STEP/REASON/VERIFY/RETURN
    s = re.sub(
        r"(MEMORY\s*\{[^}]*)\n(?=\s*(?:STEP|REASON|VERIFY|RETURN)\b)",
        r"\1\n}\n",
        s,
    )

    # REASON ON $x OUTPUT y + VERIFY $y > N  →  VERIFY $x > N
    # (models often verify the reason string instead of the numeric input)
    s = re.sub(
        r'(REASON\s+(?:"[^"]*"|\'[^\']*\')\s+ON\s+\$([A-Za-z_][A-Za-z0-9_]*)\s+OUTPUT\s+([A-Za-z_][A-Za-z0-9_]*)'
        r')([\s\S]*?\bVERIFY\s+)\$\3(\s*(?:>=|<=|!=|==|>|<))',
        r"\1\4$\2\5",
        s,
    )

    # Collapse Windows newlines
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    # Ensure newline before major keywords for readability / lex stability
    for kw in ("MEMORY", "STEP", "REASON", "VERIFY", "RETURN", "FOR", "IF", "ELSE", "END", "YIELD"):
        s = _sub_code(rf"([^\n])\s*({kw}\b)", rf"\1\n\2", s)

    return s.strip() + "\n"


def repair_json_ast(text: str) -> str:
    """Best-effort salvage of near-valid JSON AST from small models."""
    raw = extract_forge(text, prefer_json=True).strip()
    if not raw.startswith("{"):
        # try to find JSON object
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end <= start:
            return raw
        raw = raw[start : end + 1]

    # Remove trailing commas before } or ]
    cleaned = re.sub(r",\s*([}\]])", r"\1", raw)
    # Strip JS-style comments
    cleaned = re.sub(r"//.*?$", "", cleaned, flags=re.M)
    cleaned = re.sub(r"/\*.*?\*/", "", cleaned, flags=re.S)

    import json as _json

    try:
        _json.loads(cleaned)
        return cleaned
    except Exception:
        pass

    # Truncate to the longest prefix that parses as JSON object
    for i in range(len(cleaned), 0, -1):
        if cleaned[i - 1] != "}":
            continue
        chunk = cleaned[:i]
        chunk = re.sub(r",\s*([}\]])", r"\1", chunk)
        try:
            _json.loads(chunk)
            return chunk
        except Exception:
            continue
    return cleaned


def normalize_llm_output(text: str, prefer_json: bool = False) -> str:
    """Full pipeline: extract → repair."""
    if prefer_json or text.strip().lstrip().startswith("{"):
        return repair_json_ast(text)
    return repair_forge(extract_forge(text, prefer_json=False))


def try_compile_repaired(source: str):
    """
    Attempt compile with progressive repair.
    Returns (program, repaired_source, notes).
    """
    from forge_core import compile_auto, ForgeError

    notes: list[str] = []
    # Prefer repaired forms first — a parseable-but-wrong unrepaired
    # program (e.g. OUTPUT value while body uses $raw) should not win.
    candidates = [
        normalize_llm_output(source),
        repair_forge(extract_forge(source)),
        repair_forge(source),
        extract_forge(source),
        source,
    ]

    seen: set[str] = set()
    last_err: Optional[Exception] = None
    for cand in candidates:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        try:
            prog = compile_auto(cand)
            if cand.strip() != source.strip():
                notes.append("repaired")
            return prog, cand, notes
        except Exception as e:
            last_err = e
            continue

    raise ForgeError(f"Unrepairable Forge output: {last_err}")
