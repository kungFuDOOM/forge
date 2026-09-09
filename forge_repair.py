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
)


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

    m = re.search(r"(AGENT\b[\s\S]*)", raw, re.I)
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

    # Uppercase reserved keywords (word boundaries)
    for kw in KEYWORDS:
        s = re.sub(rf"\b{kw}\b", kw, s, flags=re.I)

    # Normalize smart quotes
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

    # Cut trailing prose after RETURN block
    # Keep from AGENT through last complete-looking RETURN ... 
    ret = re.search(r"\bRETURN\b[\s\S]*", s)
    if ret:
        head = s[: ret.start()]
        tail = ret.group(0)
        # Stop at first blank-line prose / markdown after RETURN value
        # Keep RETURN line(s) until we hit a line that looks like English prose
        lines = tail.splitlines()
        kept = [lines[0]] if lines else []
        brace_depth = lines[0].count("{") - lines[0].count("}") if lines else 0
        for line in lines[1:]:
            brace_depth += line.count("{") - line.count("}")
            kept.append(line)
            if brace_depth <= 0 and line.strip().endswith("}"):
                break
            if brace_depth <= 0 and re.match(r"^[A-Za-z].*[.!?]$", line.strip()):
                kept.pop()
                break
        s = head + "\n".join(kept)

    # key=value → key: value (MEMORY / INPUT / objects)
    s = re.sub(
        r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*",
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

    def _inject_output(match: re.Match) -> str:
        step_name = match.group(1)
        tool_chunk = match.group(0)
        if re.search(r"\bOUTPUT\b", tool_chunk):
            return tool_chunk
        tm = re.search(r"\bTOOL\s+([A-Za-z_][A-Za-z0-9_]*)", tool_chunk)
        tool = tm.group(1) if tm else ""
        out_var = _TOOL_OUTPUT_DEFAULTS.get(tool, step_name)
        return f"{tool_chunk.rstrip()} OUTPUT {out_var}"

    s = re.sub(
        r"STEP\s+([A-Za-z_][A-Za-z0-9_]*)\s+TOOL\b[\s\S]*?\bINPUT\s*\{[^{}]*\}(?=\s*(?:STEP|REASON|VERIFY|RETURN|$))",
        _inject_output,
        s,
    )

    # Collapse Windows newlines
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    # If VERIFY appears after RETURN, swap (common LLM order bug)
    if re.search(r"\bRETURN\b[\s\S]*\bVERIFY\b", s) and not re.search(
        r"\bVERIFY\b[\s\S]*\bRETURN\b", s
    ):
        parts = re.split(r"(\bVERIFY\b[\s\S]*?)(\bRETURN\b[\s\S]*)", s)
        # fragile — only when VERIFY somehow after; usually RETURN ends program
        pass

    # Ensure newline before major keywords for readability / lex stability
    for kw in ("MEMORY", "STEP", "REASON", "VERIFY", "RETURN"):
        s = re.sub(rf"([^\n])\s*({kw}\b)", rf"\1\n\2", s)

    return s.strip() + "\n"


def normalize_llm_output(text: str, prefer_json: bool = False) -> str:
    """Full pipeline: extract → repair."""
    return repair_forge(extract_forge(text, prefer_json=prefer_json))


def try_compile_repaired(source: str):
    """
    Attempt compile with progressive repair.
    Returns (program, repaired_source, notes).
    """
    from forge_core import compile_auto, ForgeError

    notes: list[str] = []
    candidates = [
        source,
        extract_forge(source),
        repair_forge(extract_forge(source)),
        normalize_llm_output(source),
    ]
    # Also try forcing keyword uppercase-only repair on raw
    candidates.append(repair_forge(source))

    seen: set[str] = set()
    last_err: Optional[Exception] = None
    for cand in candidates:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        try:
            prog = compile_auto(cand)
            if cand != source:
                notes.append("repaired")
            return prog, cand, notes
        except Exception as e:
            last_err = e
            continue

    raise ForgeError(f"Unrepairable Forge output: {last_err}")
