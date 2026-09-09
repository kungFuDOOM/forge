"""
Forge v0.1 — Core language infrastructure
==========================================
AST nodes, lexer, parser, compile pipeline, JSON AST dual path.

Tagline: "JavaScript for AI"
Design: dual representation (text syntax ↔ JSON AST), agent-first primitives.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Union


# =============================================================================
# EXCEPTIONS
# =============================================================================

class ForgeError(Exception):
    """Base Forge error."""


class ForgeLexError(ForgeError):
    """Lexer error."""


class ForgeParseError(ForgeError):
    """Parser error."""


class ForgeValidateError(ForgeError):
    """JSON AST validation error."""


# =============================================================================
# AST NODES
# =============================================================================

@dataclass
class ASTNode:
    """Base AST node. Subclasses set node_type via default field."""

    def to_dict(self) -> dict:
        return _ast_to_dict(self)


@dataclass
class Literal(ASTNode):
    value: Any
    node_type: str = field(default="literal")


@dataclass
class Variable(ASTNode):
    name: str
    node_type: str = field(default="variable")


@dataclass
class ObjectLiteral(ASTNode):
    properties: dict  # str -> ASTNode
    node_type: str = field(default="object_literal")


@dataclass
class Comparison(ASTNode):
    left: Any  # ASTNode or field name (str) for FILTER field refs
    operator: str
    right: Any  # ASTNode
    left_kind: str = "expr"  # "expr" | "field"
    node_type: str = field(default="comparison")


@dataclass
class MemoryEntry(ASTNode):
    key: str
    value: ASTNode
    node_type: str = field(default="memory_entry")


@dataclass
class MemoryBlock(ASTNode):
    entries: list
    node_type: str = field(default="memory_block")


@dataclass
class AgentDecl(ASTNode):
    name: str
    node_type: str = field(default="agent_decl")


@dataclass
class ToolCall(ASTNode):
    tool_name: str
    inputs: dict  # str -> ASTNode
    output_var: str
    node_type: str = field(default="tool_call")


@dataclass
class Filter(ASTNode):
    condition: Comparison
    input_var: str
    output_var: str
    node_type: str = field(default="filter")


@dataclass
class Step(ASTNode):
    step_name: str
    action: Union[ToolCall, Filter]
    node_type: str = field(default="step")


@dataclass
class Reason(ASTNode):
    prompt: str
    input_var: str
    output_var: str
    node_type: str = field(default="reason")


@dataclass
class Verify(ASTNode):
    condition: Comparison
    node_type: str = field(default="verify")


@dataclass
class Return(ASTNode):
    value: ASTNode
    node_type: str = field(default="return")


@dataclass
class Program(ASTNode):
    agent: AgentDecl
    memory: Optional[MemoryBlock]
    steps: list
    reason: Optional[Reason]
    verify: Optional[Verify]
    return_stmt: Return
    node_type: str = field(default="program")

    # Alias for JSON key "return" (Python reserved word handled in serialization)
    @property
    def return_(self) -> Return:
        return self.return_stmt


# =============================================================================
# SERIALIZATION
# =============================================================================

def _ast_to_dict(node: Any) -> Any:
    if node is None:
        return None
    if isinstance(node, list):
        return [_ast_to_dict(n) for n in node]
    if isinstance(node, dict):
        return {k: _ast_to_dict(v) for k, v in node.items()}
    if not isinstance(node, ASTNode):
        return node

    d: dict[str, Any] = {"node_type": node.node_type}

    if isinstance(node, Literal):
        d["value"] = node.value
    elif isinstance(node, Variable):
        d["name"] = node.name
    elif isinstance(node, ObjectLiteral):
        d["properties"] = {k: _ast_to_dict(v) for k, v in node.properties.items()}
    elif isinstance(node, Comparison):
        d["left"] = _ast_to_dict(node.left) if isinstance(node.left, ASTNode) else node.left
        d["operator"] = node.operator
        d["right"] = _ast_to_dict(node.right)
        d["left_kind"] = node.left_kind
    elif isinstance(node, MemoryEntry):
        d["key"] = node.key
        d["value"] = _ast_to_dict(node.value)
    elif isinstance(node, MemoryBlock):
        d["entries"] = [_ast_to_dict(e) for e in node.entries]
    elif isinstance(node, AgentDecl):
        d["name"] = node.name
    elif isinstance(node, ToolCall):
        d["tool_name"] = node.tool_name
        d["inputs"] = {k: _ast_to_dict(v) for k, v in node.inputs.items()}
        d["output_var"] = node.output_var
    elif isinstance(node, Filter):
        d["condition"] = _ast_to_dict(node.condition)
        d["input_var"] = node.input_var
        d["output_var"] = node.output_var
    elif isinstance(node, Step):
        d["step_name"] = node.step_name
        d["action"] = _ast_to_dict(node.action)
    elif isinstance(node, Reason):
        d["prompt"] = node.prompt
        d["input_var"] = node.input_var
        d["output_var"] = node.output_var
    elif isinstance(node, Verify):
        d["condition"] = _ast_to_dict(node.condition)
    elif isinstance(node, Return):
        d["value"] = _ast_to_dict(node.value)
    elif isinstance(node, Program):
        d["agent"] = _ast_to_dict(node.agent)
        d["memory"] = _ast_to_dict(node.memory)
        d["steps"] = [_ast_to_dict(s) for s in node.steps]
        d["reason"] = _ast_to_dict(node.reason)
        d["verify"] = _ast_to_dict(node.verify)
        d["return"] = _ast_to_dict(node.return_stmt)
    else:
        raise ForgeError(f"Unknown AST node for serialization: {type(node)}")

    return d


def ast_to_json(node: ASTNode, indent: int = 2) -> str:
    return json.dumps(_ast_to_dict(node), indent=indent)


# =============================================================================
# JSON AST DESERIALIZER (direct mode — LLMs can emit JSON)
# =============================================================================

def dict_to_ast(data: Any) -> ASTNode:
    """Deserialize a JSON-compatible dict into AST nodes. Source of truth path."""
    if not isinstance(data, dict):
        raise ForgeValidateError(f"Expected object, got {type(data).__name__}")

    nt = data.get("node_type")
    if not nt:
        raise ForgeValidateError("Missing required field: node_type")

    if nt == "literal":
        return Literal(value=data.get("value"))

    if nt == "variable":
        name = data.get("name")
        if not isinstance(name, str) or not name:
            raise ForgeValidateError("variable.name must be a non-empty string")
        return Variable(name=name)

    if nt == "object_literal":
        props = data.get("properties")
        if not isinstance(props, dict):
            raise ForgeValidateError("object_literal.properties must be an object")
        return ObjectLiteral(properties={k: dict_to_ast(v) for k, v in props.items()})

    if nt == "comparison":
        left_raw = data.get("left")
        left_kind = data.get("left_kind", "expr")
        if left_kind == "field":
            left: Any = left_raw
            if not isinstance(left, str):
                raise ForgeValidateError("comparison.left (field) must be a string")
        else:
            left = dict_to_ast(left_raw) if isinstance(left_raw, dict) else left_raw
        op = data.get("operator")
        if op not in (">", "<", ">=", "<=", "==", "!="):
            raise ForgeValidateError(f"Invalid comparison operator: {op}")
        right = dict_to_ast(data["right"]) if isinstance(data.get("right"), dict) else data.get("right")
        if not isinstance(right, ASTNode):
            raise ForgeValidateError("comparison.right must be an AST node")
        return Comparison(left=left, operator=op, right=right, left_kind=left_kind)

    if nt == "memory_entry":
        key = data.get("key")
        if not isinstance(key, str):
            raise ForgeValidateError("memory_entry.key must be a string")
        return MemoryEntry(key=key, value=dict_to_ast(data["value"]))

    if nt == "memory_block":
        entries = data.get("entries", [])
        if not isinstance(entries, list):
            raise ForgeValidateError("memory_block.entries must be a list")
        return MemoryBlock(entries=[dict_to_ast(e) for e in entries])

    if nt == "agent_decl":
        name = data.get("name")
        if not isinstance(name, str) or not name:
            raise ForgeValidateError("agent_decl.name must be a non-empty string")
        return AgentDecl(name=name)

    if nt == "tool_call":
        tool_name = data.get("tool_name")
        inputs = data.get("inputs", {})
        output_var = data.get("output_var")
        if not isinstance(tool_name, str) or not tool_name:
            raise ForgeValidateError("tool_call.tool_name required")
        if not isinstance(inputs, dict):
            raise ForgeValidateError("tool_call.inputs must be an object")
        if not isinstance(output_var, str) or not output_var:
            raise ForgeValidateError("tool_call.output_var required")
        return ToolCall(
            tool_name=tool_name,
            inputs={k: dict_to_ast(v) for k, v in inputs.items()},
            output_var=output_var,
        )

    if nt == "filter":
        return Filter(
            condition=dict_to_ast(data["condition"]),  # type: ignore
            input_var=data["input_var"],
            output_var=data["output_var"],
        )

    if nt == "step":
        return Step(step_name=data["step_name"], action=dict_to_ast(data["action"]))  # type: ignore

    if nt == "reason":
        return Reason(
            prompt=data["prompt"],
            input_var=data["input_var"],
            output_var=data["output_var"],
        )

    if nt == "verify":
        cond = data["condition"]
        # Allow legacy string form "$sum > 0" for friendliness
        if isinstance(cond, str):
            cond = _parse_condition_string(cond)
            return Verify(condition=cond)
        return Verify(condition=dict_to_ast(cond))  # type: ignore

    if nt == "return":
        return Return(value=dict_to_ast(data["value"]))

    if nt == "program":
        memory = data.get("memory")
        reason = data.get("reason")
        verify = data.get("verify")
        ret = data.get("return")
        if ret is None:
            raise ForgeValidateError("program.return is required")
        steps = data.get("steps", [])
        if not steps:
            raise ForgeValidateError("program.steps must contain at least one step")
        return Program(
            agent=dict_to_ast(data["agent"]),  # type: ignore
            memory=dict_to_ast(memory) if memory else None,  # type: ignore
            steps=[dict_to_ast(s) for s in steps],
            reason=dict_to_ast(reason) if reason else None,  # type: ignore
            verify=dict_to_ast(verify) if verify else None,  # type: ignore
            return_stmt=dict_to_ast(ret),  # type: ignore
        )

    raise ForgeValidateError(f"Unknown node_type: {nt}")


def _parse_condition_string(s: str) -> Comparison:
    """Parse a simple condition string like '$sum > 0' or 'relevance > 0.8'."""
    s = s.strip()
    m = re.match(
        r"^(\$[A-Za-z_][A-Za-z0-9_]*|[A-Za-z_][A-Za-z0-9_]*)\s*(>=|<=|!=|==|>|<)\s*(.+)$",
        s,
    )
    if not m:
        raise ForgeValidateError(f"Cannot parse condition string: {s!r}")
    left_s, op, right_s = m.group(1), m.group(2), m.group(3).strip()
    if left_s.startswith("$"):
        left: Any = Variable(name=left_s[1:])
        left_kind = "expr"
    else:
        left = left_s
        left_kind = "field"
    right = _parse_value_token(right_s)
    return Comparison(left=left, operator=op, right=right, left_kind=left_kind)


def _parse_value_token(s: str) -> ASTNode:
    s = s.strip()
    if s.startswith("$"):
        return Variable(name=s[1:])
    if s in ("true", "false"):
        return Literal(value=s == "true")
    if s == "null":
        return Literal(value=None)
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return Literal(value=s[1:-1])
    try:
        if "." in s:
            return Literal(value=float(s))
        return Literal(value=int(s))
    except ValueError:
        raise ForgeValidateError(f"Cannot parse value: {s!r}")


# =============================================================================
# VALIDATOR
# =============================================================================

def validate_ast(node: ASTNode, path: str = "program") -> list[str]:
    """Structural validation. Returns list of error messages (empty = valid)."""
    errors: list[str] = []

    def err(msg: str) -> None:
        errors.append(f"{path}: {msg}")

    if not isinstance(node, Program):
        err(f"root must be program, got {getattr(node, 'node_type', type(node))}")
        return errors

    if not isinstance(node.agent, AgentDecl) or not node.agent.name:
        err("agent declaration with non-empty name required")

    if not node.steps:
        err("at least one STEP required")

    for i, step in enumerate(node.steps):
        sp = f"{path}.steps[{i}]"
        if not isinstance(step, Step):
            errors.append(f"{sp}: expected step node")
            continue
        if not step.step_name:
            errors.append(f"{sp}: step_name required")
        action = step.action
        if isinstance(action, ToolCall):
            if not action.tool_name:
                errors.append(f"{sp}.action: tool_name required")
            if not action.output_var:
                errors.append(f"{sp}.action: output_var required")
        elif isinstance(action, Filter):
            if not action.output_var or not action.input_var:
                errors.append(f"{sp}.action: input_var and output_var required")
            if not isinstance(action.condition, Comparison):
                errors.append(f"{sp}.action: condition must be comparison")
        else:
            errors.append(f"{sp}.action: must be tool_call or filter")

    if node.reason is not None:
        if not isinstance(node.reason, Reason):
            err("reason must be reason node")
        elif not node.reason.prompt or not node.reason.output_var:
            err("reason requires prompt and output_var")

    if node.verify is not None:
        if not isinstance(node.verify, Verify):
            err("verify must be verify node")
        elif not isinstance(node.verify.condition, Comparison):
            err("verify.condition must be comparison")

    if not isinstance(node.return_stmt, Return):
        err("return is required")
    elif node.return_stmt.value is None:
        err("return.value is required")

    return errors


def validate_dict(data: dict) -> list[str]:
    """Validate a JSON AST dict without full execution. Raises or returns errors."""
    try:
        node = dict_to_ast(data)
    except ForgeError as e:
        return [str(e)]
    return validate_ast(node)


# =============================================================================
# LEXER
# =============================================================================

KEYWORDS = {
    "AGENT", "MEMORY", "STEP", "TOOL", "FILTER", "INPUT", "OUTPUT",
    "ON", "REASON", "VERIFY", "RETURN",
    "true", "false", "null",
}

# Order matters: longer ops first
OPERATORS = [">=", "<=", "!=", "==", ">", "<"]


@dataclass
class Token:
    type: str
    value: Any
    line: int
    col: int


class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.col = 1
        self.length = len(source)

    def _peek(self, n: int = 0) -> str:
        i = self.pos + n
        if i >= self.length:
            return ""
        return self.source[i]

    def _advance(self) -> str:
        ch = self.source[self.pos]
        self.pos += 1
        if ch == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return ch

    def _skip_ws_and_comments(self) -> None:
        while self.pos < self.length:
            ch = self._peek()
            if ch in " \t\r\n":
                self._advance()
            elif ch == "#" or (ch == "/" and self._peek(1) == "/"):
                while self.pos < self.length and self._peek() != "\n":
                    self._advance()
            else:
                break

    def tokenize(self) -> list[Token]:
        tokens: list[Token] = []
        while self.pos < self.length:
            self._skip_ws_and_comments()
            if self.pos >= self.length:
                break

            line, col = self.line, self.col
            ch = self._peek()

            # String (double or single quotes — AIs often emit either)
            if ch in ('"', "'"):
                tokens.append(self._read_string(ch))
                continue

            # Variable $name
            if ch == "$":
                self._advance()
                name = self._read_ident()
                if not name:
                    raise ForgeLexError(f"Expected identifier after $ at {line}:{col}")
                tokens.append(Token("VARIABLE", name, line, col))
                continue

            # Number (including negative)
            if ch.isdigit() or (ch == "-" and self._peek(1).isdigit()):
                tokens.append(self._read_number())
                continue

            # Operators
            matched_op = None
            for op in OPERATORS:
                if self.source[self.pos:self.pos + len(op)] == op:
                    matched_op = op
                    break
            if matched_op:
                for _ in matched_op:
                    self._advance()
                tokens.append(Token("OP", matched_op, line, col))
                continue

            # Punctuation
            if ch in "{}:,":
                self._advance()
                tokens.append(Token(ch, ch, line, col))
                continue

            # Identifier / keyword
            if ch.isalpha() or ch == "_":
                ident = self._read_ident()
                if ident in ("true", "false", "null"):
                    val = True if ident == "true" else (False if ident == "false" else None)
                    tokens.append(Token("LITERAL", val, line, col))
                elif ident in KEYWORDS:
                    tokens.append(Token("KEYWORD", ident, line, col))
                else:
                    tokens.append(Token("IDENT", ident, line, col))
                continue

            raise ForgeLexError(f"Unexpected character {ch!r} at {line}:{col}")

        tokens.append(Token("EOF", None, self.line, self.col))
        return tokens

    def _read_string(self, quote: str = '"') -> Token:
        line, col = self.line, self.col
        self._advance()  # opening quote
        chars: list[str] = []
        while self.pos < self.length:
            ch = self._peek()
            if ch == quote:
                self._advance()
                return Token("STRING", "".join(chars), line, col)
            if ch == "\\":
                self._advance()
                esc = self._advance()
                mapping = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "'": "'", "\\": "\\"}
                chars.append(mapping.get(esc, esc))
            else:
                chars.append(self._advance())
        raise ForgeLexError(f"Unterminated string starting at {line}:{col}")

    def _read_number(self) -> Token:
        line, col = self.line, self.col
        start = self.pos
        if self._peek() == "-":
            self._advance()
        while self._peek().isdigit():
            self._advance()
        if self._peek() == "." and self._peek(1).isdigit():
            self._advance()
            while self._peek().isdigit():
                self._advance()
        raw = self.source[start:self.pos]
        value: Union[int, float] = float(raw) if "." in raw else int(raw)
        return Token("NUMBER", value, line, col)

    def _read_ident(self) -> str:
        start = self.pos
        while self.pos < self.length:
            ch = self._peek()
            if ch.isalnum() or ch == "_":
                self._advance()
            else:
                break
        return self.source[start:self.pos]


# =============================================================================
# PARSER
# =============================================================================

class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def _cur(self) -> Token:
        return self.tokens[self.pos]

    def _peek(self, n: int = 0) -> Token:
        i = self.pos + n
        if i >= len(self.tokens):
            return self.tokens[-1]
        return self.tokens[i]

    def _advance(self) -> Token:
        tok = self.tokens[self.pos]
        if tok.type != "EOF":
            self.pos += 1
        return tok

    def _expect(self, type_: str, value: Any = None) -> Token:
        tok = self._cur()
        if tok.type != type_ and not (type_ == "KEYWORD" and tok.type == "KEYWORD" and tok.value == value):
            if value is not None and not (tok.type == type_ and tok.value == value):
                raise ForgeParseError(
                    f"Expected {type_} {value!r}, got {tok.type} {tok.value!r} at {tok.line}:{tok.col}"
                )
            if value is None and tok.type != type_:
                raise ForgeParseError(
                    f"Expected {type_}, got {tok.type} {tok.value!r} at {tok.line}:{tok.col}"
                )
        if value is not None and tok.value != value and tok.type != type_:
            pass
        if value is not None and tok.type == type_ and tok.value != value:
            # For KEYWORD matching value is required
            if type_ == "KEYWORD":
                raise ForgeParseError(
                    f"Expected keyword {value}, got {tok.value} at {tok.line}:{tok.col}"
                )
        if type_ == "KEYWORD":
            if tok.type != "KEYWORD" or (value is not None and tok.value != value):
                raise ForgeParseError(
                    f"Expected keyword {value}, got {tok.type} {tok.value!r} at {tok.line}:{tok.col}"
                )
        elif tok.type != type_:
            raise ForgeParseError(
                f"Expected {type_}, got {tok.type} {tok.value!r} at {tok.line}:{tok.col}"
            )
        return self._advance()

    def _match_keyword(self, name: str) -> bool:
        tok = self._cur()
        return tok.type == "KEYWORD" and tok.value == name

    def parse(self) -> Program:
        agent = self._parse_agent()

        # Multiple MEMORY blocks allowed (AIs often emit more than one)
        mem_entries: list = []
        while self._match_keyword("MEMORY"):
            block = self._parse_memory()
            mem_entries.extend(block.entries)
        memory = MemoryBlock(entries=mem_entries) if mem_entries else None

        steps: list[Step] = []
        reason = None

        # STEPs, and allow REASON interleaved or as a STEP shorthand
        while True:
            if self._match_keyword("STEP"):
                step, maybe_reason = self._parse_step()
                if maybe_reason is not None:
                    # REASON emitted as STEP — hoist to program.reason (last wins)
                    reason = maybe_reason
                else:
                    steps.append(step)  # type: ignore
                continue
            if self._match_keyword("REASON"):
                reason = self._parse_reason()
                continue
            break

        if not steps:
            raise ForgeParseError("At least one STEP is required")

        verify = None
        if self._match_keyword("VERIFY"):
            verify = self._parse_verify()

        if not self._match_keyword("RETURN"):
            raise ForgeParseError(
                f"Expected RETURN, got {self._cur().type} {self._cur().value!r} "
                f"at {self._cur().line}:{self._cur().col}"
            )
        return_stmt = self._parse_return()

        if self._cur().type != "EOF":
            raise ForgeParseError(
                f"Unexpected token after RETURN: {self._cur().type} {self._cur().value!r} "
                f"at {self._cur().line}:{self._cur().col}"
            )

        return Program(
            agent=agent,
            memory=memory,
            steps=steps,
            reason=reason,
            verify=verify,
            return_stmt=return_stmt,
        )

    def _parse_agent(self) -> AgentDecl:
        self._expect("KEYWORD", "AGENT")
        name_tok = self._expect("STRING")
        return AgentDecl(name=name_tok.value)

    def _parse_memory(self) -> MemoryBlock:
        """
        MEMORY { k: v ... }   — canonical
        MEMORY k: v, k2: v2   — AI-friendly flat form (until next keyword)
        """
        self._expect("KEYWORD", "MEMORY")
        entries: list[MemoryEntry] = []

        if self._cur().type == "{":
            self._advance()
            while self._cur().type != "}":
                if self._cur().type == "EOF":
                    raise ForgeParseError("Unterminated MEMORY block")
                key = self._expect("IDENT").value
                self._expect(":")
                value = self._parse_value()
                entries.append(MemoryEntry(key=key, value=value))
                if self._cur().type == ",":
                    self._advance()
            self._expect("}")
            return MemoryBlock(entries=entries)

        # Flat form: key: value pairs until a structural keyword
        stop = {"STEP", "REASON", "VERIFY", "RETURN", "MEMORY", "AGENT"}
        while self._cur().type == "IDENT":
            key = self._advance().value
            self._expect(":")
            value = self._parse_value()
            entries.append(MemoryEntry(key=key, value=value))
            if self._cur().type == ",":
                self._advance()
            if self._match_keyword_any(stop):
                break
        if not entries:
            raise ForgeParseError(
                f"Expected MEMORY block, got {self._cur().type} {self._cur().value!r} "
                f"at {self._cur().line}:{self._cur().col}"
            )
        return MemoryBlock(entries=entries)

    def _match_keyword_any(self, names: set) -> bool:
        tok = self._cur()
        return tok.type == "KEYWORD" and tok.value in names

    def _parse_step(self):
        """
        Returns (Step, None) for TOOL/FILTER steps.
        Returns (None, Reason) when AI writes REASON as a STEP:
          STEP name REASON "prompt" ON $x OUTPUT y
          STEP name "prompt" ON $x OUTPUT y
        """
        self._expect("KEYWORD", "STEP")
        step_name = self._expect("IDENT").value

        if self._match_keyword("TOOL"):
            action = self._parse_tool_call()
            return Step(step_name=step_name, action=action), None
        if self._match_keyword("FILTER"):
            action = self._parse_filter()
            return Step(step_name=step_name, action=action), None
        # AI-friendly: reason-as-step
        if self._match_keyword("REASON") or self._cur().type == "STRING":
            if self._match_keyword("REASON"):
                self._advance()
            prompt = self._expect("STRING").value
            self._expect("KEYWORD", "ON")
            input_var = self._expect("VARIABLE").value
            self._expect("KEYWORD", "OUTPUT")
            output_var = self._expect("IDENT").value
            self._skip_optional_annotation()
            return None, Reason(prompt=prompt, input_var=input_var, output_var=output_var)

        raise ForgeParseError(
            f"Expected TOOL, FILTER, or REASON after STEP name, got {self._cur().value!r} "
            f"at {self._cur().line}:{self._cur().col}"
        )

    def _parse_tool_call(self) -> ToolCall:
        self._expect("KEYWORD", "TOOL")
        tool_name = self._expect("IDENT").value
        self._expect("KEYWORD", "INPUT")
        self._expect("{")
        inputs: dict[str, ASTNode] = {}
        while self._cur().type != "}":
            if self._cur().type == "EOF":
                raise ForgeParseError("Unterminated INPUT block")
            key = self._expect("IDENT").value
            self._expect(":")
            inputs[key] = self._parse_value()
            if self._cur().type == ",":
                self._advance()
        self._expect("}")
        self._expect("KEYWORD", "OUTPUT")
        output_var = self._expect("IDENT").value
        self._skip_optional_annotation()
        return ToolCall(tool_name=tool_name, inputs=inputs, output_var=output_var)

    def _parse_filter(self) -> Filter:
        self._expect("KEYWORD", "FILTER")
        condition = self._parse_comparison()
        self._expect("KEYWORD", "ON")
        input_var = self._expect("VARIABLE").value
        self._expect("KEYWORD", "OUTPUT")
        output_var = self._expect("IDENT").value
        self._skip_optional_annotation()
        return Filter(condition=condition, input_var=input_var, output_var=output_var)

    def _parse_reason(self) -> Reason:
        self._expect("KEYWORD", "REASON")
        prompt = self._expect("STRING").value
        self._expect("KEYWORD", "ON")
        input_var = self._expect("VARIABLE").value
        self._expect("KEYWORD", "OUTPUT")
        output_var = self._expect("IDENT").value
        self._skip_optional_annotation()
        return Reason(prompt=prompt, input_var=input_var, output_var=output_var)

    def _skip_optional_annotation(self) -> None:
        """AIs sometimes append a label string after OUTPUT var — ignore it."""
        if self._cur().type == "STRING":
            self._advance()

    def _parse_verify(self) -> Verify:
        self._expect("KEYWORD", "VERIFY")
        condition = self._parse_comparison()
        return Verify(condition=condition)

    def _parse_return(self) -> Return:
        self._expect("KEYWORD", "RETURN")
        value = self._parse_value()
        return Return(value=value)

    def _parse_comparison(self) -> Comparison:
        """Parse left OP right. Left may be $var, field name, or literal."""
        left_kind = "expr"
        tok = self._cur()

        if tok.type == "VARIABLE":
            left: Any = Variable(name=self._advance().value)
        elif tok.type == "IDENT":
            # Field reference for FILTER (e.g. relevance > 0.8) OR bare name
            # Heuristic: IDENT before OP is a field path
            left = self._advance().value
            left_kind = "field"
        elif tok.type in ("NUMBER", "STRING", "LITERAL"):
            left = self._parse_value()
        else:
            raise ForgeParseError(
                f"Expected comparison left operand, got {tok.type} {tok.value!r} "
                f"at {tok.line}:{tok.col}"
            )

        op_tok = self._cur()
        if op_tok.type != "OP":
            raise ForgeParseError(
                f"Expected comparison operator, got {op_tok.type} {op_tok.value!r} "
                f"at {op_tok.line}:{op_tok.col}"
            )
        op = self._advance().value
        right = self._parse_value()
        return Comparison(left=left, operator=op, right=right, left_kind=left_kind)

    def _parse_value(self) -> ASTNode:
        tok = self._cur()
        if tok.type == "VARIABLE":
            self._advance()
            return Variable(name=tok.value)
        if tok.type == "NUMBER":
            self._advance()
            return Literal(value=tok.value)
        if tok.type == "STRING":
            self._advance()
            return Literal(value=tok.value)
        if tok.type == "LITERAL":
            self._advance()
            return Literal(value=tok.value)
        if tok.type == "{":
            return self._parse_object()
        raise ForgeParseError(
            f"Expected value, got {tok.type} {tok.value!r} at {tok.line}:{tok.col}"
        )

    def _parse_object(self) -> ObjectLiteral:
        self._expect("{")
        props: dict[str, ASTNode] = {}
        while self._cur().type != "}":
            if self._cur().type == "EOF":
                raise ForgeParseError("Unterminated object literal")
            key = self._expect("IDENT").value
            self._expect(":")
            props[key] = self._parse_value()
            if self._cur().type == ",":
                self._advance()
        self._expect("}")
        return ObjectLiteral(properties=props)


# =============================================================================
# PUBLIC API
# =============================================================================

def compile_forge(source: str) -> Program:
    """Text syntax → AST."""
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    program = parser.parse()
    errors = validate_ast(program)
    if errors:
        raise ForgeParseError("Validation failed:\n" + "\n".join(errors))
    return program


def compile_forge_json(data: Union[str, dict]) -> Program:
    """JSON AST → AST (direct mode for LLMs)."""
    if isinstance(data, str):
        data = json.loads(data)
    node = dict_to_ast(data)
    errors = validate_ast(node)
    if errors:
        raise ForgeValidateError("Validation failed:\n" + "\n".join(errors))
    if not isinstance(node, Program):
        raise ForgeValidateError("Root must be a program node")
    return node


def compile_auto(source: str) -> Program:
    """Accept either text Forge or JSON AST string."""
    stripped = source.strip()
    if stripped.startswith("{"):
        return compile_forge_json(stripped)
    return compile_forge(source)


# =============================================================================
# EXAMPLE PROGRAMS (canonical suite)
# =============================================================================

EXAMPLES: dict[str, str] = {
    "basic-calculator": '''
AGENT "basic-calculator"

MEMORY {
  a: 15
  b: 7
}

STEP add TOOL arithmetic_add INPUT { x: $a, y: $b } OUTPUT sum

REASON "Explain what the sum represents in a short sentence" ON $sum OUTPUT explanation

VERIFY $sum > 0

RETURN { result: $sum, note: $explanation }
''',
    "web-researcher": '''
AGENT "web-researcher"

MEMORY {
  query: "latest AI safety papers"
  max_results: 5
}

STEP search TOOL web_search INPUT { q: $query, n: $max_results } OUTPUT results

STEP filter_papers FILTER relevance > 0.8 ON $results OUTPUT top_papers

REASON "Summarize the key findings from these papers" ON $top_papers OUTPUT summary

VERIFY $top_papers != null

RETURN { papers: $top_papers, summary: $summary }
''',
    "sales-analyzer": '''
AGENT "sales-analyzer"

MEMORY {
  region: "north"
  quarter: "Q4"
}

STEP get_sales TOOL sales_data INPUT { region: $region, period: $quarter } OUTPUT raw_sales

STEP filter FILTER amount > 10000 ON $raw_sales OUTPUT large_deals

REASON "Identify the trend in these large deals" ON $large_deals OUTPUT trend

VERIFY $large_deals != null

RETURN { deals: $large_deals, trend: $trend }
''',
    "simple-filter": '''
AGENT "simple-filter"

MEMORY {
  threshold: 100
}

STEP check TOOL get_value INPUT { } OUTPUT raw

STEP filtered FILTER $raw > $threshold ON $raw OUTPUT result

VERIFY $result != $raw

RETURN { final: $result }
''',
    "tweet-creator": '''
AGENT "tweet-creator"

MEMORY {
  topic: "AI safety"
  max_chars: 280
}

STEP research TOOL web_search INPUT { q: $topic, n: 3 } OUTPUT articles

REASON "Create an engaging tweet about this topic based on the articles" ON $articles OUTPUT tweet

VERIFY $tweet != ""

RETURN { tweet: $tweet, source_count: 3 }
''',
}


def run_parser_tests() -> None:
    print("=" * 60)
    print("Forge Core — Parser Tests")
    print("=" * 60)
    passed = 0
    failed = 0
    for name, source in EXAMPLES.items():
        try:
            program = compile_forge(source)
            # Round-trip via JSON AST
            d = _ast_to_dict(program)
            program2 = compile_forge_json(d)
            assert program2.agent.name == program.agent.name
            assert len(program2.steps) == len(program.steps)
            js = ast_to_json(program)
            assert "node_type" in js
            print(f"  PASS  {name}  (agent={program.agent.name}, steps={len(program.steps)})")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            failed += 1
    print("-" * 60)
    print(f"Results: {passed} passed, {failed} failed / {passed + failed} total")
    if failed:
        raise SystemExit(1)
    print("All parser tests passed.")


if __name__ == "__main__":
    run_parser_tests()
