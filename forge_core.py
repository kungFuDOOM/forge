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
    name: str  # "doc" or dotted path "doc.body" / "hits.0.title"
    node_type: str = field(default="variable")


@dataclass
class ObjectLiteral(ASTNode):
    properties: dict  # str -> ASTNode
    node_type: str = field(default="object_literal")


@dataclass
class ArrayLiteral(ASTNode):
    items: list  # ASTNode
    node_type: str = field(default="array_literal")


@dataclass
class Comparison(ASTNode):
    left: Any  # ASTNode or field name (str) for FILTER field refs
    operator: str
    right: Any  # ASTNode
    left_kind: str = "expr"  # "expr" | "field"
    node_type: str = field(default="comparison")


@dataclass
class Logical(ASTNode):
    operator: str  # "AND" | "OR"
    left: Any  # Comparison | Logical
    right: Any
    node_type: str = field(default="logical")


CONDITION_OPS = (">", "<", ">=", "<=", "==", "!=", "CONTAINS")


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
    retries: int = 0  # extra attempts on failure (RETRY n)
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
    action: Union[ToolCall, Filter, Reason, Verify, "ForEach", "If", "Yield", "Try", "RunProgram"]
    node_type: str = field(default="step")


@dataclass
class Reason(ASTNode):
    prompt: str
    input_var: str
    output_var: str
    retries: int = 0
    node_type: str = field(default="reason")


@dataclass
class Verify(ASTNode):
    condition: Comparison
    node_type: str = field(default="verify")


@dataclass
class ForEach(ASTNode):
    var: str  # loop variable name (without $)
    source: ASTNode  # list to iterate
    body: list  # Step
    output_var: Optional[str] = None  # list of YIELDed values (or last output)
    parallel: int = 0  # >0: run up to this many iterations at once
    node_type: str = field(default="for_each")


@dataclass
class If(ASTNode):
    condition: Any  # Comparison | Logical
    then: list  # Step
    else_: list  # Step
    node_type: str = field(default="if")


@dataclass
class Try(ASTNode):
    body: list  # Step
    handler: list  # Step; runs with $error set if body fails
    error_var: str = "error"
    node_type: str = field(default="try")


@dataclass
class RunProgram(ASTNode):
    path: str  # another .forge file
    inputs: dict  # str -> ASTNode; override that program's MEMORY
    output_var: str
    retries: int = 0
    node_type: str = field(default="run_program")


MAX_RETRIES = 10
MAX_PARALLEL = 32


@dataclass
class Yield(ASTNode):
    value: ASTNode
    node_type: str = field(default="yield")


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
    elif isinstance(node, ArrayLiteral):
        d["items"] = [_ast_to_dict(v) for v in node.items]
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
        if node.retries:
            d["retries"] = node.retries
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
        if node.retries:
            d["retries"] = node.retries
    elif isinstance(node, Verify):
        d["condition"] = _ast_to_dict(node.condition)
    elif isinstance(node, Return):
        d["value"] = _ast_to_dict(node.value)
    elif isinstance(node, Logical):
        d["operator"] = node.operator
        d["left"] = _ast_to_dict(node.left)
        d["right"] = _ast_to_dict(node.right)
    elif isinstance(node, ForEach):
        d["var"] = node.var
        d["source"] = _ast_to_dict(node.source)
        d["body"] = [_ast_to_dict(s) for s in node.body]
        d["output_var"] = node.output_var
        if node.parallel:
            d["parallel"] = node.parallel
    elif isinstance(node, If):
        d["condition"] = _ast_to_dict(node.condition)
        d["then"] = [_ast_to_dict(s) for s in node.then]
        d["else"] = [_ast_to_dict(s) for s in node.else_]
    elif isinstance(node, Yield):
        d["value"] = _ast_to_dict(node.value)
    elif isinstance(node, Try):
        d["body"] = [_ast_to_dict(s) for s in node.body]
        d["handler"] = [_ast_to_dict(s) for s in node.handler]
        d["error_var"] = node.error_var
    elif isinstance(node, RunProgram):
        d["path"] = node.path
        d["inputs"] = {k: _ast_to_dict(v) for k, v in node.inputs.items()}
        d["output_var"] = node.output_var
        if node.retries:
            d["retries"] = node.retries
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

    if nt == "array_literal":
        items = data.get("items")
        if not isinstance(items, list):
            raise ForgeValidateError("array_literal.items must be a list")
        return ArrayLiteral(items=[dict_to_ast(v) for v in items])

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
        if op not in CONDITION_OPS:
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
            retries=_count(data, "retries", MAX_RETRIES),
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
            retries=_count(data, "retries", MAX_RETRIES),
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

    if nt == "logical":
        op = data.get("operator")
        if op not in ("AND", "OR"):
            raise ForgeValidateError(f"logical.operator must be AND or OR, got {op!r}")
        return Logical(operator=op, left=dict_to_ast(data.get("left")), right=dict_to_ast(data.get("right")))

    if nt == "for_each":
        var = data.get("var")
        if not isinstance(var, str) or not var:
            raise ForgeValidateError("for_each.var must be a non-empty string")
        body = data.get("body")
        if not isinstance(body, list) or not body:
            raise ForgeValidateError("for_each.body must be a non-empty list of steps")
        return ForEach(
            var=var.lstrip("$"),
            source=dict_to_ast(data.get("source")),
            body=[dict_to_ast(s) for s in body],
            output_var=data.get("output_var") or None,
            parallel=_count(data, "parallel", MAX_PARALLEL),
        )

    if nt == "if":
        then = data.get("then")
        if not isinstance(then, list) or not then:
            raise ForgeValidateError("if.then must be a non-empty list of steps")
        return If(
            condition=dict_to_ast(data.get("condition")),
            then=[dict_to_ast(s) for s in then],
            else_=[dict_to_ast(s) for s in data.get("else") or []],
        )

    if nt == "yield":
        return Yield(value=dict_to_ast(data.get("value")))

    if nt == "try":
        body, handler = data.get("body"), data.get("handler")
        if not isinstance(body, list) or not body or not isinstance(handler, list) or not handler:
            raise ForgeValidateError("try.body and try.handler must be non-empty lists of steps")
        return Try(
            body=[dict_to_ast(s) for s in body],
            handler=[dict_to_ast(s) for s in handler],
            error_var=str(data.get("error_var") or "error"),
        )

    if nt == "run_program":
        path, out = data.get("path"), data.get("output_var")
        if not isinstance(path, str) or not path or not isinstance(out, str) or not out:
            raise ForgeValidateError("run_program needs path and output_var strings")
        inputs = data.get("inputs") or {}
        if not isinstance(inputs, dict):
            raise ForgeValidateError("run_program.inputs must be an object")
        return RunProgram(
            path=path,
            inputs={k: dict_to_ast(v) for k, v in inputs.items()},
            output_var=out,
            retries=_count(data, "retries", MAX_RETRIES),
        )

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


def _count(data: dict, key: str, limit: int) -> int:
    value = data.get(key) or 0
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= limit:
        raise ForgeValidateError(f"{data.get('node_type')}.{key} must be an integer 0..{limit}")
    return value


def _parse_condition_string(s: str) -> Comparison:
    """Parse a simple condition string like '$sum > 0' or 'relevance > 0.8'."""
    s = s.strip()
    m = re.match(
        r"^(\$[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*|[A-Za-z_][A-Za-z0-9_]*)\s*(>=|<=|!=|==|>|<)\s*(.+)$",
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

    def is_condition(c: Any) -> bool:
        if isinstance(c, Logical):
            return c.operator in ("AND", "OR") and is_condition(c.left) and is_condition(c.right)
        return isinstance(c, Comparison)

    def check_steps(steps: list, sp_base: str) -> None:
        for i, step in enumerate(steps):
            sp = f"{sp_base}[{i}]"
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
                if not is_condition(action.condition):
                    errors.append(f"{sp}.action: condition must be comparison")
            elif isinstance(action, Reason):
                if not action.prompt or not action.output_var:
                    errors.append(f"{sp}.action: reason requires prompt and output_var")
            elif isinstance(action, Verify):
                if not is_condition(action.condition):
                    errors.append(f"{sp}.action: verify.condition must be comparison")
            elif isinstance(action, ForEach):
                if not action.body:
                    errors.append(f"{sp}.action: FOR body must not be empty")
                check_steps(action.body, f"{sp}.body")
            elif isinstance(action, If):
                if not is_condition(action.condition):
                    errors.append(f"{sp}.action: IF condition must be comparison")
                if not action.then:
                    errors.append(f"{sp}.action: IF body must not be empty")
                check_steps(action.then, f"{sp}.then")
                check_steps(action.else_, f"{sp}.else")
            elif isinstance(action, Yield):
                if action.value is None:
                    errors.append(f"{sp}.action: YIELD needs a value")
            elif isinstance(action, Try):
                if not action.body or not action.handler:
                    errors.append(f"{sp}.action: TRY and ON ERROR blocks must not be empty")
                check_steps(action.body, f"{sp}.body")
                check_steps(action.handler, f"{sp}.handler")
            elif isinstance(action, RunProgram):
                if not action.path or not action.output_var:
                    errors.append(f"{sp}.action: RUN needs a file path and OUTPUT")
            else:
                errors.append(
                    f"{sp}.action: must be tool_call, filter, reason, verify, for_each, if, yield, try, or run_program"
                )

    check_steps(node.steps, f"{path}.steps")

    if node.reason is not None:
        if not isinstance(node.reason, Reason):
            err("reason must be reason node")
        elif not node.reason.prompt or not node.reason.output_var:
            err("reason requires prompt and output_var")

    if node.verify is not None:
        if not isinstance(node.verify, Verify):
            err("verify must be verify node")
        elif not is_condition(node.verify.condition):
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


def check_program(program: Program, tool_names: Optional[list] = None) -> list[str]:
    """
    Static checks in execution order: every $var is defined before use and
    (if tool_names given) every TOOL exists. Runs before execution so a broken
    program fails before spending any tool or LLM calls.
    """
    errors: list[str] = []
    known_tools = set(tool_names) if tool_names is not None else None

    def use(name: str, where: str, defined: set) -> None:
        base = name.split(".", 1)[0]
        if base not in defined:
            hint = f" (defined so far: {', '.join(sorted(defined)) or 'none'})"
            errors.append(f"{where}: ${name} used before it is defined{hint}")

    def use_value(node: Any, where: str, defined: set) -> None:
        if isinstance(node, Variable):
            use(node.name, where, defined)
        elif isinstance(node, ObjectLiteral):
            for v in node.properties.values():
                use_value(v, where, defined)
        elif isinstance(node, ArrayLiteral):
            for v in node.items:
                use_value(v, where, defined)

    def use_condition(cond: Any, where: str, defined: set, in_filter: bool = False) -> None:
        if isinstance(cond, Logical):
            use_condition(cond.left, where, defined, in_filter)
            use_condition(cond.right, where, defined, in_filter)
        elif isinstance(cond, Comparison):
            if cond.left_kind == "field":
                # In FILTER a bare name is a field of each item; elsewhere it
                # can only mean a memory variable
                if not in_filter:
                    use(str(cond.left), where, defined)
            else:
                use_value(cond.left, where, defined)
            use_value(cond.right, where, defined)

    def check_block(steps: list, defined: set, in_loop: bool) -> None:
        for step in steps:
            check_action(step.action, f"STEP {step.step_name}", defined, in_loop)

    def check_action(action: Any, where: str, defined: set, in_loop: bool) -> None:
        if isinstance(action, ToolCall):
            if known_tools is not None and action.tool_name not in known_tools:
                errors.append(
                    f"{where}: unknown tool {action.tool_name!r} "
                    f"(available: {', '.join(sorted(known_tools))})"
                )
            for v in action.inputs.values():
                use_value(v, where, defined)
            defined.add(action.output_var)
        elif isinstance(action, Filter):
            use(action.input_var, where, defined)
            use_condition(action.condition, where, defined, in_filter=True)
            defined.add(action.output_var)
        elif isinstance(action, Reason):
            use(action.input_var, where, defined)
            defined.add(action.output_var)
        elif isinstance(action, Verify):
            use_condition(action.condition, where, defined)
        elif isinstance(action, ForEach):
            use_value(action.source, f"FOR {action.var}", defined)
            # Loop body is its own scope; only OUTPUT is visible afterwards
            check_block(action.body, defined | {action.var}, True)
            if action.output_var:
                defined.add(action.output_var)
        elif isinstance(action, If):
            use_condition(action.condition, "IF", defined)
            then_defs, else_defs = set(defined), set(defined)
            check_block(action.then, then_defs, in_loop)
            check_block(action.else_, else_defs, in_loop)
            # Visible after END only if set on every path
            defined |= then_defs & else_defs
        elif isinstance(action, Yield):
            if not in_loop:
                errors.append(f"{where}: YIELD is only allowed inside FOR EACH ... END")
            use_value(action.value, "YIELD", defined)
        elif isinstance(action, Try):
            body_defs, handler_defs = set(defined), defined | {action.error_var}
            check_block(action.body, body_defs, in_loop)
            check_block(action.handler, handler_defs, in_loop)
            # Visible after END only if set whether or not the body failed
            defined |= body_defs & handler_defs
        elif isinstance(action, RunProgram):
            for v in action.inputs.values():
                use_value(v, where, defined)
            defined.add(action.output_var)

    defined: set[str] = set()
    if program.memory:
        for entry in program.memory.entries:
            use_value(entry.value, f"MEMORY {entry.key}", defined)
            defined.add(entry.key)
    check_block(program.steps, defined, False)
    if program.reason:
        check_action(program.reason, "REASON", defined, False)
    if program.verify:
        check_action(program.verify, "VERIFY", defined, False)
    use_value(program.return_stmt.value, "RETURN", defined)
    return errors


# =============================================================================
# LEXER
# =============================================================================

KEYWORDS = {
    "AGENT", "MEMORY", "STEP", "TOOL", "FILTER", "INPUT", "OUTPUT",
    "ON", "REASON", "VERIFY", "RETURN",
    "FOR", "EACH", "IN", "IF", "ELSE", "END", "YIELD",
    "AND", "OR", "CONTAINS",
    "PARALLEL", "TRY", "ERROR", "RETRY", "RUN",
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

            # Variable $name, with optional field/index path: $doc.body, $hits.0.title
            if ch == "$":
                self._advance()
                name = self._read_ident()
                if not name:
                    raise ForgeLexError(f"Expected identifier after $ at {line}:{col}")
                while self._peek() == "." and (self._peek(1).isalnum() or self._peek(1) == "_"):
                    self._advance()
                    name += "." + self._read_ident()
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

            # Punctuation — '=' accepted as alias for ':' (common LLM slip)
            if ch in "{}[]:,=":
                self._advance()
                tok_ch = ":" if ch == "=" else ch
                tokens.append(Token(tok_ch, tok_ch, line, col))
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
        if tok.type != type_ or (value is not None and tok.value != value):
            want = f"{type_} {value!r}" if value is not None else type_
            raise ForgeParseError(
                f"Expected {want}, got {tok.type} {tok.value!r} at {tok.line}:{tok.col}"
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

        steps = self._parse_block()

        # The last VERIFY right before RETURN stays program.verify, keeping the
        # AST shape of single-VERIFY programs unchanged.
        verify = None
        if steps and isinstance(steps[-1].action, Verify):
            verify = steps.pop().action

        if not steps:
            raise ForgeParseError("At least one STEP is required")

        if not self._match_keyword("RETURN"):
            tok = self._cur()
            hint = " (END/ELSE without a matching FOR or IF)" if tok.value in ("END", "ELSE") else ""
            raise ForgeParseError(
                f"Expected RETURN, got {tok.type} {tok.value!r} at {tok.line}:{tok.col}{hint}"
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
            reason=None,  # REASON runs in-order as steps (no double-exec)
            verify=verify,
            return_stmt=return_stmt,
        )

    def _parse_block(self) -> list:
        """STEP / REASON / VERIFY / FOR / IF / YIELD in any order, until anything else."""
        steps: list[Step] = []
        while True:
            n = len(steps) + 1
            if self._match_keyword("STEP"):
                steps.append(self._parse_step())
            elif self._match_keyword("REASON"):
                # Mid-flow REASON becomes a step so later STEPs can use its output.
                steps.append(Step(step_name=f"reason_{n}", action=self._parse_reason()))
            elif self._match_keyword("VERIFY"):
                steps.append(Step(step_name=f"verify_{n}", action=self._parse_verify()))
            elif self._match_keyword("FOR"):
                steps.append(Step(step_name=f"for_{n}", action=self._parse_for()))
            elif self._match_keyword("PARALLEL"):
                self._advance()
                width = 8
                if self._cur().type == "NUMBER":
                    width = self._parse_count(MAX_PARALLEL, "PARALLEL", minimum=1)
                if not self._match_keyword("FOR"):
                    tok = self._cur()
                    raise ForgeParseError(f"Expected FOR EACH after PARALLEL at {tok.line}:{tok.col}")
                steps.append(Step(step_name=f"for_{n}", action=self._parse_for(parallel=width)))
            elif self._match_keyword("TRY"):
                steps.append(Step(step_name=f"try_{n}", action=self._parse_try()))
            elif self._match_keyword("IF"):
                steps.append(Step(step_name=f"if_{n}", action=self._parse_if()))
            elif self._match_keyword("YIELD"):
                self._advance()
                steps.append(Step(step_name=f"yield_{n}", action=Yield(value=self._parse_value())))
            else:
                return steps

    def _parse_try(self) -> Try:
        """TRY ... ON ERROR ... END   ($error holds the failure message)"""
        start = self._expect("KEYWORD", "TRY")
        body = self._parse_block()
        if not (self._match_keyword("ON") and self._peek(1).type == "KEYWORD" and self._peek(1).value == "ERROR"):
            tok = self._cur()
            raise ForgeParseError(
                f"Expected ON ERROR for TRY from {start.line}:{start.col}, got {tok.type} {tok.value!r} at {tok.line}:{tok.col}"
            )
        self._advance()
        self._advance()
        handler = self._parse_block()
        self._expect_end("TRY", start)
        if not body or not handler:
            raise ForgeParseError(f"TRY at {start.line}:{start.col} needs steps in both TRY and ON ERROR")
        return Try(body=body, handler=handler)

    def _parse_count(self, limit: int, what: str, minimum: int = 0) -> int:
        tok = self._expect("NUMBER")
        if not isinstance(tok.value, int) or not minimum <= tok.value <= limit:
            raise ForgeParseError(f"{what} needs a whole number {minimum}..{limit} at {tok.line}:{tok.col}")
        return tok.value

    def _parse_retry(self) -> int:
        """Optional `RETRY n` after a step's OUTPUT name."""
        if self._match_keyword("RETRY"):
            self._advance()
            return self._parse_count(MAX_RETRIES, "RETRY")
        return 0

    def _parse_for(self, parallel: int = 0) -> ForEach:
        """FOR EACH item IN $list [OUTPUT results] ... END"""
        start = self._expect("KEYWORD", "FOR")
        if self._match_keyword("EACH"):
            self._advance()
        tok = self._cur()
        if tok.type not in ("IDENT", "VARIABLE") or "." in str(tok.value):
            raise ForgeParseError(f"Expected loop variable name after FOR EACH at {tok.line}:{tok.col}")
        var = self._advance().value
        self._expect("KEYWORD", "IN")
        source = self._parse_value()
        output_var = None
        if self._match_keyword("OUTPUT"):
            self._advance()
            output_var = self._expect("IDENT").value
        body = self._parse_block()
        self._expect_end("FOR", start)
        if not body:
            raise ForgeParseError(f"FOR EACH at {start.line}:{start.col} has an empty body")
        return ForEach(var=var, source=source, body=body, output_var=output_var, parallel=parallel)

    def _parse_if(self) -> If:
        """IF condition ... [ELSE ...] END"""
        start = self._expect("KEYWORD", "IF")
        condition = self._parse_condition()
        then = self._parse_block()
        else_: list = []
        if self._match_keyword("ELSE"):
            self._advance()
            else_ = self._parse_block()
        self._expect_end("IF", start)
        if not then:
            raise ForgeParseError(f"IF at {start.line}:{start.col} has an empty body")
        return If(condition=condition, then=then, else_=else_)

    def _expect_end(self, opener: str, start: Token) -> None:
        if not self._match_keyword("END"):
            tok = self._cur()
            raise ForgeParseError(
                f"Expected END to close {opener} from {start.line}:{start.col}, "
                f"got {tok.type} {tok.value!r} at {tok.line}:{tok.col}"
            )
        self._advance()

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

    def _parse_step(self) -> Step:
        """
        STEP name TOOL ... | FILTER ... | REASON "..." ON $x OUTPUT y
        Also: STEP name "prompt" ON $x OUTPUT y  (AI shorthand for REASON)
        """
        self._expect("KEYWORD", "STEP")
        step_name = self._expect("IDENT").value

        if self._match_keyword("TOOL"):
            return Step(step_name=step_name, action=self._parse_tool_call())
        if self._match_keyword("FILTER"):
            return Step(step_name=step_name, action=self._parse_filter())
        if self._match_keyword("REASON") or self._cur().type == "STRING":
            if self._match_keyword("REASON"):
                self._advance()
            prompt = self._expect("STRING").value
            self._expect("KEYWORD", "ON")
            input_var = self._expect("VARIABLE").value
            self._expect("KEYWORD", "OUTPUT")
            output_var = self._expect("IDENT").value
            self._skip_optional_annotation()
            return Step(
                step_name=step_name,
                action=Reason(prompt=prompt, input_var=input_var, output_var=output_var,
                              retries=self._parse_retry()),
            )
        if self._match_keyword("RUN"):
            return Step(step_name=step_name, action=self._parse_run())

        raise ForgeParseError(
            f"Expected TOOL, FILTER, REASON or RUN after STEP name, got {self._cur().value!r} "
            f"at {self._cur().line}:{self._cur().col}"
        )

    def _parse_inputs(self) -> dict:
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
        return inputs

    def _parse_tool_call(self) -> ToolCall:
        self._expect("KEYWORD", "TOOL")
        tool_name = self._expect("IDENT").value
        inputs = self._parse_inputs()
        self._expect("KEYWORD", "OUTPUT")
        output_var = self._expect("IDENT").value
        self._skip_optional_annotation()
        return ToolCall(tool_name=tool_name, inputs=inputs, output_var=output_var,
                        retries=self._parse_retry())

    def _parse_run(self) -> RunProgram:
        """RUN "other.forge" [INPUT { k: $v }] OUTPUT x [RETRY n]"""
        self._expect("KEYWORD", "RUN")
        path = self._expect("STRING").value
        inputs = self._parse_inputs() if self._match_keyword("INPUT") else {}
        self._expect("KEYWORD", "OUTPUT")
        output_var = self._expect("IDENT").value
        return RunProgram(path=path, inputs=inputs, output_var=output_var, retries=self._parse_retry())

    def _parse_filter(self) -> Filter:
        self._expect("KEYWORD", "FILTER")
        condition = self._parse_condition()
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
        return Reason(prompt=prompt, input_var=input_var, output_var=output_var,
                      retries=self._parse_retry())

    def _skip_optional_annotation(self) -> None:
        """AIs sometimes append a label string after OUTPUT var — ignore it."""
        if self._cur().type == "STRING":
            self._advance()

    def _parse_verify(self) -> Verify:
        self._expect("KEYWORD", "VERIFY")
        condition = self._parse_condition()
        return Verify(condition=condition)

    def _parse_return(self) -> Return:
        self._expect("KEYWORD", "RETURN")
        value = self._parse_value()
        return Return(value=value)

    def _parse_condition(self) -> Any:
        """comparison [AND|OR comparison]...  — AND binds tighter than OR."""
        left = self._parse_and()
        while self._match_keyword("OR"):
            self._advance()
            left = Logical(operator="OR", left=left, right=self._parse_and())
        return left

    def _parse_and(self) -> Any:
        left = self._parse_comparison()
        while self._match_keyword("AND"):
            self._advance()
            left = Logical(operator="AND", left=left, right=self._parse_comparison())
        return left

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
        if op_tok.type == "KEYWORD" and op_tok.value == "CONTAINS":
            self._advance()
            return Comparison(left=left, operator="CONTAINS", right=self._parse_value(), left_kind=left_kind)
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
        if tok.type == "[":
            return self._parse_array()
        raise ForgeParseError(
            f"Expected value, got {tok.type} {tok.value!r} at {tok.line}:{tok.col}"
        )

    def _parse_array(self) -> ArrayLiteral:
        self._expect("[")
        items: list[ASTNode] = []
        while self._cur().type != "]":
            if self._cur().type == "EOF":
                raise ForgeParseError("Unterminated list literal")
            items.append(self._parse_value())
            if self._cur().type == ",":
                self._advance()
        self._expect("]")
        return ArrayLiteral(items=items)

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
    "deal-triage": '''
AGENT "deal-triage"

MEMORY {
  region: "north"
  watchlist: ["Acme", "Gamma Inc"]
}

STEP get TOOL sales_data INPUT { region: $region, period: "Q4" } OUTPUT deals

VERIFY $deals != []

FOR EACH d IN $deals OUTPUT flagged
  IF $d.amount > 10000 AND $watchlist CONTAINS $d.deal
    REASON "In one line, why does this deal need attention?" ON $d OUTPUT why
    YIELD { deal: $d.deal, amount: $d.amount, why: $why }
  END
END

STEP top TOOL sort INPUT { list: $flagged, by: "amount", desc: true } OUTPUT ranked
STEP total TOOL sum INPUT { list: $flagged, field: "amount" } OUTPUT at_stake

RETURN { flagged: $ranked, at_stake: $at_stake }
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
