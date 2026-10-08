"""
Forge test suite — stdlib only.

  python3 -m unittest discover -s tests -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from forge_core import (  # noqa: E402
    EXAMPLES,
    ForgeParseError,
    Verify,
    _ast_to_dict,
    check_program,
    compile_forge,
    compile_forge_json,
)
from forge_repair import repair_forge, try_compile_repaired  # noqa: E402
from forge_runtime import (  # noqa: E402
    ForgeCheckError,
    ForgeRuntimeError,
    ForgeVerifyError,
    MockLLMClient,
    ToolRegistry,
    run_forge,
    run_program,
)


def run(src: str, tools: ToolRegistry | None = None):
    return run_forge(src, tools=tools or ToolRegistry(), llm=MockLLMClient())


class CountingTools(ToolRegistry):
    """Registry that records every tool call."""

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def call(self, name, inputs):
        self.calls.append(name)
        return super().call(name, inputs)


class TestExamples(unittest.TestCase):
    def test_builtin_examples_parse_roundtrip_and_run(self):
        for name, src in EXAMPLES.items():
            with self.subTest(name=name):
                prog = compile_forge(src)
                again = compile_forge_json(_ast_to_dict(prog))
                self.assertEqual(_ast_to_dict(prog), _ast_to_dict(again))
                run(src)

    def test_example_files_run(self):
        tools = ToolRegistry()
        # Offline stand-in for the network tool
        tools.register("http_get", lambda i: {"status": 200, "url": i["url"], "body": "<h1>Example</h1>"})
        cwd = os.getcwd()
        os.chdir(ROOT)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                for path in sorted((ROOT / "examples").glob("*.forge")):
                    with self.subTest(file=path.name):
                        src = path.read_text(encoding="utf-8")
                        if "write_file" in src:
                            os.chdir(tmp)  # keep the repo clean
                        try:
                            run(src, tools)
                        finally:
                            os.chdir(ROOT)
        finally:
            os.chdir(cwd)

    def test_basic_calculator_result(self):
        self.assertEqual(run(EXAMPLES["basic-calculator"])["result"], 22)


class TestFieldAccess(unittest.TestCase):
    SRC = '''
AGENT "fields"
MEMORY { q: "x" }
STEP s TOOL web_search INPUT { q: $q, n: 3 } OUTPUT hits
RETURN { first: $hits.0.title, url: $hits.2.url }
'''

    def test_dict_and_list_paths(self):
        out = run(self.SRC)
        self.assertEqual(out["first"], "Result 1 for 'x'")
        self.assertEqual(out["url"], "https://example.com/r/3")

    def test_path_in_reason_and_verify(self):
        out = run('''
AGENT "p"
MEMORY { q: "x" }
STEP s TOOL web_search INPUT { q: $q, n: 2 } OUTPUT hits
VERIFY $hits.0.relevance > 0.9
REASON "Summarize" ON $hits.0.snippet OUTPUT summary
RETURN $summary
''')
        self.assertIn('data="Snippet about x (#1)"', out)

    def test_missing_field_lists_available_fields(self):
        with self.assertRaisesRegex(ForgeRuntimeError, r"has no field 'nope'.*title"):
            run(self.SRC.replace("$hits.0.title", "$hits.0.nope"))

    def test_index_out_of_range(self):
        with self.assertRaisesRegex(ForgeRuntimeError, "out of range"):
            run(self.SRC.replace("$hits.0.title", "$hits.9.title"))

    def test_path_survives_json_roundtrip(self):
        prog = compile_forge(self.SRC)
        out = run_program(compile_forge_json(_ast_to_dict(prog)), ToolRegistry(), MockLLMClient())
        self.assertEqual(out["first"], "Result 1 for 'x'")


class TestVerify(unittest.TestCase):
    def test_multiple_verify(self):
        src = '''
AGENT "v"
MEMORY { a: 1 }
STEP add TOOL arithmetic_add INPUT { x: $a, y: $a } OUTPUT sum
VERIFY $sum > 0
VERIFY $sum < 10
RETURN $sum
'''
        self.assertEqual(run(src), 2)
        with self.assertRaises(ForgeVerifyError):
            run(src.replace("$sum < 10", "$sum < 2"))

    def test_mid_flow_verify_stops_before_later_steps(self):
        tools = CountingTools()
        src = '''
AGENT "early"
MEMORY { a: 1 }
STEP add TOOL arithmetic_add INPUT { x: $a, y: $a } OUTPUT sum
VERIFY $sum > 100
STEP search TOOL web_search INPUT { q: "costly" } OUTPUT hits
RETURN $hits
'''
        with self.assertRaises(ForgeVerifyError):
            run(src, tools)
        self.assertEqual(tools.calls, ["arithmetic_add"])

    def test_single_trailing_verify_keeps_ast_shape(self):
        prog = compile_forge(EXAMPLES["basic-calculator"])
        self.assertIsInstance(prog.verify, Verify)
        self.assertFalse(any(isinstance(s.action, Verify) for s in prog.steps))

    def test_verify_alone_is_not_a_step(self):
        with self.assertRaisesRegex(ForgeParseError, "At least one STEP"):
            compile_forge('AGENT "x"\nMEMORY { a: 1 }\nVERIFY $a > 0\nRETURN $a')


class TestStaticCheck(unittest.TestCase):
    def test_undefined_var_fails_before_any_tool_call(self):
        tools = CountingTools()
        src = '''
AGENT "bad"
STEP s TOOL web_search INPUT { q: "x" } OUTPUT hits
RETURN { r: $result }
'''
        with self.assertRaisesRegex(ForgeCheckError, r"\$result used before it is defined"):
            run(src, tools)
        self.assertEqual(tools.calls, [])

    def test_use_before_define(self):
        prog = compile_forge('''
AGENT "order"
STEP a TOOL arithmetic_add INPUT { x: $later, y: 1 } OUTPUT later
RETURN $later
''')
        self.assertEqual(len(check_program(prog)), 1)

    def test_unknown_tool(self):
        prog = compile_forge('AGENT "t"\nSTEP s TOOL launch_rocket INPUT {} OUTPUT r\nRETURN $r')
        errs = check_program(prog, ToolRegistry().names())
        self.assertTrue(errs and "unknown tool 'launch_rocket'" in errs[0])

    def test_filter_field_refs_are_not_vars(self):
        self.assertEqual(check_program(compile_forge(EXAMPLES["sales-analyzer"])), [])


class TestRepair(unittest.TestCase):
    def test_double_equals_preserved(self):
        fixed = repair_forge('AGENT "t"\nMEMORY { a = 1 }\nSTEP s TOOL get_value INPUT {} OUTPUT v\nVERIFY $v == 42\nRETURN $v')
        self.assertIn("VERIFY $v == 42", fixed)
        self.assertIn("a: 1", fixed)

    def test_strings_untouched(self):
        prompt = "Is a=b? Explain the step output and return value, then RETURN it"
        src = f'agent t\nmemory {{ a: 1 }}\nstep s tool get_value input {{}} output v\nreason "{prompt}" on $v output note\nreturn {{ n: $note }}'
        prog, _, notes = try_compile_repaired(src)
        self.assertEqual(notes, ["repaired"])
        self.assertEqual(prog.steps[1].action.prompt, prompt)

    def test_trailing_prose_and_fences(self):
        src = "Here you go:\n```forge\n" + EXAMPLES["basic-calculator"] + "```\nHope this helps!"
        prog, _, _ = try_compile_repaired(src)
        self.assertEqual(run_program(prog, ToolRegistry(), MockLLMClient())["result"], 22)

    def test_smart_quotes(self):
        prog, _, _ = try_compile_repaired(
            "AGENT “q”\nSTEP s TOOL get_value INPUT {} OUTPUT v\nRETURN $v"
        )
        self.assertEqual(prog.agent.name, "q")


class TestFileSandbox(unittest.TestCase):
    def setUp(self):
        self.cwd = os.getcwd()
        self.tmp = tempfile.TemporaryDirectory()
        os.chdir(self.tmp.name)

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def test_write_then_read(self):
        tools = ToolRegistry()
        tools.call("write_file", {"path": "out/a.txt", "body": "hi"})
        self.assertEqual(tools.call("read_file", {"path": "out/a.txt"})["body"], "hi")

    def test_escape_rejected(self):
        tools = ToolRegistry()
        for name, inputs in (
            ("read_file", {"path": "../etc/passwd"}),
            ("read_file", {"path": "/etc/passwd"}),
            ("write_file", {"path": "../evil.txt", "body": "x"}),
        ):
            with self.subTest(name=name, inputs=inputs):
                with self.assertRaisesRegex(ForgeRuntimeError, "under current directory"):
                    tools.call(name, inputs)


class TestCreditSandbox(unittest.TestCase):
    def test_generated_python_is_sandboxed(self):
        from forge_credit_test import eval_python

        ok, err, _ = eval_python("def run(tools, llm):\n    return {'x': open('/etc/hostname').read()}")
        self.assertFalse(ok)
        self.assertIn("open", err)
        ok, err, _ = eval_python("def run(tools, llm):\n    return {'x': ().__class__}")
        self.assertFalse(ok)

    def test_normal_python_still_runs(self):
        from forge_credit_test import eval_python

        ok, err, _ = eval_python(
            "def run(tools, llm):\n"
            "    deals = tools.call('sales_data', {'region': 'n', 'period': 'Q1'})\n"
            "    big = [d for d in deals if d['amount'] > 10000]\n"
            "    assert len(big) > 0\n"
            "    return {'total': sum(d['amount'] for d in big)}\n"
        )
        self.assertTrue(ok, err)


if __name__ == "__main__":
    unittest.main()
