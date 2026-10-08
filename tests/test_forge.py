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
        tools.register("http_get", lambda i: {"status": 200, "url": i["url"], "body": "<h1>Example</h1>", "text": "Example"})
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



class TestLists(unittest.TestCase):
    def test_list_literals_in_memory_filter_and_return(self):
        out = run('''
AGENT "lists"
MEMORY {
  deals: [
    { name: "Acme", amount: 15000 },
    { name: "Beta", amount: 800 }
  ]
  tags: ["x", "y"]
}
STEP big FILTER amount > 1000 ON $deals OUTPUT big
RETURN { big: $big, first: $tags.0, pair: [$tags.1, 2] }
''')
        self.assertEqual(out, {"big": [{"name": "Acme", "amount": 15000}], "first": "x", "pair": ["y", 2]})

    def test_list_json_roundtrip(self):
        prog = compile_forge('AGENT "l"\nMEMORY { xs: [1, [2, 3]] }\nSTEP s TOOL get_value INPUT {} OUTPUT v\nRETURN $xs')
        again = compile_forge_json(_ast_to_dict(prog))
        self.assertEqual(run_program(again, ToolRegistry(), MockLLMClient()), [1, [2, 3]])

    def test_vars_inside_lists_are_checked(self):
        prog = compile_forge('AGENT "l"\nSTEP s TOOL get_value INPUT {} OUTPUT v\nRETURN [$v, $missing]')
        self.assertIn("$missing", check_program(prog)[0])


class TestRepairKeepsLists(unittest.TestCase):
    def test_multiline_list_survives_and_junk_is_dropped(self):
        src = """Sure! Here is the program:
agent list-demo
memory {
  items: [
    { name: "a", score: 0.9 },
    { name: "b", score: 0.2 }
  ]
  TOP PAPER 1
  limit = 0.5
}
step keep filter score > 0.5 on $items output good
verify $good.0.name == "a"
return { good: $good }
Hope that helps"""
        prog, _, _ = try_compile_repaired(src)
        self.assertEqual(run_program(prog, ToolRegistry(), MockLLMClient()), {"good": [{"name": "a", "score": 0.9}]})

    def test_one_line_memory_then_bare_keep_filter(self):
        prog, _, _ = try_compile_repaired(
            'AGENT "k"\nMEMORY { xs: [{v: 2}] }\nkeep filter v > 1 on $xs output out\nRETURN $out'
        )
        self.assertEqual(run_program(prog, ToolRegistry(), MockLLMClient()), [{"v": 2}])


class TestGenerateRetry(unittest.TestCase):
    def test_error_is_fed_back_and_fixed(self):
        from forge_generate import generate_and_run

        prompts: list[str] = []
        replies = iter([
            # attempt 1: uses an undefined variable
            'AGENT "a"\nSTEP s TOOL get_value INPUT {} OUTPUT v\nRETURN { v: $value }',
            # attempt 2: fixed
            'AGENT "a"\nSTEP s TOOL get_value INPUT {} OUTPUT v\nRETURN { v: $v }',
        ])

        def fake_llm(prompt):
            prompts.append(prompt)
            return next(replies), {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}

        out = generate_and_run("return the value", llm_call=fake_llm, retries=2)
        self.assertTrue(out["runtime_ok"], out.get("error"))
        self.assertEqual(out["result"], {"v": 42})
        self.assertEqual(out["attempts"], 2)
        self.assertEqual(out["usage"]["total_tokens"], 30)
        self.assertIn("$value used before it is defined", prompts[1])

    def test_gives_up_after_retries(self):
        from forge_generate import generate_and_run

        calls = []

        def bad_llm(prompt):
            calls.append(prompt)
            return "not forge at all", {}

        out = generate_and_run("x", llm_call=bad_llm, retries=1)
        self.assertFalse(out["parse_ok"])
        self.assertEqual(len(calls), 2)


class TestToolsFile(unittest.TestCase):
    def test_load_functions_and_docs(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "my_tools.py"
            f.write_text(
                "import json\n"
                "def word_count(inputs):\n"
                "    \"\"\"INPUT { text } → word count\"\"\"\n"
                "    return len(inputs['text'].split())\n"
                "def _private(inputs):\n    return 1\n"
            )
            tools = ToolRegistry()
            self.assertEqual(tools.load_file(str(f)), ["word_count"])
            self.assertEqual(tools.docs()["word_count"], "INPUT { text } → word count")
            out = run('AGENT "w"\nSTEP c TOOL word_count INPUT { text: "a b c" } OUTPUT n\nRETURN $n', tools)
            self.assertEqual(out, 3)

    def test_tools_dict_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "t.py"
            f.write_text("def helper(i):\n    return 1\nTOOLS = {'double': lambda i: i['x'] * 2}\n")
            tools = ToolRegistry()
            self.assertEqual(tools.load_file(str(f)), ["double"])
            self.assertNotIn("helper", tools.names())


class TestSpec(unittest.TestCase):
    def test_spec_lists_tools_and_syntax(self):
        from forge_runtime import language_spec

        spec = language_spec()
        for needle in ("AGENT", "VERIFY", "$var.field", "http_get", "write_file"):
            self.assertIn(needle, spec)
        self.assertLess(len(spec) // 4, 900, "spec should stay small enough to keep in context")


class TestMCP(unittest.TestCase):
    def setUp(self):
        from forge_mcp import ForgeMCPServer

        self.server = ForgeMCPServer(llm="mock")

    def call(self, method, params=None, msg_id=1):
        return self.server.handle({"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params or {}})

    def test_initialize_and_list(self):
        init = self.call("initialize", {"protocolVersion": "2025-06-18"})["result"]
        self.assertEqual(init["protocolVersion"], "2025-06-18")
        self.assertIn("tools", init["capabilities"])
        names = [t["name"] for t in self.call("tools/list")["result"]["tools"]]
        self.assertEqual(names, ["forge_run", "forge_check"])

    def test_run_and_check(self):
        res = self.call("tools/call", {"name": "forge_run", "arguments": {"source": EXAMPLES["basic-calculator"]}})["result"]
        self.assertFalse(res["isError"])
        self.assertIn('"result": 22', res["content"][0]["text"])
        res = self.call("tools/call", {"name": "forge_check", "arguments": {"source": 'AGENT "x"\nSTEP s TOOL nope INPUT {} OUTPUT r\nRETURN $r'}})["result"]
        self.assertTrue(res["isError"])
        self.assertIn("unknown tool 'nope'", res["content"][0]["text"])

    def test_errors_and_notifications(self):
        self.assertIsNone(self.server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}))
        self.assertEqual(self.call("nope")["error"]["code"], -32601)
        res = self.call("tools/call", {"name": "forge_run", "arguments": {"source": "garbage"}})["result"]
        self.assertTrue(res["isError"])

    def test_stdio_subprocess(self):
        import json
        import subprocess

        msgs = [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {
                "name": "forge_run", "arguments": {"source": EXAMPLES["sales-analyzer"], "llm": "mock"}}},
        ]
        proc = subprocess.run(
            [sys.executable, str(ROOT / "forge_cli.py"), "mcp", "--llm", "mock"],
            input="".join(json.dumps(m) + "\n" for m in msgs),
            capture_output=True, text=True, timeout=30,
        )
        lines = [json.loads(line) for line in proc.stdout.splitlines()]
        self.assertEqual([m["id"] for m in lines], [1, 2])
        self.assertIn("Gamma Inc", lines[1]["result"]["content"][0]["text"])


class TestChatCompletion(unittest.TestCase):
    def test_openai_compatible_http(self):
        import json
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        from forge_runtime import chat_completion

        seen = {}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                seen["path"] = self.path
                seen["auth"] = self.headers["Authorization"]
                seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                payload = json.dumps({
                    "choices": [{"message": {"content": "hello"}}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *a):
                pass

        httpd = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            text, usage = chat_completion(
                [{"role": "user", "content": "hi"}],
                api_key="k", base_url=f"http://127.0.0.1:{httpd.server_port}/v1", model="m",
                max_tokens=5, stop=["\n"],
            )
        finally:
            httpd.shutdown()
            httpd.server_close()
        self.assertEqual(text, "hello")
        self.assertEqual(usage["total_tokens"], 4)
        self.assertEqual(seen["path"], "/v1/chat/completions")
        self.assertEqual(seen["auth"], "Bearer k")
        self.assertEqual(seen["body"]["max_tokens"], 5)


class TestCLI(unittest.TestCase):
    def test_init_templates_check_clean(self):
        import contextlib
        import io

        from forge_cli import main

        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            for template in ("basic", "http", "file"):
                with self.subTest(template=template):
                    out = Path(tmp) / f"{template}.forge"
                    self.assertEqual(main(["init", "demo", "-o", str(out), "--template", template]), 0)
                    self.assertEqual(main(["check", str(out)]), 0)



TRIAGE = '''
AGENT "triage"
MEMORY { region: "north", watch: ["Acme", "Gamma Inc"] }
STEP get TOOL sales_data INPUT { region: $region, period: "Q4" } OUTPUT deals
FOR EACH d IN $deals OUTPUT flagged
  IF $d.amount > 10000 AND $watch CONTAINS $d.deal
    YIELD $d.deal
  ELSE
    IF $d.deal CONTAINS "DELTA" OR $d.amount < 0
      YIELD "small"
    END
  END
END
RETURN $flagged
'''


class TestLoopsAndBranches(unittest.TestCase):
    def test_for_if_else_yield_and_or_contains(self):
        self.assertEqual(run(TRIAGE), ["Acme", "Gamma Inc", "small"])

    def test_implicit_collect_without_yield(self):
        out = run('''
AGENT "x"
MEMORY { xs: [1, 2, 3] }
FOR EACH n IN $xs OUTPUT doubled
  STEP d TOOL calc INPUT { op: "mul", x: $n, y: 2 } OUTPUT twice
END
RETURN $doubled
''')
        self.assertEqual(out, [2, 4, 6])

    def test_nested_loops_and_loop_var_with_dollar(self):
        out = run('''
AGENT "grid"
MEMORY { rows: [1, 2], cols: [10, 20] }
FOR EACH $r IN $rows OUTPUT grid
  FOR EACH c IN $cols OUTPUT cells
    STEP s TOOL arithmetic_add INPUT { x: $r, y: $c } OUTPUT cell
  END
  YIELD $cells
END
RETURN $grid
''')
        self.assertEqual(out, [[11, 21], [12, 22]])

    def test_loop_over_object_gives_key_value(self):
        out = run('''
AGENT "kv"
MEMORY { o: { a: 1, b: 2 } }
FOR EACH e IN $o OUTPUT keys
  YIELD $e.key
END
RETURN $keys
''')
        self.assertEqual(out, ["a", "b"])

    def test_loop_body_is_scoped(self):
        src = '''
AGENT "scope"
MEMORY { xs: [1] }
FOR EACH n IN $xs OUTPUT out
  STEP s TOOL get_value INPUT {} OUTPUT inner
END
RETURN $inner
'''
        with self.assertRaisesRegex(ForgeCheckError, r"\$inner used before it is defined"):
            run(src)

    def test_if_defines_only_what_every_branch_defines(self):
        base = '''
AGENT "branch"
MEMORY { n: 5 }
IF $n > 3
  STEP a TOOL get_value INPUT {} OUTPUT v
{else}END
RETURN $v
'''
        with self.assertRaises(ForgeCheckError):
            run(base.replace("{else}", ""))
        self.assertEqual(run(base.replace("{else}", "ELSE\n  STEP b TOOL get_value INPUT {} OUTPUT v\n")), 42)

    def test_yield_outside_loop_rejected(self):
        with self.assertRaisesRegex(ForgeCheckError, "YIELD is only allowed inside FOR"):
            run('AGENT "y"\nSTEP s TOOL get_value INPUT {} OUTPUT v\nYIELD $v\nRETURN $v')

    def test_bare_name_in_if_is_checked(self):
        with self.assertRaisesRegex(ForgeCheckError, r"\$missing used before"):
            run('AGENT "y"\nSTEP s TOOL get_value INPUT {} OUTPUT v\nIF missing > 1\nYIELD 1\nEND\nRETURN $v')

    def test_parse_errors_name_the_problem(self):
        with self.assertRaisesRegex(ForgeParseError, "Expected END to close FOR"):
            compile_forge('AGENT "e"\nMEMORY { xs: [1] }\nFOR EACH x IN $xs\nSTEP s TOOL get_value INPUT {} OUTPUT v\nRETURN 1')
        with self.assertRaisesRegex(ForgeParseError, "without a matching FOR or IF"):
            compile_forge('AGENT "e"\nSTEP s TOOL get_value INPUT {} OUTPUT v\nEND\nRETURN 1')

    def test_json_roundtrip_with_loops(self):
        prog = compile_forge(TRIAGE)
        again = compile_forge_json(_ast_to_dict(prog))
        self.assertEqual(_ast_to_dict(again), _ast_to_dict(prog))
        self.assertEqual(run_program(again, ToolRegistry(), MockLLMClient()), ["Acme", "Gamma Inc", "small"])

    def test_filter_with_and(self):
        out = run('''
AGENT "f"
STEP get TOOL sales_data INPUT { region: "n", period: "Q1" } OUTPUT deals
STEP big FILTER amount > 6000 AND deal CONTAINS "co" ON $deals OUTPUT hits
RETURN $hits
''')
        self.assertEqual([d["deal"] for d in out], ["BetaCo"])

    def test_verify_failure_shows_value(self):
        with self.assertRaisesRegex(ForgeVerifyError, "left side was 42"):
            run('AGENT "v"\nSTEP s TOOL get_value INPUT {} OUTPUT v\nVERIFY $v > 100\nRETURN $v')

    def test_lowercase_loop_is_repaired(self):
        src = """agent triage
memory { region: "north", input: 3 }
step get tool sales_data input { region: $region, period: "Q4" } output end
for each d in $end output big
  if $d.amount > 10000 and $d.deal contains "a"
    yield $d.deal
  end
end
verify $input == 3
return { big: $big, n: $input }"""
        prog, _, _ = try_compile_repaired(src)
        self.assertEqual(run_program(prog, ToolRegistry(), MockLLMClient()), {"big": ["Acme", "Gamma Inc"], "n": 3})


class TestLimits(unittest.TestCase):
    SPIN = '''
AGENT "spin"
MEMORY { xs: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] }
FOR EACH a IN $xs
  FOR EACH b IN $xs
    STEP s TOOL get_value INPUT {} OUTPUT v
  END
END
RETURN 1
'''

    def test_step_budget(self):
        self.assertEqual(run_forge(self.SPIN, max_steps=1000), 1)
        with self.assertRaisesRegex(ForgeRuntimeError, "Step budget exceeded"):
            run_forge(self.SPIN, max_steps=50)

    def test_time_budget(self):
        tools = ToolRegistry()
        import time as _time

        tools.register("slow", lambda i: _time.sleep(0.05) or 1)
        src = self.SPIN.replace("TOOL get_value", "TOOL slow")
        with self.assertRaisesRegex(ForgeRuntimeError, "Time budget exceeded"):
            run_forge(src, tools=tools, max_seconds=0.2)

    def test_reason_context_is_capped(self):
        from forge_runtime import REASON_CONTEXT_CHARS

        seen = {}

        class Spy(MockLLMClient):
            def complete(self, prompt, context=None):
                seen["context"] = context
                return "ok"

        tools = ToolRegistry()
        tools.register("big", lambda i: "x" * (REASON_CONTEXT_CHARS + 500))
        run_forge('AGENT "b"\nSTEP s TOOL big INPUT {} OUTPUT t\nREASON "sum" ON $t OUTPUT r\nRETURN $r', tools=tools, llm=Spy())
        self.assertIn("truncated 500 chars", seen["context"])
        self.assertLess(len(seen["context"]), REASON_CONTEXT_CHARS + 100)


class TestDataTools(unittest.TestCase):
    def setUp(self):
        self.t = ToolRegistry()
        self.deals = [{"n": "a", "v": 3}, {"n": "b", "v": None}, {"n": "c", "v": 7}]

    def test_list_tools(self):
        t = self.t
        self.assertEqual(t.call("count", {"list": self.deals}), 3)
        self.assertEqual(t.call("pick", {"list": self.deals, "field": "n"}), ["a", "b", "c"])
        self.assertEqual([d["n"] for d in t.call("sort", {"list": self.deals, "by": "v", "desc": True})], ["c", "a", "b"])
        self.assertEqual(t.call("sort", {"list": [3, 1, 2], "limit": 2}), [1, 2])
        self.assertEqual(t.call("sum", {"list": self.deals, "field": "v"}), 10)
        self.assertEqual(t.call("join", {"list": ["a", "b"], "sep": ", "}), "a, b")

    def test_text_tools(self):
        t = self.t
        self.assertEqual(t.call("format", {"template": "{a}-{b}", "a": 1, "b": "x"}), "1-x")
        with self.assertRaisesRegex(ForgeRuntimeError, "template needs 'zz'"):
            t.call("format", {"template": "{zz}"})
        self.assertEqual(t.call("regex_find", {"text": "a1 b22", "pattern": r"\d+"}), ["1", "22"])
        self.assertEqual(t.call("json_parse", {"text": '{"k": [1]}'}), {"k": [1]})
        self.assertEqual(t.call("calc", {"op": "div", "x": 7, "y": 2}), 3.5)
        with self.assertRaisesRegex(ForgeRuntimeError, "calc div"):
            t.call("calc", {"op": "div", "x": 1, "y": 0})

    def test_extract_text(self):
        html = ("<html><head><title>T</title><style>x{}</style></head><body><h1>Big</h1>"
                "<script>evil()</script><p>Some <b>bold</b> &amp; more</p></body></html>")
        self.assertEqual(self.t.call("extract_text", {"html": html}), "T\nBig\nSome bold & more")

    def test_http_tools_against_local_server(self):
        import json
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                body = b"<html><title>Local</title><body><p>Hello <i>there</i></p><script>x</script></body></html>"
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                body = json.dumps({"echo": data}).encode()
                self.send_response(201)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        httpd = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{httpd.server_port}"
        try:
            page = self.t.call("http_get", {"url": base + "/"})
            posted = self.t.call("http_post", {"url": base + "/api", "json": {"a": 1}})
        finally:
            httpd.shutdown()
            httpd.server_close()
        self.assertEqual(page["text"], "Local\nHello there")
        self.assertEqual(posted["status"], 201)
        self.assertEqual(posted["json"], {"echo": {"a": 1}})

    def test_brave_search_when_key_set(self):
        import json
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer
        from unittest import mock

        seen = {}

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                seen["path"] = self.path
                seen["token"] = self.headers["X-Subscription-Token"]
                body = json.dumps({"web": {"results": [
                    {"title": "One", "url": "https://1", "description": "<strong>first</strong> hit"},
                    {"title": "Two", "url": "https://2", "description": "second"},
                ]}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *a):
                pass

        httpd = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        env = {"BRAVE_SEARCH_API_KEY": "tok", "FORGE_BRAVE_URL": f"http://127.0.0.1:{httpd.server_port}/search"}
        try:
            with mock.patch.dict(os.environ, env):
                hits = self.t.call("web_search", {"q": "forge lang", "n": 2})
        finally:
            httpd.shutdown()
            httpd.server_close()
        self.assertEqual(seen["token"], "tok")
        self.assertIn("q=forge+lang", seen["path"])
        self.assertEqual([h["title"] for h in hits], ["One", "Two"])
        self.assertEqual(hits[0]["snippet"], "first hit")
        self.assertGreater(hits[0]["relevance"], hits[1]["relevance"])



class TestWebsitePlayground(unittest.TestCase):
    """docs/index.html runs this repo's Python in the browser (Pyodide), loading it
    from main. Run its bridge code and examples here so a runtime change can't
    silently break the live site."""

    @staticmethod
    def _js_template(text: str) -> str:
        return text.replace("\\\\", "\\")  # JS template literal: \\ -> \

    def test_bridge_and_examples(self):
        import json
        import re

        page = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")
        bridge = self._js_template(re.search(r"const BRIDGE = `([\s\S]*?)`;", page).group(1))
        block = re.search(r"const EXAMPLES = \{([\s\S]*?)\n    \};", page).group(1)
        examples = {k: self._js_template(v) for k, v in re.findall(r'"([^"]+)": `([\s\S]*?)`,', block)}
        self.assertGreaterEqual(len(examples), 4)

        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                ns: dict = {}
                exec(compile(bridge, "playground-bridge", "exec"), ns)
                play = ns["forge_play"]
                for name, src in examples.items():
                    with self.subTest(example=name):
                        run = json.loads(play("run", src))
                        ast = json.loads(play("ast", src))
                        if "Errors" in name:
                            self.assertFalse(run["ok"])
                            check = json.loads(play("check", src))
                            self.assertIn("unknown tool 'web_serch'", check["output"])
                        else:
                            self.assertTrue(run["ok"], run["output"])
                            self.assertTrue(ast["ok"])
                offline = json.loads(play("run", 'AGENT "n"\nSTEP g TOOL http_get INPUT { url: "https://x" } OUTPUT p\nRETURN $p'))
                self.assertIn("browser playground", offline["output"])
            finally:
                os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()
