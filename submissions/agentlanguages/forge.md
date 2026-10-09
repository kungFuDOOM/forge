---
name: Forge
camp: orchestration
spans_camps: [syntactic]
one_liner: "Small agent-workflow language run through an MCP tool: the agent submits one program that chains tool calls, loops, parallel fetches and retries instead of making one tool call per step. Text syntax or JSON AST; Python standard library only."
url: https://kungfudoom.github.io/forge/
repo: kungFuDOOM/forge
paper: null
author: kungFuDOOM
implementation_language: Python
compilation_target: Tree-walking interpreter (CPython; also Pyodide in the browser)
license: MIT
maturity: early_implementation
date_appeared: 2026-09
agent_tooling:
  - MCP server (forge_run, forge_check)
  - Claude Code plugin with SKILL.md
  - "`forge spec`: compact language and tool reference for context injection"
  - JSON AST input mode
  - Pre-run static check (unknown tools, undefined variables) with errors that list the valid names
  - Repair pass for near-miss model output
key_idea: |
  Forge targets the round-trip cost of tool calling: each tool call re-sends the
  agent's conversation, so a six-step job pays for its context six times. The
  agent instead submits one program to a single MCP tool, and the program can only
  call the tools registered with the runtime. Capability is limited by what the
  language can express rather than by a sandbox around a general-purpose
  language, so the runtime needs no isolation layer to run model-written code.
crossrefs:
  - slug: quasar
    name: Quasar
    camp: orchestration
    relation: "Same target, opposite bet on the surface language. Quasar keeps a Python subset the model already knows and adds parallelisation and approval gates underneath; Forge replaces the surface with a small keyword language and pays for it with a language reference in every prompt."
  - slug: pel
    name: Pel
    camp: orchestration
    relation: "Both make the grammar the capability boundary and pair it with automated error correction. Pel is Lisp-shaped, constrains generation against its grammar and uses Common Lisp-style restarts; Forge is an uppercase keyword language that repairs near-miss output after generation and feeds structured errors back to the model."
  - slug: plasm
    name: Plasm
    camp: orchestration
    relation: "Both expose a compact program surface to agents through an MCP server instead of raw tool schemas. Plasm compiles path expressions over typed API graphs into reviewable dry-run plans; Forge runs imperative workflows (loops, branches, retries) over a flat registry of untyped tools."
history:
  - when: "September 2026"
    what: "First release: a linear AGENT, MEMORY, STEP, REASON, VERIFY, RETURN pipeline with a JSON AST twin."
  - when: "October 2026"
    what: "MCP server, field access and lists, then loops, branches, PARALLEL, TRY/ON ERROR, RETRY and RUN, plus an in-browser playground."
---

## The thesis.

Forge treats the cost of an agent's tool use as a language-design problem. In the usual loop, an agent calls one tool, reads the result, and decides on the next call; every round trip re-sends the whole conversation and the full set of tool schemas. Forge collapses a multi-step job into one program submitted to a single MCP tool, `forge_run`, so the context is sent once and intermediate results stay inside the runtime. The same diagnosis drives "code mode" approaches that let models write Python or TypeScript in a sandbox. Forge's difference is that a program can only call the tools registered with the runtime: there is no general-purpose language underneath to contain, so there is no sandbox to run.

<p class="pullquote">The agent loop is the program.</p>

## What it looks like.

<div class="code-sample">
  <div class="code">
<pre><span class="kw">AGENT</span> <span class="str">"digest"</span>
<span class="kw">MEMORY</span> { urls: [<span class="str">"https://example.com"</span>, <span class="str">"https://www.iana.org"</span>] }
<span class="cm"># fetch every page at once; one failure doesn't stop the rest</span>
<span class="kw">PARALLEL FOR EACH</span> url <span class="kw">IN</span> <span class="sl">$urls</span> <span class="kw">OUTPUT</span> pages
  <span class="kw">TRY</span>
    <span class="kw">STEP</span> get <span class="kw">TOOL</span> http_get <span class="kw">INPUT</span> { url: <span class="sl">$url</span> } <span class="kw">OUTPUT</span> page <span class="kw">RETRY</span> <span class="num">2</span>
    <span class="kw">VERIFY</span> <span class="sl">$page.status</span> <span class="op">==</span> <span class="num">200</span>
    <span class="kw">YIELD</span> { url: <span class="sl">$url</span>, text: <span class="sl">$page.text</span> }
  <span class="kw">ON ERROR</span>
    <span class="kw">YIELD</span> { url: <span class="sl">$url</span>, error: <span class="sl">$error</span> }
  <span class="kw">END</span>
<span class="kw">END</span>
<span class="kw">STEP</span> save <span class="kw">TOOL</span> write_file <span class="kw">INPUT</span> { path: <span class="str">"out/pages.json"</span>, body: <span class="sl">$pages</span> } <span class="kw">OUTPUT</span> saved
<span class="kw">RETURN</span> { saved: <span class="sl">$saved.path</span>, pages: <span class="sl">$pages</span> }</pre>
  </div>
  <p class="caption">One <code>forge_run</code> call: parallel fetches with retries, an early check, a per-item fallback and a file write. With one tool call per step, the same job is a fetch round trip per URL plus a write.</p>
</div>

## Distinctive moves.

- **Grammar as the capability boundary.** A program can call registered tools, branch, loop and return values, and nothing else. The built-in file tools are confined to the working directory, and web requests refuse cloud-metadata and link-local addresses, including after redirects.
- **Checked before anything runs.** A static pass rejects unknown tools and variables used before they are defined, with scope rules for loop bodies, branches and `TRY` handlers. Errors list what is defined or available, so the model has the valid names in hand when it retries.
- **Dual representation.** Every program has a text form and a JSON AST form with the same semantics, for models that emit structured output more reliably than syntax.
- **Repair for small models.** Before parsing, a deterministic pass normalises common near-misses in model output, such as lowercase keywords, markdown fences, trailing prose and `=` for `:`, without editing string literals.
- **Budgets the program can't escape.** Step and time budgets are shared across `PARALLEL` iterations and nested `RUN` calls, and `TRY` and `RETRY` cannot catch them.

## Maturity.

Early implementation. Single author, MIT-licensed, about a month old at cataloguing, written with heavy AI assistance (the README's own framing: "Human holds the vision. AI builds the language."). A Python 3.10+ interpreter with no third-party dependencies, a CLI, an MCP server and around 90 unit tests; the website runs the same interpreter in the browser through Pyodide. The token-savings claim is not yet backed by published numbers. The repository ships an agent benchmark comparing one call per step, Python code mode and Forge on the same tasks and model, but no results have been published. Earlier project-reported generation runs on a 1B local model predate most of the current language. The project's own benchmark notes record that the language reference in each prompt makes a Forge turn about 460 tokens more expensive than a Python code-mode turn, so Forge has to recover that cost in fewer or shorter turns. `REASON` steps call a configured LLM backend and fall back to a mock when none is set.

## Agent tooling.

The MCP server exposes `forge_run` and `forge_check`, with the language reference (about 800 tokens including the built-in tool list) embedded in the tool description, so a client needs no other documentation. A Claude Code plugin bundles the server with a `SKILL.md` that describes when one program beats several tool calls. `forge spec` prints the same reference for injection into other agents' context, and `forge ask` turns English into a program, feeding parse, check and runtime errors back to the model for up to two repair rounds. Custom tools are plain Python functions loaded with `--tools`, and their docstrings appear in the spec.
