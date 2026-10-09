# Security

Forge programs are often written by an AI and run by another AI agent, so the
runtime treats every program as untrusted input. This page says what Forge
protects against, what it deliberately does not, and the settings that tighten it.

## What a Forge program can do

A program can only call the tools in its registry. It cannot import code, run
shell commands or touch Python objects. With the built-in tools, that means:

| Capability | Limits |
|------------|--------|
| Read and write files (`read_file`, `write_file`) | Only inside the current working directory. Paths are resolved (including symlinks) before the check. Max 10 MB per file |
| Make web requests (`http_get`, `http_post`) | `http`/`https` only. Cloud metadata and link-local addresses are always blocked, including after redirects. Responses capped at 10 MB |
| Run other programs (`RUN "x.forge"`) | `.forge`/`.json` files inside the working directory or the calling program's folder, nested at most 8 deep |
| Use an LLM (`REASON`) | The backend the operator configured (mock by default). Each REASON sends at most 40,000 characters |
| Run for a while | Step budget (default 10,000; `--max-steps`) and optional time budget (`--timeout`; the MCP server defaults to 300 s). Budgets are shared by parallel iterations and `RUN` children, and `TRY`/`RETRY` can't catch them |

Before anything runs, a static check rejects unknown tools and undefined variables.

## Hardening options

| Setting | Effect |
|---------|--------|
| `FORGE_HTTP_BLOCK_PRIVATE=1` | Also block loopback and private networks (10/8, 172.16/12, 192.168/16, ::1, …). Use this when programs come from untrusted sources and the machine can reach internal services |
| `FORGE_REASON_MAX_CHARS` | Lower the per-REASON context cap |
| `--max-steps`, `--timeout` | Tighter budgets for `forge run` and `forge mcp` |
| Run in a container or VM | The working directory is the file sandbox, so start Forge in a directory that holds only what programs should see |

## What is out of scope

- **`--tools my_tools.py` runs your Python.** Custom tools are trusted code, with
  whatever access they implement.
- **DNS rebinding.** The URL check resolves the host before connecting; a hostile
  DNS server could answer differently on the second lookup. Use
  `FORGE_HTTP_BLOCK_PRIVATE=1` plus network-level egress rules if that matters.
- **Behind an HTTP proxy**, the proxy resolves hostnames, so only literal IP
  addresses and known metadata hostnames can be checked locally.
- **`forge credit-test` executes model-written Python** to grade the Python leg.
  It removes imports, private and frame attributes, and most builtins, but a
  Python-in-Python sandbox is best effort. Run the credit test in a container.
- **The website playground** runs Forge entirely in your browser (Pyodide) and
  loads the runtime from this repository's `main` branch.
- **Regular expressions** in `regex_find` can't be interrupted mid-match; input
  is capped at 1,000 characters of pattern and 1,000,000 of text.

## Reporting a vulnerability

Please open a private security advisory on GitHub
(**Security → Report a vulnerability** on the repository page) rather than a
public issue. Include a program or steps that reproduce the problem.
