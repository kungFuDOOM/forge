# Forge — JavaScript for AI

An **AI-native agent language**. Write a short program. Run it.  
Built for how AIs generate code — not 1990s human IDE habits.

**Repo:** https://github.com/kungFuDOOM/forge  
**Landing:** [docs/index.html](docs/index.html)

---

## 60-second start (usable)

```bash
git clone https://github.com/kungFuDOOM/forge.git
cd forge
chmod +x forge
./forge quickstart
```

That runs a demo and creates `my-first-agent.forge`. Then:

```bash
./forge run my-first-agent.forge
./forge check my-first-agent.forge
./forge tools
```

No API key required for demos (`--llm mock` is the default).

---

## Useful agents (real tools)

| Command | What it does |
|---------|----------------|
| `./forge run examples/url_fetcher.forge` | **HTTP GET** a URL, summarize |
| `./forge run examples/file_summarizer.forge` | **Read a file**, summarize |
| `./forge run examples/save_report.forge` | Research mock + **write file** |
| `./forge init blog --template http` | Scaffold an HTTP agent |

Built-in tools: `http_get`, `read_file`, `write_file`, `web_search`, `sales_data`, `arithmetic_add`, `get_value`

---

## Example

```
AGENT "url-fetcher"

MEMORY {
  url: "https://example.com"
}

STEP fetch TOOL http_get INPUT { url: $url } OUTPUT page

REASON "In one sentence, what is this page about?" ON $page OUTPUT summary

VERIFY $page != null

RETURN { url: $url, summary: $summary }
```

---

## Commands

```
./forge                  # welcome
./forge quickstart       # guided first run
./forge run FILE.forge   # execute
./forge check FILE.forge # validate
./forge init NAME        # new agent (--template basic|http|file)
./forge examples         # list/run demos
./forge tools            # tool reference
./forge tokens FILE      # rough token vs Python/LangChain
./forge doctor           # setup check
./forge repl             # interactive
```

Same via `python3 forge_cli.py …`.

---

## Optional: local LLM

```bash
./start_ollama.sh
ollama pull llama3.2:1b
./forge run my-first-agent.forge --llm ollama
python3 forge_generate.py --backend ollama "Add 3 and 9, return total"
```

---

## Design

- Primitives: `AGENT` `MEMORY` `STEP` `TOOL` `FILTER` `REASON` `VERIFY` `RETURN`
- Dual path: text syntax **or** JSON AST
- Pipeline: emit → repair → validate → run
- See [VISION.md](VISION.md) · [BENCH.md](BENCH.md)

## Reliability (local `llama3.2:1b`)

| Suite | Result |
|-------|--------|
| 5 tasks (text) | 100% |
| 18 tasks (text) | 77.8% |
| 18 tasks (JSON AST) | 0% on 1B (use 3B+ for JSON) |
