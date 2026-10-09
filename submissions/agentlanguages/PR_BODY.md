## New language submission

### Inclusion checklist

- [x] The project is designed for **LLMs or agents to author code** (not a tool that uses LLMs at runtime — chatbots, autocomplete plug-ins, code-completion models are out of scope)
- [x] One Markdown file added at `src/content/languages/forge.md` (the filename is the URL slug — lowercase, hyphen-separated, no spaces)
- [x] Frontmatter validates against the Zod schema in `src/content.config.ts` (checked with `astro build`: 44 pages built, the detail page renders)
- [x] One-liner is descriptive and neutral, not promotional
- [x] No star counts, fork counts, version numbers, or commit counts hardcoded
- [x] Self-classified camp with justification in the PR body below

### Project summary

- **Name:** Forge
- **URL:** https://kungfudoom.github.io/forge/
- **Repo:** kungFuDOOM/forge
- **Camp:** orchestration (secondary: syntactic)

### Justification

Forge is written by agents to orchestrate tool calls: an agent submits one program to an MCP tool (`forge_run`) instead of making one tool call per step, and the grammar is the capability boundary (a program can only call registered tools). It ships agent-facing tooling with the runtime: an MCP server, a Claude Code plugin with SKILL.md, a compact spec for context injection, a pre-run static check whose errors list valid names, and a repair pass for near-miss model output. The secondary syntactic camp reflects its token-oriented surface (uppercase keywords, `$variables`) and JSON-as-AST input mode.

The entry tries to be candid about maturity: single author with heavy AI assistance, about a month old, and no published real-model benchmark results yet. The project's own notes say its per-turn prompt cost is higher than Python code mode's.
