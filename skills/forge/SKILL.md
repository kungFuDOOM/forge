---
name: forge
description: Run multi-step tool work as ONE Forge program through the forge_run MCP tool instead of many separate tool calls. Use when a job has 3+ dependent steps or repeats work over a list, such as fetching several URLs, searching then filtering then summarizing, processing every item of a list, or chaining fetch, transform and write-file steps. Every separate tool call re-sends the whole conversation, so batching the steps into one program saves tokens and time.
---

# Forge: one call instead of many

Forge is a small agent language. A program chains tool calls, loops, checks and
file writes, and runs completely inside **one** `forge_run` call, so the
conversation is sent once instead of once per step.

## When to use it

- 3 or more tool steps where each step feeds the next
- The same work over a list (`FOR EACH`), especially network fetches (`PARALLEL FOR EACH`)
- Fetch, filter and sort, then save to a file
- Anything where you'd otherwise make a chain of calls just to move data around

Skip it for a single quick lookup, or when you need to see an intermediate
result before deciding what to do next.

## How

1. Write the program. The `forge_run` tool description contains the full
   language reference and the tool list.
2. Optionally call `forge_check` first: it reports syntax errors, undefined
   `$vars` and unknown tools without running anything.
3. Call `forge_run`. On an error, the message names what's defined or
   available; fix it and run again.

```
AGENT "digest"
MEMORY { urls: ["https://example.com", "https://www.iana.org"] }

PARALLEL FOR EACH url IN $urls OUTPUT pages
  TRY
    STEP get TOOL http_get INPUT { url: $url } OUTPUT page RETRY 2
    YIELD { url: $url, status: $page.status, text: $page.text }
  ON ERROR
    YIELD { url: $url, error: $error }
  END
END

STEP save TOOL write_file INPUT { path: "out/pages.json", body: $pages } OUTPUT saved
RETURN { saved: $saved.path, pages: $pages }
```

## Tips

- RETURN only what you need. `$page.text` is readable page text and much
  smaller than `$page.body` (raw HTML).
- REASON steps call the server's configured LLM (or a mock). When you are the
  reasoner, RETURN the data and reason over it yourself.
- VERIFY early (`VERIFY $page.status == 200`) to stop before costly steps.
- Files are limited to the working directory. Runs have step and time budgets.
