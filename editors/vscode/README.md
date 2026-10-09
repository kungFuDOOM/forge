# Forge for VS Code

Syntax highlighting, comment toggling, bracket matching, auto-indent and folding
(`FOR` / `IF` / `TRY` … `END`) for `.forge` files.

## Install

Copy this folder into your VS Code extensions directory and restart VS Code:

```bash
# macOS / Linux
cp -r editors/vscode ~/.vscode/extensions/kungfudoom.forge-lang-0.4.0
# Windows (PowerShell)
Copy-Item -Recurse editors\vscode "$env:USERPROFILE\.vscode\extensions\kungfudoom.forge-lang-0.4.0"
```

Or build a `.vsix` and install that: `npx @vscode/vsce package` in this folder, then
**Extensions → … → Install from VSIX**.

The same grammar (`syntaxes/forge.tmLanguage.json`) works in any TextMate-compatible
editor, such as Sublime Text, Zed and JetBrains IDEs (via TextMate bundles), and in
GitHub-style highlighters such as Shiki.
