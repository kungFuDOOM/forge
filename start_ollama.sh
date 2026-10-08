#!/usr/bin/env bash
# Start local Ollama server (user-space install, no sudo).
set -e
export PATH="${HOME}/.local/bin:${PATH}"
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "ollama not found at ~/.local/bin/ollama — reinstall needed."
  exit 1
fi

if curl -sf "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
  echo "Ollama already running at ${OLLAMA_HOST}"
  ollama list
  exit 0
fi

mkdir -p "${HOME}/.ollama"
nohup ollama serve > "${HOME}/.ollama/server.log" 2>&1 &
echo "Started ollama serve (PID $!)"
for i in 1 2 3 4 5 6 7 8 9 10; do
  if curl -sf "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
    echo "Ready."
    ollama list
    exit 0
  fi
  sleep 1
done
echo "Server did not become ready. Check ~/.ollama/server.log"
exit 1
