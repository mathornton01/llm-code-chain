#!/bin/bash
# Start Ollama in background, then the web server

echo "Starting Ollama..."
ollama serve &
OLLAMA_PID=$!

# Wait for Ollama to be ready
for i in $(seq 1 30); do
    if curl -s http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama ready."
        break
    fi
    sleep 1
done

# Ensure model is available (should already be pulled from build)
ollama list | grep -q "qwen2.5:1.5b" || ollama pull qwen2.5:1.5b

echo "Starting LLM Code Chain server on port ${PORT:-5050}..."
exec python -m gunicorn server:app --bind 0.0.0.0:${PORT:-5050} --workers 2 --timeout 120
