#!/bin/bash
# Start Ollama in background, then the web server
set -e

echo "=== LLM Code Chain Startup ==="
echo "Starting Ollama daemon..."
ollama serve > /tmp/ollama.log 2>&1 &
OLLAMA_PID=$!
echo "Ollama PID: $OLLAMA_PID"

# Wait for Ollama to be ready (up to 60s)
echo "Waiting for Ollama to respond..."
for i in $(seq 1 60); do
    if curl -sf http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama ready after ${i}s."
        break
    fi
    if ! kill -0 $OLLAMA_PID 2>/dev/null; then
        echo "ERROR: Ollama process died. Log:"
        cat /tmp/ollama.log
        echo "Continuing without Ollama..."
        break
    fi
    sleep 1
done

# Check/pull models
echo "Checking models..."
if curl -sf http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
    MODELS=$(ollama list 2>/dev/null || echo "")
    echo "Current models: $MODELS"
    if ! echo "$MODELS" | grep -q "qwen2.5:1.5b"; then
        echo "Pulling qwen2.5:1.5b..."
        ollama pull qwen2.5:1.5b || echo "WARNING: Failed to pull model"
    fi
    echo "Ollama is operational with models:"
    ollama list 2>/dev/null || true
else
    echo "WARNING: Ollama not responding, server will start without it"
fi

echo "Starting LLM Code Chain server on port ${PORT:-5050}..."
exec gunicorn server:app --bind 0.0.0.0:${PORT:-5050} --workers 2 --timeout 120
