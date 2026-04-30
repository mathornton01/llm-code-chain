#!/bin/bash
# Start Ollama in background, then the web server

echo "=== Prefrontal Compressor Startup ==="
echo "Date: $(date)"
echo "Memory:"
grep -E 'MemTotal|MemAvailable|MemFree' /proc/meminfo 2>/dev/null || echo "  (unavailable)"

# Ensure model storage is consistent with build
export OLLAMA_MODELS="${OLLAMA_MODELS:-/app/ollama_models}"
export OLLAMA_HOST="${OLLAMA_HOST:-0.0.0.0:11434}"
export OLLAMA_URL="${OLLAMA_URL:-http://127.0.0.1:11434}"

# Fix empty OLLAMA_URL (Railway may set it to "")
if [ -z "$OLLAMA_URL" ]; then
    export OLLAMA_URL="http://127.0.0.1:11434"
fi

echo "OLLAMA_URL=$OLLAMA_URL"
echo "OLLAMA_HOST=$OLLAMA_HOST"
echo "OLLAMA_MODELS=$OLLAMA_MODELS"
echo "PORT=${PORT:-5050}"

# Check ollama binary
OLLAMA_BIN=$(which ollama 2>/dev/null)
echo "Ollama binary: $OLLAMA_BIN"
if [ -z "$OLLAMA_BIN" ]; then
    echo "ERROR: Ollama binary not found in PATH"
    echo "PATH=$PATH"
    ls -la /usr/local/bin/ollama 2>/dev/null || echo "Not at /usr/local/bin/ollama"
    ls -la /usr/bin/ollama 2>/dev/null || echo "Not at /usr/bin/ollama"
    echo "Starting server without Ollama..."
    exec gunicorn server:app --bind 0.0.0.0:${PORT:-5050} --workers 2 --timeout 120
fi

# Check pre-pulled models
echo "Model directory contents:"
ls -la "$OLLAMA_MODELS" 2>/dev/null || echo "  (empty or missing)"
du -sh "$OLLAMA_MODELS" 2>/dev/null || true

echo "Starting Ollama daemon..."
ollama serve > /tmp/ollama.log 2>&1 &
OLLAMA_PID=$!
echo "Ollama PID: $OLLAMA_PID"

# Wait for Ollama to be ready (up to 90s -- model loading can be slow)
echo "Waiting for Ollama to respond..."
OLLAMA_READY=false
for i in $(seq 1 90); do
    if curl -sf http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama ready after ${i}s"
        OLLAMA_READY=true
        break
    fi
    if ! kill -0 $OLLAMA_PID 2>/dev/null; then
        echo "ERROR: Ollama process died after ${i}s"
        echo "=== Ollama Log ==="
        cat /tmp/ollama.log 2>/dev/null
        echo "=== End Log ==="
        echo "Continuing without Ollama..."
        break
    fi
    # Print progress every 10s
    if [ $((i % 10)) -eq 0 ]; then
        echo "  Still waiting... ${i}s (PID $OLLAMA_PID alive: $(kill -0 $OLLAMA_PID 2>/dev/null && echo yes || echo no))"
    fi
    sleep 1
done

if [ "$OLLAMA_READY" = true ]; then
    echo "Checking models..."
    MODELS=$(ollama list 2>/dev/null || echo "(list failed)")
    echo "Current models: $MODELS"

    # Pull model if not present
    if ! echo "$MODELS" | grep -q "qwen2.5:1.5b"; then
        echo "Pulling qwen2.5:1.5b (this may take a few minutes)..."
        ollama pull qwen2.5:1.5b 2>&1 || echo "WARNING: Failed to pull model"
    fi

    echo "=== Ollama operational ==="
    ollama list 2>/dev/null || true
else
    if kill -0 $OLLAMA_PID 2>/dev/null; then
        echo "WARNING: Ollama process is alive but not responding after 90s"
        echo "=== Ollama Log (last 50 lines) ==="
        tail -50 /tmp/ollama.log 2>/dev/null
        echo "=== End Log ==="
    else
        echo "WARNING: Ollama process is dead"
    fi
fi

echo "Starting Prefrontal Compressor server on port ${PORT:-5050}..."
exec gunicorn server:app --bind 0.0.0.0:${PORT:-5050} --workers 2 --timeout 120
