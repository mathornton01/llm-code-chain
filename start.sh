#!/bin/bash
# Start Ollama in background, then the web server

echo "=== Prefrontal Compressor Startup ==="
echo "Date: $(date)"
grep -E 'MemTotal|MemAvailable|MemFree' /proc/meminfo 2>/dev/null || echo "Memory info unavailable"

export OLLAMA_MODELS="${OLLAMA_MODELS:-/app/ollama_models}"
export OLLAMA_HOST="${OLLAMA_HOST:-0.0.0.0:11434}"

# Force correct OLLAMA_URL regardless of env
export OLLAMA_URL="http://127.0.0.1:11434"

echo "OLLAMA_URL=$OLLAMA_URL"
echo "OLLAMA_HOST=$OLLAMA_HOST"
echo "OLLAMA_MODELS=$OLLAMA_MODELS"

# Verify binary
echo "Ollama binary: $(which ollama 2>/dev/null || echo 'NOT FOUND')"
ollama --version 2>/dev/null || echo "WARNING: ollama --version failed"

# Check pre-pulled models
echo "Model dir size: $(du -sh $OLLAMA_MODELS 2>/dev/null || echo 'empty')"

# Start Ollama
echo "Starting Ollama serve..."
ollama serve > /tmp/ollama.log 2>&1 &
OLLAMA_PID=$!
echo "Ollama PID: $OLLAMA_PID"

# Wait for it
for i in $(seq 1 60); do
    if curl -sf http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
        echo "Ollama ready after ${i}s"
        break
    fi
    if ! kill -0 $OLLAMA_PID 2>/dev/null; then
        echo "ERROR: Ollama died after ${i}s"
        echo "--- Ollama Log ---"
        cat /tmp/ollama.log 2>/dev/null
        echo "--- End Log ---"
        break
    fi
    sleep 1
done

# Check what we have
if curl -sf http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then
    echo "=== Ollama OK ==="
    ollama list 2>/dev/null || true

    # Pull model if not present (baked in during build, but just in case)
    if ! ollama list 2>/dev/null | grep -q "qwen2.5"; then
        echo "Model not found, pulling qwen2.5:1.5b..."
        ollama pull qwen2.5:1.5b 2>&1 &
    fi
else
    echo "WARNING: Ollama not responding"
    echo "--- Ollama Log ---"
    cat /tmp/ollama.log 2>/dev/null
    echo "--- End Log ---"
fi

echo "Starting gunicorn on port ${PORT:-5050}..."
exec gunicorn server:app --bind 0.0.0.0:${PORT:-5050} --workers 2 --timeout 120
