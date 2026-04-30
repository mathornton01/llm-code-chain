FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates zstd procps \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama -- download binary directly (more reliable in containers than install.sh)
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set model storage location explicitly
ENV OLLAMA_MODELS=/app/ollama_models
RUN mkdir -p /app/ollama_models

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-pull model during build
RUN OLLAMA_MODELS=/app/ollama_models ollama serve & \
    OLLAMA_PID=$! && \
    echo "Waiting for Ollama to start for model pull..." && \
    for i in $(seq 1 30); do \
        if curl -sf http://127.0.0.1:11434/api/tags > /dev/null 2>&1; then \
            echo "Ollama ready after ${i}s"; \
            break; \
        fi; \
        sleep 1; \
    done && \
    echo "Pulling qwen2.5:1.5b..." && \
    ollama pull qwen2.5:1.5b && \
    echo "Model pulled successfully" && \
    ollama list && \
    kill $OLLAMA_PID 2>/dev/null; \
    wait $OLLAMA_PID 2>/dev/null; \
    echo "Build-time Ollama stopped"

# Ensure start script is executable
RUN chmod +x start.sh

ENV OLLAMA_URL=http://127.0.0.1:11434
ENV OLLAMA_HOST=0.0.0.0:11434

EXPOSE 5050

CMD ["./start.sh"]
