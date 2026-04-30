FROM python:3.11-slim

WORKDIR /app

# Install system deps including zstd for Ollama
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates zstd procps \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama -- download the tar.zst archive and extract
RUN curl -fsSL -o /tmp/ollama.tar.zst https://github.com/ollama/ollama/releases/download/v0.22.0/ollama-linux-amd64.tar.zst \
    && zstd -d /tmp/ollama.tar.zst -o /tmp/ollama.tar \
    && tar xf /tmp/ollama.tar -C /usr --strip-components=0 \
    && rm -f /tmp/ollama.tar.zst /tmp/ollama.tar \
    && ollama --version

# Set model storage location
ENV OLLAMA_MODELS=/app/ollama_models
RUN mkdir -p /app/ollama_models

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-pull model during build so it's baked into the image
RUN ollama serve & \
    sleep 5 && \
    for i in $(seq 1 30); do \
        curl -sf http://127.0.0.1:11434/api/tags > /dev/null 2>&1 && break; \
        sleep 1; \
    done && \
    echo "Pulling qwen2.5:1.5b..." && \
    ollama pull qwen2.5:1.5b && \
    echo "Model pulled:" && \
    ollama list && \
    pkill ollama; \
    sleep 2; \
    echo "Build-time pull complete"

RUN chmod +x start.sh

ENV OLLAMA_URL=http://127.0.0.1:11434
ENV OLLAMA_HOST=0.0.0.0:11434

EXPOSE 5050

CMD ["./start.sh"]
