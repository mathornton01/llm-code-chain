FROM python:3.11-slim

WORKDIR /app

# Install system deps + curl for Ollama install
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates zstd \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-pull the smallest model during build so startup is fast
RUN ollama serve & sleep 3 && ollama pull qwen2.5:1.5b && kill %1 2>/dev/null || true

ENV OLLAMA_URL=http://127.0.0.1:11434

EXPOSE 5050

COPY start.sh .
RUN chmod +x start.sh
CMD ["./start.sh"]
