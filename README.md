# LLM Code Chain

A compression-style encoding system that uses small local LLMs to establish a shared "code" 
between encoder and decoder instances. The goal: take a natural language prompt and encode it 
into the minimal number of tokens that can exchange the maximal amount of information.

## Architecture

```
[Natural Language Prompt]
        |
   [Encoder LLM] -- learns to compress using codebook
        |
   [Compact Code] -- minimal token representation
        |
   [Decoder LLM] -- reconstructs meaning from code
        |
[Reconstructed Meaning]
```

## Components

- **Codebook**: A negotiated mapping of concepts/phrases to short codes
- **Encoder**: Compresses prompts using codebook + LLM understanding
- **Decoder**: Reconstructs meaning from compressed codes
- **Trainer**: Iterative loop where encoder/decoder negotiate better codes
- **Server**: Flask API for encoding/decoding via HTTP
- **CLI**: Command-line interface for interactive use

## Quick Start

```bash
# Run the CLI
/usr/bin/python3 cli.py

# Run the API server
/usr/bin/python3 server.py

# Run the training loop to improve compression
/usr/bin/python3 train.py
```

## Model

Uses TinyLlama 1.1B (Q4_K_M quantization) via llama-cpp-python.
