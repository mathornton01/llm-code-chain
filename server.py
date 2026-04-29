"""
Prefrontal Compressor -- LLM Code Chain v2

Each encoding layer uses a configurable Ollama model to semantically compress text.
The output is a transmission package: encoded text + decode manifest (codebook,
layer configs, reconstruction instructions) so the receiving end can decode.

Endpoints:
  GET  /              -- web GUI
  GET  /api           -- API info
  GET  /models        -- list available Ollama models
  POST /chain         -- multi-layer compression with per-layer model config
  POST /decode        -- decode a transmission package back to natural language
  POST /submit        -- submit package to external LLM (Claude/GPT/Grok)
  GET  /codebook      -- view current codebook
  GET  /stats         -- compression statistics
"""

from flask import Flask, request, jsonify, render_template
from codebook import Codebook
import os
import time
import json
import requests as http_req

app = Flask(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
codebook = Codebook()

# Stats tracking
stats = {
    "total_compressions": 0,
    "total_input_chars": 0,
    "total_output_chars": 0,
    "total_packages_sent": 0,
}


def ollama_generate(model: str, prompt: str, max_tokens: int = 128, temperature: float = 0.2) -> str:
    """Call Ollama generate API."""
    r = http_req.post(f"{OLLAMA_URL}/api/generate", json={
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": temperature,
        }
    }, timeout=120)
    r.raise_for_status()
    return r.json().get("response", "").strip()


def ollama_models() -> list:
    """Get available Ollama models."""
    try:
        r = http_req.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        models = r.json().get("models", [])
        return [{"name": m["name"], "size": m.get("size", 0)} for m in models]
    except Exception:
        return []


def compress_layer(text: str, model: str, layer_num: int, total_layers: int) -> dict:
    """
    Run one compression layer using an Ollama model.

    The model produces:
      1. A compressed representation of the text
      2. A decode hint: key mappings needed to reconstruct
    """
    start = time.time()

    prompt = f"""You are a text compressor. Compress the following text into the shortest possible form while preserving ALL meaning.

Rules:
- Use abbreviations, acronyms, symbols (+, >, =, &, |, @, #)
- Drop articles (a, an, the), filler words, redundant phrases
- Use single letters for common words: u=you, r=are, w/=with, b/c=because, ->= leads to, <=from
- Keep domain-specific terms but abbreviate them
- Keep numbers and proper nouns intact
- The compressed form must be unambiguously decodable

After the compressed text, on a new line starting with "KEYS:", list the non-obvious abbreviations you used as key=value pairs separated by |

TEXT: {text}

COMPRESSED:"""

    try:
        raw = ollama_generate(model, prompt, max_tokens=256, temperature=0.1)
    except Exception as e:
        # Fallback to rule-based compression
        compressed = _rule_compress(text)
        elapsed = (time.time() - start) * 1000
        return {
            "layer": layer_num,
            "model": model,
            "input": text,
            "output": compressed,
            "keys": {},
            "method": f"rule-fallback ({e})",
            "input_len": len(text),
            "output_len": len(compressed),
            "compression": round(1.0 - len(compressed) / max(1, len(text)), 3),
            "time_ms": round(elapsed, 1),
        }

    # Parse compressed text and keys
    compressed = raw
    keys = {}

    if "KEYS:" in raw:
        parts = raw.split("KEYS:", 1)
        compressed = parts[0].strip()
        key_str = parts[1].strip()
        for pair in key_str.split("|"):
            pair = pair.strip()
            if "=" in pair:
                k, v = pair.split("=", 1)
                keys[k.strip()] = v.strip()
    else:
        # Try to find keys on last line
        lines = raw.strip().split("\n")
        if len(lines) > 1 and ("=" in lines[-1] and "|" in lines[-1]):
            compressed = "\n".join(lines[:-1]).strip()
            for pair in lines[-1].split("|"):
                pair = pair.strip()
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    keys[k.strip()] = v.strip()
        else:
            compressed = raw.strip()

    # Clean up the compressed text
    for prefix in ["COMPRESSED:", "Compressed:", "Output:"]:
        if compressed.startswith(prefix):
            compressed = compressed[len(prefix):].strip()
    compressed = compressed.strip('"').strip("'").strip()

    # If compression actually expanded, fall back to rules
    if len(compressed) >= len(text):
        compressed = _rule_compress(text)
        keys = {}

    elapsed = (time.time() - start) * 1000

    return {
        "layer": layer_num,
        "model": model,
        "input": text,
        "output": compressed,
        "keys": keys,
        "method": "llm",
        "input_len": len(text),
        "output_len": len(compressed),
        "compression": round(1.0 - len(compressed) / max(1, len(text)), 3),
        "time_ms": round(elapsed, 1),
    }


def _rule_compress(text: str) -> str:
    """Fallback rule-based compression."""
    import re
    result = text.strip()
    removals = [
        "can you ", "could you ", "would you ", "please ",
        "i would like to ", "i want to ", "i need to ",
        " that ", " which ", " very ", " really ", " just ",
        " basically ", " essentially ", " simply ",
        " the ", " a ", " an ",
    ]
    r = " " + result + " "
    for filler in removals:
        r = r.replace(filler, " ")
    result = re.sub(r'\s+', ' ', r).strip()

    # Apply codebook
    words = result.split()
    coded = []
    for w in words:
        coded.append(codebook.encode_token(w))

    result = ' '.join(coded)
    result = result.replace(' and ', '&').replace(' or ', '|').replace(' to ', '>')
    return result.strip()


def decode_package(package: dict) -> str:
    """
    Decode a transmission package back to natural language.

    Uses the decode manifest (layer keys, models) to reconstruct
    by running each layer's model in reverse.
    """
    encoded = package.get("encoded", "")
    layers = package.get("layers", [])

    if not layers:
        return encoded

    # Reverse through layers, using each layer's model to reconstruct
    current = encoded
    decode_steps = []

    for layer_info in reversed(layers):
        model = layer_info.get("model", "qwen2.5:1.5b")
        keys = layer_info.get("keys", {})

        # Build key reference for the decoder
        key_ref = ""
        if keys:
            key_ref = "\nAbbreviation key: " + ", ".join(f"{k}={v}" for k, v in keys.items())

        prompt = f"""You are a text decompressor. Expand the following compressed text back into clear, natural English.

The text was compressed using abbreviations and symbols:
- Common: u=you, r=are, w/=with, b/c=because, ->=leads to, <=from
- Symbols: & = and, | = or, + = more/also, > = to/leads to, = equals
- Intent: Q=question, C=command, R=request, X=explain{key_ref}

COMPRESSED: {current}

Expand this into a clear, complete English sentence or paragraph:
EXPANDED:"""

        try:
            expanded = ollama_generate(model, prompt, max_tokens=256, temperature=0.3)
            for prefix in ["EXPANDED:", "Expanded:", "Output:"]:
                if expanded.startswith(prefix):
                    expanded = expanded[len(prefix):].strip()
            expanded = expanded.strip('"').strip("'").strip()
            if expanded:
                current = expanded
        except Exception:
            # Manual expansion from keys
            for k, v in keys.items():
                current = current.replace(k, v)

        decode_steps.append({
            "layer": layer_info.get("layer", 0),
            "model": model,
            "output": current,
        })

    return current


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/api", methods=["GET"])
def api_info():
    return jsonify({
        "service": "Prefrontal Compressor",
        "version": "2.0.0",
        "description": "Multi-layer LLM semantic compression with transmission packages",
    })


@app.route("/models", methods=["GET"])
def list_models():
    models = ollama_models()
    return jsonify({"models": models, "ollama_url": OLLAMA_URL})


@app.route("/chain", methods=["POST"])
def chain_encode():
    """
    Multi-layer compression chain.

    Each layer uses its configured Ollama model to semantically compress.
    Returns a transmission package with encoded text + full decode manifest.
    """
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"]
    layer_configs = data.get("layer_configs", [])
    num_layers = len(layer_configs) if layer_configs else min(max(int(data.get("layers", 3)), 1), 10)

    # Default layer configs if not provided
    if not layer_configs:
        default_model = data.get("model", "qwen2.5:1.5b")
        layer_configs = [{"model": default_model} for _ in range(num_layers)]

    total_start = time.time()
    layers = []
    current = text
    all_keys = {}

    for i, cfg in enumerate(layer_configs):
        model = cfg.get("model", "qwen2.5:1.5b")
        layer_result = compress_layer(current, model, i + 1, len(layer_configs))
        layers.append(layer_result)

        # Accumulate keys
        if layer_result["keys"]:
            for k, v in layer_result["keys"].items():
                all_keys[k] = v

        current = layer_result["output"]

    total_time = (time.time() - total_start) * 1000
    total_compression = 1.0 - (len(current) / max(1, len(text)))

    # Build the transmission package
    # This is what gets sent to the target LLM -- it includes everything needed to decode
    transmission_package = {
        "encoded": current,
        "manifest": {
            "version": "2.0",
            "original_length": len(text),
            "encoded_length": len(current),
            "compression_ratio": round(total_compression, 3),
            "num_layers": len(layers),
            "layers": [
                {
                    "layer": l["layer"],
                    "model": l["model"],
                    "keys": l["keys"],
                    "compression": l["compression"],
                }
                for l in layers
            ],
            "combined_keys": all_keys,
        },
    }

    # Decode to check fidelity
    fidelity = None
    final_decoded = None
    try:
        final_decoded = decode_package({
            "encoded": current,
            "layers": layers,
        })
        if final_decoded:
            orig_words = set(text.lower().split())
            dec_words = set(final_decoded.lower().split())
            key_orig = {w for w in orig_words if len(w) > 4}
            key_dec = {w for w in dec_words if len(w) > 4}
            fidelity = len(key_orig & key_dec) / max(1, len(key_orig))
    except Exception:
        pass

    # Update stats
    stats["total_compressions"] += 1
    stats["total_input_chars"] += len(text)
    stats["total_output_chars"] += len(current)

    return jsonify({
        "original": text,
        "layers": layers,
        "final_encoded": current,
        "final_decoded": final_decoded,
        "total_compression": round(total_compression, 3),
        "total_time_ms": round(total_time, 1),
        "fidelity": round(fidelity, 3) if fidelity is not None else None,
        "num_layers": len(layers),
        "transmission_package": transmission_package,
    })


@app.route("/decode", methods=["POST"])
def decode():
    """Decode a transmission package."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing request body"}), 400

    start = time.time()

    # Accept either a full package or just encoded text
    if "transmission_package" in data:
        pkg = data["transmission_package"]
        decoded = decode_package({
            "encoded": pkg["encoded"],
            "layers": pkg.get("manifest", {}).get("layers", []),
        })
    elif "encoded" in data and "layers" in data:
        decoded = decode_package(data)
    elif "code" in data:
        # Legacy compat
        decoded = decode_package({"encoded": data["code"], "layers": []})
    else:
        return jsonify({"error": "Missing 'transmission_package' or 'encoded' field"}), 400

    elapsed = (time.time() - start) * 1000

    return jsonify({
        "decoded": decoded,
        "time_ms": round(elapsed, 1),
    })


@app.route("/submit", methods=["POST"])
def submit_to_llm():
    """Submit a transmission package to an external LLM (Claude, GPT, or Grok)."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing request body"}), 400

    target = data.get("target", "claude")
    model = data.get("model", "")
    api_key = data.get("api_key", "")
    mode = data.get("mode", "package")  # package, decoded, original

    original = data.get("original", "")
    package = data.get("transmission_package", None)
    decoded_text = data.get("decoded", "")

    if not api_key:
        return jsonify({"error": "Missing API key"}), 400

    # Build the prompt based on mode
    if mode == "package" and package:
        # Send the encoded text + decode manifest so the LLM can decode
        encoded = package.get("encoded", "")
        manifest = package.get("manifest", {})
        combined_keys = manifest.get("combined_keys", {})

        key_section = ""
        if combined_keys:
            key_section = "\n\nDecode key:\n" + "\n".join(f"  {k} = {v}" for k, v in combined_keys.items())

        prompt = f"""The following message was compressed through a {manifest.get('num_layers', 1)}-layer semantic compressor.
Compression ratio: {manifest.get('compression_ratio', 0):.0%}
{key_section}

Compressed message: {encoded}

Please decode this compressed message and respond to it as if the user had sent the original uncompressed version."""

        system_msg = (
            "You are receiving a semantically compressed prompt. Use the provided decode key "
            "and context clues to reconstruct the original meaning, then respond helpfully."
        )
    elif mode == "decoded" and decoded_text:
        prompt = decoded_text
        system_msg = "Respond helpfully to the following message."
    elif mode == "original" and original:
        prompt = original
        system_msg = "Respond helpfully to the following message."
    else:
        return jsonify({"error": "No content to submit"}), 400

    start = time.time()

    try:
        if target == "claude":
            response_text, tokens = _call_claude(prompt, system_msg, model, api_key)
        elif target == "gpt":
            response_text, tokens = _call_openai(prompt, system_msg, model, api_key)
        elif target == "grok":
            response_text, tokens = _call_grok(prompt, system_msg, model, api_key)
        else:
            return jsonify({"error": f"Unknown target: {target}"}), 400

        latency = (time.time() - start) * 1000
        stats["total_packages_sent"] += 1

        return jsonify({
            "response": response_text,
            "target": target,
            "model": model,
            "tokens": tokens,
            "latency_ms": round(latency, 1),
            "mode": mode,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _call_claude(prompt, system_msg, model, api_key):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model or "claude-sonnet-4-20250514",
        max_tokens=2048,
        system=system_msg,
        messages=[{"role": "user", "content": prompt}],
    )
    text = message.content[0].text
    tokens = message.usage.output_tokens if hasattr(message, 'usage') else None
    return text, tokens


def _call_openai(prompt, system_msg, model, api_key):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model or "gpt-4o",
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 2048,
    }
    r = http_req.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    text = data["choices"][0]["message"]["content"]
    tokens = data.get("usage", {}).get("completion_tokens")
    return text, tokens


def _call_grok(prompt, system_msg, model, api_key):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model or "grok-3",
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 2048,
    }
    r = http_req.post("https://api.x.ai/v1/chat/completions", headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    text = data["choices"][0]["message"]["content"]
    tokens = data.get("usage", {}).get("completion_tokens")
    return text, tokens


@app.route("/codebook", methods=["GET"])
def get_codebook():
    return jsonify({"codes": codebook.codes, "stats": codebook.get_stats()})


@app.route("/stats", methods=["GET"])
def get_stats():
    return jsonify({
        "compression": dict(stats),
        "codebook": codebook.get_stats(),
        "ollama": {"url": OLLAMA_URL, "models": ollama_models()},
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"Prefrontal Compressor v2 starting on port {port}")
    print(f"Ollama: {OLLAMA_URL}")
    print(f"GUI: http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
