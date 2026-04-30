"""
Prefrontal Compressor v3 -- Multi-Strategy LLM Compression Chain

Three distinct layer types that each compress DIFFERENTLY:
  1. DISTILL  -- LLM rephrases to preserve meaning in fewer words
  2. REFERENCE -- Deterministic codebook substitution (fully reversible)
  3. COMPACT  -- LLM compresses syntax/structure of already-short text

The shared reference file is the key to decompression. Both encoder and
decoder must have the same version. It gets refined through training.

Endpoints:
  GET  /              -- web GUI
  GET  /api           -- API info
  GET  /models        -- list available Ollama models
  POST /chain         -- multi-layer compression
  POST /decode        -- decode a transmission package
  POST /submit        -- submit to external LLM (Claude/GPT/Grok)
  GET  /reference     -- view current reference file
  GET  /references    -- list all reference files
  POST /reference/new -- create a new reference file
  POST /reference/train -- run a training round
  GET  /stats         -- server stats
"""

from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from codebook import SharedReference, list_references
import os
import time
import json
import re
import requests as http_req

app = Flask(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")

# Active reference file
active_ref = SharedReference("default")

# Stats tracking
stats = {
    "total_compressions": 0,
    "total_input_chars": 0,
    "total_output_chars": 0,
    "total_packages_sent": 0,
}


def ollama_generate(model: str, prompt: str, max_tokens: int = 256, temperature: float = 0.2) -> str:
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


def commercial_generate(provider: str, model: str, api_key: str, prompt: str,
                        max_tokens: int = 256, temperature: float = 0.2) -> str:
    """Call a commercial LLM API (Claude, GPT, Grok) for compression layers."""
    if provider == "claude":
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=model or "claude-sonnet-4-20250514",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    elif provider == "gpt":
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model or "gpt-4o",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        r = http_req.post("https://api.openai.com/v1/chat/completions",
                          headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    elif provider == "grok":
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model or "grok-3",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        r = http_req.post("https://api.x.ai/v1/chat/completions",
                          headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    else:
        raise ValueError(f"Unknown provider: {provider}")


def generate_text(provider: str, model: str, prompt: str,
                  max_tokens: int = 256, temperature: float = 0.2,
                  api_key: str = "") -> str:
    """Unified text generation -- routes to Ollama or commercial APIs."""
    if provider in ("claude", "gpt", "grok"):
        if not api_key:
            raise ValueError(f"API key required for {provider}")
        return commercial_generate(provider, model, api_key, prompt, max_tokens, temperature)
    else:
        return ollama_generate(model, prompt, max_tokens, temperature)


def ollama_models() -> list:
    """Get available Ollama models."""
    try:
        r = http_req.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        models = r.json().get("models", [])
        return [{"name": m["name"], "size": m.get("size", 0)} for m in models]
    except Exception:
        return []


# ===== LAYER TYPES =====

def layer_distill(text: str, model: str, layer_num: int,
                   provider: str = "ollama", api_key: str = "",
                   temperature: float = 0.15) -> dict:
    """
    DISTILL layer: LLM rephrases the text to preserve meaning in fewer words.

    This is the FIRST layer -- works on natural language input.
    The LLM understands the content and produces a shorter version.
    Supports both Ollama and commercial models (Claude/GPT/Grok).
    """
    start = time.time()

    prompt = f"""You are a text compressor. Rewrite the input in as few characters as possible while keeping ALL meaning. Be extremely aggressive.

RULES:
1. Remove ALL filler: the, a, an, very, just, really, basically, actually, simply, please, could you, would you, I would like, I need you to
2. Abbreviate aggressively: function->fn, implement->impl, configure->cfg, information->info, development->dev, environment->env, database->db, authentication->auth, application->app, message->msg, request->req, response->res, parameter->param, argument->arg, variable->var, number->num, string->str, array->arr, object->obj, error->err, exception->ex, library->lib, package->pkg, dependency->dep, documentation->doc, repository->repo, directory->dir, network->net, because->bc, between->btwn, through->thru, without->w/o, with->w/, should->shd, would->wd, about->abt, before->b4, after->aft
3. Drop ALL articles, pronouns, copulas (is/are/was/were) where meaning survives
4. Merge short phrases: "in order to" -> "to", "as well as" -> "&", "due to the fact" -> "bc"
5. Keep technical terms and proper nouns intact
6. Output ONLY the compressed text. No quotes, labels, explanations, or prefixes.

INPUT: {text}
OUTPUT:"""

    # Word analysis before
    input_words = text.split()
    input_word_count = len(input_words)
    input_unique = len(set(w.lower().strip(".,!?;:") for w in input_words))

    try:
        raw = generate_text(provider, model, prompt, max_tokens=256,
                            temperature=temperature, api_key=api_key)
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        err_msg = str(e)
        # Make error message more helpful
        if "Connection" in err_msg or "timeout" in err_msg.lower():
            err_msg = f"Cannot reach {provider} ({model}). Is the service running? " + err_msg
        elif "API key" in err_msg or "401" in err_msg or "403" in err_msg:
            err_msg = f"Authentication failed for {provider}. Check your API key. " + err_msg
        return {
            "layer": layer_num,
            "type": "distill",
            "model": model,
            "input": text,
            "output": text,
            "method": f"error-passthrough",
            "error": err_msg,
            "input_len": len(text),
            "output_len": len(text),
            "compression": 0,
            "time_ms": round(elapsed, 1),
            "decode_info": {"type": "distill", "original": text},
            "detail": {
                "description": "LLM rephrases natural language to be shorter while preserving all key information.",
                "status": "error",
                "error": err_msg,
                "prompt_used": prompt,
                "input_words": input_word_count,
                "output_words": input_word_count,
            },
        }

    # Clean LLM output
    raw_before_clean = raw
    distilled = raw.strip()
    for prefix in ["SHORT VERSION:", "Short version:", "Shortened:", "Output:", "Result:", "OUTPUT:", "COMPRESSED:"]:
        if distilled.lower().startswith(prefix.lower()):
            distilled = distilled[len(prefix):].strip()
    distilled = distilled.strip('"').strip("'").strip()

    # If multi-line, take the shortest meaningful line
    if "\n" in distilled:
        lines = [l.strip() for l in distilled.split("\n") if l.strip()]
        if lines:
            min_len = max(3, len(text) * 0.15)
            candidates = [l for l in lines if len(l) >= min_len and len(l) < len(text)]
            if candidates:
                distilled = min(candidates, key=len)
            else:
                distilled = lines[0]

    # If LLM expanded instead of compressing, keep original but flag it
    kept_original = False
    if len(distilled) >= len(text) or not distilled:
        kept_original = True
        distilled = text

    elapsed = (time.time() - start) * 1000

    # Word analysis after
    output_words = distilled.split()
    output_word_count = len(output_words)
    output_unique = len(set(w.lower().strip(".,!?;:") for w in output_words))

    # Find which key words survived
    orig_key = {w.lower().strip(".,!?;:") for w in input_words if len(w) > 3}
    dist_key = {w.lower().strip(".,!?;:") for w in output_words if len(w) > 3}
    words_kept = sorted(orig_key & dist_key)
    words_lost = sorted(orig_key - dist_key)
    words_added = sorted(dist_key - orig_key)

    return {
        "layer": layer_num,
        "type": "distill",
        "model": model,
        "input": text,
        "output": distilled,
        "method": "llm-distill",
        "input_len": len(text),
        "output_len": len(distilled),
        "compression": round(1.0 - len(distilled) / max(1, len(text)), 3),
        "time_ms": round(elapsed, 1),
        "decode_info": {
            "type": "distill",
            "model": model,
            "original_text": text,
            "distilled_text": distilled,
        },
        "detail": {
            "description": "LLM rephrases natural language to be shorter while preserving all key information.",
            "status": "kept-original" if kept_original else "distilled",
            "prompt_used": prompt,
            "raw_llm_output": raw_before_clean,
            "input_words": input_word_count,
            "output_words": output_word_count,
            "input_unique_words": input_unique,
            "output_unique_words": output_unique,
            "words_reduction": input_word_count - output_word_count,
            "key_words_kept": words_kept,
            "key_words_lost": words_lost,
            "key_words_added": words_added,
            "key_word_retention": f"{len(words_kept)}/{len(orig_key)}" if orig_key else "n/a",
        },
    }


def layer_reference(text: str, ref: SharedReference, layer_num: int) -> dict:
    """
    REFERENCE layer: Deterministic substitution using the shared reference file.

    This is FULLY REVERSIBLE -- the substitution log captures every change.
    Both encoder and decoder need the same reference file to work.
    """
    start = time.time()

    encoded, substitution_log = ref.full_encode(text)

    elapsed = (time.time() - start) * 1000

    # Categorize substitutions (handles both 3-tuple and 2-tuple formats)
    vocab_subs = []
    phrase_subs = []
    learned_subs = []
    removal_subs = []

    for entry in substitution_log:
        if len(entry) == 3:
            stype, orig, code = entry
        else:
            orig, code = entry[0], entry[1]
            stype = "phrase" if " " in orig else ("removal" if not code or not code.strip() else "vocab")

        if stype == "removal" or (not code or not code.strip()):
            removal_subs.append((orig, code))
        elif stype == "phrase":
            phrase_subs.append((orig, code))
        elif orig.lower() in ref.learned:
            learned_subs.append((orig, code))
        else:
            vocab_subs.append((orig, code))

    # Calculate chars saved per substitution
    chars_saved_breakdown = []
    for entry in substitution_log:
        orig = entry[1] if len(entry) == 3 else entry[0]
        code = entry[2] if len(entry) == 3 else entry[1]
        saved = len(orig) - len(code)
        chars_saved_breakdown.append({"original": orig, "code": code, "chars_saved": saved})

    # Sort by most chars saved
    chars_saved_breakdown.sort(key=lambda x: x["chars_saved"], reverse=True)

    # Input word analysis
    input_words = text.split()
    total_words = len(input_words)
    words_hit = len(substitution_log)
    hit_rate = round(words_hit / max(1, total_words), 3)

    return {
        "layer": layer_num,
        "type": "reference",
        "model": f"ref:{ref.name} v{ref.version}",
        "input": text,
        "output": encoded,
        "method": "reference-substitution",
        "input_len": len(text),
        "output_len": len(encoded),
        "compression": round(1.0 - len(encoded) / max(1, len(text)), 3),
        "time_ms": round(elapsed, 1),
        "substitutions_applied": len(substitution_log),
        "decode_info": {
            "type": "reference",
            "reference_name": ref.name,
            "reference_version": ref.version,
            "reference_fingerprint": ref.get_fingerprint(),
            "substitution_log": substitution_log,
        },
        "detail": {
            "description": "Deterministic codebook substitution. Replaces known words/phrases with short codes. Fully reversible via substitution log.",
            "status": "applied",
            "reference_file": ref.name,
            "reference_version": ref.version,
            "reference_fingerprint": ref.get_fingerprint(),
            "total_vocab_size": len(ref.vocabulary),
            "total_phrase_size": len(ref.phrases),
            "total_learned_size": len(ref.learned),
            "input_word_count": total_words,
            "substitutions_total": len(substitution_log),
            "substitutions_vocab": len(vocab_subs),
            "substitutions_phrase": len(phrase_subs),
            "substitutions_learned": len(learned_subs),
            "substitutions_removals": len(removal_subs),
            "hit_rate": hit_rate,
            "vocab_subs": vocab_subs[:15],
            "phrase_subs": phrase_subs[:10],
            "learned_subs": learned_subs[:10],
            "removal_subs": removal_subs[:10],
            "top_chars_saved": chars_saved_breakdown[:10],
            "total_chars_saved": sum(s["chars_saved"] for s in chars_saved_breakdown),
        },
    }


def layer_compact(text: str, model: str, layer_num: int,
                   provider: str = "ollama", api_key: str = "",
                   temperature: float = 0.1) -> dict:
    """
    COMPACT layer: LLM compresses already-shortened text further.

    This works on text that's already been through distill + reference encoding.
    The prompt is different from distill -- it works on abbreviated/coded text
    and tries to find structural patterns to compress further.
    Supports both Ollama and commercial models (Claude/GPT/Grok).
    """
    start = time.time()

    prompt = f"""Compress this text further using symbols and merging. Make it as short as possible.

SYMBOL MAP: and/also=+ then/leads_to=> is/equals== or=| but/however=^ at=@ about/topic=#
DROP: of, to, for, in, on, by, it, do, if, the, a, an
MERGE: adjacent words with > when sequential

Output ONLY the compressed result. No labels, quotes, explanation, or examples.

TEXT: {text}
COMPRESSED:"""

    input_tokens = text.split()

    try:
        raw = generate_text(provider, model, prompt, max_tokens=192,
                            temperature=temperature, api_key=api_key)
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        err_msg = str(e)
        if "Connection" in err_msg or "timeout" in err_msg.lower():
            err_msg = f"Cannot reach {provider} ({model}). Is the service running? " + err_msg
        elif "API key" in err_msg or "401" in err_msg or "403" in err_msg:
            err_msg = f"Authentication failed for {provider}. Check your API key. " + err_msg
        return {
            "layer": layer_num,
            "type": "compact",
            "model": model,
            "input": text,
            "output": text,
            "method": f"error-passthrough",
            "error": err_msg,
            "input_len": len(text),
            "output_len": len(text),
            "compression": 0,
            "time_ms": round(elapsed, 1),
            "decode_info": {"type": "compact", "pre_compact": text},
            "detail": {
                "description": "LLM compresses already-coded/abbreviated text by merging codes and using symbols.",
                "status": "error",
                "error": err_msg,
                "prompt_used": prompt,
            },
        }

    raw_before_clean = raw
    compacted = raw.strip()
    for prefix in ["OUTPUT:", "Output:", "Result:", "Compressed:", "COMPRESSED:"]:
        if compacted.lower().startswith(prefix.lower()):
            compacted = compacted[len(prefix):].strip()
    compacted = compacted.strip('"').strip("'").strip()

    # Take the shortest non-empty line -- small models often ramble or repeat examples
    truncated = False
    if "\n" in compacted:
        lines = [l.strip() for l in compacted.split("\n") if l.strip()]
        if lines:
            # Pick the shortest line that's at least 20% of input length
            # (filters out garbage ultra-short lines while avoiding the repeated examples)
            min_len = max(3, len(text) * 0.15)
            candidates = [l for l in lines if len(l) >= min_len]
            if candidates:
                compacted = min(candidates, key=len)
            else:
                compacted = lines[-1]  # last line as fallback
            truncated = len(lines) > 1
        else:
            compacted = text

    # If it expanded, keep original but flag it clearly
    kept_original = False
    if len(compacted) >= len(text) or not compacted:
        kept_original = True
        compacted = text

    elapsed = (time.time() - start) * 1000

    output_tokens = compacted.split()

    # Detect symbols used
    symbols_used = []
    symbol_map = {"+": "and/more", ">": "leads to", "=": "is", "|": "or", "^": "but", "@": "at", "#": "topic"}
    for sym, meaning in symbol_map.items():
        count = compacted.count(sym) - text.count(sym)
        if count > 0:
            symbols_used.append({"symbol": sym, "meaning": meaning, "count": count})

    return {
        "layer": layer_num,
        "type": "compact",
        "model": model,
        "input": text,
        "output": compacted,
        "method": "llm-compact",
        "input_len": len(text),
        "output_len": len(compacted),
        "compression": round(1.0 - len(compacted) / max(1, len(text)), 3),
        "time_ms": round(elapsed, 1),
        "decode_info": {
            "type": "compact",
            "model": model,
            "pre_compact": text,
            "post_compact": compacted,
        },
        "detail": {
            "description": "LLM compresses already-coded/abbreviated text by merging codes and using symbols for connectors.",
            "status": "kept-original" if kept_original else "compacted",
            "prompt_used": prompt,
            "raw_llm_output": raw_before_clean,
            "truncated_multiline": truncated,
            "input_tokens": len(input_tokens),
            "output_tokens": len(output_tokens),
            "tokens_reduction": len(input_tokens) - len(output_tokens),
            "symbols_introduced": symbols_used,
        },
    }


# ===== DECODING =====

def decode_layer(layer_info: dict, current_text: str, ref: SharedReference) -> str:
    """Decode a single layer based on its type and stored decode_info."""
    decode_info = layer_info.get("decode_info", {})
    layer_type = decode_info.get("type", layer_info.get("type", "unknown"))

    if layer_type == "reference":
        # Deterministic reversal using substitution log
        sub_log = decode_info.get("substitution_log", [])
        if sub_log:
            return ref.decode_text(current_text, sub_log)
        else:
            # Fallback: use reference reverse lookup
            return ref.decode_text(current_text)

    elif layer_type == "distill":
        # We stored the original text -- that's the decode
        original = decode_info.get("original_text", "")
        if original:
            return original
        # Fallback: LLM expansion
        model = decode_info.get("model", layer_info.get("model", ""))
        if model and not model.startswith("ref:"):
            try:
                prompt = f"""Expand this shortened text back into clear, complete English. Add back any articles, connectors, and grammar needed for natural reading.

SHORTENED: {current_text}

FULL TEXT:"""
                expanded = ollama_generate(model, prompt, max_tokens=256, temperature=0.3)
                for pfx in ["FULL TEXT:", "Full text:", "Expanded:", "Output:"]:
                    if expanded.startswith(pfx):
                        expanded = expanded[len(pfx):].strip()
                return expanded.strip('"').strip("'").strip() or current_text
            except Exception:
                pass
        return current_text

    elif layer_type == "compact":
        # We stored the pre-compact text
        pre = decode_info.get("pre_compact", "")
        if pre:
            return pre
        # Fallback: LLM expansion
        model = decode_info.get("model", layer_info.get("model", ""))
        if model and not model.startswith("ref:"):
            try:
                prompt = f"""Expand these compressed abbreviations and symbols back into readable text.
Symbols: + (and/more) > (leads to) = (is) | (or) ^ (but) @ (at) # (topic)

COMPRESSED: {current_text}

EXPANDED:"""
                expanded = ollama_generate(model, prompt, max_tokens=256, temperature=0.3)
                for pfx in ["EXPANDED:", "Expanded:", "Output:"]:
                    if expanded.startswith(pfx):
                        expanded = expanded[len(pfx):].strip()
                return expanded.strip('"').strip("'").strip() or current_text
            except Exception:
                pass
        return current_text

    return current_text


def decode_full(package: dict, ref: SharedReference) -> str:
    """Decode a full transmission package by reversing layers."""
    encoded = package.get("encoded", "")
    layers = package.get("layers", [])

    if not layers:
        return ref.decode_text(encoded)

    current = encoded
    for layer_info in reversed(layers):
        current = decode_layer(layer_info, current, ref)

    return current


# ===== ROUTES =====

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/api", methods=["GET"])
def api_info():
    return jsonify({
        "service": "Prefrontal Compressor",
        "version": "3.0.0",
        "description": "Multi-strategy LLM compression with shared reference files",
    })


@app.route("/models", methods=["GET"])
def get_models():
    models = ollama_models()
    ollama_available = len(models) > 0
    return jsonify({"models": models, "ollama_url": OLLAMA_URL, "ollama_available": ollama_available})


@app.route("/chain", methods=["POST"])
def chain_encode():
    """
    Multi-strategy compression chain.

    Each layer has a TYPE (distill/reference/compact) and optionally a model.
    The shared reference file is used for the reference layer.
    """
    global active_ref
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"]
    layer_configs = data.get("layer_configs", [])

    # Default chain: distill -> reference -> compact
    if not layer_configs:
        default_model = data.get("model", "qwen2.5:1.5b")
        layer_configs = [
            {"type": "distill", "model": default_model},
            {"type": "reference"},
            {"type": "compact", "model": default_model},
        ]

    # Switch reference file if specified
    ref_name = data.get("reference", active_ref.name)
    if ref_name != active_ref.name:
        active_ref = SharedReference(ref_name)

    total_start = time.time()
    layers = []
    current = text
    layer_num = 0

    for cfg in layer_configs:
        layer_num += 1
        ltype = cfg.get("type", "distill")
        model = cfg.get("model", "qwen2.5:1.5b")
        provider = cfg.get("provider", "ollama")
        layer_api_key = cfg.get("api_key", "")
        layer_temp = cfg.get("temperature", None)

        if ltype == "distill":
            kwargs = {"temperature": layer_temp} if layer_temp is not None else {}
            result = layer_distill(current, model, layer_num, provider, layer_api_key, **kwargs)
        elif ltype == "reference":
            result = layer_reference(current, active_ref, layer_num)
        elif ltype == "compact":
            kwargs = {"temperature": layer_temp} if layer_temp is not None else {}
            result = layer_compact(current, model, layer_num, provider, layer_api_key, **kwargs)
        else:
            kwargs = {"temperature": layer_temp} if layer_temp is not None else {}
            result = layer_distill(current, model, layer_num, provider, layer_api_key, **kwargs)

        layers.append(result)
        current = result["output"]

    total_time = (time.time() - total_start) * 1000
    total_compression = 1.0 - (len(current) / max(1, len(text)))

    # Build transmission package
    transmission_package = {
        "encoded": current,
        "reference": {
            "name": active_ref.name,
            "version": active_ref.version,
            "fingerprint": active_ref.get_fingerprint(),
        },
        "layers": [
            {
                "layer": l["layer"],
                "type": l["type"],
                "model": l["model"],
                "compression": l["compression"],
                "decode_info": l["decode_info"],
            }
            for l in layers
        ],
        "manifest": {
            "version": "3.0",
            "original_length": len(text),
            "encoded_length": len(current),
            "compression_ratio": round(total_compression, 3),
            "num_layers": len(layers),
        },
    }

    # Decode to check fidelity
    fidelity = None
    final_decoded = None
    try:
        final_decoded = decode_full(transmission_package, active_ref)
        if final_decoded:
            orig_words = set(text.lower().split())
            dec_words = set(final_decoded.lower().split())
            key_orig = {w for w in orig_words if len(w) > 3}
            key_dec = {w for w in dec_words if len(w) > 3}
            fidelity = len(key_orig & key_dec) / max(1, len(key_orig))
    except Exception:
        pass

    # Update stats
    stats["total_compressions"] += 1
    stats["total_input_chars"] += len(text)
    stats["total_output_chars"] += len(current)

    # Save reference (updates frequency counts)
    active_ref.save()

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


@app.route("/chain/stream", methods=["POST"])
def chain_stream():
    """Streaming compression chain -- sends ndjson events as each layer completes."""
    global active_ref
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    text = data["text"]
    layer_configs = data.get("layer_configs", [])

    if not layer_configs:
        default_model = data.get("model", "qwen2.5:1.5b")
        layer_configs = [
            {"type": "distill", "model": default_model},
            {"type": "reference"},
            {"type": "compact", "model": default_model},
        ]

    ref_name = data.get("reference", active_ref.name)
    if ref_name != active_ref.name:
        active_ref = SharedReference(ref_name)

    def generate():
        total_start = time.time()
        layers = []
        current = text
        layer_num = 0

        yield json.dumps({
            "event": "chain_start",
            "total_layers": len(layer_configs),
            "input_len": len(text),
            "input_words": len(text.split()),
            "configs": [{"type": c.get("type", "distill"), "model": c.get("model", "")} for c in layer_configs],
        }) + "\n"

        for cfg in layer_configs:
            layer_num += 1
            ltype = cfg.get("type", "distill")
            model = cfg.get("model", "qwen2.5:1.5b")
            provider = cfg.get("provider", "ollama")
            layer_api_key = cfg.get("api_key", "")
            layer_temp = cfg.get("temperature", None)

            display_model = model if ltype != "reference" else f"ref:{active_ref.name} v{active_ref.version}"
            if provider != "ollama" and ltype != "reference":
                display_model = f"{provider}:{model}"

            yield json.dumps({
                "event": "layer_start",
                "layer": layer_num,
                "type": ltype,
                "model": display_model,
                "provider": provider,
                "input_len": len(current),
                "input_words": len(current.split()),
            }) + "\n"

            if ltype == "distill":
                kwargs = {"temperature": layer_temp} if layer_temp is not None else {}
                result = layer_distill(current, model, layer_num, provider, layer_api_key, **kwargs)
            elif ltype == "reference":
                result = layer_reference(current, active_ref, layer_num)
            elif ltype == "compact":
                kwargs = {"temperature": layer_temp} if layer_temp is not None else {}
                result = layer_compact(current, model, layer_num, provider, layer_api_key, **kwargs)
            else:
                kwargs = {"temperature": layer_temp} if layer_temp is not None else {}
                result = layer_distill(current, model, layer_num, provider, layer_api_key, **kwargs)

            layers.append(result)
            current = result["output"]

            yield json.dumps({
                "event": "layer_complete",
                "layer": layer_num,
                "result": result,
                "running_compression": round(1.0 - len(current) / max(1, len(text)), 3),
                "running_chars": len(current),
            }) + "\n"

        total_time = (time.time() - total_start) * 1000
        total_compression = 1.0 - (len(current) / max(1, len(text)))

        transmission_package = {
            "encoded": current,
            "reference": {
                "name": active_ref.name,
                "version": active_ref.version,
                "fingerprint": active_ref.get_fingerprint(),
            },
            "layers": [
                {
                    "layer": l["layer"],
                    "type": l["type"],
                    "model": l["model"],
                    "compression": l["compression"],
                    "decode_info": l["decode_info"],
                }
                for l in layers
            ],
            "manifest": {
                "version": "3.0",
                "original_length": len(text),
                "encoded_length": len(current),
                "compression_ratio": round(total_compression, 3),
                "num_layers": len(layers),
            },
        }

        fidelity = None
        final_decoded = None
        try:
            final_decoded = decode_full(transmission_package, active_ref)
            if final_decoded:
                orig_words = set(text.lower().split())
                dec_words = set(final_decoded.lower().split())
                key_orig = {w for w in orig_words if len(w) > 3}
                key_dec = {w for w in dec_words if len(w) > 3}
                fidelity = len(key_orig & key_dec) / max(1, len(key_orig))
        except Exception:
            pass

        stats["total_compressions"] += 1
        stats["total_input_chars"] += len(text)
        stats["total_output_chars"] += len(current)
        active_ref.save()

        yield json.dumps({
            "event": "chain_complete",
            "original": text,
            "layers": layers,
            "final_encoded": current,
            "final_decoded": final_decoded,
            "total_compression": round(total_compression, 3),
            "total_time_ms": round(total_time, 1),
            "fidelity": round(fidelity, 3) if fidelity is not None else None,
            "num_layers": len(layers),
            "transmission_package": transmission_package,
        }) + "\n"

    return Response(
        stream_with_context(generate()),
        mimetype='application/x-ndjson',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )


@app.route("/decode", methods=["POST"])
def decode():
    """Decode a transmission package."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing request body"}), 400

    start = time.time()

    if "transmission_package" in data:
        pkg = data["transmission_package"]
    elif "encoded" in data:
        pkg = {"encoded": data["encoded"], "layers": data.get("layers", [])}
    else:
        return jsonify({"error": "Missing 'transmission_package' or 'encoded'"}), 400

    # Use specified reference or active
    ref_name = pkg.get("reference", {}).get("name", active_ref.name)
    ref = SharedReference(ref_name) if ref_name != active_ref.name else active_ref

    decoded = decode_full(pkg, ref)
    elapsed = (time.time() - start) * 1000

    return jsonify({
        "decoded": decoded,
        "time_ms": round(elapsed, 1),
        "reference_used": ref.name,
        "reference_version": ref.version,
    })


@app.route("/submit", methods=["POST"])
def submit_to_llm():
    """Submit to external LLM (Claude, GPT, or Grok)."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing request body"}), 400

    target = data.get("target", "claude")
    model = data.get("model", "")
    api_key = data.get("api_key", "")
    mode = data.get("mode", "package")

    original = data.get("original", "")
    package = data.get("transmission_package", None)
    decoded_text = data.get("decoded", "")

    if not api_key:
        return jsonify({"error": "Missing API key"}), 400

    if mode == "package" and package:
        encoded = package.get("encoded", "")
        ref_info = package.get("reference", {})
        layer_info = package.get("layers", [])

        # Build decode instructions from the layer decode_info
        decode_section = "\n\nDECODE REFERENCE:\n"
        for l in layer_info:
            di = l.get("decode_info", {})
            if di.get("type") == "reference":
                sub_log = di.get("substitution_log", [])
                if sub_log:
                    decode_section += "Substitutions (code -> meaning):\n"
                    for entry in sub_log:
                        if isinstance(entry, (list, tuple)) and len(entry) == 3:
                            _stype, orig, code = entry
                        elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                            orig, code = entry
                        else:
                            continue
                        if code and code.strip():
                            decode_section += f"  {code} = {orig}\n"
            elif di.get("type") == "distill":
                decode_section += f"Layer {l['layer']} (distill): LLM-shortened from original\n"
            elif di.get("type") == "compact":
                decode_section += f"Layer {l['layer']} (compact): Symbols: + (and) > (to) = (is) | (or) ^ (but) @ (at) # (topic)\n"

        prompt = f"""The following message was compressed through a {len(layer_info)}-layer semantic compressor.
Reference file: {ref_info.get('name', 'default')} v{ref_info.get('version', 1)}
{decode_section}

Compressed message: {encoded}

Decode this and respond to the original intent."""

        system_msg = (
            "You are receiving a semantically compressed prompt. Use the provided decode reference "
            "to reconstruct the meaning, then respond helpfully to the original request."
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


# ===== REFERENCE FILE MANAGEMENT =====

@app.route("/reference", methods=["GET"])
def get_reference():
    """View the active reference file."""
    return jsonify(active_ref.to_dict())


@app.route("/reference/full", methods=["GET"])
def get_reference_full():
    """Return the complete reference file with all mappings for the codebook browser."""
    freq = {}
    if hasattr(active_ref, 'frequency'):
        freq = dict(active_ref.frequency) if active_ref.frequency else {}
    return jsonify({
        "name": active_ref.name,
        "version": active_ref.version,
        "fingerprint": active_ref.get_fingerprint(),
        "training_rounds": getattr(active_ref, 'training_rounds', 0),
        "total_compressions": getattr(active_ref, 'total_compressions', 0),
        "vocabulary": active_ref.vocabulary,
        "phrases": active_ref.phrases,
        "learned": active_ref.learned if hasattr(active_ref, 'learned') else {},
        "frequency": freq,
    })


@app.route("/references", methods=["GET"])
def get_references():
    """List all available reference files."""
    refs = list_references()
    return jsonify({
        "references": refs,
        "active": active_ref.name,
    })


@app.route("/reference/new", methods=["POST"])
def create_reference():
    """Create a new reference file."""
    global active_ref
    data = request.get_json()
    name = data.get("name", f"ref_{int(time.time())}")
    ref = SharedReference(name)
    ref.save()
    active_ref = ref
    return jsonify({"created": name, "stats": ref.get_stats()})


@app.route("/reference/select", methods=["POST"])
def select_reference():
    """Switch to a different reference file."""
    global active_ref
    data = request.get_json()
    name = data.get("name", "default")
    active_ref = SharedReference(name)
    return jsonify({"active": name, "stats": active_ref.get_stats()})


@app.route("/reference/train", methods=["POST"])
def train_reference():
    """
    Run a training round: encode sample texts, decode them, compare,
    and learn new patterns for the reference file.
    """
    data = request.get_json() or {}
    samples = data.get("samples", [
        "Can you explain how neural networks learn from data?",
        "Create a Python function that sorts a list in descending order",
        "The patient has elevated blood pressure and should reduce sodium intake",
        "What is the difference between supervised and unsupervised learning?",
        "Explain the central limit theorem and why it matters in statistics",
        "How does the attention mechanism work in transformer models?",
        "Calculate the p-value for a two-tailed t-test with 30 degrees of freedom",
        "Deploy a machine learning model to production using Kubernetes",
        "What is the Bayesian approach to parameter estimation?",
        "Implement a binary search tree with insert and delete operations",
    ])

    results = []
    new_learned = []

    for text in samples:
        # Encode through reference layer
        encoded, sub_log = active_ref.full_encode(text)

        # Decode
        decoded = active_ref.decode_text(encoded, sub_log)

        # Score
        orig_words = set(text.lower().split())
        dec_words = set(decoded.lower().split())
        key_orig = {w for w in orig_words if len(w) > 3}
        key_dec = {w for w in dec_words if len(w) > 3}
        fidelity = len(key_orig & key_dec) / max(1, len(key_orig))

        compression = 1.0 - len(encoded) / max(1, len(text))

        # Find missed words (candidates for new codes)
        missed = active_ref.learn_from_comparison(text, encoded, decoded)

        results.append({
            "original": text,
            "encoded": encoded,
            "decoded": decoded,
            "compression": round(compression, 3),
            "fidelity": round(fidelity, 3),
            "substitutions": len(sub_log),
            "missed_words": missed,
        })

        # Learn from missed words
        for word in missed[:3]:
            code = active_ref.suggest_code(word)
            while code in active_ref.vocabulary or code in active_ref.learned:
                code = code + str(len(code))
            active_ref.learn(word, code)
            new_learned.append({"phrase": word, "code": code})

    active_ref.training_rounds += 1
    active_ref.save()

    avg_compression = sum(r["compression"] for r in results) / max(1, len(results))
    avg_fidelity = sum(r["fidelity"] for r in results) / max(1, len(results))

    return jsonify({
        "round": active_ref.training_rounds,
        "samples": len(samples),
        "avg_compression": round(avg_compression, 3),
        "avg_fidelity": round(avg_fidelity, 3),
        "new_codes_learned": new_learned,
        "reference_version": active_ref.version,
        "results": results,
    })


@app.route("/reference/entry", methods=["PUT"])
def update_entry():
    """Add or update a codebook entry."""
    global active_ref
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing request body"}), 400

    word = data.get("word", "").strip()
    code = data.get("code", "").strip()
    entry_type = data.get("type", "learned")
    old_word = data.get("old_word", "").strip()  # for edits

    if not word:
        return jsonify({"error": "Missing 'word' field"}), 400
    if not code:
        return jsonify({"error": "Missing 'code' field"}), 400

    # If editing, remove old entry first
    if old_word:
        active_ref.vocabulary.pop(old_word, None)
        active_ref.vocabulary.pop(old_word.lower(), None)
        active_ref.phrases.pop(old_word, None)
        active_ref.phrases.pop(old_word.lower(), None)
        active_ref.learned.pop(old_word, None)
        active_ref.learned.pop(old_word.lower(), None)

    # Add to appropriate dict
    if entry_type == "vocab":
        active_ref.vocabulary[word.lower()] = code
    elif entry_type == "phrase":
        active_ref.phrases[word.lower()] = code
    else:
        active_ref.learned[word.lower()] = code

    active_ref.version += 1
    active_ref._build_reverse()
    active_ref.save()

    return jsonify({"ok": True, "word": word, "code": code, "type": entry_type,
                     "stats": active_ref.get_stats()})


@app.route("/reference/entry", methods=["DELETE"])
def delete_entry():
    """Delete a codebook entry."""
    global active_ref
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing request body"}), 400

    word = data.get("word", "").strip()
    if not word:
        return jsonify({"error": "Missing 'word' field"}), 400

    removed = False
    for key in [word, word.lower()]:
        if key in active_ref.vocabulary:
            del active_ref.vocabulary[key]
            removed = True
        if key in active_ref.phrases:
            del active_ref.phrases[key]
            removed = True
        if key in active_ref.learned:
            del active_ref.learned[key]
            removed = True

    if removed:
        active_ref.version += 1
        active_ref._build_reverse()
        active_ref.save()

    return jsonify({"ok": True, "removed": removed, "word": word,
                     "stats": active_ref.get_stats()})


@app.route("/reference/import", methods=["POST"])
def import_codebook():
    """Import a codebook JSON file, merging entries into the active reference."""
    global active_ref
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing request body"}), 400

    imported = 0
    if "vocabulary" in data:
        active_ref.vocabulary.update(data["vocabulary"])
        imported += len(data["vocabulary"])
    if "phrases" in data:
        active_ref.phrases.update(data["phrases"])
        imported += len(data["phrases"])
    if "learned" in data:
        active_ref.learned.update(data["learned"])
        imported += len(data["learned"])

    if imported > 0:
        active_ref.version += 1
        active_ref._build_reverse()
        active_ref.save()

    return jsonify({"ok": True, "imported": imported, "stats": active_ref.get_stats()})


@app.route("/stats", methods=["GET"])
def get_stats():
    return jsonify({
        "compression": dict(stats),
        "reference": active_ref.get_stats(),
        "ollama": {"url": OLLAMA_URL, "models": ollama_models()},
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"Prefrontal Compressor v3 starting on port {port}")
    print(f"Ollama: {OLLAMA_URL}")
    print(f"Reference: {active_ref.name} v{active_ref.version} ({active_ref.get_fingerprint()})")
    print(f"GUI: http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
