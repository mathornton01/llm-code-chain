"""
Prefrontal Compressor v5 -- Fully Reversible Multi-Level Compression Chain

Three layers, each working at a DIFFERENT level of abstraction:
  1. DISCOVER -- LLM identifies compressible patterns, learns new codebook
               entries, then codebook encodes words/phrases deterministically
  2. META-PATTERN -- Analyzes the coded output to find recurring CODE
               SEQUENCES (bigrams/trigrams of codes) and creates higher-order
               meta-codes (e.g. "= v imp" -> "M.1"). Patterns of patterns.
  3. STRUCTURAL -- Packs the coded text tighter: abbreviates remaining
               uncoded words, removes spaces around operators, collapses
               domain-prefix spacing.

Key design: each layer produces genuinely different compression because
they work at different granularity levels (words -> code sequences -> structure).
All layers are fully reversible via substitution logs.

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

OLLAMA_URL = os.environ.get("OLLAMA_URL", "") or "http://127.0.0.1:11434"

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

def _parse_llm_suggestions(raw: str) -> list:
    """Parse LLM output into a list of {original, code} suggestion dicts.
    Handles JSON arrays, JSON-in-markdown, and line-by-line fallback."""
    raw = raw.strip()

    # Try to extract JSON array from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', raw, re.DOTALL)
    if json_match:
        raw = json_match.group(1)

    # Try direct JSON parse
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            results = []
            for item in parsed:
                if isinstance(item, dict) and "original" in item and "code" in item:
                    results.append(item)
                elif isinstance(item, dict) and "word" in item and "code" in item:
                    results.append({"original": item["word"], "code": item["code"]})
            return results
    except (json.JSONDecodeError, ValueError):
        pass

    # Try to find a JSON array anywhere in the text
    bracket_match = re.search(r'\[.*\]', raw, re.DOTALL)
    if bracket_match:
        try:
            parsed = json.loads(bracket_match.group())
            if isinstance(parsed, list):
                results = []
                for item in parsed:
                    if isinstance(item, dict) and "original" in item and "code" in item:
                        results.append(item)
                return results
        except (json.JSONDecodeError, ValueError):
            pass

    # Line-by-line fallback: "word -> code" or "word => code" or "word: code"
    results = []
    for line in raw.split("\n"):
        line = line.strip().strip("-").strip("*").strip()
        for sep in ["->", "=>", ":"]:
            if sep in line:
                parts = line.split(sep, 1)
                orig = parts[0].strip().strip('"').strip("'").lower()
                code = parts[1].strip().strip('"').strip("'")
                if orig and code and len(code) < len(orig) and len(orig) > 2:
                    results.append({"original": orig, "code": code})
                break
    return results


def layer_distill(text: str, model: str, layer_num: int,
                   provider: str = "ollama", api_key: str = "",
                   temperature: float = 0.3) -> dict:
    """
    DISCOVER layer: LLM analyzes text and suggests new codebook entries.
    Then the codebook encodes EVERYTHING deterministically with full logging.

    This is FULLY REVERSIBLE -- every substitution is in the log.
    The LLM never rewrites text; it only suggests abbreviations to learn.
    """
    global active_ref
    start = time.time()

    # Build a sample of what the codebook already knows
    known_samples = list(active_ref.vocabulary.items())[:20]
    known_str = ", ".join(f'"{w}"->"{c}"' for w, c in known_samples)

    prompt = f"""You are a codebook builder for a text compression system. Analyze the input text and suggest NEW abbreviation rules for words/phrases NOT already in the codebook.

EXISTING CODEBOOK SAMPLES (do NOT repeat these):
{known_str}

RULES FOR SUGGESTIONS:
1. Only suggest words/phrases that appear in the input text
2. Each code must be shorter than the original (at least 2 chars saved)
3. Use consonant-based abbreviations: "learning" -> "lrn", "network" -> "ntwk"
4. For multi-word phrases, use domain prefixes: "machine learning" -> "a.ml"
5. Domain prefixes: a.=AI, c.=code, m.=math, b.=bio, w.=web, d.=devops, ds.=data, s.=security
6. Codes must be unambiguous -- different words need different codes
7. Skip words shorter than 4 characters (not worth encoding)
8. Focus on domain-specific terms and repeated phrases

Return ONLY a JSON array of objects. No explanation, no markdown, no extra text.
Format: [{{"original": "word or phrase", "code": "short_code"}}, ...]
If nothing new to suggest, return: []

INPUT TEXT: {text}
JSON:"""

    input_words = text.split()
    input_word_count = len(input_words)
    new_entries_learned = []

    try:
        raw = generate_text(provider, model, prompt, max_tokens=512,
                            temperature=temperature, api_key=api_key)
        suggestions = _parse_llm_suggestions(raw)

        # Validate and learn new entries
        all_codes = set(active_ref.vocabulary.values()) | set(active_ref.phrases.values()) | set(active_ref.learned.values())
        for s in suggestions:
            orig = s["original"].lower().strip()
            code = s["code"].strip()

            # Skip if already known or invalid
            if not orig or not code:
                continue
            if len(code) >= len(orig):
                continue
            if orig in active_ref.vocabulary or orig in active_ref.phrases or orig in active_ref.learned:
                continue
            if code in all_codes:
                # Try appending a counter to avoid collision
                for i in range(1, 10):
                    candidate = f"{code}{i}"
                    if candidate not in all_codes:
                        code = candidate
                        break
                else:
                    continue
            # Verify the word/phrase actually appears in the text
            if orig not in text.lower():
                continue

            # Learn it
            if " " in orig:
                active_ref.phrases[orig] = code
                active_ref.decode_phrases[code] = orig
            else:
                active_ref.learned[orig] = code
                active_ref.decode_learned[code] = orig
            all_codes.add(code)
            new_entries_learned.append({"original": orig, "code": code})
            active_ref.version += 1

        llm_raw = raw
        llm_error = None
    except Exception as e:
        llm_raw = None
        llm_error = str(e)
        suggestions = []
        if "Connection" in llm_error or "timeout" in llm_error.lower():
            llm_error = f"Cannot reach {provider} ({model}). " + llm_error

    # Now encode deterministically using the (possibly expanded) codebook
    encoded, substitution_log = active_ref.full_encode(text)

    elapsed = (time.time() - start) * 1000

    # Categorize substitutions
    vocab_subs = []
    phrase_subs = []
    learned_subs = []
    for entry in substitution_log:
        stype, orig, code = entry[0], entry[1], entry[2] if len(entry) == 3 else entry[1]
        if stype == "phrase":
            phrase_subs.append((orig, code))
        elif orig.lower() in active_ref.learned:
            learned_subs.append((orig, code))
        else:
            vocab_subs.append((orig, code))

    return {
        "layer": layer_num,
        "type": "distill",
        "model": model,
        "input": text,
        "output": encoded,
        "method": "discover-encode",
        "input_len": len(text),
        "output_len": len(encoded),
        "compression": round(1.0 - len(encoded) / max(1, len(text)), 3),
        "time_ms": round(elapsed, 1),
        "substitutions_applied": len(substitution_log),
        "decode_info": {
            "type": "reference",  # decode same as reference -- it's all codebook
            "reference_name": active_ref.name,
            "reference_version": active_ref.version,
            "reference_fingerprint": active_ref.get_fingerprint(),
            "substitution_log": substitution_log,
        },
        "detail": {
            "description": "LLM discovers new codebook entries, then codebook encodes deterministically. 100% reversible via substitution log.",
            "status": "error" if llm_error else "discovered",
            "error": llm_error,
            "llm_raw_output": llm_raw,
            "new_entries_learned": new_entries_learned,
            "llm_suggestions_count": len(suggestions),
            "entries_accepted": len(new_entries_learned),
            "input_words": input_word_count,
            "substitutions_vocab": len(vocab_subs),
            "substitutions_phrase": len(phrase_subs),
            "substitutions_learned": len(learned_subs),
            "vocab_subs": vocab_subs[:15],
            "phrase_subs": phrase_subs[:10],
            "learned_subs": learned_subs[:10],
        },
    }


def layer_reference(text: str, ref: SharedReference, layer_num: int,
                    model: str = "", provider: str = "ollama", api_key: str = "",
                    temperature: float = 0.3) -> dict:
    """
    META-PATTERN layer: Finds patterns OF codes in already-encoded text.

    After Layer 1 encodes words/phrases into codes, this layer analyzes
    the coded output to find recurring code SEQUENCES (bigrams/trigrams)
    and creates higher-order meta-codes for them.

    Example: if "= v imp" (is very important) appears, it becomes "M.1"

    Also applies any existing meta-patterns from the codebook.
    This is FULLY REVERSIBLE via the substitution log.
    """
    start = time.time()

    # Step 1: Apply any existing meta-patterns first
    encoded, meta_subs = ref.encode_meta(text)

    # Step 2: Find new recurring code sequences in the (meta-encoded) text
    bigram_candidates = ref.find_code_bigrams(encoded)

    # Step 3: If we have an LLM, ask it to suggest which bigrams to learn
    new_meta_learned = []
    llm_raw = None
    llm_error = None

    if bigram_candidates and model:
        # Show the LLM the coded text and the bigram candidates
        candidate_str = "\n".join(f'  "{seq}" (appears {count}x, saves {(len(seq) - 3) * count} chars)'
                                   for seq, count in bigram_candidates[:15])

        existing_meta_str = ""
        if ref.meta_patterns:
            existing_meta_str = "\nEXISTING META-CODES (do NOT repeat):\n" + \
                "\n".join(f'  "{seq}" -> "{mc}"' for seq, mc in ref.meta_patterns.items())

        prompt = f"""You are a code sequence optimizer. The text below has been encoded using a codebook (words -> short codes). Your job is to find PATTERNS OF CODES -- recurring sequences that can be combined into even shorter meta-codes.

CODED TEXT: {encoded}

CANDIDATE CODE SEQUENCES (recurring patterns found):
{candidate_str}
{existing_meta_str}

RULES:
1. Only suggest sequences that genuinely recur or save significant space
2. Each meta-code format: M.1, M.2, M.3, etc.
3. The sequence must appear in the coded text above
4. Prefer sequences that save the most characters
5. Don't suggest single-token sequences -- they must be 2+ tokens

Return ONLY a JSON array. No explanation, no markdown.
Format: [{{"original": "code sequence", "code": "M.N"}}, ...]
If nothing worth combining, return: []
JSON:"""

        try:
            raw = generate_text(provider, model, prompt, max_tokens=384,
                                temperature=temperature, api_key=api_key)
            suggestions = _parse_llm_suggestions(raw)

            # Validate and learn meta-patterns
            for s in suggestions:
                seq = s["original"].strip()
                meta_code = s["code"].strip()

                if not seq or not meta_code:
                    continue
                if len(meta_code) >= len(seq):
                    continue
                if seq in ref.meta_patterns:
                    continue
                # Must be a multi-token sequence
                if " " not in seq:
                    continue
                # Must appear in the coded text
                if seq not in encoded:
                    continue
                # Ensure meta_code doesn't collide
                all_meta = set(ref.meta_patterns.values())
                if meta_code in all_meta:
                    meta_code = ref.next_meta_code()

                ref.learn_meta(seq, meta_code)
                new_meta_learned.append({"sequence": seq, "meta_code": meta_code,
                                          "chars_saved": len(seq) - len(meta_code)})

            llm_raw = raw
        except Exception as e:
            llm_error = str(e)
            if "Connection" in str(e) or "timeout" in str(e).lower():
                llm_error = f"Cannot reach {provider} ({model}). " + llm_error
    elif bigram_candidates and not model:
        # No LLM available -- auto-learn top candidates deterministically
        for seq, count in bigram_candidates[:5]:
            if seq in ref.meta_patterns:
                continue
            if " " not in seq:
                continue
            savings = (len(seq) - 3) * count
            if savings >= 3:  # worth at least 3 chars savings
                meta_code = ref.next_meta_code()
                ref.learn_meta(seq, meta_code)
                new_meta_learned.append({"sequence": seq, "meta_code": meta_code,
                                          "chars_saved": len(seq) - len(meta_code)})

    # Step 4: Re-apply meta-patterns (including newly learned ones)
    if new_meta_learned:
        encoded, meta_subs = ref.encode_meta(text)

    elapsed = (time.time() - start) * 1000

    # Calculate savings
    chars_saved = len(text) - len(encoded)
    meta_applied = len(meta_subs)

    return {
        "layer": layer_num,
        "type": "reference",
        "model": f"ref:{ref.name} v{ref.version}" + (f" + {provider}:{model}" if model else ""),
        "input": text,
        "output": encoded,
        "method": "meta-pattern-encoding",
        "input_len": len(text),
        "output_len": len(encoded),
        "compression": round(1.0 - len(encoded) / max(1, len(text)), 3),
        "time_ms": round(elapsed, 1),
        "substitutions_applied": meta_applied,
        "decode_info": {
            "type": "meta",
            "reference_name": ref.name,
            "reference_version": ref.version,
            "reference_fingerprint": ref.get_fingerprint(),
            "substitution_log": meta_subs,
        },
        "detail": {
            "description": "Meta-pattern encoding: finds recurring CODE SEQUENCES in already-encoded text and combines them into higher-order meta-codes. This is 'patterns of patterns'.",
            "status": "error" if llm_error else ("meta-learned" if new_meta_learned else ("meta-applied" if meta_subs else "passthrough")),
            "error": llm_error,
            "llm_raw_output": llm_raw,
            "reference_file": ref.name,
            "reference_version": ref.version,
            "reference_fingerprint": ref.get_fingerprint(),
            "total_meta_patterns": len(ref.meta_patterns),
            "existing_meta_patterns": dict(list(ref.meta_patterns.items())[:20]),
            "bigram_candidates_found": len(bigram_candidates),
            "top_bigram_candidates": [{"sequence": s, "count": c} for s, c in bigram_candidates[:10]],
            "new_meta_learned": new_meta_learned,
            "meta_subs_applied": len(meta_subs),
            "meta_subs_detail": [{"sequence": s[1], "meta_code": s[2]} for s in meta_subs],
            "chars_saved": chars_saved,
        },
    }


def layer_compact(text: str, model: str, layer_num: int,
                   provider: str = "ollama", api_key: str = "",
                   temperature: float = 0.25) -> dict:
    """
    STRUCTURAL PACKING layer: Deterministic compression of coded text.

    After Layer 1 (word/phrase encoding) and Layer 2 (meta-pattern encoding),
    this layer applies structural rules to pack the output tighter:
    - Remove spaces around operator codes
    - Collapse domain-prefix spacing
    - Find remaining uncoded words and abbreviate them

    Optionally uses LLM to find remaining compressible words.
    This is FULLY REVERSIBLE via the substitution log.
    """
    global active_ref
    start = time.time()

    # Step 1: Find remaining uncoded words (words not in any codebook section)
    words = text.split()
    uncoded_words = []
    all_known = set()
    all_known.update(active_ref.vocabulary.keys())
    all_known.update(active_ref.vocabulary.values())
    all_known.update(active_ref.learned.keys())
    all_known.update(active_ref.learned.values())
    all_known.update(active_ref.meta_patterns.values())
    # Add phrase codes
    all_known.update(active_ref.phrases.values())

    for w in words:
        clean = w.strip(".,!?;:()[]{}\"'").lower()
        if (clean not in all_known
            and len(clean) > 3
            and clean.isalpha()):
            uncoded_words.append(clean)

    # Step 2: If LLM available, ask for abbreviations of remaining uncoded words
    new_entries_learned = []
    llm_raw = None
    llm_error = None

    if uncoded_words and model:
        from collections import Counter
        word_freq = Counter(uncoded_words)
        frequent = [w for w, c in word_freq.most_common(15)]

        prompt = f"""You are a text abbreviator. These words survived two rounds of encoding and need short abbreviations.

WORDS TO ABBREVIATE: {', '.join(frequent)}

RULES:
1. Use consonant-based abbreviations: "learning" -> "lrn", "algorithm" -> "alg"
2. Each abbreviation must be shorter than the original (save at least 2 chars)
3. Must be unambiguous and readable
4. Skip words shorter than 4 characters

Return ONLY a JSON array. No explanation.
Format: [{{"original": "word", "code": "abbrev"}}, ...]
If nothing to abbreviate, return: []
JSON:"""

        try:
            raw = generate_text(provider, model, prompt, max_tokens=256,
                                temperature=temperature, api_key=api_key)
            suggestions = _parse_llm_suggestions(raw)

            all_codes = (set(active_ref.vocabulary.values()) |
                         set(active_ref.phrases.values()) |
                         set(active_ref.learned.values()) |
                         set(active_ref.meta_patterns.values()))

            for s in suggestions:
                orig = s["original"].lower().strip()
                code = s["code"].strip()

                if not orig or not code or len(code) >= len(orig):
                    continue
                if orig in active_ref.vocabulary or orig in active_ref.learned:
                    continue
                if code in all_codes:
                    for i in range(1, 10):
                        candidate = f"{code}{i}"
                        if candidate not in all_codes:
                            code = candidate
                            break
                    else:
                        continue
                if orig not in text.lower():
                    continue

                active_ref.learned[orig] = code
                active_ref.decode_learned[code] = orig
                all_codes.add(code)
                new_entries_learned.append({"original": orig, "code": code})
                active_ref.version += 1

            llm_raw = raw
        except Exception as e:
            llm_error = str(e)

    # Step 3: Apply any newly-learned word codes
    encoded = text
    word_subs = []
    if new_entries_learned:
        for entry in new_entries_learned:
            orig, code = entry["original"], entry["code"]
            if orig in encoded.lower():
                # Case-insensitive replacement
                import re
                encoded = re.sub(re.escape(orig), code, encoded, flags=re.IGNORECASE)
                word_subs.append(["vocab", orig, code])

    # Step 4: Structural packing (deterministic)
    packed, pack_subs = active_ref.pack_structural(encoded)

    all_subs = word_subs + pack_subs

    elapsed = (time.time() - start) * 1000

    return {
        "layer": layer_num,
        "type": "compact",
        "model": model if model else "structural",
        "input": text,
        "output": packed,
        "method": "structural-packing",
        "input_len": len(text),
        "output_len": len(packed),
        "compression": round(1.0 - len(packed) / max(1, len(text)), 3),
        "time_ms": round(elapsed, 1),
        "substitutions_applied": len(all_subs),
        "decode_info": {
            "type": "compact",
            "reference_name": active_ref.name,
            "reference_version": active_ref.version,
            "reference_fingerprint": active_ref.get_fingerprint(),
            "substitution_log": all_subs,
            "pre_pack_text": encoded,  # text before structural packing
        },
        "detail": {
            "description": "Structural packing: abbreviates remaining uncoded words, removes spaces around operators, packs coded text tighter.",
            "status": "error" if llm_error else ("packed" if pack_subs or new_entries_learned else "passthrough"),
            "error": llm_error,
            "llm_raw_output": llm_raw,
            "uncoded_words_found": len(uncoded_words),
            "uncoded_words": uncoded_words[:15],
            "new_abbreviations_learned": new_entries_learned,
            "structural_rules_applied": len(pack_subs),
            "pack_rules": [{"rule": s[1], "action": s[2]} for s in pack_subs],
            "total_chars_saved": len(text) - len(packed),
        },
    }


# ===== DECODING =====

def decode_layer(layer_info: dict, current_text: str, ref: SharedReference) -> str:
    """Decode a single layer based on its type and stored decode_info.

    Layer types and their decode strategies:
      - reference/distill: codebook substitution log (vocab + phrase reversal)
      - meta: meta-pattern reversal (meta_code -> code sequence)
      - compact: structural unpacking + word abbreviation reversal
    """
    decode_info = layer_info.get("decode_info", {})
    layer_type = decode_info.get("type", layer_info.get("type", "unknown"))

    if layer_type == "reference":
        # Codebook reversal using substitution log
        sub_log = decode_info.get("substitution_log", [])
        if sub_log:
            return ref.decode_text(current_text, sub_log)
        else:
            return ref.decode_text(current_text)

    elif layer_type == "meta":
        # Meta-pattern reversal (M.1 -> "code1 code2")
        sub_log = decode_info.get("substitution_log", [])
        return ref.decode_meta_text(current_text, sub_log)

    elif layer_type == "distill":
        # v4: distill stores codebook substitution_log
        sub_log = decode_info.get("substitution_log", [])
        if sub_log:
            return ref.decode_text(current_text, sub_log)
        original = decode_info.get("original_text", "")
        if original:
            return original
        return ref.decode_text(current_text)

    elif layer_type == "compact":
        # Structural unpacking + abbreviation reversal
        sub_log = decode_info.get("substitution_log", [])
        result = current_text

        # First, unpack structural rules
        pack_subs = [s for s in sub_log if isinstance(s, (list, tuple)) and len(s) == 3 and s[0] == "pack"]
        if pack_subs:
            result = ref.unpack_structural(result, pack_subs)

        # Then reverse word abbreviations
        vocab_subs = [s for s in sub_log if isinstance(s, (list, tuple)) and len(s) == 3 and s[0] == "vocab"]
        if vocab_subs:
            # Word-by-word reversal
            words = result.split()
            code_to_orig = {}
            for entry in vocab_subs:
                code_to_orig[entry[2]] = entry[1]
            decoded_words = []
            for w in words:
                if w in code_to_orig:
                    decoded_words.append(code_to_orig[w])
                else:
                    decoded_words.append(w)
            result = " ".join(decoded_words)

        # Fallback to pre_pack_text if available
        if not sub_log:
            pre = decode_info.get("pre_pack_text", "")
            if pre:
                return pre
            return ref.decode_text(current_text)

        return result

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
        "version": "5.0.0",
        "description": "Multi-level compression: word encoding -> meta-patterns (patterns of codes) -> structural packing",
    })


@app.route("/models", methods=["GET"])
def get_models():
    models = ollama_models()
    # Check if Ollama is reachable even if no models are pulled yet
    ollama_available = False
    try:
        r = http_req.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        ollama_available = r.status_code == 200
    except Exception:
        pass
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
            kwargs = {"temperature": layer_temp} if layer_temp is not None else {}
            result = layer_reference(current, active_ref, layer_num,
                                     model=model, provider=provider,
                                     api_key=layer_api_key, **kwargs)
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
            "version": "5.0",
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
                kwargs = {"temperature": layer_temp} if layer_temp is not None else {}
                result = layer_reference(current, active_ref, layer_num,
                                         model=model, provider=provider,
                                         api_key=layer_api_key, **kwargs)
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
                "version": "5.0",
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
        "meta_patterns": active_ref.meta_patterns if hasattr(active_ref, 'meta_patterns') else {},
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


@app.route("/debug/ollama", methods=["GET"])
def debug_ollama():
    """Debug endpoint -- shows Ollama startup log and system info."""
    import subprocess
    import shutil

    ollama_log = ""
    try:
        with open("/tmp/ollama.log", "r") as f:
            ollama_log = f.read()[-5000:]  # last 5KB
    except Exception as e:
        ollama_log = f"Could not read log: {e}"

    # Check if ollama binary exists
    ollama_bin = shutil.which("ollama")

    # Check if ollama process is running
    ollama_running = False
    try:
        result = subprocess.run(["pgrep", "-f", "ollama"], capture_output=True, text=True, timeout=5)
        ollama_running = result.returncode == 0
        ollama_pids = result.stdout.strip()
    except Exception:
        ollama_pids = "pgrep failed"

    # Try to reach ollama
    ollama_reachable = False
    ollama_api_error = ""
    try:
        r = http_req.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        ollama_reachable = r.status_code == 200
    except Exception as e:
        ollama_api_error = str(e)

    # System info
    mem_info = ""
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if any(k in line for k in ["MemTotal", "MemAvailable", "MemFree"]):
                    mem_info += line
    except Exception:
        mem_info = "unavailable"

    # Check OLLAMA_MODELS directory
    models_dir = os.environ.get("OLLAMA_MODELS", "/root/.ollama/models")
    models_exist = os.path.isdir(models_dir)
    model_files = []
    if models_exist:
        try:
            for root, dirs, files in os.walk(models_dir):
                for f in files:
                    fp = os.path.join(root, f)
                    model_files.append({"path": fp, "size": os.path.getsize(fp)})
        except Exception:
            pass

    return jsonify({
        "ollama_binary": ollama_bin,
        "ollama_running": ollama_running,
        "ollama_pids": ollama_pids,
        "ollama_reachable": ollama_reachable,
        "ollama_api_error": ollama_api_error,
        "ollama_url": OLLAMA_URL,
        "ollama_log": ollama_log,
        "memory": mem_info,
        "models_dir": models_dir,
        "models_dir_exists": models_exist,
        "model_files_count": len(model_files),
        "model_files": model_files[:20],
        "env_OLLAMA_URL": os.environ.get("OLLAMA_URL", "NOT SET"),
        "env_OLLAMA_MODELS": os.environ.get("OLLAMA_MODELS", "NOT SET"),
        "env_OLLAMA_HOST": os.environ.get("OLLAMA_HOST", "NOT SET"),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"Prefrontal Compressor v4 starting on port {port}")
    print(f"Ollama: {OLLAMA_URL}")
    print(f"Reference: {active_ref.name} v{active_ref.version} ({active_ref.get_fingerprint()})")
    print(f"GUI: http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
