#!/usr/bin/env python3
"""
CLI interface for LLM Code Chain.

Interactive REPL for encoding/decoding prompts.
Uses Ollama for local LLM inference.
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(__file__))

from encoder import Encoder
from decoder import Decoder
from codebook import Codebook
from trainer import Trainer

DEFAULT_MODEL = os.environ.get("LCC_MODEL", "qwen2.5:1.5b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")

HELP = """
LLM Code Chain -- Interactive CLI (Ollama backend)
====================================================

Commands:
  encode <text>     Compress text to minimal code
  decode <code>     Expand code back to natural language
  roundtrip <text>  Encode then decode (test fidelity)
  train [n]         Run n training rounds (default: 1)
  codebook          Show current codebook
  stats             Show compression statistics
  model <name>      Switch Ollama model (e.g. phi3:mini, llama3.2:1b)
  fast              Toggle LLM off (rule-based only, instant)
  slow              Toggle LLM on (better compression, slower)
  help              Show this help
  quit              Exit

Shortcuts:
  e <text>          Same as encode
  d <code>          Same as decode
  r <text>          Same as roundtrip
  t [n]             Same as train
"""


def main():
    print(f"\nLLM Code Chain v0.2.0 (Ollama)")
    print(f"Model: {DEFAULT_MODEL} @ {OLLAMA_URL}")

    codebook = Codebook()
    model = DEFAULT_MODEL
    encoder = Encoder(model=model, codebook=codebook, ollama_url=OLLAMA_URL)
    decoder = Decoder(model=model, codebook=codebook, ollama_url=OLLAMA_URL)

    use_llm = True
    print(f"Mode: {'LLM-assisted' if use_llm else 'Rule-based (fast)'}")
    print("Type 'help' for commands.\n")

    while True:
        try:
            line = input(">> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not line:
            continue

        parts = line.split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if cmd in ("quit", "exit", "q"):
            print("Bye.")
            break

        elif cmd in ("help", "h", "?"):
            print(HELP)

        elif cmd in ("model", "m"):
            if not arg:
                print(f"  Current model: {model}")
                continue
            model = arg.strip()
            encoder = Encoder(model=model, codebook=codebook, ollama_url=OLLAMA_URL)
            decoder = Decoder(model=model, codebook=codebook, ollama_url=OLLAMA_URL)
            print(f"  Switched to model: {model}")

        elif cmd in ("encode", "e"):
            if not arg:
                print("Usage: encode <text>")
                continue
            result = encoder.encode(arg, use_llm=use_llm)
            print(f"\n  Input:       {result['original']}")
            print(f"  Code:        {result['code']}")
            print(f"  Compression: {result['compression_ratio']:.0%}")
            print(f"  Tokens:      {result['token_reduction']}")
            print(f"  Method:      {result['method']}")
            print(f"  Time:        {result['time_ms']:.0f}ms\n")

        elif cmd in ("decode", "d"):
            if not arg:
                print("Usage: decode <code>")
                continue
            result = decoder.decode(arg, use_llm=use_llm)
            print(f"\n  Code:      {result['code']}")
            print(f"  Expanded:  {result['expanded']}")
            print(f"  Decoded:   {result['decoded']}")
            print(f"  Method:    {result['method']}")
            print(f"  Time:      {result['time_ms']:.0f}ms\n")

        elif cmd in ("roundtrip", "r"):
            if not arg:
                print("Usage: roundtrip <text>")
                continue
            enc = encoder.encode(arg, use_llm=use_llm)
            dec = decoder.decode(enc["code"], use_llm=use_llm)

            orig_words = set(arg.lower().split())
            dec_words = set(dec["decoded"].lower().split())
            key_orig = {w for w in orig_words if len(w) > 4}
            key_dec = {w for w in dec_words if len(w) > 4}
            fidelity = len(key_orig & key_dec) / max(1, len(key_orig))

            print(f"\n  Original:    {arg}")
            print(f"  Encoded:     {enc['code']}")
            print(f"  Decoded:     {dec['decoded']}")
            print(f"  Compression: {enc['compression_ratio']:.0%}")
            print(f"  Fidelity:    {fidelity:.0%}")
            print(f"  Time:        {enc['time_ms'] + dec['time_ms']:.0f}ms\n")

        elif cmd in ("train", "t"):
            rounds = int(arg) if arg.isdigit() else 1
            trainer = Trainer(model=model, codebook=codebook, ollama_url=OLLAMA_URL)
            trainer.train(rounds=rounds, use_llm=use_llm)

        elif cmd in ("codebook", "cb"):
            stats = codebook.get_stats()
            print(f"\n  Codebook v{stats['version']}")
            print(f"  Total codes:   {stats['total_codes']}")
            print(f"  Vocabulary:    {stats['vocabulary_count']}")
            print(f"  Phrases:       {stats['phrase_count']}")
            print(f"  Learned codes: {stats['learned_count']}")
            if stats['top_used']:
                print(f"\n  Top used codes:")
                for code, count in stats['top_used'][:10]:
                    meaning = codebook.decode_vocab.get(code, codebook.decode_phrases.get(code, codebook.decode_learned.get(code, '?')))
                    print(f"    {code:8s} = {meaning:30s} (used {count}x)")
            print()

        elif cmd in ("stats", "s"):
            enc_stats = encoder.get_stats()
            dec_stats = decoder.get_stats()
            print(f"\n  Encoder:")
            print(f"    Total encoded:     {enc_stats['total_encoded']}")
            print(f"    Avg compression:   {enc_stats['avg_compression']:.0%}")
            print(f"  Decoder:")
            print(f"    Total decoded:     {dec_stats['total_decoded']}")
            print()

        elif cmd == "fast":
            use_llm = False
            print("  Mode: Rule-based (fast, no LLM)")

        elif cmd == "slow":
            use_llm = True
            print("  Mode: LLM-assisted (better compression)")

        else:
            # Default: treat as encode
            result = encoder.encode(line, use_llm=use_llm)
            print(f"\n  Code:        {result['code']}")
            print(f"  Compression: {result['compression_ratio']:.0%}")
            print(f"  Time:        {result['time_ms']:.0f}ms\n")


if __name__ == "__main__":
    main()
