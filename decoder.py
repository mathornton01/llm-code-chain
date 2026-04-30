"""
Decoder: reconstructs meaning from compact code representations.

Strategy:
  1. Code parsing -- identify structure markers, operators, codes
  2. Codebook expansion -- expand known codes to meanings
  3. LLM reconstruction -- use Ollama to produce natural language
"""

import requests
from codebook import Codebook
from typing import Dict, Optional
import re
import time

OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5:1.5b"

DECODE_PROMPT = """Expand this compressed text back into a clear English sentence. The symbols mean: & (and), | (or), > (leads to), < (from), = (equals), Q (question), C (command), R (request), X (explain).

Compressed: {code}
Hints: {expanded}
English:"""


class Decoder:
    """Decodes compact codes back into natural language."""

    def __init__(self, model: str = DEFAULT_MODEL, codebook: Optional[Codebook] = None,
                 ollama_url: str = OLLAMA_URL):
        self.codebook = codebook or Codebook()
        self.model = model
        self.ollama_url = ollama_url
        self.stats = {
            "total_decoded": 0,
            "total_input_chars": 0,
            "total_output_chars": 0,
        }

    def _ollama_generate(self, prompt: str, max_tokens: int = 128, temperature: float = 0.3) -> str:
        """Call Ollama generate API."""
        r = requests.post(f"{self.ollama_url}/api/generate", json={
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "stop": ["\n\n", "\n"],
            }
        }, timeout=120)
        r.raise_for_status()
        return r.json().get("response", "").strip()

    def _expand_codes(self, code: str) -> str:
        """Rule-based expansion of codes to meanings."""
        result = code
        result = result.replace('&', ' and ')
        result = result.replace('|', ' or ')
        result = result.replace('^', ' but ')
        result = result.replace('!', ' not ')
        result = result.replace('>', ' to ')
        result = result.replace('<', ' from ')
        result = result.replace('=', ' equals ')

        for prefix in ['Q ', 'C ', 'R ', 'X ', 'D ', 'A ']:
            if result.startswith(prefix):
                result = result[2:]
                break

        # Use codebook's decode_text for reverse lookup
        result = self.codebook.decode_text(result)

        # Also handle domain prefixes
        tokens = result.split()
        expanded = []
        for token in tokens:
            if '.' in token and len(token.split('.')[0]) == 1:
                parts = token.split('.', 1)
                prefix_map = {
                    'c': 'programming', 'm': 'math', 'b': 'biology',
                    'p': 'physics', 'l': 'language', 'w': 'web',
                    'h': 'hardware', 'a': 'AI/ML',
                }
                domain = prefix_map.get(parts[0], '')
                expanded.append(f"{domain} {parts[1]}" if domain else token)
            else:
                expanded.append(token)

        return ' '.join(expanded)

    def _llm_decode(self, code: str, expanded: str) -> str:
        """Use Ollama to reconstruct natural language."""
        prompt = DECODE_PROMPT.format(code=code, expanded=expanded)
        result = self._ollama_generate(prompt, max_tokens=128, temperature=0.3)

        for prefix in ["English:", "Decoded:", "Output:", "Expanded:"]:
            if result.startswith(prefix):
                result = result[len(prefix):].strip()
        result = result.strip('"').strip("'")

        if not result:
            return expanded

        return result

    def decode(self, code: str, use_llm: bool = True) -> Dict:
        """Decode a compact code back to natural language."""
        start = time.time()

        expanded = self._expand_codes(code)

        if use_llm:
            try:
                decoded = self._llm_decode(code, expanded)
                method = "llm"
            except Exception as e:
                decoded = expanded
                method = f"rule-based (llm error: {e})"
        else:
            decoded = expanded
            method = "rule-based"

        elapsed = (time.time() - start) * 1000

        expansion = len(decoded) / max(1, len(code))

        self.stats["total_decoded"] += 1
        self.stats["total_input_chars"] += len(code)
        self.stats["total_output_chars"] += len(decoded)

        return {
            "decoded": decoded,
            "code": code,
            "expanded": expanded,
            "method": method,
            "expansion_ratio": round(expansion, 2),
            "time_ms": round(elapsed, 1),
        }

    def get_stats(self) -> dict:
        return dict(self.stats)
