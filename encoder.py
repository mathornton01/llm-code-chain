"""
Encoder: compresses natural language prompts into minimal token codes.

Strategy:
  1. Structural analysis -- identify intent, entities, relationships
  2. Codebook lookup -- replace known phrases with short codes
  3. LLM compression -- use Ollama to further compress novel content
  4. Packaging -- produce final compact representation
"""

import requests
from codebook import Codebook
from typing import Dict, List, Optional
import re
import time

OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen2.5:1.5b"

ENCODE_PROMPT = """Compress this text to the shortest possible form. Keep key words, drop filler. Use symbols: & (and), | (or), > (causes), = (equals), Q (question), C (command).

Text: "{text}"
Short:"""


class Encoder:
    """Encodes natural language into compact code representations."""

    def __init__(self, model: str = DEFAULT_MODEL, codebook: Optional[Codebook] = None,
                 ollama_url: str = OLLAMA_URL):
        self.codebook = codebook or Codebook()
        self.model = model
        self.ollama_url = ollama_url
        self.stats = {
            "total_encoded": 0,
            "total_input_chars": 0,
            "total_output_chars": 0,
            "avg_compression": 0.0,
        }

    def _ollama_generate(self, prompt: str, max_tokens: int = 64, temperature: float = 0.1) -> str:
        """Call Ollama generate API."""
        r = requests.post(f"{self.ollama_url}/api/generate", json={
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
                "stop": ["\n", "\n\n"],
            }
        }, timeout=120)
        r.raise_for_status()
        return r.json().get("response", "").strip()

    def _pre_compress(self, text: str) -> str:
        """Aggressive rule-based compression."""
        intent = ""
        lower = text.lower().strip()
        if lower.startswith(("what ", "how ", "why ", "when ", "where ", "who ")):
            intent = "Q "
        elif lower.startswith(("can you", "could you", "would you", "please")):
            intent = "R "
        elif lower.startswith(("create", "write", "build", "make", "implement", "set up", "deploy")):
            intent = "C "
        elif lower.startswith(("explain", "describe")):
            intent = "X "

        result = text.strip()

        removals = [
            "can you ", "could you ", "would you ", "please ",
            "i would like to ", "i want to ", "i need to ",
            "how do i ", "how does ", "how do ",
            "what is the ", "what are the ", "what is ", "what are ",
            "explain the ", "explain how ", "explain ",
            "describe the ", "describe how ", "describe ",
            "the process of ", "the difference between ",
            "in order to ", "in terms of ",
            " that ", " which ", " who ", " whom ",
            " very ", " really ", " just ", " actually ",
            " basically ", " essentially ", " simply ",
            " the ", " a ", " an ",
            " is ", " are ", " was ", " were ",
            " has ", " have ", " had ",
            " do ", " does ", " did ",
            " will ", " would ", " can ", " could ", " should ", " might ",
            " been ", " being ",
        ]

        r = " " + result + " "
        for filler in removals:
            r = r.replace(filler, " ")
        result = r.strip()

        result = re.sub(r'\s+', ' ', result).strip()

        # Use the codebook's full_encode for phrase + vocabulary substitution
        result, _subs = self.codebook.full_encode(result)

        # Abbreviate remaining long words not in codebook
        all_codes = set(self.codebook.vocabulary.values()) | set(self.codebook.phrases.values()) | set(self.codebook.learned.values())
        words = result.split()
        final = []
        for token in words:
            if len(token) > 6 and token not in all_codes:
                final.append(token[:4])
            else:
                final.append(token)

        compressed = intent + ' '.join(final)

        compressed = compressed.replace(' and ', '&')
        compressed = compressed.replace(' or ', '|')
        compressed = compressed.replace(' but ', '^')
        compressed = compressed.replace(' not ', '!')
        compressed = compressed.replace(' versus ', '|')
        compressed = compressed.replace(' vs ', '|')
        compressed = compressed.replace(' from ', '<')
        compressed = compressed.replace(' into ', '>')
        compressed = compressed.replace(' to ', '>')

        return compressed.strip()

    def _llm_compress(self, text: str, pre_compressed: str) -> str:
        """Use Ollama to further compress."""
        prompt = ENCODE_PROMPT.format(text=text)
        result = self._ollama_generate(prompt, max_tokens=64, temperature=0.1)

        # Clean artifacts
        result = result.replace('"', '').replace("'", "")
        for prefix in ["Short:", "Compressed:", "Code:", "Output:"]:
            if result.startswith(prefix):
                result = result[len(prefix):].strip()

        if not result or len(result) >= len(pre_compressed):
            return pre_compressed

        return result

    def encode(self, text: str, use_llm: bool = True) -> Dict:
        """Encode a natural language prompt into compact code."""
        start = time.time()

        pre_compressed = self._pre_compress(text)

        if use_llm:
            try:
                llm_result = self._llm_compress(text, pre_compressed)
                if len(llm_result) < len(pre_compressed):
                    code = llm_result
                    method = "llm"
                else:
                    code = pre_compressed
                    method = "rule-based (llm not shorter)"
            except Exception as e:
                code = pre_compressed
                method = f"rule-based (llm error: {e})"
        else:
            code = pre_compressed
            method = "rule-based"

        elapsed = (time.time() - start) * 1000

        tokens_orig = max(1, len(text) // 4)
        tokens_enc = max(1, len(code) // 4)
        ratio = 1.0 - (len(code) / max(1, len(text)))

        self.stats["total_encoded"] += 1
        self.stats["total_input_chars"] += len(text)
        self.stats["total_output_chars"] += len(code)
        if self.stats["total_input_chars"] > 0:
            self.stats["avg_compression"] = 1.0 - (
                self.stats["total_output_chars"] / self.stats["total_input_chars"]
            )

        return {
            "code": code,
            "original": text,
            "pre_compressed": pre_compressed,
            "compression_ratio": round(ratio, 3),
            "tokens_original": tokens_orig,
            "tokens_encoded": tokens_enc,
            "token_reduction": f"{tokens_orig} -> {tokens_enc} ({round(ratio * 100)}% smaller)",
            "method": method,
            "time_ms": round(elapsed, 1),
        }

    def encode_batch(self, texts: List[str], use_llm: bool = True) -> List[Dict]:
        return [self.encode(t, use_llm=use_llm) for t in texts]

    def get_stats(self) -> dict:
        return dict(self.stats)
