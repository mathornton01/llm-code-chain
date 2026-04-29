"""
Encoder: compresses natural language prompts into minimal token codes.

Strategy:
  1. Structural analysis -- identify intent, entities, relationships
  2. Codebook lookup -- replace known phrases with short codes
  3. LLM compression -- use the model to further compress novel content
  4. Packaging -- produce final compact representation
"""

from llama_cpp import Llama
from codebook import Codebook
from typing import Dict, List, Tuple, Optional
import re
import json
import time


ENCODE_PROMPT = """Compress this text to the shortest possible form. Keep key words, drop filler. Use symbols: & (and), | (or), > (causes), = (equals), Q (question), C (command).

Text: "{text}"
Short:"""


class Encoder:
    """Encodes natural language into compact code representations."""

    def __init__(self, model_path: str, codebook: Optional[Codebook] = None):
        self.codebook = codebook or Codebook()
        self.model_path = model_path
        self.llm = None
        self.stats = {
            "total_encoded": 0,
            "total_input_chars": 0,
            "total_output_chars": 0,
            "avg_compression": 0.0,
        }

    def _ensure_model(self):
        """Lazy-load the model."""
        if self.llm is None:
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=1024,
                n_threads=4,
                verbose=False,
            )

    def _pre_compress(self, text: str) -> str:
        """Aggressive rule-based compression."""
        # Detect intent prefix
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

        # Remove filler words (order matters -- longer phrases first)
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

        # Collapse whitespace
        result = re.sub(r'\s+', ' ', result).strip()

        # Apply codebook substitutions
        words = result.split()
        coded = []
        skip_next = 0
        for i, w in enumerate(words):
            if skip_next > 0:
                skip_next -= 1
                continue

            # Try trigram match
            if i + 2 < len(words):
                tri = f"{w} {words[i+1]} {words[i+2]}".lower()
                if tri in self.codebook.reverse:
                    coded.append(self.codebook.reverse[tri])
                    skip_next = 2
                    continue

            # Try bigram match
            if i + 1 < len(words):
                bi = f"{w} {words[i+1]}".lower()
                if bi in self.codebook.reverse:
                    coded.append(self.codebook.reverse[bi])
                    skip_next = 1
                    continue

            # Single word
            coded.append(self.codebook.encode_token(w))

        # Abbreviate remaining long words
        final = []
        for token in coded:
            if len(token) > 6 and token.lower() not in self.codebook.reverse:
                # Keep first 4 chars of long unknown words
                final.append(token[:4])
            else:
                final.append(token)

        compressed = intent + ' '.join(final)

        # Replace common connectors with symbols
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
        """Use the LLM to further compress."""
        self._ensure_model()

        prompt = ENCODE_PROMPT.format(text=text)

        output = self.llm(
            prompt,
            max_tokens=64,
            temperature=0.1,
            stop=["\n", "\n\n"],
        )

        result = output["choices"][0]["text"].strip()

        # Clean artifacts
        result = result.replace('"', '').replace("'", "")
        for prefix in ["Short:", "Compressed:", "Code:", "Output:"]:
            if result.startswith(prefix):
                result = result[len(prefix):].strip()

        # If LLM returned empty or longer than pre-compressed, fall back
        if not result or len(result) >= len(pre_compressed):
            return pre_compressed

        return result

    def encode(self, text: str, use_llm: bool = True) -> Dict:
        """
        Encode a natural language prompt into compact code.
        """
        start = time.time()

        pre_compressed = self._pre_compress(text)

        if use_llm:
            try:
                llm_result = self._llm_compress(text, pre_compressed)
                # Use whichever is shorter
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
        ratio = self.codebook.compress_ratio(text, code)

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
