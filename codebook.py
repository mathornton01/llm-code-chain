"""
Codebook: the shared vocabulary between encoder and decoder.

Maps common phrases, concepts, and patterns to short code tokens.
The codebook is built in layers:
  1. Base codes: single characters/symbols for the most common concepts
  2. Domain codes: short prefixes for domain-specific terminology
  3. Structure codes: encoding for grammar, relationships, intent
  4. Learned codes: discovered through training iterations
"""

import json
import os
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

CODEBOOK_PATH = os.path.join(os.path.dirname(__file__), "codebook.json")


# Base concept codes -- single/double char for highest frequency concepts
BASE_CODES = {
    # Intent markers
    "Q": "question",
    "A": "answer",
    "C": "command/instruction",
    "D": "description",
    "R": "request",
    "X": "explanation",
    "N": "negation/not",
    "Y": "affirmation/yes",

    # Relationship operators
    ">": "leads to / causes",
    "<": "comes from / caused by",
    "=": "equals / is the same as",
    "~": "similar to / approximately",
    "!": "important / emphasis",
    "&": "and / additionally",
    "|": "or / alternatively",
    "^": "contrasts with / but",

    # Scope markers
    "(": "begin group",
    ")": "end group",
    "[": "begin list",
    "]": "end list",
    "{": "begin conditional",
    "}": "end conditional",

    # Quantifiers
    "1": "one / single / a",
    "*": "many / all / every",
    "0": "none / zero / nothing",
    "+": "more / increase",
    "-": "less / decrease",

    # Time
    "t.now": "present / currently",
    "t.past": "previously / before",
    "t.fut": "will / future",

    # Common verbs (2-char)
    "mk": "make / create / build",
    "gt": "get / obtain / retrieve",
    "st": "set / assign / configure",
    "rm": "remove / delete",
    "mv": "move / transfer",
    "fn": "find / search / locate",
    "rn": "run / execute",
    "sh": "show / display",
    "wr": "write / output",
    "rd": "read / input",
    "tx": "transform / convert",
    "cmp": "compare",
    "hlp": "help / assist",

    # Common nouns (2-3 char)
    "f": "file",
    "d": "directory / folder",
    "s": "string / text",
    "n": "number / integer",
    "ls": "list / array",
    "fn": "function / method",
    "cl": "class / object",
    "er": "error / exception",
    "msg": "message",
    "usr": "user / person",
    "sys": "system",
    "dat": "data",
    "cfg": "config / settings",
    "net": "network / internet",
    "db": "database",
    "api": "interface / endpoint",
    "ui": "interface / display",
    "doc": "document",

    # Adjectives
    "bg": "big / large",
    "sm": "small / little",
    "nw": "new",
    "od": "old",
    "gd": "good",
    "bd": "bad",
    "fs": "fast / quick",
    "sl": "slow",
}

# Domain-specific prefix codes
DOMAIN_PREFIXES = {
    "c.": "code/programming",
    "m.": "math/statistics",
    "b.": "biology/medical",
    "p.": "physics",
    "l.": "language/linguistics",
    "w.": "web/internet",
    "h.": "hardware/electronics",
    "a.": "AI/machine learning",
}


class Codebook:
    """Manages the encoding/decoding codebook with learning capabilities."""

    def __init__(self, path: Optional[str] = None):
        self.path = path or CODEBOOK_PATH
        self.codes: Dict[str, str] = dict(BASE_CODES)
        self.reverse: Dict[str, str] = {}
        self.frequency: Dict[str, int] = defaultdict(int)
        self.learned: Dict[str, str] = {}
        self.version = 1
        self._build_reverse()
        self._load()

    def _build_reverse(self):
        """Build reverse lookup (meaning -> code)."""
        self.reverse = {}
        for code, meaning in self.codes.items():
            # Handle multiple meanings separated by /
            for m in meaning.split(" / "):
                m = m.strip().lower()
                self.reverse[m] = code

    def _load(self):
        """Load learned codes from disk."""
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    data = json.load(f)
                self.learned = data.get("learned", {})
                self.frequency = defaultdict(int, data.get("frequency", {}))
                self.version = data.get("version", 1)
                self.codes.update(self.learned)
                self._build_reverse()
            except (json.JSONDecodeError, IOError):
                pass

    def save(self):
        """Persist learned codes and frequencies to disk."""
        data = {
            "version": self.version,
            "learned": self.learned,
            "frequency": dict(self.frequency),
        }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def encode_token(self, word: str) -> str:
        """Look up the shortest code for a word/phrase."""
        w = word.strip().lower()
        if w in self.reverse:
            code = self.reverse[w]
            self.frequency[code] += 1
            return code
        return word  # no code found, pass through

    def decode_token(self, code: str) -> str:
        """Expand a code back to its meaning."""
        if code in self.codes:
            return self.codes[code]
        return code

    def learn(self, phrase: str, code: str):
        """Add a new learned code mapping."""
        self.learned[code] = phrase
        self.codes[code] = phrase
        self.reverse[phrase.lower()] = code
        self.version += 1

    def get_stats(self) -> dict:
        """Return codebook statistics."""
        return {
            "total_codes": len(self.codes),
            "base_codes": len(BASE_CODES),
            "learned_codes": len(self.learned),
            "version": self.version,
            "top_used": sorted(
                self.frequency.items(), key=lambda x: x[1], reverse=True
            )[:20],
        }

    def suggest_code(self, phrase: str) -> str:
        """Generate a short code suggestion for a phrase."""
        words = phrase.lower().split()
        if len(words) == 1:
            w = words[0]
            # Try first 2-3 consonants
            consonants = [c for c in w if c not in "aeiou"]
            if len(consonants) >= 2:
                return "".join(consonants[:3])
            return w[:3]
        else:
            # Initials of each word
            return "".join(w[0] for w in words if w)

    def compress_ratio(self, original: str, encoded: str) -> float:
        """Calculate compression ratio."""
        if not original:
            return 0.0
        return 1.0 - (len(encoded) / len(original))
