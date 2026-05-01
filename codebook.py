"""
Shared Reference File: the bidirectional dictionary shared between encoder and decoder.

Both sides must have the same version of this file to encode/decode.
It gets refined through training cycles (encode -> decode -> compare -> learn).

Sections:
  1. vocabulary: word -> code mappings (bidirectional)
  2. phrases: multi-word phrase -> code mappings
  3. domains: domain-prefixed codes (c.=code, m.=math, b.=bio, etc.)
  4. learned: codes discovered through training
  5. meta: version, stats, training history
"""

import json
import os
import hashlib
import time
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

REFERENCES_DIR = os.path.join(os.path.dirname(__file__), "references")


def ensure_references_dir():
    os.makedirs(REFERENCES_DIR, exist_ok=True)


# Default vocabulary: high-frequency words -> short codes
DEFAULT_VOCABULARY = {
    # Articles - mapped to short unique tokens (not removed)
    # Removing articles entirely caused decode artifacts since position info was lost
    "the": "T",
    "an": "N",
    # "a" is 1 char -- no savings from encoding, so skip it
    # Pronouns
    "you": "u",
    "your": "ur",
    "are": "r",
    "is": "=",
    "was": "ws",
    "were": "wr",
    "will": "wl",
    "would": "wd",
    "could": "cd",
    "should": "shd",
    "have": "hv",
    "has": "hs",
    "had": "hd",
    "been": "bn",
    "being": "bng",
    "that": "th",
    "this": "ths",
    "with": "w/",
    "from": "<",
    "into": "->",
    "between": "btw",
    "through": "thru",
    "about": "abt",
    "because": "b/c",
    "however": "hwvr",
    "therefore": ":..",
    "although": "altho",
    "whether": "whthr",
    # Conjunctions
    "and": "&",
    "or": "|",
    "but": "^",
    "not": "!",
    # Common verbs
    "make": "mk",
    "create": "mk",
    "build": "mk",
    "get": "gt",
    "obtain": "gt",
    "retrieve": "gt",
    "set": "st",
    "configure": "cfg",
    "remove": "rm",
    "delete": "rm",
    "find": "fn",
    "search": "fn",
    "locate": "fn",
    "run": "rn",
    "execute": "rn",
    "show": "sh",
    "display": "sh",
    "write": "wr",
    "read": "rd",
    "use": "us",
    "using": "us",
    "used": "usd",
    "implement": "impl",
    "explain": "xpln",
    "describe": "desc",
    "understand": "undst",
    "analyze": "anlz",
    "compare": "cmp",
    "calculate": "calc",
    "provide": "prov",
    "include": "incl",
    "require": "req",
    "define": "def",
    "apply": "aply",
    "determine": "dtrm",
    "develop": "dev",
    "generate": "gen",
    "process": "proc",
    "produce": "prod",
    "consider": "cnsd",
    "contain": "cntn",
    "represent": "repr",
    "function": "fn",
    "following": "fllw",
    "different": "diff",
    "important": "imp",
    "specific": "spec",
    "possible": "psbl",
    "available": "avail",
    "necessary": "nec",
    "significant": "sig",
    "additional": "addl",
    "information": "info",
    "example": "ex",
    "system": "sys",
    "number": "num",
    "data": "dat",
    "which": "wh",
    "where": "whr",
    "when": "whn",
    "what": "wt",
    "how": "hw",
    "does": "ds",
    "their": "thr",
    "there": "thr",
    "these": "ths",
    "those": "thos",
    "other": "othr",
    "each": "ea",
    "also": "+",
    "more": "+",
    "very": "v",
    "just": "jst",
    "only": "onl",
    "then": "thn",
    "than": "thn",
    "some": "sm",
    "such": "sch",
    "like": "lk",
    "well": "wl",
    "even": "evn",
    "most": "mst",
    "many": "mny",
    "much": "mch",
    "both": "bth",
    "before": "bf",
    "after": "af",
    "over": "ovr",
    "under": "undr",
    "during": "dur",
    "while": "whl",
    "since": "snc",
    "without": "w/o",
    "within": "w/in",
    "above": "abv",
    "below": "blw",
    "need": "nd",
    "want": "wnt",
    "help": "hlp",
    "know": "knw",
    "think": "thnk",
    "work": "wrk",
    "give": "gv",
    "take": "tk",
    "come": "cm",
    "look": "lk",
    "call": "cl",
    "keep": "kp",
    "let": "lt",
    "begin": "bgn",
    "start": "strt",
    "seem": "sm",
    "might": "mgt",
    "must": "mst",
    "shall": "shl",
    "cannot": "cnt",
    "can": "cn",
    "able": "abl",
    "please": "pls",
    # Technical terms
    "function": "fn",
    "variable": "var",
    "parameter": "param",
    "argument": "arg",
    "string": "str",
    "integer": "int",
    "boolean": "bool",
    "array": "arr",
    "object": "obj",
    "class": "cls",
    "method": "mthd",
    "module": "mod",
    "package": "pkg",
    "library": "lib",
    "framework": "fwk",
    "interface": "ifc",
    "exception": "exc",
    "error": "err",
    "warning": "wrn",
    "message": "msg",
    "request": "req",
    "response": "res",
    "server": "srv",
    "client": "clt",
    "network": "ntwk",
    "connection": "conn",
    "session": "sess",
    "token": "tok",
    "memory": "mem",
    "performance": "perf",
    "application": "app",
    "environment": "env",
    "development": "dev",
    "production": "prod",
    "testing": "tst",
    "deployment": "dply",
    "configuration": "cfg",
    "documentation": "docs",
    "repository": "repo",
    "directory": "dir",
    "algorithm": "algo",
    "iteration": "iter",
    "recursive": "rcrs",
    "container": "ctnr",
    "component": "cmpnt",
    "template": "tmpl",
    "instance": "inst",
    "property": "prop",
    "attribute": "attr",
    "default": "dflt",
    "maximum": "max",
    "minimum": "min",
    "average": "avg",
    "frequency": "freq",
    "probability": "prob",
    "distribution": "dist",
    "optimization": "opt",
    "authentication": "auth",
    "authorization": "authz",
    "encryption": "enc",
    "compression": "cmp",
    "operation": "op",
    "transaction": "txn",
    "validation": "valid",
    "notification": "notif",
    "temperature": "temp",
    "approximately": "~",
    "between": "btw",
    "currently": "cur",
    "especially": "esp",
    "originally": "orig",
    "previously": "prev",
    "typically": "typ",
    "usually": "usu",
    "always": "alw",
    "never": "nvr",
    "sometimes": "smt",
    "everything": "evryth",
    "something": "smth",
    "anything": "anyth",
    "nothing": "nth",
    "someone": "smo",
    "everyone": "evryo",
    "already": "alrdy",
    "actually": "actly",
    "probably": "prob",
    "different": "diff",
    "important": "imp",
    "necessary": "nec",
    "possible": "psbl",
    "available": "avail",
    "specific": "spec",
    "significant": "sig",
    "additional": "addl",
    "example": "eg",
    "through": "thru",
    "whether": "whthr",
    "because": "b/c",
    "although": "altho",
    "however": "hwvr",
    "therefore": ":..",
    "according": "acc",
    "regarding": "re:",
    "including": "incl",
    "following": "fllw",
}

# Multi-word phrases -> short codes
DEFAULT_PHRASES = {
    "machine learning": "a.ml",
    "deep learning": "a.dl",
    "neural network": "a.nn",
    "neural networks": "a.nns",
    "natural language": "a.nlp",
    "artificial intelligence": "a.ai",
    "large language model": "a.llm",
    "gradient descent": "a.gd",
    "attention mechanism": "a.attn",
    "transformer model": "a.xfmr",
    "random forest": "a.rf",
    "gradient boosting": "a.gb",
    "convolutional neural network": "a.cnn",
    "recurrent neural network": "a.rnn",
    "reinforcement learning": "a.rl",
    "supervised learning": "a.sl",
    "unsupervised learning": "a.ul",
    "transfer learning": "a.tl",
    "binary search tree": "c.bst",
    "data structure": "c.ds",
    "data pipeline": "c.dp",
    "rest api": "c.rest",
    "api endpoint": "c.ep",
    "source code": "c.src",
    "database": "c.db",
    "version control": "c.vc",
    "command line": "c.cli",
    "file system": "c.fs",
    "operating system": "c.os",
    "open source": "c.oss",
    "blood pressure": "b.bp",
    "heart rate": "b.hr",
    "side effects": "b.se",
    "clinical trial": "b.ct",
    "gene expression": "b.gx",
    "protein folding": "b.pf",
    "amino acid": "b.aa",
    "dna sequence": "b.dna",
    "central limit theorem": "m.clt",
    "standard deviation": "m.sd",
    "confidence interval": "m.ci",
    "null hypothesis": "m.h0",
    "p-value": "m.pv",
    "type i error": "m.t1e",
    "type ii error": "m.t2e",
    "hypothesis testing": "m.ht",
    "fourier transform": "m.ft",
    "linear regression": "m.lr",
    "bayesian": "m.bay",
    "high availability": "w.ha",
    "load balancer": "w.lb",
    "microservice": "w.ms",
    "container orchestration": "w.k8s",
    "continuous integration": "w.ci",
    "continuous deployment": "w.cd",
    "infrastructure as code": "w.iac",
    "domain name": "w.dns",
    # DevOps / Cloud
    "docker container": "d.dkr",
    "kubernetes cluster": "d.k8s",
    "virtual machine": "d.vm",
    "cloud computing": "d.cld",
    "object storage": "d.s3",
    "message queue": "d.mq",
    "key value store": "d.kv",
    "gpu acceleration": "d.gpu",
    # Data Science
    "feature engineering": "ds.fe",
    "cross validation": "ds.cv",
    "training data": "ds.td",
    "test data": "ds.tstd",
    "data warehouse": "ds.dwh",
    "data lake": "ds.dlk",
    "batch processing": "ds.bp",
    "real time": "ds.rt",
    "time series": "ds.ts",
    # Security
    "access control": "s.acl",
    "two factor": "s.2fa",
    "single sign on": "s.sso",
    "rate limiter": "s.rl",
    "token bucket": "s.tb",
    # Programming patterns
    "design pattern": "p.dp",
    "dependency injection": "p.di",
    "event driven": "p.ed",
    "publish subscribe": "p.ps",
    "observer pattern": "p.obs",
    "factory pattern": "p.fac",
    "in order to": ">",
    "as well as": "&",
    "on the other hand": "^",
    "for example": "eg",
    "such as": "eg",
    "in addition": "+",
    "can you": "Q",
    "could you": "Q",
    "would you": "Q",
    "i would like to": "R",
    "i want to": "R",
    "i need to": "R",
    "how do i": "Q hw",
    "how does": "Q hw",
    "what is": "Q wt=",
    "what are": "Q wt=",
}

# Domain prefix descriptions (for display/documentation)
DOMAIN_PREFIXES = {
    "c.": "code/programming",
    "m.": "math/statistics",
    "b.": "biology/medical",
    "p.": "physics",
    "l.": "language/linguistics",
    "w.": "web/infrastructure",
    "h.": "hardware/electronics",
    "a.": "AI/machine learning",
}


class SharedReference:
    """
    The shared reference file between encoder and decoder.

    Both sides must have the same version. The reference contains all the
    mappings needed to deterministically encode and decode text.

    Five encoding levels (hierarchical):
      1. vocabulary: word -> code  (single words)
      2. phrases: multi-word -> code  (multi-word expressions)
      3. learned: LLM-discovered word/phrase -> code
      4. meta_patterns: code_sequence -> meta_code  (patterns OF codes)
      5. meta2_patterns: meta_sequence -> meta2_code  (patterns OF meta-patterns)
    """

    def __init__(self, name: str = "default"):
        ensure_references_dir()
        self.name = name
        self.path = os.path.join(REFERENCES_DIR, f"{name}.json")
        self.version = 1
        self.created_at = time.time()
        self.updated_at = time.time()

        # Forward: word/phrase -> code
        self.vocabulary: Dict[str, str] = dict(DEFAULT_VOCABULARY)
        self.phrases: Dict[str, str] = dict(DEFAULT_PHRASES)
        self.learned: Dict[str, str] = {}

        # Meta-patterns: code_sequence -> meta_code (patterns OF codes)
        # e.g. "= v imp" -> "M.1", "T a.ml" -> "M.2"
        self.meta_patterns: Dict[str, str] = {}

        # Meta²-patterns: meta_sequence -> meta2_code (patterns OF meta-patterns)
        # e.g. "M.1 M.3" -> "MM.1", "M.2 algo M.1" -> "MM.2"
        self.meta2_patterns: Dict[str, str] = {}

        # Reverse: code -> word/phrase (built automatically)
        self.decode_vocab: Dict[str, str] = {}
        self.decode_phrases: Dict[str, str] = {}
        self.decode_learned: Dict[str, str] = {}
        self.decode_meta: Dict[str, str] = {}
        self.decode_meta2: Dict[str, str] = {}

        # Stats
        self.frequency: Dict[str, int] = defaultdict(int)
        self.training_rounds: int = 0
        self.total_compressions: int = 0

        self._build_reverse()
        self._load()

    def _build_reverse(self):
        """Build reverse lookup tables."""
        self.decode_vocab = {}
        for word, code in self.vocabulary.items():
            if code and code not in self.decode_vocab:
                self.decode_vocab[code] = word

        self.decode_phrases = {}
        for phrase, code in self.phrases.items():
            if code not in self.decode_phrases:
                self.decode_phrases[code] = phrase

        self.decode_learned = {}
        for item, code in self.learned.items():
            if code not in self.decode_learned:
                self.decode_learned[code] = item

        self.decode_meta = {}
        for seq, meta_code in self.meta_patterns.items():
            if meta_code not in self.decode_meta:
                self.decode_meta[meta_code] = seq

        self.decode_meta2 = {}
        for seq, meta2_code in self.meta2_patterns.items():
            if meta2_code not in self.decode_meta2:
                self.decode_meta2[meta2_code] = seq

    def _load(self):
        """Load from disk if exists."""
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r") as f:
                data = json.load(f)
            self.version = data.get("version", 1)
            self.created_at = data.get("created_at", self.created_at)
            self.updated_at = data.get("updated_at", self.updated_at)
            self.training_rounds = data.get("training_rounds", 0)
            self.total_compressions = data.get("total_compressions", 0)

            # Merge loaded data (loaded takes priority over defaults)
            if "vocabulary" in data:
                self.vocabulary.update(data["vocabulary"])
            if "phrases" in data:
                self.phrases.update(data["phrases"])
            self.learned = data.get("learned", {})
            self.meta_patterns = data.get("meta_patterns", {})
            self.meta2_patterns = data.get("meta2_patterns", {})
            self.frequency = defaultdict(int, data.get("frequency", {}))

            self._build_reverse()
        except (json.JSONDecodeError, IOError):
            pass

    def save(self):
        """Persist to disk."""
        self.updated_at = time.time()
        data = {
            "name": self.name,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "training_rounds": self.training_rounds,
            "total_compressions": self.total_compressions,
            "vocabulary": self.vocabulary,
            "phrases": self.phrases,
            "learned": self.learned,
            "meta_patterns": self.meta_patterns,
            "meta2_patterns": self.meta2_patterns,
            "frequency": dict(self.frequency),
        }
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

    def get_fingerprint(self) -> str:
        """Short hash identifying this exact version of the reference."""
        content = json.dumps({
            "v": self.version,
            "vocab_count": len(self.vocabulary),
            "phrase_count": len(self.phrases),
            "learned_count": len(self.learned),
            "meta2_count": len(self.meta2_patterns),
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    # ----- Encoding (text -> codes) -----

    def encode_phrases(self, text: str) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Apply phrase substitutions. Returns (encoded_text, substitution_log).
        Substitution log is a list of (original, code) pairs -- everything
        needed to reverse the operation.

        IMPORTANT: Only multi-word phrases use substring replacement.
        Single-word learned entries are collected and returned separately
        so they can be encoded word-by-word in encode_vocabulary.
        """
        import re as _re
        result = text.lower()
        subs = []

        # Sort phrases by length (longest first) to avoid partial matches
        sorted_phrases = sorted(self.phrases.items(), key=lambda x: len(x[0]), reverse=True)
        for phrase, code in sorted_phrases:
            phrase_lower = phrase.lower()
            # Use word-boundary matching for phrases to avoid partial word matches
            pattern = r'\b' + _re.escape(phrase_lower) + r'\b'
            if _re.search(pattern, result):
                result = _re.sub(pattern, code, result)
                subs.append((phrase, code))
                self.frequency[code] = self.frequency.get(code, 0) + 1

        # Only apply MULTI-WORD learned entries here (with word boundaries)
        # Single-word learned entries go through encode_vocabulary instead
        sorted_learned = sorted(self.learned.items(), key=lambda x: len(x[0]), reverse=True)
        for item, code in sorted_learned:
            item_lower = item.lower()
            if " " not in item_lower:
                continue  # single-word -> handled in encode_vocabulary
            pattern = r'\b' + _re.escape(item_lower) + r'\b'
            if _re.search(pattern, result):
                result = _re.sub(pattern, code, result)
                subs.append((item, code))
                self.frequency[code] = self.frequency.get(code, 0) + 1

        return result, subs

    def encode_vocabulary(self, text: str) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Apply word-level vocabulary substitutions. Returns (encoded_text, sub_log).
        Also handles single-word learned entries (word-by-word, not substring).
        """
        import re as _re
        # Build single-word learned lookup
        single_learned = {k.lower(): v for k, v in self.learned.items() if " " not in k}

        words = text.split()
        encoded = []
        subs = []

        for word in words:
            # Strip punctuation but preserve it for reattachment
            clean = word.strip(".,!?;:()[]{}\"'").lower()
            trailing = ""
            if word and word[-1] in ".,!?;:":
                trailing = word[-1]

            if clean in self.vocabulary:
                code = self.vocabulary[clean]
                if code != clean:
                    subs.append((clean, code))
                    self.frequency[code] = self.frequency.get(code, 0) + 1
                encoded.append(code + trailing)
            elif clean in single_learned:
                code = single_learned[clean]
                if code != clean:
                    subs.append((clean, code))
                    self.frequency[code] = self.frequency.get(code, 0) + 1
                encoded.append(code + trailing)
            else:
                encoded.append(word)

        result = " ".join(w for w in encoded if w)
        result = _re.sub(r'\s+', ' ', result).strip()
        return result, subs

    def full_encode(self, text: str) -> Tuple[str, list]:
        """
        Apply all reference substitutions (phrases first, then vocabulary).
        Returns (encoded, tagged_substitution_log).
        Log entries are 3-tuples: [type, original, code]
        where type is "phrase", "vocab", or "removal".
        """
        # Phrases first (longer matches)
        encoded, phrase_subs = self.encode_phrases(text)
        # Then vocabulary
        encoded, vocab_subs = self.encode_vocabulary(encoded)

        # Tag each substitution with its type for proper decode
        tagged = []
        for orig, code in phrase_subs:
            tagged.append(["phrase", orig, code])
        for orig, code in vocab_subs:
            if code == "" or code.strip() == "":
                tagged.append(["removal", orig, ""])
            else:
                tagged.append(["vocab", orig, code])

        self.total_compressions += 1
        return encoded, tagged

    # ----- Meta-pattern encoding (code sequences -> meta codes) -----

    def find_code_bigrams(self, coded_text: str, min_len: int = 3) -> List[Tuple[str, int]]:
        """
        Find recurring 2-3 token sequences in coded text.
        Returns [(sequence, count), ...] sorted by savings potential.
        """
        tokens = coded_text.split()
        if len(tokens) < 2:
            return []

        from collections import Counter

        # Find bigrams (2-token sequences)
        bigrams = []
        for i in range(len(tokens) - 1):
            bg = f"{tokens[i]} {tokens[i+1]}"
            # Skip if either token is very short (1 char) -- not worth meta-coding
            if len(bg) >= min_len:
                bigrams.append(bg)

        # Find trigrams (3-token sequences)
        trigrams = []
        for i in range(len(tokens) - 2):
            tg = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
            if len(tg) >= min_len + 2:
                trigrams.append(tg)

        # Count frequencies and calculate savings
        bg_counts = Counter(bigrams)
        tg_counts = Counter(trigrams)

        candidates = []
        for seq, count in bg_counts.items():
            # Already a meta-pattern?
            if seq in self.meta_patterns:
                continue
            # Savings = (original_len - meta_code_len) * occurrences
            meta_code_len = 3  # estimated: "M.x"
            savings = (len(seq) - meta_code_len) * count
            if savings > 0 and count >= 1:
                candidates.append((seq, count, savings, "bigram"))

        for seq, count in tg_counts.items():
            if seq in self.meta_patterns:
                continue
            meta_code_len = 4  # estimated: "M.xx"
            savings = (len(seq) - meta_code_len) * count
            if savings > 0 and count >= 1:
                candidates.append((seq, count, savings, "trigram"))

        # Sort by savings potential
        candidates.sort(key=lambda x: x[2], reverse=True)
        return [(c[0], c[1]) for c in candidates[:20]]

    def next_meta_code(self) -> str:
        """Generate the next available meta-code (M.1, M.2, ...)."""
        existing = set(self.meta_patterns.values())
        i = 1
        while True:
            code = f"M.{i}"
            if code not in existing:
                return code
            i += 1

    def learn_meta(self, sequence: str, meta_code: str = None) -> str:
        """Learn a new meta-pattern (code sequence -> meta code)."""
        if not meta_code:
            meta_code = self.next_meta_code()
        self.meta_patterns[sequence] = meta_code
        self.decode_meta[meta_code] = sequence
        self.version += 1
        return meta_code

    def encode_meta(self, text: str) -> Tuple[str, list]:
        """
        Apply meta-pattern substitutions to coded text.
        Returns (encoded_text, substitution_log).
        Log entries: ["meta", sequence, meta_code]
        """
        result = text
        subs = []

        # Sort by length (longest patterns first) to avoid partial matches
        sorted_meta = sorted(self.meta_patterns.items(),
                             key=lambda x: len(x[0]), reverse=True)

        for sequence, meta_code in sorted_meta:
            if sequence in result:
                count = result.count(sequence)
                result = result.replace(sequence, meta_code)
                subs.append(["meta", sequence, meta_code])
                self.frequency[meta_code] = self.frequency.get(meta_code, 0) + count

        return result, subs

    def decode_meta_text(self, text: str, sub_log: list = None) -> str:
        """Reverse meta-pattern substitutions."""
        result = text
        if sub_log:
            # Reverse in reverse order (undo last applied first)
            for entry in reversed(sub_log):
                if isinstance(entry, (list, tuple)) and len(entry) == 3 and entry[0] == "meta":
                    sequence, meta_code = entry[1], entry[2]
                    result = result.replace(meta_code, sequence)
        else:
            # Fallback: use reverse lookup (longest codes first)
            sorted_rev = sorted(self.decode_meta.items(),
                                key=lambda x: len(x[0]), reverse=True)
            for meta_code, sequence in sorted_rev:
                if meta_code in result:
                    result = result.replace(meta_code, sequence)
        return result

    # ----- Meta²-pattern encoding (meta-code sequences -> meta² codes) -----

    def find_meta2_sequences(self, coded_text: str) -> List[Tuple[str, int]]:
        """
        Find recurring sequences in meta-encoded text that contain M-codes or MM-codes.
        These are "patterns of patterns" -- sequences where at least one token is
        a meta-code (M.N) or another higher-order code.
        Returns [(sequence, count), ...] sorted by savings potential.
        """
        tokens = coded_text.split()
        if len(tokens) < 2:
            return []

        from collections import Counter
        import re

        # A token is "interesting" for meta² if it's an M-code, MM-code, or domain code
        def is_meta_token(t):
            return bool(re.match(r'^M\.\d+$', t) or re.match(r'^MM\.\d+$', t))

        # Find bigrams that contain at least one meta-token
        bigrams = []
        for i in range(len(tokens) - 1):
            bg = f"{tokens[i]} {tokens[i+1]}"
            if is_meta_token(tokens[i]) or is_meta_token(tokens[i+1]):
                bigrams.append(bg)

        # Find trigrams that contain at least one meta-token
        trigrams = []
        for i in range(len(tokens) - 2):
            tg = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
            has_meta = any(is_meta_token(tokens[i+j]) for j in range(3))
            if has_meta:
                trigrams.append(tg)

        # Also find ANY recurring bigram (even without explicit M-codes)
        # since at this level all tokens are already codes
        all_bigrams = []
        for i in range(len(tokens) - 1):
            bg = f"{tokens[i]} {tokens[i+1]}"
            if len(bg) >= 4:
                all_bigrams.append(bg)

        bg_counts = Counter(bigrams)
        tg_counts = Counter(trigrams)
        all_bg_counts = Counter(all_bigrams)

        candidates = []

        # Priority 1: sequences with M-codes (true patterns of patterns)
        for seq, count in bg_counts.items():
            if seq in self.meta2_patterns:
                continue
            if count < 2:
                continue
            meta2_len = 4  # "MM.x"
            savings = (len(seq) - meta2_len) * count
            if savings >= 3:
                candidates.append((seq, count, savings, "meta-bigram"))

        for seq, count in tg_counts.items():
            if seq in self.meta2_patterns:
                continue
            if count < 2:
                continue
            meta2_len = 5  # "MM.xx"
            savings = (len(seq) - meta2_len) * count
            if savings >= 3:
                candidates.append((seq, count, savings, "meta-trigram"))

        # Priority 2: any recurring bigram at this compression level
        for seq, count in all_bg_counts.items():
            if seq in self.meta2_patterns or seq in self.meta_patterns:
                continue
            if count < 2:
                continue
            # Only if not already covered by meta-bigrams
            if seq in bg_counts:
                continue
            meta2_len = 4
            savings = (len(seq) - meta2_len) * count
            if savings >= 3:
                candidates.append((seq, count, savings, "code-bigram"))

        candidates.sort(key=lambda x: x[2], reverse=True)
        return [(c[0], c[1]) for c in candidates[:20]]

    def next_meta2_code(self) -> str:
        """Generate the next available meta²-code (MM.1, MM.2, ...)."""
        existing = set(self.meta2_patterns.values())
        i = 1
        while True:
            code = f"MM.{i}"
            if code not in existing:
                return code
            i += 1

    def learn_meta2(self, sequence: str, meta2_code: str = None) -> str:
        """Learn a new meta²-pattern (meta-code sequence -> meta² code)."""
        if not meta2_code:
            meta2_code = self.next_meta2_code()
        self.meta2_patterns[sequence] = meta2_code
        self.decode_meta2[meta2_code] = sequence
        self.version += 1
        return meta2_code

    def encode_meta2(self, text: str) -> Tuple[str, list]:
        """
        Apply meta²-pattern substitutions to meta-encoded text.
        Returns (encoded_text, substitution_log).
        Log entries: ["meta2", sequence, meta2_code]
        """
        result = text
        subs = []

        sorted_meta2 = sorted(self.meta2_patterns.items(),
                               key=lambda x: len(x[0]), reverse=True)

        for sequence, meta2_code in sorted_meta2:
            if sequence in result:
                count = result.count(sequence)
                result = result.replace(sequence, meta2_code)
                subs.append(["meta2", sequence, meta2_code])
                self.frequency[meta2_code] = self.frequency.get(meta2_code, 0) + count

        return result, subs

    def decode_meta2_text(self, text: str, sub_log: list = None) -> str:
        """Reverse meta²-pattern substitutions."""
        result = text
        if sub_log:
            for entry in reversed(sub_log):
                if isinstance(entry, (list, tuple)) and len(entry) == 3 and entry[0] == "meta2":
                    sequence, meta2_code = entry[1], entry[2]
                    result = result.replace(meta2_code, sequence)
        else:
            sorted_rev = sorted(self.decode_meta2.items(),
                                key=lambda x: len(x[0]), reverse=True)
            for meta2_code, sequence in sorted_rev:
                if meta2_code in result:
                    result = result.replace(meta2_code, sequence)
        return result

    # ----- Structural packing -----

    def pack_structural(self, text: str) -> Tuple[str, list]:
        """
        Apply deterministic structural compression to coded text:
        - Remove redundant whitespace
        - Compact conjunction operators (&, |, ^)
        - Remove trailing articles before codes
        Returns (packed_text, substitution_log).
        Log entries: ["pack", description, action]
        """
        import re
        subs = []
        result = text

        # Rule 1: Multiple spaces -> single space
        original = result
        result = re.sub(r'\s+', ' ', result).strip()
        if result != original:
            subs.append(["pack", "multi-space", "single-space"])

        # Rule 2: Remove space around SAFE conjunction operators only
        # Only &, |, ^ -- NOT < > = ! + which are word-meaning codes
        safe_ops = ['&', '|', '^']
        for op in safe_ops:
            pattern = f' {re.escape(op)} '
            if pattern in result:
                old_result = result
                result = result.replace(pattern, op)
                if result != old_result:
                    subs.append(["pack", f"spaces around '{op}'", "removed"])

        # Rule 3: Remove redundant "T" (the) before domain codes
        # "T a.ml" -> "a.ml" (the article adds no info before a code)
        old_result = result
        result = re.sub(r'\bT ([a-z]+\.\w+)', r'\1', result)
        if result != old_result:
            subs.append(["pack", "article before domain code", "removed"])

        return result, subs

    def unpack_structural(self, text: str, sub_log: list = None) -> str:
        """Reverse structural packing."""
        import re
        result = text
        if not sub_log:
            return result

        # Reverse in reverse order
        for entry in reversed(sub_log):
            if not isinstance(entry, (list, tuple)) or len(entry) != 3:
                continue
            if entry[0] != "pack":
                continue
            desc = entry[1]

            if "spaces around" in desc:
                # Extract the operator
                op = desc.split("'")[1] if "'" in desc else ""
                if op:
                    # Add spaces back around operator
                    # Find bare operator not already surrounded by spaces
                    result = result.replace(op, f' {op} ')

            if "space after dot-prefix" in desc:
                # Re-add space after domain prefixes: "a.ml" -> "a. ml" -- actually don't,
                # this is hard to reverse perfectly. The decode_meta + decode_text handles it.
                pass

        # Clean up any double spaces from re-adding
        result = re.sub(r'\s+', ' ', result).strip()
        return result

    # ----- Decoding (codes -> text) -----

    def decode_text(self, encoded: str, substitution_log: list = None) -> str:
        """
        Decode encoded text using the substitution log (preferred) or reverse lookup.

        Handles both legacy 2-tuple logs [(orig, code), ...] and new
        3-tuple logs [["type", orig, code], ...].

        Vocab subs are decoded word-by-word to avoid substring collisions.
        Phrase subs are decoded via full-text replacement (longest codes first).
        """
        result = encoded

        if substitution_log:
            # Parse log entries into phrase_subs and vocab_subs
            phrase_subs = []
            vocab_subs = []  # ordered -- preserves left-to-right encoding order

            for entry in substitution_log:
                if isinstance(entry, (list, tuple)) and len(entry) == 3:
                    stype, orig, code = entry
                elif isinstance(entry, (list, tuple)) and len(entry) == 2:
                    orig, code = entry
                    # Legacy: guess type from content
                    if " " in orig:
                        stype = "phrase"
                    elif code == "" or (code and code.strip() == ""):
                        stype = "removal"
                    else:
                        stype = "vocab"
                else:
                    continue

                if stype == "phrase":
                    phrase_subs.append((orig, code))
                elif stype == "removal":
                    pass  # removals can't be positioned, skip
                else:
                    vocab_subs.append((orig, code))

            # Step 1: Reverse vocab subs word-by-word
            # Build queues: for each code, the list of original words (in encoding order)
            code_queues = {}
            for orig, code in vocab_subs:
                if code not in code_queues:
                    code_queues[code] = []
                code_queues[code].append(orig)
            code_idx = {k: 0 for k in code_queues}

            words = result.split()
            decoded_words = []
            for w in words:
                # Strip trailing punctuation for lookup, reattach after
                trailing = ""
                clean_w = w
                while clean_w and clean_w[-1] in ".,!?;:":
                    trailing = clean_w[-1] + trailing
                    clean_w = clean_w[:-1]

                if clean_w in code_queues:
                    qi = code_idx[clean_w]
                    originals = code_queues[clean_w]
                    if qi < len(originals):
                        decoded_words.append(originals[qi] + trailing)
                        code_idx[clean_w] = qi + 1
                    else:
                        decoded_words.append((originals[-1] if originals else clean_w) + trailing)
                elif w in code_queues:
                    # Exact match including punctuation
                    qi = code_idx[w]
                    originals = code_queues[w]
                    if qi < len(originals):
                        decoded_words.append(originals[qi])
                        code_idx[w] = qi + 1
                    else:
                        decoded_words.append(originals[-1] if originals else w)
                else:
                    decoded_words.append(w)
            result = " ".join(decoded_words)

            # Step 2: Reverse phrase subs (longest code first for safety)
            sorted_psubs = sorted(phrase_subs, key=lambda x: len(x[1]), reverse=True)
            for orig, code in sorted_psubs:
                if code and code in result:
                    result = result.replace(code, orig)
        else:
            # Fallback: use reverse lookup tables
            # Phrases first (they're longer codes, more specific)
            for code, phrase in sorted(self.decode_phrases.items(), key=lambda x: len(x[0]), reverse=True):
                if code in result:
                    result = result.replace(code, phrase)
            for code, phrase in sorted(self.decode_learned.items(), key=lambda x: len(x[0]), reverse=True):
                if code in result:
                    result = result.replace(code, phrase)
            # Then vocabulary (word-by-word)
            words = result.split()
            decoded = []
            for w in words:
                if w in self.decode_vocab:
                    decoded.append(self.decode_vocab[w])
                else:
                    decoded.append(w)
            result = " ".join(decoded)

        return result

    # ----- Learning -----

    def learn(self, phrase: str, code: str):
        """Add a new learned mapping."""
        self.learned[phrase] = code
        self.decode_learned[code] = phrase
        self.version += 1

    def edit_entry(self, old_word: str, new_word: str = None, new_code: str = None,
                   entry_type: str = "auto") -> dict:
        """
        Edit or move an existing entry. Can change the word, the code, or both.
        Returns {"status": "ok"/"error", "details": ...}
        """
        # Find where the entry lives
        found_in = None
        current_code = None
        for section_name, section in [("vocabulary", self.vocabulary),
                                       ("phrases", self.phrases),
                                       ("learned", self.learned)]:
            if old_word in section:
                found_in = section_name
                current_code = section[old_word]
                break

        if not found_in:
            return {"status": "error", "details": f"'{old_word}' not found in any section"}

        section = getattr(self, found_in)
        reverse = {"vocabulary": "decode_vocab", "phrases": "decode_phrases",
                    "learned": "decode_learned"}[found_in]
        rev_section = getattr(self, reverse)

        # Remove old entry
        del section[old_word]
        if current_code in rev_section:
            del rev_section[current_code]

        # Add updated entry
        final_word = new_word if new_word else old_word
        final_code = new_code if new_code else current_code
        section[final_word] = final_code
        rev_section[final_code] = final_word

        self.version += 1
        self.save()
        return {"status": "ok", "details": f"Updated: '{final_word}' -> '{final_code}' in {found_in}"}

    def delete_entry(self, word: str) -> dict:
        """Delete an entry from any section."""
        for section_name, section in [("vocabulary", self.vocabulary),
                                       ("phrases", self.phrases),
                                       ("learned", self.learned)]:
            if word in section:
                code = section[word]
                del section[word]
                reverse = {"vocabulary": "decode_vocab", "phrases": "decode_phrases",
                            "learned": "decode_learned"}[section_name]
                rev_section = getattr(self, reverse)
                if code in rev_section:
                    del rev_section[code]
                self.version += 1
                self.save()
                return {"status": "ok", "details": f"Deleted '{word}' -> '{code}' from {section_name}"}
        return {"status": "error", "details": f"'{word}' not found"}

    def add_entry(self, word: str, code: str, entry_type: str = "learned") -> dict:
        """Add a new entry to the specified section."""
        if entry_type == "vocabulary":
            if word in self.vocabulary:
                return {"status": "error", "details": f"'{word}' already exists in vocabulary"}
            self.vocabulary[word] = code
            self.decode_vocab[code] = word
        elif entry_type == "phrase":
            if word in self.phrases:
                return {"status": "error", "details": f"'{word}' already exists in phrases"}
            self.phrases[word] = code
            self.decode_phrases[code] = word
        else:
            if word in self.learned:
                return {"status": "error", "details": f"'{word}' already exists in learned"}
            self.learned[word] = code
            self.decode_learned[code] = word
        self.version += 1
        self.save()
        return {"status": "ok", "details": f"Added '{word}' -> '{code}' to {entry_type}"}

    def learn_from_comparison(self, original: str, encoded: str, decoded: str):
        """
        Analyze an encode-decode cycle and learn new patterns.
        Identifies words/phrases that survived encoding well vs poorly.
        Also looks for repeated bigrams as phrase candidates.
        """
        orig_words = original.lower().split()
        dec_words = decoded.lower().split()

        # Find words in original that didn't make it to decoded
        missed = set(orig_words) - set(dec_words)
        # Filter out already-known words and short words
        missed = {w for w in missed
                  if w not in self.vocabulary
                  and w not in self.learned
                  and len(w) > 4
                  and w.isalpha()}

        # Also find bigrams that might be good phrase candidates
        bigram_candidates = []
        for i in range(len(orig_words) - 1):
            bg = f"{orig_words[i]} {orig_words[i+1]}"
            if (bg not in self.phrases
                and bg not in self.learned
                and len(bg) > 8  # only worth encoding if savings > 3 chars
                and all(w.isalpha() for w in bg.split())):
                bigram_candidates.append(bg)

        return list(missed) + bigram_candidates[:3]

    def suggest_code(self, phrase: str) -> str:
        """Generate a short code for a phrase. Avoids collisions."""
        words = phrase.lower().split()
        if len(words) == 1:
            w = words[0]
            consonants = [c for c in w if c not in "aeiou"]
            if len(consonants) >= 3:
                code = "".join(consonants[:3])
            elif len(consonants) >= 2:
                code = "".join(consonants[:2]) + w[-1]
            else:
                code = w[:3]
        else:
            code = "".join(w[0] for w in words if w)

        # Ensure no collision
        all_codes = set(self.vocabulary.values()) | set(self.phrases.values()) | set(self.learned.values())
        original_code = code
        counter = 0
        while code in all_codes:
            counter += 1
            code = f"{original_code}{counter}"
        return code

    def get_all_entries(self) -> list:
        """Get all entries across all sections as a flat list for the GUI."""
        entries = []
        for word, code in sorted(self.vocabulary.items()):
            entries.append({
                "word": word, "code": code, "type": "vocabulary",
                "freq": self.frequency.get(code, 0),
                "savings": len(word) - len(code),
            })
        for word, code in sorted(self.phrases.items()):
            entries.append({
                "word": word, "code": code, "type": "phrase",
                "freq": self.frequency.get(code, 0),
                "savings": len(word) - len(code),
            })
        for word, code in sorted(self.learned.items()):
            entries.append({
                "word": word, "code": code, "type": "learned",
                "freq": self.frequency.get(code, 0),
                "savings": len(word) - len(code),
            })
        for seq, code in sorted(self.meta_patterns.items()):
            entries.append({
                "word": seq, "code": code, "type": "meta",
                "freq": self.frequency.get(code, 0),
                "savings": len(seq) - len(code),
            })
        for seq, code in sorted(self.meta2_patterns.items()):
            entries.append({
                "word": seq, "code": code, "type": "meta2",
                "freq": self.frequency.get(code, 0),
                "savings": len(seq) - len(code),
            })
        return entries

    # ----- Info -----

    def get_stats(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "fingerprint": self.get_fingerprint(),
            "vocabulary_count": len(self.vocabulary),
            "phrase_count": len(self.phrases),
            "learned_count": len(self.learned),
            "meta_count": len(self.meta_patterns),
            "meta2_count": len(self.meta2_patterns),
            "total_codes": len(self.vocabulary) + len(self.phrases) + len(self.learned) + len(self.meta_patterns) + len(self.meta2_patterns),
            "training_rounds": self.training_rounds,
            "total_compressions": self.total_compressions,
            "top_used": sorted(
                self.frequency.items(), key=lambda x: x[1], reverse=True
            )[:20],
        }

    def export_for_transmission(self) -> dict:
        """
        Export a minimal version of the reference for inclusion in
        transmission packages. The receiver needs this to decode.
        """
        return {
            "name": self.name,
            "version": self.version,
            "fingerprint": self.get_fingerprint(),
            "decode_phrases": self.decode_phrases,
            "decode_learned": self.decode_learned,
            "decode_vocab": self.decode_vocab,
        }

    def to_dict(self) -> dict:
        """Full export for viewing."""
        return {
            "name": self.name,
            "version": self.version,
            "fingerprint": self.get_fingerprint(),
            "vocabulary": self.vocabulary,
            "phrases": self.phrases,
            "learned": self.learned,
            "meta_patterns": self.meta_patterns,
            "meta2_patterns": self.meta2_patterns,
            "domains": DOMAIN_PREFIXES,
            "stats": self.get_stats(),
        }


def list_references() -> List[dict]:
    """List all available reference files."""
    ensure_references_dir()
    refs = []
    for fname in os.listdir(REFERENCES_DIR):
        if fname.endswith(".json"):
            name = fname[:-5]
            try:
                ref = SharedReference(name)
                refs.append(ref.get_stats())
            except Exception:
                refs.append({"name": name, "error": True})
    return refs


# Backward compatibility alias
Codebook = SharedReference
