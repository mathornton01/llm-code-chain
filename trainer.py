"""
Trainer: iterative loop that improves the codebook through encode-decode cycles.

Process:
  1. Take a corpus of sample prompts
  2. Encode each one
  3. Decode the encoded version
  4. Compare decoded vs original (semantic similarity)
  5. Identify patterns that compress well vs poorly
  6. Update codebook with new learned codes for frequent patterns
  7. Repeat
"""

from encoder import Encoder
from decoder import Decoder
from codebook import Codebook
from typing import List, Dict, Tuple
from collections import Counter
import re
import json
import os
import time


# Sample training corpus -- diverse prompt types
TRAINING_CORPUS = [
    "Can you explain how neural networks learn from data?",
    "Create a Python function that sorts a list of numbers in descending order",
    "The patient has elevated blood pressure and should reduce sodium intake",
    "What is the difference between supervised and unsupervised learning?",
    "Write a bash script that finds all files larger than 100MB",
    "Explain the central limit theorem and why it matters in statistics",
    "How do I configure a Docker container to use GPU acceleration?",
    "The genetic sequence ATCG represents the four nucleotide bases",
    "Compare the performance of random forests versus gradient boosting",
    "Build a REST API endpoint that handles user authentication",
    "What are the side effects of metformin in diabetic patients?",
    "Implement a binary search tree with insert and delete operations",
    "The Fourier transform decomposes a signal into frequency components",
    "How does attention mechanism work in transformer models?",
    "Calculate the p-value for a two-tailed t-test with 30 degrees of freedom",
    "Set up a PostgreSQL database with replication for high availability",
    "Explain the difference between Type I and Type II errors in hypothesis testing",
    "Write a recursive function to compute the nth Fibonacci number",
    "What is the role of epigenetics in cancer development?",
    "Deploy a machine learning model to production using Kubernetes",
    "Describe the process of protein folding and misfolding diseases",
    "How do convolutional neural networks detect features in images?",
    "Create a data pipeline that ingests CSV files and loads them into a data warehouse",
    "What is the Bayesian approach to parameter estimation?",
    "Implement a rate limiter using the token bucket algorithm",
]


class Trainer:
    """Trains the codebook through iterative encode-decode cycles."""

    def __init__(self, model_path: str, codebook: Codebook = None):
        self.codebook = codebook or Codebook()
        self.encoder = Encoder(model_path, self.codebook)
        self.decoder = Decoder(model_path, self.codebook)
        self.history: List[Dict] = []
        self.log_path = os.path.join(os.path.dirname(__file__), "training_log.json")

    def _similarity_score(self, original: str, decoded: str) -> float:
        """
        Simple word-overlap similarity score.
        Returns 0-1 where 1 is perfect reconstruction.
        """
        orig_words = set(original.lower().split())
        dec_words = set(decoded.lower().split())

        if not orig_words:
            return 0.0

        # Jaccard similarity
        intersection = orig_words & dec_words
        union = orig_words | dec_words
        jaccard = len(intersection) / len(union) if union else 0

        # Key term preservation (nouns, verbs, technical terms tend to be longer)
        key_orig = {w for w in orig_words if len(w) > 4}
        key_dec = {w for w in dec_words if len(w) > 4}
        if key_orig:
            key_recall = len(key_orig & key_dec) / len(key_orig)
        else:
            key_recall = 1.0

        return 0.4 * jaccard + 0.6 * key_recall

    def _find_patterns(self, results: List[Dict]) -> List[Tuple[str, int]]:
        """Find frequently occurring multi-word patterns not yet in codebook."""
        all_text = " ".join(r["original"] for r in results).lower()
        words = all_text.split()

        # Find bigrams and trigrams
        bigrams = Counter()
        trigrams = Counter()
        for i in range(len(words) - 1):
            bg = f"{words[i]} {words[i+1]}"
            if bg not in self.codebook.reverse:
                bigrams[bg] += 1
        for i in range(len(words) - 2):
            tg = f"{words[i]} {words[i+1]} {words[i+2]}"
            if tg not in self.codebook.reverse:
                trigrams[tg] += 1

        # Return patterns that appear 2+ times
        patterns = []
        for phrase, count in trigrams.most_common(10):
            if count >= 2:
                patterns.append((phrase, count))
        for phrase, count in bigrams.most_common(10):
            if count >= 2:
                patterns.append((phrase, count))

        return patterns

    def train_round(
        self, corpus: List[str] = None, use_llm: bool = True
    ) -> Dict:
        """
        Run one training round.

        Returns round summary with compression and fidelity stats.
        """
        corpus = corpus or TRAINING_CORPUS
        results = []
        round_start = time.time()

        print(f"\n{'='*60}")
        print(f"  Training Round {len(self.history) + 1}")
        print(f"  Codebook: {len(self.codebook.codes)} codes (v{self.codebook.version})")
        print(f"  Corpus: {len(corpus)} samples")
        print(f"{'='*60}\n")

        for i, text in enumerate(corpus):
            # Encode
            enc_result = self.encoder.encode(text, use_llm=use_llm)
            code = enc_result["code"]

            # Decode
            dec_result = self.decoder.decode(code, use_llm=use_llm)
            decoded = dec_result["decoded"]

            # Score
            score = self._similarity_score(text, decoded)

            result = {
                "original": text,
                "code": code,
                "decoded": decoded,
                "compression_ratio": enc_result["compression_ratio"],
                "similarity": round(score, 3),
                "encode_ms": enc_result["time_ms"],
                "decode_ms": dec_result["time_ms"],
            }
            results.append(result)

            status = "OK" if score > 0.5 else "!!"
            print(
                f"  [{status}] {i+1:2d}. ratio={enc_result['compression_ratio']:.0%} "
                f"fidelity={score:.0%} | {text[:50]}..."
            )

        # Analyze patterns and update codebook
        patterns = self._find_patterns(results)
        new_codes = 0
        for phrase, count in patterns[:5]:
            code = self.codebook.suggest_code(phrase)
            # Make sure code doesn't collide
            while code in self.codebook.codes:
                code = code + str(len(code))
            self.codebook.learn(phrase, code)
            new_codes += 1
            print(f"  [LEARN] '{phrase}' -> '{code}' (seen {count}x)")

        # Summary
        avg_compression = sum(r["compression_ratio"] for r in results) / len(results)
        avg_fidelity = sum(r["similarity"] for r in results) / len(results)
        total_time = (time.time() - round_start) * 1000

        summary = {
            "round": len(self.history) + 1,
            "samples": len(corpus),
            "avg_compression": round(avg_compression, 3),
            "avg_fidelity": round(avg_fidelity, 3),
            "new_codes_learned": new_codes,
            "total_codes": len(self.codebook.codes),
            "time_ms": round(total_time, 1),
            "results": results,
        }

        self.history.append(summary)
        self.codebook.save()
        self._save_log()

        print(f"\n  Summary:")
        print(f"    Avg compression: {avg_compression:.0%}")
        print(f"    Avg fidelity:    {avg_fidelity:.0%}")
        print(f"    New codes:       {new_codes}")
        print(f"    Time:            {total_time/1000:.1f}s")
        print(f"{'='*60}\n")

        return summary

    def train(self, rounds: int = 3, corpus: List[str] = None, use_llm: bool = True):
        """Run multiple training rounds."""
        for r in range(rounds):
            summary = self.train_round(corpus, use_llm=use_llm)
            print(f"  Round {r+1} complete: "
                  f"compression={summary['avg_compression']:.0%} "
                  f"fidelity={summary['avg_fidelity']:.0%}")

        return self.history

    def _save_log(self):
        """Save training history."""
        # Save only summaries (without full results to keep file small)
        log = []
        for h in self.history:
            entry = {k: v for k, v in h.items() if k != "results"}
            log.append(entry)
        with open(self.log_path, "w") as f:
            json.dump(log, f, indent=2)


if __name__ == "__main__":
    import sys
    model_path = "/home/herald/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

    use_llm = "--no-llm" not in sys.argv
    rounds = 1

    for arg in sys.argv[1:]:
        if arg.startswith("--rounds="):
            rounds = int(arg.split("=")[1])

    print(f"LLM Code Chain Trainer")
    print(f"Model: {model_path}")
    print(f"LLM: {'enabled' if use_llm else 'disabled (rule-based only)'}")
    print(f"Rounds: {rounds}")

    trainer = Trainer(model_path)
    trainer.train(rounds=rounds, use_llm=use_llm)
