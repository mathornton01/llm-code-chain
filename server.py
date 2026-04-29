"""
Flask API server for LLM Code Chain.

Endpoints:
  POST /encode     -- encode natural language to compact code
  POST /decode     -- decode compact code to natural language
  POST /roundtrip  -- encode then decode (test fidelity)
  GET  /codebook   -- view current codebook
  GET  /stats      -- compression statistics
  POST /train      -- run a training round
"""

from flask import Flask, request, jsonify
from encoder import Encoder
from decoder import Decoder
from codebook import Codebook
from trainer import Trainer
import os

app = Flask(__name__)

MODEL_PATH = "/home/herald/models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
codebook = Codebook()
encoder = Encoder(MODEL_PATH, codebook)
decoder = Decoder(MODEL_PATH, codebook)


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "LLM Code Chain",
        "version": "0.1.0",
        "endpoints": {
            "POST /encode": "Encode text to compact code",
            "POST /decode": "Decode compact code to text",
            "POST /roundtrip": "Encode + decode roundtrip test",
            "GET /codebook": "View codebook",
            "GET /stats": "View statistics",
            "POST /train": "Run training round",
        }
    })


@app.route("/encode", methods=["POST"])
def encode():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    use_llm = data.get("use_llm", True)
    result = encoder.encode(data["text"], use_llm=use_llm)
    return jsonify(result)


@app.route("/decode", methods=["POST"])
def decode():
    data = request.get_json()
    if not data or "code" not in data:
        return jsonify({"error": "Missing 'code' field"}), 400

    use_llm = data.get("use_llm", True)
    result = decoder.decode(data["code"], use_llm=use_llm)
    return jsonify(result)


@app.route("/roundtrip", methods=["POST"])
def roundtrip():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    use_llm = data.get("use_llm", True)

    enc = encoder.encode(data["text"], use_llm=use_llm)
    dec = decoder.decode(enc["code"], use_llm=use_llm)

    # Simple fidelity check
    orig_words = set(data["text"].lower().split())
    dec_words = set(dec["decoded"].lower().split())
    key_orig = {w for w in orig_words if len(w) > 4}
    key_dec = {w for w in dec_words if len(w) > 4}
    fidelity = len(key_orig & key_dec) / max(1, len(key_orig))

    return jsonify({
        "original": data["text"],
        "encoded": enc["code"],
        "decoded": dec["decoded"],
        "compression_ratio": enc["compression_ratio"],
        "fidelity": round(fidelity, 3),
        "encode_time_ms": enc["time_ms"],
        "decode_time_ms": dec["time_ms"],
        "total_time_ms": round(enc["time_ms"] + dec["time_ms"], 1),
    })


@app.route("/codebook", methods=["GET"])
def get_codebook():
    return jsonify({
        "codes": codebook.codes,
        "stats": codebook.get_stats(),
    })


@app.route("/stats", methods=["GET"])
def stats():
    return jsonify({
        "encoder": encoder.get_stats(),
        "decoder": decoder.get_stats(),
        "codebook": codebook.get_stats(),
    })


@app.route("/train", methods=["POST"])
def train():
    data = request.get_json() or {}
    rounds = min(data.get("rounds", 1), 5)  # cap at 5
    use_llm = data.get("use_llm", True)
    corpus = data.get("corpus", None)

    trainer = Trainer(MODEL_PATH, codebook)
    history = trainer.train(rounds=rounds, corpus=corpus, use_llm=use_llm)

    # Refresh encoder/decoder codebook
    encoder.codebook = trainer.codebook
    decoder.codebook = trainer.codebook

    return jsonify({
        "rounds_completed": len(history),
        "final_avg_compression": history[-1]["avg_compression"],
        "final_avg_fidelity": history[-1]["avg_fidelity"],
        "total_codes": len(trainer.codebook.codes),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"LLM Code Chain server starting on port {port}")
    print(f"Model: {MODEL_PATH}")
    app.run(host="0.0.0.0", port=port, debug=False)
