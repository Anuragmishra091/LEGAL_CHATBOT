from flask import Flask, request, jsonify, render_template
from main import LegalChatbot
import json
import threading
from pathlib import Path

app = Flask(__name__)

# Initialise chatbot once at startup
chatbot = LegalChatbot()

# ── Eval state (shared across threads) ──────────────────────────────────────
_eval_state: dict = {"status": "idle", "progress": 0, "total": 0, "error": None}
_eval_lock = threading.Lock()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    question = (data.get("question") or "").strip()
    chat_history = data.get("chat_history", [])  # list of {role, content}
    
    if not question:
       return jsonify({"error": "No question provided"}), 400

    result = chatbot.ask(
        question,
        chat_history=chat_history if chat_history else None,
    )

    # Keep only serialisable fields from sources
    sources = [
        {
            "act": s["act"],
            "section": s["section"],
            "heading": s["heading"],
            "relevance_score": round(s["relevance_score"], 4),
            "text": s["text"][:300],
        }
        for s in result["sources"]
    ]

    return jsonify({
        "answer": result["answer"],
        "sources": sources,
        "context_used": result["context_used"],
    })


# ── Evaluation endpoints ─────────────────────────────────────────────────────

@app.route("/eval/run", methods=["POST"])
def eval_run():
    data = request.get_json(force=True) or {}
    limit    = data.get("limit")
    category = data.get("category") or None
    skip_llm = bool(data.get("skip_llm", False))

    with _eval_lock:
        if _eval_state["status"] == "running":
            return jsonify({"error": "Evaluation already running"}), 409
        _eval_state.update({"status": "running", "progress": 0, "total": 0, "error": None})

    def _run():
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from eval_runner import run_evaluation

        def _progress(done, total):
            with _eval_lock:
                _eval_state["progress"] = done
                _eval_state["total"]    = total

        try:
            run_evaluation(
                limit=limit,
                category=category,
                skip_llm=skip_llm,
                chatbot=chatbot,
                on_progress=_progress,
            )
            with _eval_lock:
                _eval_state["status"] = "done"
        except Exception as exc:
            with _eval_lock:
                _eval_state.update({"status": "error", "error": str(exc)})

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "started"})


@app.route("/eval/status")
def eval_status():
    with _eval_lock:
        return jsonify(dict(_eval_state))


@app.route("/eval/results")
def eval_results_route():
    results_path = Path(__file__).parent / "eval_results.json"
    if results_path.exists():
        with open(results_path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify({"error": "No results yet. Run evaluation first."}), 404


if __name__ == "__main__":
    app.run(debug=True, port=5000)
