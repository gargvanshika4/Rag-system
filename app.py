#!/usr/bin/env python3
"""Flask API and UI entry point for the research-grade Graph RAG system."""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from graph_rag import KnowledgeGraphRAG

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False
app.config["MAX_CONTENT_LENGTH"] = 128 * 1024 * 1024

_rag: KnowledgeGraphRAG | None = None
_build_lock = threading.Lock()


def _init_system(force: bool = False) -> KnowledgeGraphRAG:
    """Initialize or rebuild the RAG system."""
    global _rag
    with _build_lock:
        if _rag is not None and not force:
            return _rag
        api_key = os.getenv("GROQ_API_KEY")
        rag = KnowledgeGraphRAG(api_key, use_ai=bool(api_key))
        rag.build_system(str(BASE_DIR / "data"), str(BASE_DIR / "data"))
        _rag = rag
        return _rag


def _get_system() -> KnowledgeGraphRAG:
    return _init_system()


@app.route("/")
def index():
    """Render the dashboard."""
    return render_template("index.html")


@app.route("/api/query", methods=["POST"])
def query():
    """Answer a question with hybrid Graph RAG."""
    data = request.get_json(force=True) or {}
    query_text = str(data.get("query", "")).strip()
    if not query_text:
        return jsonify({"error": "No query provided"}), 400
    conversation_id = data.get("conversation_id") or "default"
    try:
        result = _get_system().search_and_answer(query_text, conversation_id=conversation_id)
        return jsonify({"success": True, **result})
    except Exception as exc:
        logger.exception("Query failed")
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/search", methods=["POST"])
def search():
    """Return retrieval-only ranked sources."""
    data = request.get_json(force=True) or {}
    query_text = str(data.get("query", "")).strip()
    if not query_text:
        return jsonify({"error": "No query provided"}), 400
    results = _get_system().search(query_text, k=int(data.get("k", 10)), conversation_id=data.get("conversation_id") or "default")
    return jsonify({"success": True, "results": results})


@app.route("/api/upload", methods=["POST"])
def upload():
    """Upload one or more documents, then rebuild the incremental index."""
    if "files" not in request.files:
        return jsonify({"success": False, "error": "No files part named 'files'"}), 400
    saved = []
    for file_storage in request.files.getlist("files"):
        filename = secure_filename(file_storage.filename or "")
        if not filename:
            continue
        target = UPLOAD_DIR / filename
        file_storage.save(target)
        saved.append(str(target.name))
    if saved:
        _init_system(force=True)
    return jsonify({"success": True, "files": saved})


@app.route("/api/graph-data", methods=["POST"])
def graph_data():
    """Return graph nodes and edges relevant to a query."""
    data = request.get_json(force=True) or {}
    return jsonify(_get_system().get_graph_data(str(data.get("query", "")).strip()))


@app.route("/api/stats")
def stats():
    """Return system statistics."""
    return jsonify(_get_system().get_system_statistics())


@app.route("/api/history/<conversation_id>")
def history(conversation_id: str):
    """Return conversation history."""
    return jsonify({"conversation_id": conversation_id, "history": _get_system().memory.history(conversation_id)})


@app.route("/api/history/<conversation_id>", methods=["DELETE"])
def clear_history(conversation_id: str):
    """Clear conversation history."""
    _get_system().memory.clear(conversation_id)
    return jsonify({"success": True})


@app.route("/api/reload", methods=["POST"])
def reload_system():
    """Force a rebuild from disk."""
    _init_system(force=True)
    return jsonify({"success": True})


@app.route("/api/clear-cache", methods=["POST"])
def clear_cache():
    """Clear cache and rebuild."""
    result = _get_system().clear_cache()
    _init_system(force=True)
    return jsonify(result)


@app.route("/api/debug")
def debug():
    """Return feature flags and key status."""
    return jsonify(
        {
            "system": "KnowledgeGraphRAG v3.0",
            "groq_api_key_set": bool(os.getenv("GROQ_API_KEY")),
            "rag_system_ready": _rag is not None,
            "hybrid_retrieval": True,
            "rrf_enabled": True,
            "adaptive_chunking": True,
            "conversation_memory": True,
            "faiss_hnsw": True,
            "request_id": str(uuid4()),
        }
    )


@app.route("/api/health")
def health():
    """Health check."""
    return jsonify({"status": "healthy", "system": "KnowledgeGraphRAG", "version": "3.0.0"})


if __name__ == "__main__":
    print("Open browser: http://localhost:5001")

    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5001)))

    app.run(debug=True, host="0.0.0.0", port=5001)

