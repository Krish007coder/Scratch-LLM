"""
Flask HTTP server — 1:1 port of the C++ httplib routes from main.cpp.

All endpoints, response shapes, and CORS headers are identical so that
the existing index.html frontend works without any modification.
"""
from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING

from flask import Flask, jsonify, request, send_file

from vectordb import DocumentDB, OllamaClient, VectorDB, get_dist_fn

if TYPE_CHECKING:
    pass

DIMS = 16  # demo vector dimension


def _cors(response):
    """Attach permissive CORS headers to every response."""
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


def _parse_vec(s: str):
    """Parse a comma-separated float string into a list of floats."""
    result = []
    for token in s.split(","):
        token = token.strip()
        try:
            result.append(float(token))
        except ValueError:
            pass
    return result


def _chunk_text(text: str, chunk_words: int = 250, overlap_words: int = 30):
    """Split text into overlapping word-count chunks (mirrors C++ chunkText)."""
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]

    chunks = []
    step = chunk_words - overlap_words
    i = 0
    while i < len(words):
        end = min(i + chunk_words, len(words))
        chunks.append(" ".join(words[i:end]))
        if end == len(words):
            break
        i += step
    return chunks


def create_app(db: VectorDB, doc_db: DocumentDB, ollama: OllamaClient) -> Flask:
    app = Flask(__name__, static_folder=None)

    # --------------------------------------------------------------------------
    # CORS preflight
    # --------------------------------------------------------------------------
    @app.before_request
    def handle_preflight():
        if request.method == "OPTIONS":
            from flask import make_response
            resp = make_response("", 204)
            _cors(resp)
            return resp

    @app.after_request
    def attach_cors(response):
        return _cors(response)

    # --------------------------------------------------------------------------
    # Serve index.html at /
    # --------------------------------------------------------------------------
    @app.route("/")
    def index():
        html_path = os.path.join(os.path.dirname(__file__), "index.html")
        return send_file(html_path, mimetype="text/html")

    # ==========================================================================
    # DEMO VECTOR ENDPOINTS
    # ==========================================================================

    @app.route("/search")
    def search():
        q = _parse_vec(request.args.get("v", ""))
        if len(q) != DIMS:
            return jsonify({"error": f"need {DIMS}D vector"}), 400

        k = int(request.args.get("k", 5))
        metric = request.args.get("metric", "cosine")
        algo = request.args.get("algo", "hnsw")

        out = db.search(q, k, metric, algo)
        return jsonify({
            "results": [
                {
                    "id": h.id,
                    "metadata": h.meta,
                    "category": h.cat,
                    "distance": round(h.dist, 6),
                    "embedding": [round(x, 4) for x in h.emb],
                }
                for h in out.hits
            ],
            "latencyUs": out.latency_us,
            "algo": out.algo,
            "metric": out.metric,
        })

    @app.route("/insert", methods=["POST"])
    def insert():
        body = request.get_json(silent=True) or {}
        meta = body.get("metadata", "")
        cat = body.get("category", "")
        emb = body.get("embedding", [])
        if not meta or not emb or len(emb) != DIMS:
            return jsonify({"error": "invalid body"}), 400
        item_id = db.insert(meta, cat, emb, get_dist_fn("cosine"))
        return jsonify({"id": item_id})

    @app.route("/delete/<int:item_id>", methods=["DELETE"])
    def delete(item_id: int):
        ok = db.remove(item_id)
        return jsonify({"ok": ok})

    @app.route("/items")
    def items():
        return jsonify([
            {
                "id": v.id,
                "metadata": v.metadata,
                "category": v.category,
                "embedding": [round(x, 4) for x in v.emb],
            }
            for v in db.all()
        ])

    @app.route("/benchmark")
    def benchmark():
        q = _parse_vec(request.args.get("v", ""))
        if len(q) != DIMS:
            return jsonify({"error": f"need {DIMS}D vector"}), 400
        k = int(request.args.get("k", 5))
        metric = request.args.get("metric", "cosine")
        b = db.benchmark(q, k, metric)
        return jsonify({
            "bruteforceUs": b.bf_us,
            "kdtreeUs": b.kd_us,
            "hnswUs": b.hnsw_us,
            "itemCount": b.item_count,
        })

    @app.route("/hnsw-info")
    def hnsw_info():
        gi = db.hnsw_info()
        return jsonify({
            "topLayer": gi.top_layer,
            "nodeCount": gi.node_count,
            "nodesPerLayer": gi.nodes_per_layer,
            "edgesPerLayer": gi.edges_per_layer,
            "nodes": gi.nodes,
            "edges": gi.edges,
        })

    @app.route("/stats")
    def stats():
        return jsonify({
            "count": db.size(),
            "dims": DIMS,
            "algorithms": ["bruteforce", "kdtree", "hnsw"],
            "metrics": ["euclidean", "cosine", "manhattan"],
        })

    # ==========================================================================
    # DOCUMENT + RAG ENDPOINTS
    # ==========================================================================

    @app.route("/doc/insert", methods=["POST"])
    def doc_insert():
        body = request.get_json(silent=True) or {}
        title = body.get("title", "").strip()
        text = body.get("text", "").strip()
        if not title or not text:
            return jsonify({"error": "need title and text"}), 400

        chunks = _chunk_text(text, 250, 30)
        ids = []
        for i, chunk in enumerate(chunks):
            emb = ollama.embed(chunk)
            if not emb:
                return jsonify({
                    "error": (
                        "Ollama unavailable. Install from https://ollama.com then run: "
                        "ollama pull nomic-embed-text && ollama pull llama3.2"
                    )
                }), 503
            chunk_title = (
                f"{title} [{i + 1}/{len(chunks)}]" if len(chunks) > 1 else title
            )
            ids.append(doc_db.insert(chunk_title, chunk, emb))

        return jsonify({"ids": ids, "chunks": len(chunks), "dims": doc_db.dims})

    @app.route("/doc/delete/<int:doc_id>", methods=["DELETE"])
    def doc_delete(doc_id: int):
        ok = doc_db.remove(doc_id)
        return jsonify({"ok": ok})

    @app.route("/doc/list")
    def doc_list():
        docs = doc_db.all()
        result = []
        for d in docs:
            preview = d.text[:120] + ("…" if len(d.text) > 120 else "")
            word_count = len(d.text.split())
            result.append({
                "id": d.id,
                "title": d.title,
                "preview": preview,
                "words": word_count,
            })
        return jsonify(result)

    @app.route("/doc/search", methods=["POST"])
    def doc_search():
        body = request.get_json(silent=True) or {}
        question = body.get("question", "").strip()
        k = int(body.get("k", 3))
        if not question:
            return jsonify({"error": "need question"}), 400

        q_emb = ollama.embed(question)
        if not q_emb:
            return jsonify({"error": "Ollama unavailable"}), 503

        hits = doc_db.search(q_emb, k)
        return jsonify({
            "contexts": [
                {
                    "id": doc.id,
                    "title": doc.title,
                    "distance": round(d, 4),
                }
                for d, doc in hits
            ]
        })

    @app.route("/doc/ask", methods=["POST"])
    def doc_ask():
        body = request.get_json(silent=True) or {}
        question = body.get("question", "").strip()
        k = int(body.get("k", 3))
        if not question:
            return jsonify({"error": "need question"}), 400

        # Step 1: embed the question
        q_emb = ollama.embed(question)
        if not q_emb:
            return jsonify({"error": "Ollama unavailable"}), 503

        # Step 2: retrieve top-k relevant chunks
        hits = doc_db.search(q_emb, k)

        # Step 3: build prompt
        context_parts = [
            f"[{i + 1}] {doc.title}:\n{doc.text}"
            for i, (_, doc) in enumerate(hits)
        ]
        context_str = "\n\n".join(context_parts)
        prompt = (
            "You are a helpful assistant. Answer the user's question directly. "
            "Use the provided context if it contains relevant information. "
            "If it doesn't, just use your own general knowledge. "
            "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like "
            "'the context doesn't mention'. Just answer the question naturally.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\n\nAnswer:"
        )

        # Step 4: generate answer
        answer = ollama.generate(prompt)

        # Step 5: return everything
        return jsonify({
            "answer": answer,
            "model": ollama.gen_model,
            "contexts": [
                {
                    "id": doc.id,
                    "title": doc.title,
                    "text": doc.text,
                    "distance": round(d, 4),
                }
                for d, doc in hits
            ],
            "docCount": doc_db.size(),
        })

    @app.route("/status")
    def status():
        up = ollama.is_available()
        return jsonify({
            "ollamaAvailable": up,
            "embedModel": ollama.embed_model,
            "genModel": ollama.gen_model,
            "docCount": doc_db.size(),
            "docDims": doc_db.dims,
            "demoDims": DIMS,
            "demoCount": db.size(),
        })

    return app
