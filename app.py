"""Flask server for the LexScan legal clause classification interface."""
from __future__ import annotations

import logging
import os
import tempfile
from collections import OrderedDict
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS

from classifier import LegalClauseClassifier

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static")
CORS(app)

_classifier: LegalClauseClassifier | None = None

# In-memory cache of extracted document text keyed by a generated id. Used by
# the lazy /api/excerpt endpoint so the client doesn't have to re-POST the
# entire document on every card expansion. Bounded LRU — oldest evicted first.
_DOC_CACHE_MAX = 32
_document_cache: "OrderedDict[str, str]" = OrderedDict()


def _cache_document(text: str) -> str:
    doc_id = uuid4().hex
    _document_cache[doc_id] = text
    while len(_document_cache) > _DOC_CACHE_MAX:
        _document_cache.popitem(last=False)
    return doc_id


def _lookup_document(doc_id: str) -> str | None:
    if doc_id in _document_cache:
        _document_cache.move_to_end(doc_id)
        return _document_cache[doc_id]
    return None


def get_classifier() -> LegalClauseClassifier:
    global _classifier
    if _classifier is None:
        _classifier = LegalClauseClassifier()
    return _classifier


def extract_text(file_path: str, filename: str) -> str:
    ext = Path(filename).suffix.lower()

    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            try:
                from PyPDF2 import PdfReader  # type: ignore[no-redef]
            except ImportError:
                raise RuntimeError("PDF support requires pypdf: pip install pypdf")
        reader = PdfReader(file_path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)

    if ext in (".docx", ".doc"):
        try:
            import docx
        except ImportError:
            raise RuntimeError("DOCX support requires python-docx: pip install python-docx")
        doc = docx.Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)

    # Plain text — try common encodings
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            with open(file_path, "r", encoding=enc) as fh:
                return fh.read()
        except UnicodeDecodeError:
            continue
    raise RuntimeError("Could not decode file as text.")


@app.route("/favicon.ico")
def favicon():
    return Response(status=204)


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/status")
def status():
    clf = get_classifier()
    return jsonify({"mode": clf.mode, "status": "ready"})


@app.route("/api/classify", methods=["POST"])
def classify():
    if "file" not in request.files:
        return jsonify({"error": "No file attached."}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename."}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in (".pdf", ".docx", ".doc", ".txt"):
        return jsonify({"error": f"Unsupported format '{ext}'. Upload PDF, DOCX, or TXT."}), 400

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp_path = tmp.name
            file.save(tmp_path)

        text = extract_text(tmp_path, file.filename)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if not text.strip():
        return jsonify({"error": "No readable text found in this file."}), 400

    try:
        result = get_classifier().classify(text)
    except Exception as exc:
        logger.exception("Classification error")
        return jsonify({"error": f"Classification failed: {exc}"}), 500

    result["word_count"] = len(text.split())
    result["char_count"] = len(text)
    result["document_id"] = _cache_document(text)
    return jsonify(result)


@app.route("/api/excerpt", methods=["POST"])
def excerpt():
    payload = request.get_json(silent=True) or {}
    doc_id = (payload.get("document_id") or "").strip()
    clause = (payload.get("clause") or "").strip()

    if not doc_id or not clause:
        return jsonify({"error": "Missing 'document_id' or 'clause'."}), 400

    text = _lookup_document(doc_id)
    if text is None:
        return jsonify({"error": "Document not in cache — please re-upload."}), 404

    try:
        snippet = get_classifier().extract_excerpt(text, clause)
    except Exception as exc:
        logger.exception("Excerpt extraction error")
        return jsonify({"error": f"Excerpt extraction failed: {exc}"}), 500

    return jsonify({"excerpt": snippet})


@app.route("/api/explain", methods=["POST"])
def explain():
    payload = request.get_json(silent=True) or {}
    excerpt = (payload.get("excerpt") or "").strip()
    clause = (payload.get("clause") or "").strip()

    if not excerpt:
        return jsonify({"error": "Missing 'excerpt' in request body."}), 400

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return jsonify({
            "error": "OpenAI API key not configured. Add OPENAI_API_KEY to the .env file."
        }), 500

    try:
        from openai import OpenAI
    except ImportError:
        return jsonify({"error": "openai package not installed: pip install openai"}), 500

    excerpt_for_prompt = excerpt[:4000]
    user_prompt = (
        f"The following is an excerpt from a legal contract"
        f"{f' identified as a \"{clause}\" clause' if clause else ''}.\n"
        f"Explain in plain, simple terms what this clause means and why it matters "
        f"to a non-lawyer (e.g., a business owner or procurement manager). "
        f"Keep it under 120 words.\n\n"
        f"Excerpt:\n\"\"\"\n{excerpt_for_prompt}\n\"\"\""
    )

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a legal-tech assistant who explains contract clauses in plain English for non-lawyers."},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        explanation = response.choices[0].message.content.strip()
    except Exception as exc:
        logger.exception("OpenAI explain error")
        return jsonify({"error": f"LLM call failed: {exc}"}), 500

    return jsonify({"explanation": explanation})


if __name__ == "__main__":
    logger.info("Initializing Legal-BERT classifier...")
    get_classifier()
    clf = get_classifier()
    logger.info("Mode: %s | Server: http://127.0.0.1:5001", clf.mode)
    app.run(host="127.0.0.1", port=5001, debug=False)
