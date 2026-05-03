"""
RAG (Retrieval-Augmented Generation) system for VibeFinder 2.0.

Two retrieval modes:
  - Semantic (default): embeds query and documents with Gemini text-embedding-004,
    ranks by cosine similarity. Embeddings are cached to data/embeddings_cache.json
    so documents are only embedded once.
  - Keyword fallback: token-overlap scoring when no API key is available.
"""

import json
import logging
import math
import os
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

KNOWLEDGE_BASE_PATH = "data/knowledge_base.json"
EMBEDDINGS_CACHE_PATH = "data/embeddings_cache.json"
EMBEDDING_MODEL = "text-embedding-004"


# ---------------------------------------------------------------------------
# Knowledge base loading
# ---------------------------------------------------------------------------

def load_knowledge_base(path: str = KNOWLEDGE_BASE_PATH) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["documents"]


# ---------------------------------------------------------------------------
# Math helpers (no numpy dependency)
# ---------------------------------------------------------------------------

def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v))


def cosine_similarity(a: List[float], b: List[float]) -> float:
    denom = _norm(a) * _norm(b)
    return _dot(a, b) / denom if denom > 1e-10 else 0.0


# ---------------------------------------------------------------------------
# Gemini embedding helpers
# ---------------------------------------------------------------------------

def _get_embedding(text: str, client) -> List[float]:
    """Call Gemini text-embedding-004 and return the embedding vector."""
    from google.genai import types as gtypes
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
    )
    return list(response.embeddings[0].values)


def _load_cache() -> Dict[str, List[float]]:
    if os.path.exists(EMBEDDINGS_CACHE_PATH):
        with open(EMBEDDINGS_CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_cache(cache: Dict[str, List[float]]) -> None:
    with open(EMBEDDINGS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f)


# ---------------------------------------------------------------------------
# Index building (semantic)
# ---------------------------------------------------------------------------

def build_index(documents: List[Dict], client) -> Dict[str, List[float]]:
    """Embed all documents and cache to disk. Returns {doc_id: embedding}."""
    cache = _load_cache()
    updated = False
    for doc in documents:
        if doc["id"] not in cache:
            text = doc["title"] + ". " + doc["content"]
            try:
                cache[doc["id"]] = _get_embedding(text, client)
                updated = True
                logger.debug("Embedded doc: %s", doc["id"])
            except Exception as exc:
                logger.warning("Could not embed doc %s: %s", doc["id"], exc)
    if updated:
        _save_cache(cache)
        logger.info("Saved %d embeddings to cache.", len(cache))
    return cache


# ---------------------------------------------------------------------------
# Semantic retrieval (requires API key + pre-built index)
# ---------------------------------------------------------------------------

def retrieve_semantic(
    query: str,
    documents: List[Dict],
    index: Dict[str, List[float]],
    client,
    k: int = 3,
) -> List[Tuple[Dict, float]]:
    """Embed query and return top-k documents by cosine similarity."""
    try:
        query_emb = _get_embedding(query, client)
    except Exception as exc:
        logger.warning("Query embedding failed, falling back to keyword: %s", exc)
        return retrieve_keyword(query, documents, k)

    scored = []
    for doc in documents:
        emb = index.get(doc["id"])
        if emb:
            sim = cosine_similarity(query_emb, emb)
            scored.append((doc, sim))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


# ---------------------------------------------------------------------------
# Keyword retrieval (fallback, no API needed)
# ---------------------------------------------------------------------------

def retrieve_keyword(
    query: str,
    documents: List[Dict],
    k: int = 3,
) -> List[Tuple[Dict, float]]:
    """Token-overlap scoring against doc tags + title + first 60 words of content."""
    q_tokens = set(query.lower().replace(",", " ").replace(".", " ").split())

    def _score(doc: Dict) -> float:
        tags = set(t.lower() for t in doc.get("tags", []))
        title_tokens = set(doc.get("title", "").lower().split())
        content_tokens = set(doc.get("content", "").lower().split()[:60])
        all_tokens = tags | title_tokens | content_tokens
        overlap = len(q_tokens & all_tokens)
        return overlap / (math.sqrt(len(q_tokens) + 1) * math.sqrt(len(all_tokens) + 1))

    scored = [(doc, _score(doc)) for doc in documents]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


# ---------------------------------------------------------------------------
# Unified retrieve entry point
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    documents: List[Dict],
    index: Optional[Dict[str, List[float]]],
    client=None,
    k: int = 3,
) -> Tuple[List[Tuple[Dict, float]], str]:
    """Retrieve top-k docs. Returns (results, mode_used).

    Uses semantic retrieval when client and index are available,
    falls back to keyword retrieval otherwise.
    """
    if client is not None and index:
        results = retrieve_semantic(query, documents, index, client, k)
        return results, "semantic"
    else:
        results = retrieve_keyword(query, documents, k)
        return results, "keyword"


def format_context(retrieved: List[Tuple[Dict, float]]) -> str:
    """Format retrieved documents into a context string for the LLM prompt."""
    lines = []
    for doc, score in retrieved:
        lines.append(f"[{doc['title']}]: {doc['content']}")
    return "\n\n".join(lines)
