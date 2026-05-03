"""Unit tests for src/rag.py."""

import json
import math
import os
import pytest
from src.rag import (
    cosine_similarity,
    retrieve_keyword,
    format_context,
    load_knowledge_base,
)


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_identical_vectors_return_one(self):
        v = [1.0, 0.5, 0.2]
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors_return_zero(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors_return_negative_one(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-6)

    def test_zero_vector_does_not_raise(self):
        result = cosine_similarity([0.0, 0.0], [1.0, 1.0])
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_partial_similarity(self):
        a = [1.0, 1.0]
        b = [1.0, 0.0]
        result = cosine_similarity(a, b)
        assert 0.0 < result < 1.0

    def test_scaling_does_not_change_result(self):
        a = [1.0, 2.0, 3.0]
        b = [2.0, 4.0, 6.0]  # b = 2*a
        assert cosine_similarity(a, b) == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# load_knowledge_base
# ---------------------------------------------------------------------------

class TestLoadKnowledgeBase:
    def test_returns_list(self):
        docs = load_knowledge_base()
        assert isinstance(docs, list)

    def test_has_substantial_number_of_docs(self):
        docs = load_knowledge_base()
        assert len(docs) >= 30

    def test_each_doc_has_required_fields(self):
        docs = load_knowledge_base()
        for doc in docs:
            assert "id" in doc, f"doc missing 'id': {doc}"
            assert "type" in doc
            assert "title" in doc
            assert "content" in doc
            assert "tags" in doc

    def test_ids_are_unique(self):
        docs = load_knowledge_base()
        ids = [d["id"] for d in docs]
        assert len(ids) == len(set(ids)), "Duplicate document IDs found"

    def test_gym_document_present(self):
        docs = load_knowledge_base()
        ids = {d["id"] for d in docs}
        assert "act_gym" in ids

    def test_study_document_present(self):
        docs = load_knowledge_base()
        ids = {d["id"] for d in docs}
        assert "act_study" in ids

    def test_all_doc_types_covered(self):
        docs = load_knowledge_base()
        types = {d["type"] for d in docs}
        assert "activity" in types
        assert "genre" in types
        assert "mood" in types


# ---------------------------------------------------------------------------
# retrieve_keyword
# ---------------------------------------------------------------------------

class TestRetrieveKeyword:
    @pytest.fixture
    def docs(self):
        return load_knowledge_base()

    def test_returns_k_results(self, docs):
        results = retrieve_keyword("gym workout music", docs, k=3)
        assert len(results) == 3

    def test_returns_tuples_of_doc_and_score(self, docs):
        results = retrieve_keyword("gym workout music", docs, k=1)
        doc, score = results[0]
        assert isinstance(doc, dict)
        assert isinstance(score, float)

    def test_scores_descending(self, docs):
        results = retrieve_keyword("jazz for relaxing", docs, k=5)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_gym_query_retrieves_gym_doc_in_top3(self, docs):
        results = retrieve_keyword("pump up music for the gym", docs, k=5)
        ids = [d["id"] for d, _ in results]
        assert "act_gym" in ids, f"Expected act_gym in top-5, got: {ids}"

    def test_sad_query_retrieves_sad_doc(self, docs):
        results = retrieve_keyword("sad heartbreak songs rainy day", docs, k=5)
        ids = [d["id"] for d, _ in results]
        assert any(i in ids for i in ["act_sad", "combo_sad_acoustic", "mood_melancholic"]), \
            f"No sad-related doc in top-5: {ids}"

    def test_party_query_retrieves_party_doc(self, docs):
        results = retrieve_keyword("dance music house party", docs, k=5)
        ids = [d["id"] for d, _ in results]
        assert "act_party" in ids, f"Expected act_party in top-5, got: {ids}"

    def test_jazz_query_retrieves_jazz_doc(self, docs):
        results = retrieve_keyword("jazz coffee shop", docs, k=5)
        ids = [d["id"] for d, _ in results]
        assert "genre_jazz" in ids or "combo_chill_jazz" in ids, \
            f"No jazz doc in top-5: {ids}"

    def test_all_scores_non_negative(self, docs):
        results = retrieve_keyword("anything", docs, k=10)
        assert all(s >= 0 for _, s in results)

    def test_k_larger_than_docs_returns_all_docs(self, docs):
        results = retrieve_keyword("music", docs, k=999)
        assert len(results) == len(docs)

    def test_empty_query_does_not_crash(self, docs):
        results = retrieve_keyword("", docs, k=3)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# format_context
# ---------------------------------------------------------------------------

class TestFormatContext:
    def test_returns_string(self):
        docs = load_knowledge_base()
        retrieved = [(docs[0], 0.9), (docs[1], 0.8)]
        ctx = format_context(retrieved)
        assert isinstance(ctx, str)

    def test_contains_doc_titles(self):
        docs = load_knowledge_base()
        retrieved = [(docs[0], 0.9)]
        ctx = format_context(retrieved)
        assert docs[0]["title"] in ctx

    def test_contains_doc_content(self):
        docs = load_knowledge_base()
        retrieved = [(docs[0], 0.9)]
        ctx = format_context(retrieved)
        # at least some part of the content is present
        first_words = docs[0]["content"].split()[:5]
        assert any(w in ctx for w in first_words)

    def test_multiple_docs_separated(self):
        docs = load_knowledge_base()
        retrieved = [(docs[0], 0.9), (docs[1], 0.8)]
        ctx = format_context(retrieved)
        # both titles present
        assert docs[0]["title"] in ctx
        assert docs[1]["title"] in ctx

    def test_empty_retrieved_returns_empty_string(self):
        ctx = format_context([])
        assert ctx == ""
