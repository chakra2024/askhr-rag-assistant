"""Tests for the Pinecone helpers (no network calls)."""

import pytest
from langchain_core.documents import Document

from src.pinecone_store import get_vector_store, make_chunk_ids


def test_chunk_ids_are_stable():
    chunks = [
        Document(page_content="a", metadata={"source": "03benefits/02leave.md"}),
        Document(page_content="b", metadata={"source": "03benefits/02leave.md"}),
        Document(page_content="c", metadata={"source": "02welcome/02first_week.md"}),
    ]

    ids = make_chunk_ids(chunks)

    assert ids == [
        "03benefits/02leave.md#chunk-0",
        "03benefits/02leave.md#chunk-1",
        "02welcome/02first_week.md#chunk-0",
    ]
    # Same input gives the same IDs, so re-indexing overwrites instead of duplicating.
    assert make_chunk_ids(chunks) == ids


def test_missing_api_key(monkeypatch):
    monkeypatch.setenv("PINECONE_API_KEY", "replace_me")

    # The key is checked before any network call is made.
    with pytest.raises(ValueError, match="PINECONE_API_KEY is missing"):
        get_vector_store()
