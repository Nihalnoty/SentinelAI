import pytest

from src.rag.chunking import chunk_text


def test_short_text_returns_one_chunk():
    assert chunk_text("Hello world", chunk_size=100, overlap=20) == [
        "Hello world"
    ]


def test_chunks_overlap():
    chunks = chunk_text("ABCDEFGHIJKLMNO", chunk_size=10, overlap=3)
    assert chunks == ["ABCDEFGHIJ", "HIJKLMNO"]


def test_empty_text_returns_no_chunks():
    assert chunk_text("   ") == []


def test_invalid_chunk_size_is_rejected():
    with pytest.raises(ValueError):
        chunk_text("Hello", chunk_size=0, overlap=0)


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("Hello", chunk_size=10, overlap=10)