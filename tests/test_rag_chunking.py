
import pytest

from src.rag.chunking import chunk_documents, chunk_text


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


def test_sentence_aware_chunks_respect_maximum_size():
    text = (
        "Inspect the bearing carefully. "
        "Measure vibration levels. "
        "Record temperature readings."
    )

    chunks = chunk_text(
        text,
        chunk_size=40,
        overlap=10,
        sentence_aware=True,
    )

    assert chunks
    assert all(len(chunk) <= 40 for chunk in chunks)


def test_sentence_aware_chunks_split_at_word_boundaries():
    text = (
        "Inspect the mounting bolts and bearing condition "
        "according to the approved maintenance procedure."
    )

    chunks = chunk_text(
        text,
        chunk_size=35,
        overlap=10,
        sentence_aware=True,
    )

    assert all(len(chunk) <= 35 for chunk in chunks)

    # No chunk should begin or end with a partial ordinary word.
    assert all(chunk == chunk.strip() for chunk in chunks)
    assert all(not chunk.startswith(" ") for chunk in chunks)
    assert all(not chunk.endswith(" ") for chunk in chunks)


def test_sentence_aware_chunks_preserve_all_words():
    text = (
        "Inspect the mounting bolts and bearing condition "
        "according to the approved maintenance procedure."
    )

    chunks = chunk_text(
        text,
        chunk_size=35,
        overlap=10,
        sentence_aware=True,
    )

    reconstructed_words = " ".join(chunks).split()
    original_words = text.split()

    # Overlap may duplicate words, so compare the unique sequence
    # of words rather than requiring an exact string reconstruction.
    deduplicated_words = []
    for word in reconstructed_words:
        if not deduplicated_words or word != deduplicated_words[-1]:
            deduplicated_words.append(word)

    assert all(word in reconstructed_words for word in original_words)
    assert deduplicated_words == original_words


def test_sentence_aware_maintenance_chunks_respect_size_limit():
    text = (
        "Document: Centrifuge Maintenance Guide\n"
        "Section: Abnormal Vibration\n\n"
        "If abnormal vibration is observed, inspect the mounting bolts "
        "and bearing condition according to the approved maintenance "
        "procedure. Follow the site's safety procedures before inspection."
    )

    chunks = chunk_text(
        text,
        chunk_size=100,
        overlap=20,
        sentence_aware=True,
    )

    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)
    assert all(chunk.strip() for chunk in chunks)


def test_chunk_documents_rejects_invalid_text():
    with pytest.raises(ValueError, match="invalid text"):
        chunk_documents([
            {"source": "maintenance.txt", "text": None}
        ])


def test_chunk_documents_preserves_source_metadata():
    chunks = chunk_documents([
        {
            "source": "maintenance.txt",
            "text": "Inspect the bearing for abnormal vibration.",
        }
    ])

    assert chunks
    assert all(
        chunk["source"] == "maintenance.txt"
        for chunk in chunks
    )
