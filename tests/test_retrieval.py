"""Tests for the TF-IDF retrieval layer."""
from chunkbench.chunkers import Chunk
from chunkbench.retrieval import TfidfIndex, tokenize


def _chunks(texts):
    return [Chunk(doc_id=f"d{i}", idx=0, text=t, start=0, end=len(t))
            for i, t in enumerate(texts)]


def test_tokenize_lowercases_and_drops_stopwords():
    toks = tokenize("The QUICK brown fox jumps over 2 lazy dogs!")
    assert "the" not in toks and "over" not in toks
    assert "quick" in toks and "2" in toks


def test_search_ranks_rare_term_first():
    chunks = _chunks([
        "the weather today is sunny and warm",
        "population stability index measures data drift with psi above 0.25",
        "another sunny day in the park",
    ])
    index = TfidfIndex(chunks)
    top = index.search("population stability index psi", k=3)
    assert top[0][0].doc_id == "d1"


def test_search_empty_query_returns_nothing():
    chunks = _chunks(["some text here"])
    assert TfidfIndex(chunks).search("   ", k=5) == []
    assert TfidfIndex(chunks).search("the and or", k=5) == []


def test_search_respects_k():
    chunks = _chunks(["apple banana", "apple cherry", "apple date"])
    index = TfidfIndex(chunks)
    assert len(index.search("apple", k=2)) == 2


def test_search_no_match_returns_empty():
    chunks = _chunks(["apples and oranges"])
    assert TfidfIndex(chunks).search("quantum entanglement", k=5) == []
