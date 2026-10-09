"""Tests for the three chunking strategies."""
import pytest

from chunkbench.chunkers import fixed_chunks, recursive_chunks, semantic_chunks


SAMPLE = (
    "First paragraph about LoRA fine-tuning and rank decomposition matrices.\n\n"
    "Second paragraph about the scaling factor of alpha divided by r and how "
    "it keeps the effective learning rate stable across ranks.\n\n"
    "Third paragraph about target modules such as q_proj and v_proj in the "
    "attention projections of a transformer model."
)


def test_fixed_covers_full_text():
    chunks = fixed_chunks(SAMPLE, size=100, overlap=20)
    assert chunks[0].start == 0
    assert chunks[-1].end == len(SAMPLE)
    # overlap is honored between consecutive chunks
    for a, b in zip(chunks, chunks[1:]):
        assert b.start == a.end - 20


def test_fixed_rejects_bad_params():
    with pytest.raises(ValueError):
        fixed_chunks("abc", size=0)
    with pytest.raises(ValueError):
        fixed_chunks("abc", size=100, overlap=100)
    with pytest.raises(ValueError):
        fixed_chunks("abc", size=100, overlap=-1)


def test_fixed_keeps_short_tail():
    chunks = fixed_chunks("hello world", size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"


def test_recursive_never_exceeds_size():
    text = "word " * 300 + "\n\n" + "another " * 300
    chunks = recursive_chunks(text, size=200)
    assert all(len(c.text) <= 200 for c in chunks)
    # every chunk is an exact substring with correct offsets
    for c in chunks:
        assert text[c.start:c.end] == c.text


def test_recursive_respects_paragraph_boundaries():
    paras = ["alpha " * 20, "beta " * 20, "gamma " * 20]
    text = "\n\n".join(paras)
    chunks = recursive_chunks(text, size=500)
    assert len(chunks) == 1  # everything fits: one clean chunk


def test_recursive_splits_long_unbroken_text():
    text = "x" * 1200
    chunks = recursive_chunks(text, size=500)
    assert len(chunks) == 3
    assert "".join(c.text for c in chunks) == text


def test_recursive_deterministic():
    a = recursive_chunks(SAMPLE, size=120)
    b = recursive_chunks(SAMPLE, size=120)
    assert [(c.text, c.start, c.end) for c in a] == [(c.text, c.start, c.end) for c in b]


def test_semantic_splits_on_topic_shift():
    lora = "LoRA injects rank decomposition matrices into transformer layers. "
    lora += "The scaling factor of alpha divided by r controls the update size. "
    drift = "Data drift means the serving distribution moved away from training. "
    drift += "A PSI above 0.25 signals significant drift worth investigating. "
    text = lora * 4 + drift * 4
    chunks = semantic_chunks(text, target=10000, threshold=0.15, min_chars=100)
    assert len(chunks) >= 2
    # the topic shift is detected: no chunk mixes both topics
    for c in chunks:
        assert not ("LoRA" in c.text and "PSI" in c.text)


def test_semantic_offsets_exact():
    chunks = semantic_chunks(SAMPLE, target=200)
    for c in chunks:
        assert SAMPLE[c.start:c.end] == c.text
    assert chunks[0].start == 0


def test_semantic_single_sentence():
    chunks = semantic_chunks("Just one sentence here.", target=500)
    assert len(chunks) == 1


def test_semantic_rejects_bad_target():
    with pytest.raises(ValueError):
        semantic_chunks("abc", target=0)
