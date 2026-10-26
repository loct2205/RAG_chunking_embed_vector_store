"""Test RRF, TF bão hòa và từ điển BM25."""

from __future__ import annotations

from rag.encode.bm25 import Bm25Encoder, document_weights, saturated_tf, tokenize
from rag.fusion import rrf


def test_rrf_breaks_ties_by_chunk_id() -> None:
    fused = rrf([["b", "a"], ["a", "b"]], k=60)
    assert [chunk_id for chunk_id, _ in fused] == ["a", "b"]
    assert fused[0][1] == fused[1][1]


def test_rrf_puts_the_shared_winner_first() -> None:
    fused = rrf([["keep", "other"], ["keep"]], k=1)
    assert fused[0][0] == "keep"
    assert fused[0][1] > fused[1][1]


def test_saturated_tf_matches_formula() -> None:
    # tf=2, dl=avgdl=4, k1=1.2, b=0.75 -> 4.4 / 3.2
    assert saturated_tf(2, 4, 4.0, 1.2, 0.75) == 4.4 / 3.2


def test_tokenize_syllable_lowercases_and_splits() -> None:
    assert tokenize("Sốt  Xuất", "syllable") == ["sốt", "xuất"]


def test_document_weights_use_real_length() -> None:
    weights = document_weights(["sốt", "sốt"], avgdl=2.0, k1=1.2, b=0.75)
    assert weights["sốt"] == saturated_tf(2, 2, 2.0, 1.2, 0.75)


def test_bm25_query_drops_unknown_tokens_and_roundtrips(tmp_path) -> None:
    encoder = Bm25Encoder("syllable", k1=1.2, b=0.75)
    encoder.fit(["sốt xuất huyết", "paracetamol 20 kg"])
    query = encoder.encode_query("Sốt 20 không-có")
    assert query.indices
    saved = tmp_path / "bm25_vocab.json"
    encoder.save(saved)
    loaded = Bm25Encoder("syllable", k1=0, b=0)
    loaded.load(saved)
    assert loaded.encode_query("Sốt 20 không-có") == query


def test_bm25_word_mode_returns_tokens() -> None:
    tokens = tokenize("trẻ sốt xuất huyết", "word")
    assert tokens
    assert all(token == token.lower() for token in tokens)
