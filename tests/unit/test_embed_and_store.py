"""Test embedder giả, BGE bọc model giả, và thử lại kết nối Qdrant."""

from __future__ import annotations

import pytest
from qdrant_client.http.exceptions import ResponseHandlingException

from rag.encode.bge_m3 import BGEM3Embedder, BgeSettings
from rag.encode.fake import FakeEmbedder
from rag.schema import SparseVec
from rag.store.qdrant_store import call_with_retry, point_id


def test_fake_embedder_is_stable_and_shares_tokens() -> None:
    embedder = FakeEmbedder()
    first = embedder.encode(["paracetamol 20 kg"])[0]
    second = embedder.encode(["paracetamol 20 kg"])[0]
    other = embedder.encode(["sốt xuất huyết"])[0]
    assert first == second
    assert len(first[0]) == embedder.dimension
    assert first[0] != other[0]


def test_bge_wrapper_passes_explicit_encode_flags(tmp_path) -> None:
    settings = BgeSettings(
        "BAAI/bge-m3", tmp_path, max_length=1024, batch_size=8, use_fp16=False
    )
    embedder = BGEM3Embedder(_StubModel(), settings)
    dense, sparse = embedder.encode(["câu"])[0]
    assert dense == [0.0, 1.0]
    assert sparse == SparseVec([7, 42], [0.5, 0.3])
    assert embedder.tokenize("ab") == ["1", "2"]


def test_point_id_is_stable() -> None:
    assert point_id("sot_chunk_0001") == point_id("sot_chunk_0001")
    assert point_id("sot_chunk_0001") != point_id("sot_chunk_0002")


def test_retry_stops_on_data_errors() -> None:
    calls = {"count": 0}

    def fail() -> None:
        calls["count"] += 1
        raise ValueError("sai dữ liệu")

    with pytest.raises(ValueError, match="sai dữ liệu"):
        call_with_retry(fail, "ghi")
    assert calls["count"] == 1


def test_retry_connects_on_the_third_attempt() -> None:
    calls = {"count": 0}

    def flaky() -> str:
        calls["count"] += 1
        if calls["count"] < 3:
            raise ResponseHandlingException("mất kết nối")
        return "ok"

    assert call_with_retry(flaky, "tìm") == "ok"
    assert calls["count"] == 3


class _StubTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [1, 2]


class _StubModel:
    tokenizer = _StubTokenizer()

    def encode(
        self,
        texts: list[str],
        batch_size: int,
        max_length: int,
        return_dense: bool,
        return_sparse: bool,
        return_colbert_vecs: bool,
    ) -> dict[str, object]:
        assert return_dense is True
        assert return_sparse is True
        assert return_colbert_vecs is False
        assert max_length == 1024
        assert batch_size == 8
        return {
            "dense_vecs": [[0.0, 1.0] for _ in texts],
            "lexical_weights": [{"7": 0.5, "9": 0.0, "42": 0.3} for _ in texts],
        }
