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


def test_e5_override_keeps_bge_sparse_and_tokenizer(tmp_path) -> None:
    from unittest.mock import Mock

    from rag.encode.e5_embedder import E5_MODEL_NAME, E5Embedder, E5Settings
    from rag.pipeline import DenseOverrideEmbedder

    bge = BGEM3Embedder(
        _StubModel(), BgeSettings("BAAI/bge-m3", tmp_path, 1024, 8, False)
    )
    model = Mock()
    model.encode.return_value = [[1.0] + [0.0] * 767, [0.0, 1.0] + [0.0] * 766]
    e5 = E5Embedder(model, E5Settings(E5_MODEL_NAME, tmp_path, 256, 2, False))
    embedder = DenseOverrideEmbedder(bge, e5)
    texts = ["first document", "second document"]
    rows = embedder.encode(texts)
    assert [dense for dense, _ in rows] == model.encode.return_value
    assert [sparse for _, sparse in rows] == [sparse for _, sparse in bge.encode(texts)]
    assert embedder.dimension == 768
    assert embedder.model_name == f"{E5_MODEL_NAME}+sparse:BAAI/bge-m3"
    assert embedder.tokenize("ab") == bge.tokenize("ab")
    model.tokenizer.encode.assert_not_called()
    assert model.max_seq_length == 256
    model.encode.assert_called_once_with(
        ["passage: first document", "passage: second document"],
        batch_size=2,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    model.encode.reset_mock()
    assert embedder.encode([]) == []
    model.encode.assert_not_called()


def test_dense_override_is_opt_in_for_existing_config() -> None:
    from dataclasses import replace
    from pathlib import Path
    from unittest.mock import patch

    from rag.config import load_config
    from rag.pipeline import configure_dense_embedder

    # The corpus YAML intentionally has none of the new E5 settings.
    config = load_config(Path(__file__).resolve().parents[2] / "config.corpus.yaml")
    base = FakeEmbedder()
    dense = FakeEmbedder(768)
    with patch("rag.pipeline.load_e5", return_value=dense) as load:
        assert configure_dense_embedder(base, config.embedding) is base
        load.assert_not_called()
        selected = replace(config.embedding, dense_model="e5-base", e5_batch_size=3)
        combined = configure_dense_embedder(base, selected)
        assert combined.encode(["document"])[0] == (
            dense.encode(["document"])[0][0],
            base.encode(["document"])[0][1],
        )
        assert load.call_args.args[0].batch_size == 3
        assert load.call_args.args[0].max_length == 512


@pytest.mark.parametrize(
    "invalid",
    [{"dense_model": "unknown"}, {"e5_max_length": 513}, {"e5_batch_size": 0}],
)
def test_invalid_dense_config_is_rejected(invalid) -> None:
    from rag.config import _embedding

    section = {
        "model_name": "BAAI/bge-m3",
        "model_path": "models/bge-m3",
        "max_length": 1024,
        "batch_size": 16,
        "use_fp16": False,
        **invalid,
    }
    with pytest.raises(ValueError):
        _embedding(section)


def test_dense_override_rejects_missing_vectors() -> None:
    from unittest.mock import Mock

    from rag.encode.base import Embedder
    from rag.pipeline import DenseOverrideEmbedder

    dense = Mock(spec=Embedder)
    dense.encode.return_value = []
    with pytest.raises(RuntimeError):
        DenseOverrideEmbedder(FakeEmbedder(), dense).encode(["document"])


def test_e5_queries_use_query_prefix_and_keep_bge_sparse(tmp_path) -> None:
    from unittest.mock import Mock

    from rag.encode.e5_embedder import E5_MODEL_NAME, E5Embedder, E5Settings
    from rag.pipeline import DenseOverrideEmbedder

    model = Mock()
    model.encode.return_value = [[1.0] + [0.0] * 767]
    base = FakeEmbedder()
    e5 = E5Embedder(model, E5Settings(E5_MODEL_NAME, tmp_path))
    selected = DenseOverrideEmbedder(base, e5)
    dense, sparse = selected.encode_queries(["retrieval question"])[0]
    assert len(dense) == 768
    assert sparse == base.encode(["retrieval question"])[0][1]
    assert model.encode.call_args.args[0] == ["query: retrieval question"]
    assert selected.encode_queries([]) == []
    model.encode.return_value = []
    with pytest.raises(RuntimeError):
        selected.encode_queries(["missing vector"])
