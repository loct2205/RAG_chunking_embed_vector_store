"""Ingest rồi search trên data/sample, FakeEmbedder và Qdrant trong bộ nhớ."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest
from qdrant_client import QdrantClient

from rag.config import PathsConfig, load_config
from rag.encode.bm25 import Bm25Encoder
from rag.encode.cache import VectorCache
from rag.encode.fake import FakeEmbedder
from rag.evaluate import Evaluator, is_relevant, load_questions
from rag.pipeline import IngestParts, IngestPipeline
from rag.search import SearchService
from rag.store.qdrant_store import QdrantSettings, QdrantStore
from rag.text.chunker import Chunker
from rag.text.loader import DocumentLoader

ROOT = Path(__file__).resolve().parents[2]


def test_ingest_then_search_finds_table_row_and_plain_note(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    config = load_config(ROOT / "config.yaml")
    config = replace(
        config,
        paths=PathsConfig(
            data_dir=ROOT / "data" / "sample",
            cache_dir=tmp_path / "cache",
            reports_dir=tmp_path / "reports",
        ),
        bm25=replace(config.bm25, vocab_path=tmp_path / "cache" / "bm25_vocab.json"),
    )
    embedder = FakeEmbedder()
    store = QdrantStore(
        QdrantClient(":memory:"),
        QdrantSettings(
            collection="test_chunks",
            upsert_batch_size=config.qdrant.upsert_batch_size,
            prefetch_limit=config.search.prefetch_limit,
            rrf_k=config.search.rrf_k,
            hnsw_m=config.qdrant.hnsw.m,
            ef_construct=config.qdrant.hnsw.ef_construct,
            full_scan_threshold=config.qdrant.hnsw.full_scan_threshold,
            indexing_threshold=config.qdrant.indexing_threshold,
            hnsw_ef=config.qdrant.hnsw_ef,
        ),
    )
    pipeline = IngestPipeline(
        config,
        IngestParts(
            loader=DocumentLoader(),
            chunker=Chunker(
                config.chunking.chunk_size,
                config.chunking.chunk_overlap,
                embedder.tokenize,
            ),
            embedder=embedder,
            bm25=Bm25Encoder(config.bm25.mode, config.bm25.k1, config.bm25.b),
            store=store,
            cache=VectorCache(),
        ),
    )
    count = pipeline.run(recreate=True)
    assert count == store.count()
    assert count > 1
    assert (tmp_path / "cache" / "vectors.json").is_file()

    service = SearchService(embedder, _loaded_bm25(config.bm25), store, top_k=10)
    table_hits = service.search(
        "Trẻ 20 kg dùng paracetamol tối đa bao nhiêu mỗi ngày?", ["dense", "sparse"]
    )
    assert any(is_relevant(hit.content, ["20 kg", "1200 mg"]) for hit in table_hits)
    note_hits = service.search("Bù nước khi trẻ nôn nhiều lần", ["bm25"])
    assert any(is_relevant(hit.content, ["nôn nhiều lần"]) for hit in note_hits)

    questions = load_questions(ROOT / "data" / "eval" / "questions.jsonl")
    _, chunks, _ = VectorCache().load(config.paths.cache_dir)
    summary = Evaluator(service, questions, chunks).score(["dense", "bm25"], top_k=10)
    assert summary["text_count"] >= 1
    assert summary["table_count"] >= 1
    assert float(summary["table_recall_at_k"]) > 0
    for kind in ("text", "table"):
        assert 0 < summary[f"{kind}_retrieval_recall_at_k"] <= 1
        assert 0 < summary[f"{kind}_ndcg_at_k"] <= 1

    from rag import cli

    config = replace(config, eval=replace(config.eval, warmup_runs=0, latency_runs=1))
    monkeypatch.setattr(cli, "load_config", lambda _: config)
    monkeypatch.setattr(cli, "_store", lambda _: store)
    for command in ("eval", "compare"):
        assert cli.main([command, "--configs", "dense+bm25", "--fake-embedder"]) == 0
        output = capsys.readouterr().out
        assert "Recall@10" in output
        assert "nDCG@10" in output
    reports = list(config.paths.reports_dir.glob("*.json"))
    assert len(reports) == 2
    for path in reports:
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = payload["results"]["dense+bm25"]
        quality = result.get("quality", result.get("client_rrf"))
        assert quality["top_k"] == 10
        assert "text_recall_at_k" in quality
        assert "text_retrieval_recall_at_k" in quality
        assert "text_ndcg_at_k" in quality


def _loaded_bm25(bm25_config: object) -> Bm25Encoder:
    encoder = Bm25Encoder(bm25_config.mode, bm25_config.k1, bm25_config.b)  # type: ignore[attr-defined]
    encoder.load(bm25_config.vocab_path)  # type: ignore[attr-defined]
    return encoder


@pytest.mark.parametrize(
    ("dense_model", "fake"), [("bge-m3", False), ("e5-base", False), ("e5-base", True)]
)
def test_cli_ingest_dense_selection(tmp_path, monkeypatch, dense_model, fake):
    from rag import cli
    from rag.encode.e5_embedder import E5_MODEL_NAME, E5Embedder, E5Settings
    from rag.text.normalizer import build_embed_text

    config = load_config(ROOT / "config.yaml")
    config = replace(
        config,
        paths=replace(config.paths, data_dir=ROOT / "data/sample", cache_dir=tmp_path),
        bm25=replace(config.bm25, vocab_path=tmp_path / "bm25_vocab.json"),
        embedding=replace(config.embedding, dense_model=dense_model, batch_size=2),
    )
    base = FakeEmbedder()
    reference_dense = FakeEmbedder(768)
    model = Mock()
    model.encode.side_effect = lambda texts, **kwargs: [
        reference_dense.encode([text.removeprefix("passage: ")])[0][0] for text in texts
    ]
    e5 = E5Embedder(model, E5Settings(E5_MODEL_NAME, tmp_path))
    load_e5 = Mock(return_value=e5)
    monkeypatch.setattr("rag.pipeline.load_e5", load_e5)
    monkeypatch.setattr(cli, "load_bge_m3", lambda settings: base)
    store = QdrantStore(
        QdrantClient(":memory:"),
        QdrantSettings(
            collection="dense_selection",
            upsert_batch_size=256,
            prefetch_limit=100,
            rrf_k=60,
            hnsw_m=16,
            ef_construct=100,
            full_scan_threshold=10,
            indexing_threshold=100,
            hnsw_ef=128,
        ),
    )
    monkeypatch.setattr(cli, "_store", lambda config: store)
    assert cli._ingest(config, fake=fake, recreate=True) == 0
    name, chunks, vectors = VectorCache().load(tmp_path)
    assert len(chunks) == len(vectors) == store.count() > 1
    use_e5 = dense_model == "e5-base" and not fake
    expected_dense = reference_dense if use_e5 else base
    expected_name = (
        f"{E5_MODEL_NAME}+sparse:{base.model_name}" if use_e5 else base.model_name
    )
    assert name == store.model_name() == expected_name
    bm25 = _loaded_bm25(config.bm25)
    texts = [
        build_embed_text(chunk.content, chunk.metadata.get("header_context", ""))
        for chunk in chunks
    ]
    expected_bm25 = bm25.encode_documents(texts)
    for text, vector, bm25_vector in zip(texts, vectors, expected_bm25, strict=True):
        assert vector.dense == expected_dense.encode([text])[0][0]
        assert vector.sparse == base.encode([text])[0][1]
        assert vector.bm25 == bm25_vector
    if use_e5:
        load_e5.assert_called_once()
        assert all(
            text.startswith("passage: ")
            for call in model.encode.call_args_list
            for text in call.args[0]
        )
        # A BGE-only query still must not search the E5 collection.
        with pytest.raises(ValueError):
            SearchService(base, bm25, store, top_k=10).check_model()
    else:
        load_e5.assert_not_called()

    # All CLI commands choose the same encoder and report its actual identity.
    config = replace(config, eval=replace(config.eval, warmup_runs=0, latency_runs=1))
    config = replace(
        config, paths=replace(config.paths, reports_dir=tmp_path / "reports")
    )
    monkeypatch.setattr(cli, "load_config", lambda _: config)
    flags = ["--fake-embedder"] if fake else []
    assert (
        cli.main(["search", "paracetamol", "--branches", "dense,sparse", *flags]) == 0
    )
    if use_e5:
        assert model.encode.call_args.args[0] == ["query: paracetamol"]
    for command in ("eval", "compare"):
        assert cli.main([command, "--configs", "dense+sparse", *flags]) == 0
    for path in (tmp_path / "reports").glob("*.json"):
        assert (
            json.loads(path.read_text(encoding="utf-8"))["model_name"] == expected_name
        )
