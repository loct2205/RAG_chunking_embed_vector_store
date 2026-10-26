"""Ingest rồi search trên data/sample, FakeEmbedder và Qdrant trong bộ nhớ."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

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


def test_ingest_then_search_finds_table_row_and_plain_note(tmp_path: Path) -> None:
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
    summary = Evaluator(service, questions).score(["dense", "bm25"], top_k=10)
    assert summary["text_count"] >= 1
    assert summary["table_count"] >= 1
    assert float(summary["table_recall_at_k"]) > 0


def _loaded_bm25(bm25_config: object) -> Bm25Encoder:
    encoder = Bm25Encoder(bm25_config.mode, bm25_config.k1, bm25_config.b)  # type: ignore[attr-defined]
    encoder.load(bm25_config.vocab_path)  # type: ignore[attr-defined]
    return encoder
