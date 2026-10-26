"""Nạp tài liệu: đọc, chuẩn hóa, cắt, BM25, encode, cache, ghi Qdrant.

Đây là chỗ duy nhất ghép loader, chunker, encoder và kho cho lệnh ingest.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from rag.config import AppConfig
from rag.encode.base import Embedder
from rag.encode.bm25 import Bm25Encoder
from rag.encode.cache import VectorCache
from rag.schema import Chunk, Document, SparseVec, Vectors
from rag.store.base import VectorStore
from rag.text.chunker import Chunker
from rag.text.loader import DocumentLoader
from rag.text.normalizer import build_embed_text, normalize

log = logging.getLogger(__name__)


@dataclass
class IngestParts:
    """Các lớp đã tạo sẵn. Pipeline không tự nạp model hay tự mở client."""

    loader: DocumentLoader
    chunker: Chunker
    embedder: Embedder
    bm25: Bm25Encoder
    store: VectorStore
    cache: VectorCache


class IngestPipeline:
    """Chạy một lượt ingest và kiểm tra số điểm bằng số chunk."""

    def __init__(self, config: AppConfig, parts: IngestParts) -> None:
        self._config = config
        self._parts = parts

    def run(self, recreate: bool) -> int:
        started = time.perf_counter()
        documents = self._load()
        chunks = self._chunk(documents)
        texts = [
            build_embed_text(chunk.content, chunk.metadata.get("header_context", ""))
            for chunk in chunks
        ]
        self._fit_bm25(texts)
        vectors = self._encode(texts)
        self._parts.cache.save(
            self._config.paths.cache_dir,
            self._parts.embedder.model_name,
            chunks,
            vectors,
        )
        self._store_all(chunks, vectors, recreate)
        log.info(
            "Ingest xong %d chunk trong %.1fs",
            len(chunks),
            time.perf_counter() - started,
        )
        return len(chunks)

    def _load(self) -> list[Document]:
        started = time.perf_counter()
        raw_documents = self._parts.loader.load(self._config.paths.data_dir)
        documents = [
            Document(
                doc_id=document.doc_id,
                source=document.source,
                text=normalize(document.text),
            )
            for document in raw_documents
        ]
        log.info(
            "Đọc %d tài liệu trong %.1fs", len(documents), time.perf_counter() - started
        )
        return documents

    def _chunk(self, documents: list[Document]) -> list[Chunk]:
        started = time.perf_counter()
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self._parts.chunker.split_document(document))
        chunks.sort(key=lambda chunk: chunk.chunk_id)
        log.info(
            "Cắt xong %d chunk trong %.1fs", len(chunks), time.perf_counter() - started
        )
        return chunks

    def _fit_bm25(self, texts: list[str]) -> None:
        started = time.perf_counter()
        self._parts.bm25.fit(texts)
        path = self._config.bm25.vocab_path
        self._parts.bm25.save(path)
        log.info(
            "BM25 fit xong %d đoạn trong %.1fs, ghi %s",
            len(texts),
            time.perf_counter() - started,
            path,
        )

    def _encode(self, texts: list[str]) -> list[Vectors]:
        started = time.perf_counter()
        batch_size = self._config.embedding.batch_size
        dense_and_sparse: list[tuple[list[float], SparseVec]] = []
        for start in range(0, len(texts), batch_size):
            dense_and_sparse.extend(
                self._parts.embedder.encode(texts[start : start + batch_size])
            )
        bm25_vectors = self._parts.bm25.encode_documents(texts)
        vectors = [
            Vectors(dense=dense, sparse=sparse, bm25=bm25)
            for (dense, sparse), bm25 in zip(
                dense_and_sparse, bm25_vectors, strict=True
            )
        ]
        log.info(
            "Encode xong %d chunk trong %.1fs (batch=%d)",
            len(texts),
            time.perf_counter() - started,
            batch_size,
        )
        return vectors

    def _store_all(
        self, chunks: list[Chunk], vectors: list[Vectors], recreate: bool
    ) -> None:
        started = time.perf_counter()
        model_name = self._parts.embedder.model_name
        dense_dim = len(vectors[0].dense) if vectors else self._parts.embedder.dimension
        self._parts.store.ensure_collection(dense_dim, model_name, recreate)
        self._parts.store.upsert(chunks, vectors, model_name)
        log.info(
            "Ghi %d điểm vào collection, batch=%d, trong %.1fs",
            len(chunks),
            self._config.qdrant.upsert_batch_size,
            time.perf_counter() - started,
        )
