"""Nạp tài liệu: đọc, chuẩn hóa, cắt, BM25, encode, cache, ghi Qdrant.

Đây là chỗ duy nhất ghép loader, chunker, encoder và kho cho lệnh ingest.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass

from rag.config import AppConfig, EmbeddingConfig
from rag.encode.base import Embedder
from rag.encode.bm25 import Bm25Encoder
from rag.encode.cache import VectorCache
from rag.encode.e5_embedder import E5_MODEL_NAME, E5Settings, load_e5
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

        # Token diagnostics are optional and apply only to the E5 experiment.
        if (
            isinstance(self._parts.embedder, DenseOverrideEmbedder)
            and self._config.embedding.e5_model_path.is_dir()
        ):
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                str(self._config.embedding.e5_model_path),
                local_files_only=True,
            )

            limit = self._config.embedding.e5_max_length  # mặc định 512
            max_tokens = 0
            over_limit = 0

            for chunk, text in zip(chunks, texts, strict=True):
                token_ids = tokenizer.encode(
                    "passage: " + text,
                    add_special_tokens=True,
                    truncation=False,  # giữ nguyên để đếm cả phần vượt giới hạn
                )
                num_tokens = len(token_ids)
                max_tokens = max(max_tokens, num_tokens)

                if num_tokens > limit:
                    over_limit += 1
                    print(
                        f"{chunk.chunk_id}: {num_tokens} tokens "
                        f"(giới hạn={limit}, vượt={num_tokens - limit})"
                    )

            print(f"Chunk dài nhất: {max_tokens} tokens")
            print(f"Số chunk vượt {limit}: {over_limit}/{len(chunks)}")
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


def configure_dense_embedder(embedder: Embedder, config: EmbeddingConfig) -> Embedder:
    """Chọn dense cho cả tài liệu và query, giữ sparse và tokenizer BGE-M3."""
    if config.dense_model == "bge-m3":
        return embedder
    if config.dense_model != "e5-base":
        raise ValueError("embedding.dense_model phải là bge-m3 hoặc e5-base.")
    dense = load_e5(
        E5Settings(
            model_name=E5_MODEL_NAME,
            model_path=config.e5_model_path,
            max_length=config.e5_max_length,
            batch_size=config.e5_batch_size,
            use_fp16=config.use_fp16,
        )
    )
    return DenseOverrideEmbedder(embedder, dense)


class DenseOverrideEmbedder(Embedder):
    """Thay dense cho tài liệu và query, giữ sparse của encoder gốc."""

    def __init__(self, embedder: Embedder, dense_embedder: Embedder) -> None:
        self._base = embedder
        self._dense = dense_embedder

    @property
    def model_name(self) -> str:
        return f"{self._dense.model_name}+sparse:{self._base.model_name}"

    @property
    def dimension(self) -> int:
        return self._dense.dimension

    def tokenize(self, text: str) -> Sequence[object]:
        return self._base.tokenize(text)

    def encode(self, texts: list[str]) -> list[tuple[list[float], SparseVec]]:
        if not texts:
            return []
        original_rows = self._base.encode(texts)
        dense_rows = self._dense.encode(texts)
        if len(original_rows) != len(texts) or len(dense_rows) != len(texts):
            raise RuntimeError("Số vector dense/sparse phải bằng số văn bản đầu vào.")
        return [
            (dense, sparse)
            for (dense, _), (_, sparse) in zip(dense_rows, original_rows, strict=True)
        ]

    def encode_queries(self, texts: list[str]) -> list[tuple[list[float], SparseVec]]:
        if not texts:
            return []
        original_rows = self._base.encode_queries(texts)
        dense_rows = self._dense.encode_queries(texts)
        if len(original_rows) != len(texts) or len(dense_rows) != len(texts):
            raise RuntimeError("Số vector query dense/sparse phải bằng số câu hỏi.")
        return [
            (dense, sparse)
            for (dense, _), (_, sparse) in zip(dense_rows, original_rows, strict=True)
        ]
