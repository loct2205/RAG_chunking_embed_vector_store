"""Tìm kiếm: kiểm tra model, chuẩn hóa câu hỏi, encode các nhánh đang bật, rồi truy vấn kho."""

from __future__ import annotations

import logging
import time

from rag.encode.base import Embedder
from rag.encode.bm25 import Bm25Encoder
from rag.schema import Hit, SearchTiming, SparseVec, Vectors
from rag.store.base import VectorStore
from rag.text.normalizer import normalize

log = logging.getLogger(__name__)


class SearchService:
    """Một lượt tìm. check_model chạy một lần rồi nhớ kết quả trong tiến trình này."""

    def __init__(
        self,
        embedder: Embedder,
        bm25: Bm25Encoder,
        store: VectorStore,
        top_k: int,
    ) -> None:
        self._embedder = embedder
        self._bm25 = bm25
        self._store = store
        self._top_k = top_k
        self._checked = False

    def check_model(self) -> None:
        """Dừng nếu collection được nạp bằng model khác với lượt tìm này."""
        if self._checked:
            return
        stored = self._store.model_name()
        current = self._embedder.model_name
        if stored != current:
            raise ValueError(
                f"Collection đang chứa model '{stored}' nhưng lượt tìm dùng '{current}'. "
                "Chạy lại ingest với cùng model, hoặc thêm --fake-embedder cho cả hai lệnh."
            )
        self._checked = True

    def search(
        self, question: str, branches: list[str], top_k: int | None = None
    ) -> list[Hit]:
        return self.search_timed(question, branches, top_k).hits

    def search_timed(
        self,
        question: str,
        branches: list[str],
        top_k: int | None = None,
        log_result: bool = True,
    ) -> SearchTiming:
        self.check_model()
        limit = self._top_k if top_k is None else top_k
        normalized = normalize(question)
        encode_started = time.perf_counter()
        query = self._encode(normalized, branches)
        encode_seconds = time.perf_counter() - encode_started
        query_started = time.perf_counter()
        hits = self._store.search(query, branches, limit)
        query_seconds = time.perf_counter() - query_started
        if log_result:
            log.info(
                "Tìm '%s' nhánh %s: %d hit, encode %.3fs, truy vấn %.3fs",
                normalized[:80],
                ",".join(branches),
                len(hits),
                encode_seconds,
                query_seconds,
            )
        return SearchTiming(
            hits=hits, encode_seconds=encode_seconds, query_seconds=query_seconds
        )

    def _encode(self, question: str, branches: list[str]) -> Vectors:
        dense: list[float] = []
        sparse = SparseVec(indices=[], values=[])
        bm25 = SparseVec(indices=[], values=[])
        if "dense" in branches or "sparse" in branches:
            dense, sparse = self._embedder.encode_queries([question])[0]
        if "bm25" in branches:
            bm25 = self._bm25.encode_query(question)
        return Vectors(dense=dense, sparse=sparse, bm25=bm25)
