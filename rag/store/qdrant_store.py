"""Qdrant 1.19: tạo collection, payload index, rồi upsert. Tìm bằng query_points.

Client được truyền từ ngoài. Test dùng QdrantClient(':memory:').
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from rag.schema import Chunk, Hit, SparseVec, Vectors
from rag.store.base import VectorStore
from rag.store.ids import point_id

log = logging.getLogger(__name__)

_MAX_ATTEMPTS = 3
T = TypeVar("T")


@dataclass(frozen=True)
class QdrantSettings:
    """Tham số collection và truy vấn, không gồm URL. URL nằm ở client."""

    collection: str
    upsert_batch_size: int
    prefetch_limit: int
    rrf_k: int
    hnsw_m: int
    ef_construct: int
    full_scan_threshold: int
    indexing_threshold: int
    hnsw_ef: int


def call_with_retry(action: Callable[[], T], description: str) -> T:
    """Thử lại tối đa 3 lần khi mất kết nối. Lỗi dữ liệu không thử lại."""
    last_error: ResponseHandlingException | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            return action()
        except ResponseHandlingException as exc:
            last_error = exc
            log.warning(
                "Qdrant lỗi kết nối (%s) lần %d/%d: %s",
                description,
                attempt,
                _MAX_ATTEMPTS,
                exc,
            )
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Qdrant không trả kết quả cho '{description}'.")


class QdrantStore(VectorStore):
    """Một collection với vector dense, sparse của BGE-M3, và bm25 có IDF."""

    def __init__(self, client: QdrantClient, settings: QdrantSettings) -> None:
        self._client = client
        self._settings = settings

    def ensure_collection(
        self, dense_dim: int, model_name: str, recreate: bool
    ) -> None:
        name = self._settings.collection
        exists = call_with_retry(
            lambda: self._client.collection_exists(name), "kiểm tra collection"
        )
        if exists and recreate:
            call_with_retry(
                lambda: self._client.delete_collection(name), "xóa collection"
            )
            exists = False
            log.info("Đã xóa collection '%s' vì có --recreate", name)
        if exists:
            self._check_existing(dense_dim, model_name)
            return
        self._create(dense_dim, model_name)

    def upsert(
        self, chunks: list[Chunk], vectors: list[Vectors], model_name: str
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Có {len(chunks)} chunk nhưng {len(vectors)} vector. Không ghi lên Qdrant."
            )
        name = self._settings.collection
        batch_size = self._settings.upsert_batch_size
        points = [
            _point(chunk, vector, model_name)
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        for start in range(0, len(points), batch_size):
            batch = points[start : start + batch_size]
            call_with_retry(
                lambda batch=batch: self._client.upsert(
                    collection_name=name, points=batch, wait=True
                ),
                "upsert",
            )
        stored = self.count()
        if stored != len(chunks):
            raise RuntimeError(
                f"Collection '{name}' có {stored} điểm sau khi nạp, nhưng có {len(chunks)} chunk. "
                "Chạy lại ingest với --recreate."
            )

    def search(self, query: Vectors, branches: list[str], top_k: int) -> list[Hit]:
        name = self._settings.collection
        search_params = models.SearchParams(hnsw_ef=self._settings.hnsw_ef)
        if len(branches) == 1:
            response = call_with_retry(
                lambda: self._client.query_points(
                    collection_name=name,
                    query=_branch_query(query, branches[0]),
                    using=branches[0],
                    query_filter=None,
                    search_params=search_params,
                    limit=top_k,
                    with_payload=True,
                ),
                "tìm một nhánh",
            )
        else:
            response = call_with_retry(
                lambda: self._client.query_points(
                    collection_name=name,
                    prefetch=_prefetches(
                        query, branches, self._settings.prefetch_limit
                    ),
                    query=models.RrfQuery(rrf=models.Rrf(k=self._settings.rrf_k)),
                    search_params=search_params,
                    limit=top_k,
                    with_payload=True,
                ),
                "tìm hybrid",
            )
        return [_hit(point) for point in response.points]

    def count(self) -> int:
        name = self._settings.collection
        result = call_with_retry(
            lambda: self._client.count(collection_name=name, exact=True),
            "đếm điểm",
        )
        return int(result.count)

    def model_name(self) -> str:
        info = self._collection_info()
        metadata = info.config.metadata or {}
        stored = metadata.get("model_name")
        if not stored:
            raise ValueError(
                f"Collection '{self._settings.collection}' không ghi tên model. "
                "Chạy lại ingest với --recreate."
            )
        return str(stored)

    def _create(self, dense_dim: int, model_name: str) -> None:
        name = self._settings.collection
        settings = self._settings
        try:
            self._create_collection(name, dense_dim, model_name, settings)
        except UnexpectedResponse as error:
            raise ValueError(
                f"Qdrant từ chối tạo collection '{name}': {error}. "
                "Sửa tham số trong config.yaml rồi chạy lại."
            ) from error
        for field_name in ("doc_id", "content_type"):
            call_with_retry(
                lambda field_name=field_name: self._client.create_payload_index(
                    collection_name=name,
                    field_name=field_name,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                ),
                "tạo payload index",
            )
        log.info(
            "Đã tạo collection '%s' (%d chiều, model %s)", name, dense_dim, model_name
        )

    def _create_collection(
        self, name: str, dense_dim: int, model_name: str, settings: QdrantSettings
    ) -> None:
        call_with_retry(
            lambda: self._client.create_collection(
                collection_name=name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=dense_dim, distance=models.Distance.COSINE
                    ),
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams(),
                    "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF),
                },
                hnsw_config=models.HnswConfigDiff(
                    m=settings.hnsw_m,
                    ef_construct=settings.ef_construct,
                    full_scan_threshold=settings.full_scan_threshold,
                ),
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=settings.indexing_threshold
                ),
                metadata={"model_name": model_name},
            ),
            "tạo collection",
        )

    def _check_existing(self, dense_dim: int, model_name: str) -> None:
        info = self._collection_info()
        got = _dense_size(info)
        if got != dense_dim:
            raise ValueError(
                f"Collection '{self._settings.collection}' có {got} chiều nhưng model "
                f"{model_name} ra {dense_dim} chiều. "
                "Đổi model trong config.yaml hoặc chạy lại ingest với --recreate."
            )
        stored = self.model_name()
        if stored != model_name:
            raise ValueError(
                f"Collection '{self._settings.collection}' đang chứa model '{stored}' "
                f"nhưng lượt nạp dùng '{model_name}'. "
                "Chạy lại ingest với --recreate."
            )

    def _collection_info(self) -> models.CollectionInfo:
        name = self._settings.collection
        if not call_with_retry(
            lambda: self._client.collection_exists(name), "kiểm tra collection"
        ):
            raise ValueError(
                f"Chưa có collection '{name}'. Chạy ingest trước khi search."
            )
        return call_with_retry(
            lambda: self._client.get_collection(name), "đọc collection"
        )


def _point(chunk: Chunk, vector: Vectors, model_name: str) -> models.PointStruct:
    metadata = chunk.metadata
    return models.PointStruct(
        id=point_id(chunk.chunk_id),
        vector={
            "dense": vector.dense,
            "sparse": _qdrant_sparse(vector.sparse),
            "bm25": _qdrant_sparse(vector.bm25),
        },
        payload={
            "chunk_id": chunk.chunk_id,
            "content": chunk.content,
            "metadata": metadata,
            "model_name": model_name,
            "doc_id": metadata.get("doc_id", ""),
            "content_type": metadata.get("content_type", ""),
        },
    )


def _qdrant_sparse(vector: SparseVec) -> models.SparseVector:
    return models.SparseVector(indices=vector.indices, values=vector.values)


def _branch_query(query: Vectors, branch: str) -> list[float] | models.SparseVector:
    if branch == "dense":
        return query.dense
    if branch == "sparse":
        return _qdrant_sparse(query.sparse)
    if branch == "bm25":
        return _qdrant_sparse(query.bm25)
    raise ValueError(
        f"Nhánh '{branch}' không có trong collection. Chỉ dùng dense, sparse, bm25."
    )


def _prefetches(
    query: Vectors, branches: list[str], limit: int
) -> list[models.Prefetch]:
    return [
        models.Prefetch(query=_branch_query(query, branch), using=branch, limit=limit)
        for branch in branches
    ]


def _hit(point: models.ScoredPoint) -> Hit:
    payload = point.payload or {}
    metadata = payload.get("metadata") or {}
    return Hit(
        chunk_id=str(payload.get("chunk_id", "")),
        score=float(point.score or 0.0),
        content=str(payload.get("content", "")),
        metadata={str(key): str(value) for key, value in metadata.items()},
    )


def _dense_size(info: models.CollectionInfo) -> int:
    vectors = info.config.params.vectors
    if isinstance(vectors, dict):
        params = vectors.get("dense")
        if params is None:
            raise ValueError(
                "Collection không có vector 'dense'. Chạy lại ingest với --recreate."
            )
        return int(params.size)
    return int(vectors.size)
