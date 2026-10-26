"""Milvus 2.5: dense HNSW cosine, sparse và bm25 là sparse inverted index.

Server không gộp nhiều nhánh trong một lần search, nên RRF chạy ở client.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from pymilvus import MilvusClient
from pymilvus.exceptions import (
    ConnectError,
    ConnectionNotExistException,
    MilvusException,
    MilvusUnavailableException,
)

from rag.schema import Chunk, Hit, Vectors
from rag.store.base import VectorStore
from rag.store.milvus_rows import (
    add_fields,
    checked_name,
    collection_description,
    hit_from_milvus,
    insert_row,
    query_payload,
)
from rag.store.ranking import fuse_hits

log = logging.getLogger(__name__)

_MAX_ATTEMPTS = 3
_CONNECTION_ERRORS = (
    ConnectError,
    MilvusUnavailableException,
    ConnectionNotExistException,
)
T = TypeVar("T")


@dataclass(frozen=True)
class MilvusSettings:
    """Tham số collection và HNSW. URI nằm ở hàm tạo client."""

    collection: str
    sparse_dimension: int
    upsert_batch_size: int
    prefetch_limit: int
    rrf_k: int
    hnsw_m: int
    ef_construct: int
    hnsw_ef: int


class MilvusStore(VectorStore):
    """Một collection với ba field vector. Client được truyền từ ngoài."""

    def __init__(
        self, connect: Callable[[], MilvusClient], settings: MilvusSettings
    ) -> None:
        self._connect = connect
        self._settings = settings
        self._cached: MilvusClient | None = None
        self._collection = checked_name(settings.collection)

    def ensure_collection(
        self, dense_dim: int, model_name: str, recreate: bool
    ) -> None:
        exists = self._call(
            "kiểm tra collection",
            lambda client: bool(client.has_collection(self._collection)),
        )
        if exists and recreate:
            self._call(
                "xóa collection",
                lambda client: client.drop_collection(self._collection),
            )
            exists = False
            log.info("Đã xóa collection '%s' vì có --recreate", self._collection)
        if exists:
            self._check_existing(dense_dim, model_name)
            return
        self._create(dense_dim, model_name)

    def upsert(
        self, chunks: list[Chunk], vectors: list[Vectors], model_name: str
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Có {len(chunks)} chunk nhưng {len(vectors)} vector. Không ghi lên Milvus."
            )
        rows = [
            insert_row(chunk, vector, model_name, self._settings.sparse_dimension)
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        name = self._collection
        batch_size = self._settings.upsert_batch_size
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            self._call("upsert", lambda client, batch=batch: client.upsert(name, batch))
        stored = self.count()
        if stored != len(chunks):
            raise RuntimeError(
                f"Collection '{name}' có {stored} điểm sau khi nạp, "
                f"nhưng có {len(chunks)} chunk. Chạy lại ingest với --recreate."
            )

    def search(self, query: Vectors, branches: list[str], top_k: int) -> list[Hit]:
        if len(branches) == 1:
            return self._search_one(query, branches[0], top_k)
        rankings = [
            self._search_one(query, branch, self._settings.prefetch_limit)
            for branch in branches
        ]
        return fuse_hits(rankings, self._settings.rrf_k, top_k)

    def count(self) -> int:
        stats = self._call(
            "đếm điểm",
            lambda client: client.get_collection_stats(self._collection),
        )
        if not isinstance(stats, dict) or "row_count" not in stats:
            raise RuntimeError(f"Milvus không trả row_count cho '{self._collection}'.")
        return int(stats["row_count"])

    def model_name(self) -> str:
        stored, _dense_dim = self._read_meta()
        return stored

    def _create(self, dense_dim: int, model_name: str) -> None:
        schema = self._call(
            "tạo schema",
            lambda client: client.create_schema(
                description=f"{model_name}|{dense_dim}"
            ),
        )
        add_fields(schema, dense_dim)
        hnsw_m = self._settings.hnsw_m
        ef_construct = self._settings.ef_construct

        def build(client: MilvusClient) -> None:
            index_params = client.prepare_index_params()
            index_params.add_index(
                field_name="dense",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": hnsw_m, "efConstruction": ef_construct},
            )
            for field_name in ("sparse", "bm25"):
                # Không truyền inverted_index_algo: Milvus Lite trên Colab chỉ nhận
                # SPARSE_INVERTED_INDEX trần; server 2.5 vẫn dùng chỉ mục này.
                index_params.add_index(
                    field_name=field_name,
                    index_type="SPARSE_INVERTED_INDEX",
                    metric_type="IP",
                )
            client.create_collection(
                self._collection,
                schema=schema,
                index_params=index_params,
                consistency_level="Strong",
            )

        self._call("tạo collection", build)
        log.info(
            "Đã tạo collection '%s' (%d chiều, model %s)",
            self._collection,
            dense_dim,
            model_name,
        )

    def _check_existing(self, dense_dim: int, model_name: str) -> None:
        stored, got = self._read_meta()
        if got != dense_dim:
            raise ValueError(
                f"Collection '{self._collection}' có {got} chiều nhưng model "
                f"{model_name} ra {dense_dim} chiều. "
                "Đổi model trong config.yaml hoặc chạy lại ingest với --recreate."
            )
        if stored != model_name:
            raise ValueError(
                f"Collection '{self._collection}' đang chứa model '{stored}' "
                f"nhưng lượt nạp dùng '{model_name}'. "
                "Chạy lại ingest với --recreate."
            )

    def _read_meta(self) -> tuple[str, int]:
        info = self._call(
            "đọc collection",
            lambda client: client.describe_collection(self._collection),
        )
        description = collection_description(info)
        model, separator, dim_text = description.partition("|")
        if separator != "|" or not dim_text.isdigit() or not model:
            raise ValueError(
                f"Collection '{self._collection}' không ghi tên model. "
                "Chạy lại ingest với --recreate."
            )
        return model, int(dim_text)

    def _search_one(self, query: Vectors, branch: str, limit: int) -> list[Hit]:
        payload = query_payload(
            query, branch, self._settings.sparse_dimension, self._settings.hnsw_ef
        )
        if payload is None:
            return []
        data, params = payload
        found = self._call(
            f"tìm {branch}",
            lambda client: client.search(
                self._collection,
                data,
                anns_field=branch,
                limit=limit,
                output_fields=["chunk_id", "content", "metadata"],
                search_params=params,
            ),
        )
        if not found:
            return []
        return [hit_from_milvus(item) for item in found[0]]

    def _call(self, description: str, action: Callable[[MilvusClient], T]) -> T:
        last: Exception | None = None
        for attempt in range(1, _MAX_ATTEMPTS + 1):
            try:
                return action(self._alive())
            except _CONNECTION_ERRORS as exc:
                self._discard()
                last = exc
                log.warning(
                    "Milvus lỗi kết nối (%s) lần %d/%d: %s",
                    description,
                    attempt,
                    _MAX_ATTEMPTS,
                    exc,
                )
        raise RuntimeError(
            f"Milvus lỗi kết nối ({description}) sau {_MAX_ATTEMPTS} lần: {last}. "
            "Kiểm tra `docker compose up -d etcd minio milvus` và milvus.uri."
        )

    def _alive(self) -> MilvusClient:
        if self._cached is None:
            self._cached = self._connect()
        return self._cached

    def _discard(self) -> None:
        client = self._cached
        self._cached = None
        if client is None:
            return
        try:
            client.close()
        except (MilvusException, OSError) as exc:
            log.warning("Đóng client Milvus lỗi: %s", exc)
