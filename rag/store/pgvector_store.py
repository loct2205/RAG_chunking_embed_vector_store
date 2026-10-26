"""pgvector: dense HNSW cosine, sparse và bm25 là sparsevec. Nhiều nhánh gộp bằng RRF.

Hàm connect được truyền từ ngoài. Server không gộp sẵn nên fusion nằm ở client.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from psycopg import Connection, sql

from rag.schema import Chunk, Hit, Vectors
from rag.store.base import VectorStore
from rag.store.pgvector_rows import (
    checked_name,
    hit_from_row,
    insert_row,
    meta_name,
    query_literal,
    score_expression,
)
from rag.store.pgvector_session import PgSession
from rag.store.ranking import fuse_hits

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PgSettings:
    """Tham số bảng và HNSW. Chuỗi kết nối nằm ở hàm connect."""

    collection: str
    sparse_dimension: int
    upsert_batch_size: int
    prefetch_limit: int
    rrf_k: int
    hnsw_m: int
    ef_construct: int
    hnsw_ef: int


class PgVectorStore(VectorStore):
    """Một bảng Postgres với extension vector. Tạo index rồi mới ghi."""

    def __init__(self, connect: Callable[[], Connection], settings: PgSettings) -> None:
        self._settings = settings
        self._session = PgSession(connect)
        self._collection = checked_name(settings.collection)

    def ensure_collection(
        self, dense_dim: int, model_name: str, recreate: bool
    ) -> None:
        if self._table_exists() and recreate:
            self._drop()
            log.info("Đã xóa bảng '%s' vì có --recreate", self._collection)
        if self._table_exists():
            self._check_existing(dense_dim, model_name)
            return
        self._create(dense_dim, model_name)

    def upsert(
        self, chunks: list[Chunk], vectors: list[Vectors], model_name: str
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Có {len(chunks)} chunk nhưng {len(vectors)} vector. Không ghi lên Postgres."
            )
        rows = [
            insert_row(chunk, vector, model_name, self._settings.sparse_dimension)
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        statement = sql.SQL(
            """
            INSERT INTO {table}
                (point_id, chunk_id, content, metadata, model_name, dense, sparse, bm25)
            VALUES (%s, %s, %s, %s::jsonb, %s, %s::vector, %s::sparsevec, %s::sparsevec)
            ON CONFLICT (point_id) DO UPDATE SET
                chunk_id = EXCLUDED.chunk_id,
                content = EXCLUDED.content,
                metadata = EXCLUDED.metadata,
                model_name = EXCLUDED.model_name,
                dense = EXCLUDED.dense,
                sparse = EXCLUDED.sparse,
                bm25 = EXCLUDED.bm25
            """
        ).format(table=sql.Identifier(self._collection))
        batch_size = self._settings.upsert_batch_size
        for start in range(0, len(rows), batch_size):
            self._session.executemany(
                statement, rows[start : start + batch_size], "upsert"
            )
        stored = self.count()
        if stored != len(chunks):
            raise RuntimeError(
                f"Bảng '{self._collection}' có {stored} dòng sau khi nạp, "
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
        row = self._session.run(
            sql.SQL("SELECT count(*) FROM {table}").format(
                table=sql.Identifier(self._collection)
            ),
            None,
            "đếm điểm",
            "one",
        )
        if not isinstance(row, tuple):
            raise TypeError(f"Không đếm được bảng '{self._collection}'.")
        return int(row[0])

    def model_name(self) -> str:
        stored, _dense_dim = self._read_meta()
        return stored

    def _create(self, dense_dim: int, model_name: str) -> None:
        self._session.run(
            "CREATE EXTENSION IF NOT EXISTS vector", None, "bật pgvector", "none"
        )
        self._create_table(dense_dim)
        meta = sql.Identifier(meta_name(self._collection))
        self._session.run(
            sql.SQL(
                "CREATE TABLE {meta} (model_name text PRIMARY KEY, dense_dim integer NOT NULL)"
            ).format(meta=meta),
            None,
            "tạo bảng meta",
            "none",
        )
        self._session.run(
            sql.SQL(
                "INSERT INTO {meta} (model_name, dense_dim) VALUES (%s, %s)"
            ).format(meta=meta),
            (model_name, dense_dim),
            "ghi tên model",
            "none",
        )
        self._create_indexes()
        log.info(
            "Đã tạo bảng '%s' (%d chiều, model %s)",
            self._collection,
            dense_dim,
            model_name,
        )

    def _create_table(self, dense_dim: int) -> None:
        statement = sql.SQL(
            """
            CREATE TABLE {table} (
                point_id uuid PRIMARY KEY,
                chunk_id text NOT NULL UNIQUE,
                content text NOT NULL,
                metadata jsonb NOT NULL,
                model_name text NOT NULL,
                dense vector({dense_dim}) NOT NULL,
                sparse sparsevec({sparse_dim}) NOT NULL,
                bm25 sparsevec({sparse_dim}) NOT NULL
            )
            """
        ).format(
            table=sql.Identifier(self._collection),
            dense_dim=sql.Literal(int(dense_dim)),
            sparse_dim=sql.Literal(int(self._settings.sparse_dimension)),
        )
        self._session.run(statement, None, "tạo bảng", "none")

    def _create_indexes(self) -> None:
        table = sql.Identifier(self._collection)
        for column, ops in (
            ("dense", "vector_cosine_ops"),
            ("sparse", "sparsevec_ip_ops"),
            ("bm25", "sparsevec_ip_ops"),
        ):
            statement = sql.SQL(
                "CREATE INDEX {idx} ON {table} USING hnsw ({column} {ops}) "
                "WITH (m = {m}, ef_construction = {ef})"
            ).format(
                idx=sql.Identifier(f"{self._collection}_{column}_hnsw"),
                table=table,
                column=sql.Identifier(column),
                ops=sql.SQL(ops),
                m=sql.Literal(int(self._settings.hnsw_m)),
                ef=sql.Literal(int(self._settings.ef_construct)),
            )
            self._session.run(statement, None, f"tạo index {column}", "none")

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
        row = self._session.run(
            sql.SQL("SELECT model_name, dense_dim FROM {meta}").format(
                meta=sql.Identifier(meta_name(self._collection))
            ),
            None,
            "đọc model",
            "one",
        )
        if not isinstance(row, tuple):
            raise TypeError(
                f"Bảng '{self._collection}' không ghi tên model. "
                "Chạy lại ingest với --recreate."
            )
        return str(row[0]), int(row[1])

    def _search_one(self, query: Vectors, branch: str, limit: int) -> list[Hit]:
        literal = query_literal(query, branch, self._settings.sparse_dimension)
        if literal is None:
            return []
        self._session.run(
            "SELECT set_config('hnsw.ef_search', %s, false)",
            (str(self._settings.hnsw_ef),),
            "đặt hnsw.ef_search",
            "one",
        )
        statement = sql.SQL(
            "SELECT chunk_id, content, metadata, {score} AS score FROM {table} "
            "ORDER BY score DESC, chunk_id LIMIT %s"
        ).format(
            score=sql.SQL(score_expression(branch)),
            table=sql.Identifier(self._collection),
        )
        rows = self._session.run(statement, (literal, limit), f"tìm {branch}", "all")
        if not isinstance(rows, list):
            return []
        return [hit_from_row(row) for row in rows]

    def _table_exists(self) -> bool:
        row = self._session.run(
            "SELECT to_regclass(%s)",
            (self._collection,),
            "kiểm tra bảng",
            "one",
        )
        return isinstance(row, tuple) and row[0] is not None

    def _drop(self) -> None:
        for name in (self._collection, meta_name(self._collection)):
            self._session.run(
                sql.SQL("DROP TABLE IF EXISTS {table}").format(
                    table=sql.Identifier(name)
                ),
                None,
                "xóa bảng",
                "none",
            )
