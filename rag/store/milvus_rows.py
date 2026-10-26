"""Dòng ghi và hit của Milvus. Không mở kết nối."""

from __future__ import annotations

import re

from pymilvus import DataType

from rag.schema import Chunk, Hit, Vectors
from rag.store.ids import point_id
from rag.store.sparse_format import collapse_sparse, sparse_pairs

_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,40}$")
CONTENT_LIMIT = 65535


def checked_name(name: str) -> str:
    if _NAME.fullmatch(name) is None:
        raise ValueError(
            f"Tên collection '{name}' chỉ được gồm chữ, số và gạch dưới. "
            "Sửa collection trong config.yaml."
        )
    return name


def hit_from_milvus(record: dict[str, object]) -> Hit:
    """Đọc một hit từ kết quả MilvusClient.search."""
    entity = record.get("entity")
    if not isinstance(entity, dict):
        entity = {}
    metadata = entity.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    distance = record.get("distance")
    score = 0.0 if distance is None else float(distance)
    return Hit(
        chunk_id=str(entity.get("chunk_id", "")),
        score=score,
        content=str(entity.get("content", "")),
        metadata={str(key): str(value) for key, value in metadata.items()},
    )


def add_fields(schema: object, dense_dim: int) -> None:
    add = schema.add_field
    add("point_id", DataType.VARCHAR, is_primary=True, max_length=64)
    add("chunk_id", DataType.VARCHAR, max_length=256)
    add("content", DataType.VARCHAR, max_length=CONTENT_LIMIT)
    add("metadata", DataType.JSON)
    add("model_name", DataType.VARCHAR, max_length=256)
    add("dense", DataType.FLOAT_VECTOR, dim=dense_dim)
    add("sparse", DataType.SPARSE_FLOAT_VECTOR)
    add("bm25", DataType.SPARSE_FLOAT_VECTOR)


def insert_row(
    chunk: Chunk, vector: Vectors, model_name: str, dimension: int
) -> dict[str, object]:
    if len(chunk.content) > CONTENT_LIMIT:
        raise ValueError(
            f"Chunk '{chunk.chunk_id}' dài {len(chunk.content)} ký tự, "
            f"Milvus chỉ nhận VARCHAR tối đa {CONTENT_LIMIT}. Giảm chunk_size."
        )
    if len(chunk.chunk_id) > 256:
        raise ValueError(
            f"chunk_id '{chunk.chunk_id}' dài quá 256 ký tự. Rút ngắn doc_id."
        )
    return {
        "point_id": point_id(chunk.chunk_id),
        "chunk_id": chunk.chunk_id,
        "content": chunk.content,
        "metadata": dict(chunk.metadata),
        "model_name": model_name,
        "dense": list(vector.dense),
        "sparse": sparse_pairs(vector.sparse, dimension),
        "bm25": sparse_pairs(vector.bm25, dimension),
    }


def query_payload(
    query: Vectors, branch: str, sparse_dimension: int, hnsw_ef: int
) -> tuple[list[object], dict[str, object]] | None:
    if branch == "dense":
        params: dict[str, object] = {
            "metric_type": "COSINE",
            "params": {"ef": hnsw_ef},
        }
        return [list(query.dense)], params
    if branch not in ("sparse", "bm25"):
        raise ValueError(
            f"Nhánh '{branch}' không có trong collection. Chỉ dùng dense, sparse, bm25."
        )
    vector = query.sparse if branch == "sparse" else query.bm25
    if not collapse_sparse(vector, sparse_dimension).indices:
        return None
    return (
        [sparse_pairs(vector, sparse_dimension)],
        {"metric_type": "IP", "params": {}},
    )


def collection_description(info: object) -> str:
    if isinstance(info, dict):
        return str(info.get("description", ""))
    return str(getattr(info, "description", ""))
