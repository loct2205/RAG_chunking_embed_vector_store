"""Câu SQL và dòng ghi của pgvector. Không mở kết nối."""

from __future__ import annotations

import json
import re
from typing import Any

from rag.schema import Chunk, Hit, Vectors
from rag.store.ids import point_id
from rag.store.sparse_format import dense_literal, sparsevec_literal

_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,40}$")
_SCORE = {
    "dense": "1 - (dense <=> %s::vector)",
    "sparse": "-(sparse <#> %s::sparsevec)",
    "bm25": "-(bm25 <#> %s::sparsevec)",
}


def checked_name(name: str) -> str:
    """Tên bảng chỉ gồm chữ, số và gạch dưới để ghép vào SQL an toàn."""
    if _NAME.fullmatch(name) is None:
        raise ValueError(
            f"Tên collection '{name}' chỉ được gồm chữ, số và gạch dưới. "
            "Sửa collection trong config.yaml."
        )
    return name


def meta_name(collection: str) -> str:
    return f"{collection}_meta"


def score_expression(branch: str) -> str:
    return _SCORE[branch]


def query_literal(query: Vectors, branch: str, dimension: int) -> str | None:
    """Literal để truy vấn. Vector thưa rỗng thì không tìm nhánh đó."""
    if branch == "dense":
        return dense_literal(query.dense)
    if branch not in _SCORE:
        raise ValueError(
            f"Nhánh '{branch}' không có trong bảng. Chỉ dùng dense, sparse, bm25."
        )
    vector = query.sparse if branch == "sparse" else query.bm25
    literal = sparsevec_literal(vector, dimension)
    if literal == "{}/" + str(dimension):
        return None
    return literal


def insert_row(
    chunk: Chunk, vector: Vectors, model_name: str, dimension: int
) -> tuple[object, ...]:
    return (
        point_id(chunk.chunk_id),
        chunk.chunk_id,
        chunk.content,
        json.dumps(chunk.metadata, ensure_ascii=False, sort_keys=True),
        model_name,
        dense_literal(vector.dense),
        sparsevec_literal(vector.sparse, dimension),
        sparsevec_literal(vector.bm25, dimension),
    )


def hit_from_row(row: tuple[Any, ...]) -> Hit:
    metadata = row[2]
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    if not isinstance(metadata, dict):
        metadata = {}
    return Hit(
        chunk_id=str(row[0]),
        score=float(row[3]),
        content=str(row[1]),
        metadata={str(key): str(value) for key, value in metadata.items()},
    )
