"""Đưa SparseVec về dạng pgvector và Milvus nhận được.

Chỉ số BGE-M3 nằm trong từ điển, còn FakeEmbedder băm ra số rất lớn.
Gộp theo modulo để vừa sparse_dimension mà không bỏ trọng số.
"""

from __future__ import annotations

from rag.schema import SparseVec


def collapse_sparse(vector: SparseVec, dimension: int) -> SparseVec:
    """Gộp chỉ số theo modulo dimension. Trọng số <= 0 bị bỏ."""
    if dimension < 1:
        raise ValueError(
            "sparse_dimension phải là số dương. Sửa khóa này trong config.yaml."
        )
    if len(vector.indices) != len(vector.values):
        raise ValueError("SparseVec có số chỉ số và số trọng số không bằng nhau.")
    buckets: dict[int, float] = {}
    for index, value in zip(vector.indices, vector.values, strict=True):
        if index < 0:
            raise ValueError(f"Chỉ số sparse {index} âm. Vector không hợp lệ.")
        if value <= 0:
            continue
        slot = index % dimension
        buckets[slot] = buckets.get(slot, 0.0) + float(value)
    pairs = sorted(buckets.items())
    return SparseVec(
        indices=[slot for slot, _ in pairs],
        values=[weight for _, weight in pairs],
    )


def sparsevec_literal(vector: SparseVec, dimension: int) -> str:
    """Literal sparsevec. SQL đếm chỉ số từ 1, SparseVec đếm từ 0."""
    fitted = collapse_sparse(vector, dimension)
    body = ",".join(
        f"{index + 1}:{_format_float(value)}"
        for index, value in zip(fitted.indices, fitted.values, strict=True)
    )
    return "{" + body + "}/" + str(dimension)


def sparse_pairs(vector: SparseVec, dimension: int) -> dict[int, float]:
    """Dict chỉ số → trọng số cho Milvus. Vector rỗng thành một trọng số rất nhỏ.

    Milvus từ chối vector thưa không có phần tử nào.
    """
    fitted = collapse_sparse(vector, dimension)
    if not fitted.indices:
        return {0: 1e-12}
    return dict(zip(fitted.indices, fitted.values, strict=True))


def dense_literal(values: list[float]) -> str:
    """Literal vector của pgvector."""
    return "[" + ",".join(_format_float(value) for value in values) + "]"


def _format_float(value: float) -> str:
    return format(value, ".8g")
