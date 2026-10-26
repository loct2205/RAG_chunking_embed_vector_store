"""Các kiểu dữ liệu đi giữa loader, chunker, encoder, kho vector và phần đánh giá.

Mọi lớp khác chỉ trao đổi các dataclass ở đây, không truyền dict tự do.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    """Một file Markdown đã đọc, trước hoặc sau khi chuẩn hóa."""

    doc_id: str
    source: str
    text: str


@dataclass(frozen=True)
class Chunk:
    """Một đoạn đã cắt. Tên trường chunk_id, content, metadata là hợp đồng với nhóm khác."""

    chunk_id: str
    content: str
    metadata: dict[str, str]


@dataclass(frozen=True)
class SparseVec:
    """Vector thưa cho Qdrant. indices không được trùng."""

    indices: list[int]
    values: list[float]


@dataclass(frozen=True)
class Vectors:
    """Ba nhánh vector của một chunk hoặc một câu hỏi."""

    dense: list[float]
    sparse: SparseVec
    bm25: SparseVec


@dataclass(frozen=True)
class Hit:
    """Một kết quả tìm kiếm."""

    chunk_id: str
    score: float
    content: str
    metadata: dict[str, str]


@dataclass(frozen=True)
class EvalQuestion:
    """Một câu hỏi đánh giá. Đáp án là cụm chữ, không phải chunk_id."""

    qid: str
    question: str
    answer_keys: list[str]
    kind: str


@dataclass(frozen=True)
class SearchTiming:
    """Kết quả tìm kèm thời gian encode và thời gian truy vấn kho."""

    hits: list[Hit]
    encode_seconds: float
    query_seconds: float


def to_sparse(lexical_weights: dict[str, float]) -> SparseVec:
    """Chuyển lexical_weights của BGE-M3 (key là chuỗi) sang SparseVec cho Qdrant."""
    pairs = sorted(
        (int(key), float(value))
        for key, value in lexical_weights.items()
        if float(value) > 0
    )
    return SparseVec(
        indices=[index for index, _ in pairs],
        values=[value for _, value in pairs],
    )
