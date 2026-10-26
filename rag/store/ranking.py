"""Gộp kết quả nhiều nhánh bằng RRF cho kho không gộp sẵn trên server.

Qdrant 1.19 gộp bằng RrfQuery. pgvector và Milvus gọi hàm này.
"""

from __future__ import annotations

from rag.fusion import rrf
from rag.schema import Hit


def fuse_hits(rankings: list[list[Hit]], rrf_k: int, top_k: int) -> list[Hit]:
    """Giữ nội dung của hit gốc, thay điểm bằng điểm RRF."""
    by_id: dict[str, Hit] = {}
    ordered: list[list[str]] = []
    for hits in rankings:
        ids: list[str] = []
        for hit in hits:
            ids.append(hit.chunk_id)
            by_id.setdefault(hit.chunk_id, hit)
        ordered.append(ids)
    fused: list[Hit] = []
    for chunk_id, score in rrf(ordered, rrf_k)[:top_k]:
        found = by_id[chunk_id]
        fused.append(
            Hit(
                chunk_id=found.chunk_id,
                score=score,
                content=found.content,
                metadata=found.metadata,
            )
        )
    return fused
