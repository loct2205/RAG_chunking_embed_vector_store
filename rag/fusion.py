"""Gộp thứ hạng bằng Reciprocal Rank Fusion cho kho không gộp sẵn.

Qdrant 1.19 gộp trên server. Hàm này dùng khi so sánh kho khác, hoặc để đối chiếu.
"""

from __future__ import annotations


def rrf(rankings: list[list[str]], k: int) -> list[tuple[str, float]]:
    """Gộp các danh sách chunk_id đã xếp hạng. Điểm cao hơn đứng trước.

    Cùng điểm thì xếp theo chunk_id để mỗi lần chạy ra cùng một thứ tự.
    """
    if k < 0:
        raise ValueError(
            "RRF k phải là số không âm. Sửa search.rrf_k trong config.yaml."
        )
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
