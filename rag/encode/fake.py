"""Embedder giả, vector cố định từ hash của token.

Chỉ để chạy thử luồng ingest/search/eval khi không có GPU và không tải model.
Không dùng kết quả này để đo chất lượng embedding.
"""

from __future__ import annotations

import hashlib
import math

from rag.encode.base import Embedder
from rag.schema import SparseVec

FAKE_DIMENSION = 32
FAKE_MODEL_NAME = "fake-embedder"


class FakeEmbedder(Embedder):
    """Cùng một token luôn rơi vào cùng một chiều, nên truy vấn còn khớp được từ."""

    def __init__(self, dimension: int = FAKE_DIMENSION) -> None:
        if dimension < 1:
            raise ValueError("dimension của FakeEmbedder phải dương.")
        self._dimension = dimension

    @property
    def model_name(self) -> str:
        return FAKE_MODEL_NAME

    @property
    def dimension(self) -> int:
        return self._dimension

    def tokenize(self, text: str) -> list[str]:
        """Tách theo khoảng trắng. Không phải tokenizer BGE-M3."""
        return text.split()

    def encode(self, texts: list[str]) -> list[tuple[list[float], SparseVec]]:
        return [self._encode_one(text) for text in texts]

    def _encode_one(self, text: str) -> tuple[list[float], SparseVec]:
        dense = [0.0] * self._dimension
        weights: dict[int, float] = {}
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            dense[int.from_bytes(digest[:4], "big") % self._dimension] += 1.0
            weights[int.from_bytes(digest[4:8], "big")] = (
                weights.get(int.from_bytes(digest[4:8], "big"), 0.0) + 1.0
            )
        if not any(dense):
            dense[0] = 1.0
        return _l2_normalize(dense), _sparse_from_weights(weights)


def _l2_normalize(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return values
    return [value / norm for value in values]


def _sparse_from_weights(weights: dict[int, float]) -> SparseVec:
    pairs = sorted((index, weight) for index, weight in weights.items() if weight > 0)
    return SparseVec(
        indices=[index for index, _ in pairs],
        values=[weight for _, weight in pairs],
    )
