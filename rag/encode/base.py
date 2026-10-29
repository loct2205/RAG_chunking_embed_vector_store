"""Giao diện embedder. Pipeline và SearchService chỉ phụ thuộc lớp này."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from rag.schema import SparseVec


class Embedder(ABC):
    """Dense và sparse, có thể ghép từ hai model. BM25 là encoder riêng."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Tên ghi vào collection để search đối chiếu trước khi tìm."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Số chiều vector dense."""

    @abstractmethod
    def tokenize(self, text: str) -> Sequence[object]:
        """Tokenizer dùng cho SentenceSplitter. Chỉ cần đếm được độ dài."""

    @abstractmethod
    def encode(self, texts: list[str]) -> list[tuple[list[float], SparseVec]]:
        """Encode một batch tài liệu."""

    def encode_queries(self, texts: list[str]) -> list[tuple[list[float], SparseVec]]:
        """Mặc định giống tài liệu; model bất đối xứng có thể override."""
        return self.encode(texts)
