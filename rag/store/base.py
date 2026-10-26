"""Giao diện kho vector. Pipeline và SearchService không gọi client của từng kho."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag.schema import Chunk, Hit, Vectors


class VectorStore(ABC):
    """Tạo collection, ghi điểm, và tìm theo các nhánh đang bật."""

    @abstractmethod
    def ensure_collection(
        self, dense_dim: int, model_name: str, recreate: bool
    ) -> None:
        """Tạo collection nếu chưa có. Chỉ xóa khi recreate=True."""

    @abstractmethod
    def upsert(
        self, chunks: list[Chunk], vectors: list[Vectors], model_name: str
    ) -> None:
        """Ghi hoặc cập nhật điểm. Cùng chunk_id thì cùng point id."""

    @abstractmethod
    def search(self, query: Vectors, branches: list[str], top_k: int) -> list[Hit]:
        """Một nhánh thì query thẳng. Nhiều nhánh thì gộp RRF."""

    @abstractmethod
    def count(self) -> int:
        """Số điểm đang có trong collection."""

    @abstractmethod
    def model_name(self) -> str:
        """Tên model đã ghi lúc tạo collection."""
