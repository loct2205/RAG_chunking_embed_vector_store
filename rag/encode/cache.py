"""Lưu vector đã encode để các kho so sánh nạp từ cùng một bản.

Pipeline ghi sau khi encode. File nằm dưới cache_dir trong config.
"""

from __future__ import annotations

import json
from pathlib import Path

from rag.schema import Chunk, SparseVec, Vectors

VECTORS_FILE = "vectors.json"


class VectorCache:
    """Một file JSON, sắp theo chunk_id trước khi ghi."""

    def save(
        self,
        directory: Path,
        model_name: str,
        chunks: list[Chunk],
        vectors: list[Vectors],
    ) -> Path:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Có {len(chunks)} chunk nhưng {len(vectors)} vector. Encode lại trước khi ghi cache."
            )
        directory.mkdir(parents=True, exist_ok=True)
        rows = []
        for chunk, vector in sorted(
            zip(chunks, vectors, strict=True), key=lambda pair: pair[0].chunk_id
        ):
            rows.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "content": chunk.content,
                    "metadata": chunk.metadata,
                    "dense": vector.dense,
                    "sparse": _sparse_payload(vector.sparse),
                    "bm25": _sparse_payload(vector.bm25),
                }
            )
        path = directory / VECTORS_FILE
        payload = {"model_name": model_name, "chunks": rows}
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

    def load(self, directory: Path) -> tuple[str, list[Chunk], list[Vectors]]:
        path = directory / VECTORS_FILE
        if not path.is_file():
            raise FileNotFoundError(
                f"Không thấy cache tại '{path}'. Chạy ingest trước khi so sánh kho."
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        chunks: list[Chunk] = []
        vectors: list[Vectors] = []
        for row in payload["chunks"]:
            chunks.append(
                Chunk(
                    chunk_id=row["chunk_id"],
                    content=row["content"],
                    metadata=dict(row["metadata"]),
                )
            )
            vectors.append(
                Vectors(
                    dense=[float(value) for value in row["dense"]],
                    sparse=_sparse_from_payload(row["sparse"]),
                    bm25=_sparse_from_payload(row["bm25"]),
                )
            )
        return str(payload["model_name"]), chunks, vectors


def _sparse_payload(vector: SparseVec) -> dict[str, list[float] | list[int]]:
    return {"indices": vector.indices, "values": vector.values}


def _sparse_from_payload(payload: dict[str, list[int] | list[float]]) -> SparseVec:
    return SparseVec(
        indices=[int(index) for index in payload["indices"]],
        values=[float(value) for value in payload["values"]],
    )
