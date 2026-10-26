"""Bọc BGE-M3 qua FlagEmbedding.

CLI chỉ gọi load_bge_m3 khi không có cờ --fake-embedder. Test truyền model giả vào lớp.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag.encode.base import Embedder
from rag.schema import SparseVec, to_sparse

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BgeSettings:
    """Tham số nạp model. Tách khỏi config.py để encode không đọc YAML."""

    model_name: str
    model_path: Path
    max_length: int
    batch_size: int
    use_fp16: bool


class BGEM3Embedder(Embedder):
    """Gọi BGEM3FlagModel.encode với dense, sparse, và max_length lấy từ config."""

    def __init__(self, model: Any, settings: BgeSettings) -> None:
        tokenizer = getattr(model, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError(
                "BGEM3FlagModel không có tokenizer. Kiểm tra FlagEmbedding dòng 1.4."
            )
        self._model = model
        self._settings = settings
        self._tokenizer = tokenizer
        self._dimension: int | None = None

    @property
    def model_name(self) -> str:
        return self._settings.model_name

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise RuntimeError("Chưa encode batch nào nên chưa biết số chiều dense.")
        return self._dimension

    def tokenize(self, text: str) -> list[str]:
        """Token id của BGE-M3, không thêm token đặc biệt, để đếm đúng độ dài chunk."""
        token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        return [str(token_id) for token_id in token_ids]

    def encode(self, texts: list[str]) -> list[tuple[list[float], SparseVec]]:
        if not texts:
            return []
        output = self._model.encode(
            texts,
            batch_size=self._settings.batch_size,
            max_length=self._settings.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense_rows = output["dense_vecs"]
        lexical_rows = output["lexical_weights"]
        encoded: list[tuple[list[float], SparseVec]] = []
        for dense, lexical in zip(dense_rows, lexical_rows, strict=True):
            vector = _as_floats(dense)
            self._dimension = len(vector)
            encoded.append(
                (
                    vector,
                    to_sparse(
                        {str(key): float(value) for key, value in lexical.items()}
                    ),
                )
            )
        return encoded


def load_bge_m3(settings: BgeSettings) -> BGEM3Embedder:
    """Nạp model từ đĩa. Không tải từ Hugging Face nếu thư mục chưa có."""
    if not settings.model_path.exists():
        raise FileNotFoundError(
            f"Không thấy model tại '{settings.model_path}'. "
            "Tải BAAI/bge-m3 vào thư mục đó, hoặc chạy lệnh với --fake-embedder."
        )
    use_fp16 = _fp16_on_gpu(settings.use_fp16)
    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel(str(settings.model_path), use_fp16=use_fp16)
    log.info(
        "Đã nạp %s từ %s (fp16=%s)", settings.model_name, settings.model_path, use_fp16
    )
    return BGEM3Embedder(model, settings)


def _fp16_on_gpu(requested: bool) -> bool:
    """Chỉ bật fp16 khi máy có GPU. Config bật sẵn nhưng CPU thì tắt."""
    if not requested:
        return False
    try:
        import torch
    except ImportError:
        log.info("Tắt use_fp16 vì không import được torch.")
        return False
    if not torch.cuda.is_available():
        log.info("Tắt use_fp16 vì không có GPU.")
        return False
    return True


def _as_floats(dense: Any) -> list[float]:
    if hasattr(dense, "tolist"):
        return [float(value) for value in dense.tolist()]
    return [float(value) for value in dense]
