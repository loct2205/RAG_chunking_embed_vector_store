"""Dense multilingual E5 embedding; hybrid composition lives in rag.pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag.encode.base import Embedder
from rag.encode.bge_m3 import _fp16_on_gpu
from rag.schema import SparseVec

log = logging.getLogger(__name__)
E5_MODEL_NAME = "intfloat/multilingual-e5-base"
E5_DIMENSION = 768


@dataclass(frozen=True)
class E5Settings:
    model_name: str
    model_path: Path
    max_length: int = 512
    batch_size: int = 16
    use_fp16: bool = True

    def __post_init__(self) -> None:
        if not 1 <= self.max_length <= 512:
            raise ValueError("E5 max_length must be between 1 and 512 tokens.")
        if self.batch_size < 1:
            raise ValueError("E5 batch_size must be positive.")


class E5Embedder(Embedder):
    """L2-normalized dense vectors, with passage/query prefixes for retrieval."""

    def __init__(self, model: Any, settings: E5Settings) -> None:
        if getattr(model, "tokenizer", None) is None:
            raise RuntimeError("SentenceTransformer must expose a tokenizer.")
        self._model = model
        self._settings = settings
        self._model.max_seq_length = settings.max_length

    @property
    def model_name(self) -> str:
        return self._settings.model_name

    @property
    def dimension(self) -> int:
        return E5_DIMENSION

    def tokenize(self, text: str) -> list[str]:
        token_ids = self._model.tokenizer.encode(text, add_special_tokens=False)
        return [str(token_id) for token_id in token_ids]

    def encode(self, texts: list[str]) -> list[tuple[list[float], SparseVec]]:
        return self._encode_with_prefix(texts, "passage: ")

    def encode_queries(self, texts: list[str]) -> list[tuple[list[float], SparseVec]]:
        return self._encode_with_prefix(texts, "query: ")

    def _encode_with_prefix(
        self, texts: list[str], prefix: str
    ) -> list[tuple[list[float], SparseVec]]:
        if not texts:
            return []
        rows = self._model.encode(
            [prefix + text for text in texts],
            batch_size=self._settings.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        if len(rows) != len(texts):
            raise RuntimeError("E5 returned a different number of vectors than inputs.")
        results = []
        for row in rows:
            dense = [float(value) for value in row]
            if len(dense) != self.dimension:
                raise ValueError(f"Expected E5 dense dimension {self.dimension}.")
            results.append((dense, SparseVec(indices=[], values=[])))
        return results


def load_e5(settings: E5Settings) -> E5Embedder:
    """Load a local model, following the existing BGE loading convention."""
    if not settings.model_path.is_dir():
        raise FileNotFoundError(
            f"E5 model not found at '{settings.model_path}'. "
            f"Download {settings.model_name} there or set embedding.e5_model_path."
        )
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(str(settings.model_path), trust_remote_code=False)
    use_fp16 = _fp16_on_gpu(settings.use_fp16)
    if use_fp16:
        model.half()
    log.info(
        "Loaded %s from %s (fp16=%s, max_length=%d)",
        settings.model_name,
        settings.model_path,
        use_fp16,
        settings.max_length,
    )
    return E5Embedder(model, settings)
