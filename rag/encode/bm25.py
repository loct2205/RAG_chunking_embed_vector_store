"""BM25 cho tài liệu và câu hỏi.

Trọng số tài liệu là TF bão hòa. IDF do Qdrant tính qua Modifier.IDF,
nên vector lưu trong collection không nhân IDF.
"""

from __future__ import annotations

import json
from pathlib import Path

from rag.schema import SparseVec


def tokenize(text: str, mode: str) -> list[str]:
    """Tách token theo mode syllable (khoảng trắng) hoặc word (pyvi)."""
    lowered = text.lower()
    if mode == "syllable":
        return [token for token in lowered.split() if token]
    if mode == "word":
        # pyvi chỉ cần khi mode word, để test syllable không phụ thuộc thư viện này.
        from pyvi import ViTokenizer

        tokenized = ViTokenizer.tokenize(lowered)
        return [token for token in tokenized.split() if token]
    raise ValueError(
        f"bm25.mode '{mode}' không hợp lệ. Chọn syllable hoặc word trong config.yaml."
    )


def saturated_tf(
    term_count: int, doc_length: int, avgdl: float, k1: float, b: float
) -> float:
    """TF bão hòa theo k1, b. avgdl là độ dài trung bình thật của corpus."""
    if avgdl <= 0:
        raise ValueError("avgdl phải dương. Corpus BM25 đang rỗng hoặc không có token.")
    numerator = term_count * (k1 + 1)
    denominator = term_count + k1 * (1 - b + b * doc_length / avgdl)
    return numerator / denominator


def document_weights(
    tokens: list[str], avgdl: float, k1: float, b: float
) -> dict[str, float]:
    """Trọng số từng token trong một tài liệu."""
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    doc_length = len(tokens)
    return {
        token: saturated_tf(count, doc_length, avgdl, k1, b)
        for token, count in counts.items()
    }


class Bm25Encoder:
    """Học từ điển lúc nạp dữ liệu và encode lúc tìm kiếm từ đúng file đó."""

    def __init__(self, mode: str, k1: float, b: float) -> None:
        self.mode = mode
        self.k1 = k1
        self.b = b
        self._token_to_index: dict[str, int] = {}
        self._avgdl = 0.0
        self._fitted = False

    def fit(self, texts: list[str]) -> None:
        """Tính avgdl và từ điển, sắp xếp token để lần chạy sau ra cùng chỉ số."""
        if not texts:
            raise ValueError(
                "BM25 cần ít nhất một đoạn văn. Kiểm tra thư mục tài liệu."
            )
        documents = [tokenize(text, self.mode) for text in texts]
        avgdl = sum(len(tokens) for tokens in documents) / len(documents)
        if avgdl <= 0:
            raise ValueError(
                "BM25 không tính được avgdl vì mọi đoạn văn rỗng sau khi tách từ."
            )
        vocabulary = sorted({token for tokens in documents for token in tokens})
        self._token_to_index = {token: index for index, token in enumerate(vocabulary)}
        self._avgdl = avgdl
        self._fitted = True

    def encode_documents(self, texts: list[str]) -> list[SparseVec]:
        """Vector tài liệu: TF bão hòa, chưa nhân IDF."""
        self._require_fitted()
        vectors: list[SparseVec] = []
        for text in texts:
            weights = document_weights(
                tokenize(text, self.mode), self._avgdl, self.k1, self.b
            )
            vectors.append(self._to_vector(weights))
        return vectors

    def encode_query(self, text: str) -> SparseVec:
        """Mỗi token của câu hỏi có trọng số 1. Token lạ bị bỏ."""
        self._require_fitted()
        weights = {
            token: 1.0
            for token in tokenize(text, self.mode)
            if token in self._token_to_index
        }
        return self._to_vector(weights)

    def save(self, path: Path) -> None:
        """Ghi từ điển để lượt tìm kiếm sau nạp lại đúng file này."""
        self._require_fitted()
        path.parent.mkdir(parents=True, exist_ok=True)
        tokens = [
            token
            for token, _ in sorted(
                self._token_to_index.items(), key=lambda item: item[1]
            )
        ]
        payload = {
            "mode": self.mode,
            "k1": self.k1,
            "b": self.b,
            "avgdl": self._avgdl,
            "tokens": tokens,
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def load(self, path: Path) -> None:
        """Nạp từ điển đã ghi lúc ingest."""
        if not path.is_file():
            raise FileNotFoundError(
                f"Không thấy từ điển BM25 tại '{path}'. Chạy ingest trước khi search."
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        tokens = payload["tokens"]
        self.mode = str(payload["mode"])
        self.k1 = float(payload["k1"])
        self.b = float(payload["b"])
        self._avgdl = float(payload["avgdl"])
        self._token_to_index = {token: index for index, token in enumerate(tokens)}
        self._fitted = True

    def _to_vector(self, weights: dict[str, float]) -> SparseVec:
        pairs = sorted(
            (self._token_to_index[token], weight)
            for token, weight in weights.items()
            if weight > 0 and token in self._token_to_index
        )
        return SparseVec(
            indices=[index for index, _ in pairs],
            values=[weight for _, weight in pairs],
        )

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise ValueError("BM25 chưa được fit hoặc load. Chạy ingest trước khi tìm.")
