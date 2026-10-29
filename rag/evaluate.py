"""Chấm retrieval và đo độ trễ. Không sinh câu trả lời.

Đáp án là cụm chữ trong chunk. Câu text và câu table được báo cáo riêng.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from math import log2
from pathlib import Path

import numpy as np

from rag.config import AppConfig
from rag.fusion import rrf
from rag.schema import Chunk, EvalQuestion, Hit
from rag.search import SearchService
from rag.text.normalizer import normalize

_TRACKED_PACKAGES = (
    "qdrant-client",
    "llama-index-core",
    "FlagEmbedding",
    "numpy",
    "torch",
    "transformers",
    "sentence-transformers",
)
_TYPE_TO_KIND = {"table": "table", "paraphrase": "text", "entity_number": "text"}


def is_relevant(content: str, answer_keys: list[str]) -> bool:
    """Chunk chứa mọi cụm trong answer_keys sau khi chuẩn hóa cả hai phía."""
    if not answer_keys:
        raise ValueError(
            "answer_keys không được rỗng. Sửa dòng tương ứng trong questions.jsonl."
        )
    normalized = normalize(content)
    return all(normalize(key) in normalized for key in answer_keys)


def success_at_k(flags_per_question: list[list[bool]], k: int) -> float:
    """Tỷ lệ câu có ít nhất một chunk đúng trong top k."""
    if k < 1:
        raise ValueError("k phải dương khi tính success@k.")
    if not flags_per_question:
        return 0.0
    hits = sum(1 for flags in flags_per_question if any(flags[:k]))
    return hits / len(flags_per_question)


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Recall của một query; relevant_ids phải lấy từ toàn bộ corpus."""
    if k < 1:
        raise ValueError("k phải dương khi tính Recall@K.")
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)


def ndcg_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """nDCG của một query với nhãn nhị phân; chunk lặp không được cộng điểm lại."""
    if k < 1:
        raise ValueError("k phải dương khi tính nDCG@K.")
    if not relevant_ids:
        return 0.0
    seen: set[str] = set()
    dcg = 0.0
    for rank, chunk_id in enumerate(ranked_ids[:k], start=1):
        if chunk_id in relevant_ids and chunk_id not in seen:
            dcg += 1.0 / log2(rank + 1)
        seen.add(chunk_id)
    ideal = sum(
        1.0 / log2(rank + 1) for rank in range(1, min(k, len(relevant_ids)) + 1)
    )
    return dcg / ideal


def mean_reciprocal_rank(flags_per_question: list[list[bool]]) -> float:
    """Trung bình 1/hạng của chunk đúng đầu tiên. Không có thì hạng đó là 0."""
    if not flags_per_question:
        return 0.0
    return sum(_reciprocal_rank(flags) for flags in flags_per_question) / len(
        flags_per_question
    )


def latency_percentiles(samples_ms: list[float]) -> dict[str, float]:
    """P50, P95, P99 bằng numpy.percentile."""
    if not samples_ms:
        raise ValueError("Không có mẫu độ trễ để tính phân vị.")
    values = np.asarray(samples_ms, dtype=float)
    return {
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def load_questions(path: Path) -> list[EvalQuestion]:
    """Đọc jsonl. kind text/table, hoặc type paraphrase, entity_number, table."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Không thấy bộ câu hỏi '{path}'. Sửa eval.questions_path trong config.yaml."
        )
    questions: list[EvalQuestion] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        questions.append(_parse_question(line, line_number, path))
    if not questions:
        raise ValueError(f"'{path}' không có câu hỏi nào.")
    questions.sort(key=lambda question: question.qid)
    return questions


class Evaluator:
    """Chấm một SearchService trên một bộ câu hỏi."""

    def __init__(
        self,
        search: SearchService,
        questions: list[EvalQuestion],
        chunks: list[Chunk] | None = None,
    ) -> None:
        self._search = search
        self._questions = questions
        # Không truyền corpus: vẫn trả các metric cũ, không suy mẫu số từ top K.
        self._corpus = None if chunks is None else {c.chunk_id: c for c in chunks}
        if self._corpus is not None and len(self._corpus) != len(chunks):
            raise ValueError("Corpus đánh giá có chunk_id trùng nhau.")
        self._relevant_ids = (
            None
            if chunks is None
            else [
                {c.chunk_id for c in chunks if is_relevant(c.content, q.answer_keys)}
                for q in questions
            ]
        )

    def score(self, branches: list[str], top_k: int) -> dict[str, float | int]:
        if top_k < 1:
            raise ValueError("top_k phải dương.")
        rankings = [
            self._search.search(question.question, branches, top_k)
            for question in self._questions
        ]
        return self._score_rankings(rankings, top_k)

    def measure_latency(
        self, branches: list[str], top_k: int, warmup: int, runs: int
    ) -> dict[str, object]:
        if not self._questions:
            raise ValueError("Không có câu hỏi để đo độ trễ.")
        for index in range(warmup):
            question = self._questions[index % len(self._questions)]
            self._search.search(question.question, branches, top_k)
        encode_ms: list[float] = []
        query_ms: list[float] = []
        total_ms: list[float] = []
        for index in range(runs):
            question = self._questions[index % len(self._questions)]
            started = time.perf_counter()
            timed = self._search.search_timed(
                question.question, branches, top_k, log_result=False
            )
            total_ms.append((time.perf_counter() - started) * 1000)
            encode_ms.append(timed.encode_seconds * 1000)
            query_ms.append(timed.query_seconds * 1000)
        return {
            "warmup_runs": warmup,
            "runs": runs,
            "encode_ms": latency_percentiles(encode_ms),
            "query_ms": latency_percentiles(query_ms),
            "total_ms": latency_percentiles(total_ms),
        }

    def fuse_branches(
        self, branches: list[str], top_k: int, prefetch_limit: int, rrf_k: int
    ) -> dict[str, float | int]:
        """Gộp thứ hạng từng nhánh bằng rrf() ở client, cho kho không gộp sẵn."""
        if top_k < 1:
            raise ValueError("top_k phải dương.")
        fused_rankings: list[list[Hit]] = []
        for question in self._questions:
            rankings = [
                self._search.search(question.question, [branch], prefetch_limit)
                for branch in branches
            ]
            fused = _fuse_hits(rankings, rrf_k, top_k)
            fused_rankings.append(fused)
        return self._score_rankings(fused_rankings, top_k)

    def _score_rankings(
        self, rankings: list[list[Hit]], top_k: int
    ) -> dict[str, float | int]:
        grouped: dict[str, list[list[bool]]] = {"text": [], "table": []}
        recalls: dict[str, list[float]] = {"text": [], "table": []}
        ndcgs: dict[str, list[float]] = {"text": [], "table": []}
        no_relevant = {"text": 0, "table": 0}
        for index, (question, hits) in enumerate(
            zip(self._questions, rankings, strict=True)
        ):
            grouped[question.kind].append(
                [is_relevant(hit.content, question.answer_keys) for hit in hits]
            )
            if self._relevant_ids is not None:
                assert self._corpus is not None
                for hit in hits:
                    chunk = self._corpus.get(hit.chunk_id)
                    if chunk is None or chunk.content != hit.content:
                        raise ValueError(
                            f"Chunk '{hit.chunk_id}' không khớp corpus đánh giá. "
                            "Dùng cache của đúng lần ingest và collection đang tìm."
                        )
                relevant = self._relevant_ids[index]
                ranked_ids = [hit.chunk_id for hit in hits]
                recalls[question.kind].append(recall_at_k(ranked_ids, relevant, top_k))
                ndcgs[question.kind].append(ndcg_at_k(ranked_ids, relevant, top_k))
                no_relevant[question.kind] += int(not relevant)
        summary = _summarize(grouped, top_k)
        if self._relevant_ids is not None:
            for kind in grouped:
                summary[f"{kind}_retrieval_recall_at_k"] = (
                    sum(recalls[kind]) / len(recalls[kind]) if recalls[kind] else 0.0
                )
                summary[f"{kind}_ndcg_at_k"] = (
                    sum(ndcgs[kind]) / len(ndcgs[kind]) if ndcgs[kind] else 0.0
                )
                summary[f"{kind}_no_relevant_count"] = no_relevant[kind]
        return summary


def write_report(directory: Path, label: str, payload: dict[str, object]) -> Path:
    """Ghi reports/<thời điểm>_<cấu hình>.json."""
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    safe_label = label.replace(",", "_").replace("+", "-")
    path = directory / f"{stamp}_{safe_label}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def report_payload(config: AppConfig, body: dict[str, object]) -> dict[str, object]:
    """Kèm tên model, phiên bản thư viện, tham số config và thời điểm chạy."""
    return {
        "model_name": config.embedding.model_name,
        "library_versions": library_versions(),
        "config": _jsonable(asdict(config)),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "results": body,
    }


def library_versions() -> dict[str, str]:
    found: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = "not-installed"
    return found


def question_kind(record: dict[str, object]) -> str:
    """kind text/table nếu có. type table giữ nguyên, paraphrase và entity_number là text."""
    kind = record.get("kind")
    if isinstance(kind, str) and kind.strip():
        return kind.strip()
    mapped = record.get("type")
    if isinstance(mapped, str):
        return _TYPE_TO_KIND.get(mapped.strip(), "")
    return ""


def _parse_question(line: str, line_number: int, path: Path) -> EvalQuestion:
    try:
        raw = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Dòng {line_number} trong '{path}' không phải JSON.") from exc
    if not isinstance(raw, dict):
        raise TypeError(f"Dòng {line_number} trong '{path}' phải là một object JSON.")
    kind = question_kind(raw)
    if kind not in ("text", "table"):
        raise ValueError(
            f"Dòng {line_number} trong '{path}' có kind/type không hợp lệ. "
            "Dùng kind text hoặc table, hoặc type paraphrase, entity_number, table."
        )
    keys = raw.get("answer_keys")
    if (
        not isinstance(keys, list)
        or not keys
        or not all(isinstance(key, str) for key in keys)
    ):
        raise ValueError(
            f"Dòng {line_number} trong '{path}' cần answer_keys là danh sách cụm chữ không rỗng."
        )
    return EvalQuestion(
        qid=str(raw.get("qid", f"line-{line_number}")),
        question=str(raw.get("question", "")),
        answer_keys=list(keys),
        kind=kind,
    )


def _summarize(
    grouped: dict[str, list[list[bool]]], top_k: int
) -> dict[str, float | int]:
    summary: dict[str, float | int] = {"top_k": top_k}
    for kind, flags in grouped.items():
        summary[f"{kind}_count"] = len(flags)
        summary[f"{kind}_recall_at_k"] = success_at_k(flags, top_k)
        summary[f"{kind}_mrr"] = mean_reciprocal_rank(flags)
        summary[f"{kind}_success_at_k"] = summary[f"{kind}_recall_at_k"]
        summary[f"{kind}_mrr_at_k"] = mean_reciprocal_rank(
            [row[:top_k] for row in flags]
        )
    return summary


def _reciprocal_rank(flags: list[bool]) -> float:
    for index, flag in enumerate(flags, start=1):
        if flag:
            return 1.0 / index
    return 0.0


def _fuse_hits(rankings: list[list[Hit]], rrf_k: int, top_k: int) -> list[Hit]:
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
        hit = by_id[chunk_id]
        fused.append(
            Hit(
                chunk_id=hit.chunk_id,
                score=score,
                content=hit.content,
                metadata=hit.metadata,
            )
        )
    return fused


def _jsonable(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value
