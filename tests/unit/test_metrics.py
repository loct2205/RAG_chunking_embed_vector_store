"""Test chỉ số chấm và đọc bộ câu hỏi."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.evaluate import (
    Evaluator,
    is_relevant,
    latency_percentiles,
    load_questions,
    mean_reciprocal_rank,
    success_at_k,
)
from rag.schema import EvalQuestion, Hit, SearchTiming


def test_is_relevant_needs_every_key_after_normalize() -> None:
    content = "Trẻ  20   kg dùng 1200\tmg mỗi ngày."
    assert is_relevant(content, ["20 kg", "1200 mg"])
    assert not is_relevant(content, ["20 kg", "999 mg"])


def test_success_and_mrr() -> None:
    flags = [[False, True], [True], [False]]
    assert success_at_k(flags, k=2) == 2 / 3
    assert mean_reciprocal_rank(flags) == pytest.approx((0.5 + 1.0 + 0.0) / 3)


def test_latency_percentiles_use_numpy() -> None:
    stats = latency_percentiles([10, 20, 30, 40, 50])
    assert stats["p50"] == 30
    assert stats["p95"] > stats["p50"]


def test_load_questions_rejects_bad_kind(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    path.write_text(
        '{"qid": "q1", "question": "x", "answer_keys": ["a"], "kind": "nope"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="kind"):
        load_questions(path)


def test_load_questions_maps_testset_type(tmp_path: Path) -> None:
    path = tmp_path / "questions.jsonl"
    path.write_text(
        '{"qid": "q2", "type": "paraphrase", "question": "ý?", "answer_keys": ["a"]}\n'
        '{"qid": "q1", "type": "table", "question": "bảng?", "answer_keys": ["b"]}\n'
        '{"qid": "q3", "type": "entity_number", "question": "số?", "answer_keys": ["c"]}\n',
        encoding="utf-8",
    )
    questions = load_questions(path)
    assert [(item.qid, item.kind) for item in questions] == [
        ("q1", "table"),
        ("q2", "text"),
        ("q3", "text"),
    ]


def test_corpus_testset_file_has_text_and_table() -> None:
    path = Path(__file__).resolve().parents[2] / "data" / "eval" / "testset.jsonl"
    questions = load_questions(path)
    assert len(questions) >= 40
    assert {item.kind for item in questions} == {"text", "table"}


def test_evaluator_scores_a_matching_hit() -> None:
    question = EvalQuestion("q1", "liều", ["20 kg"], "table")
    evaluator = Evaluator(_StubSearch(), [question])
    summary = evaluator.score(["dense"], top_k=1)
    assert summary["table_recall_at_k"] == 1.0
    assert summary["text_count"] == 0


class _StubSearch:
    def search(
        self, question: str, branches: list[str], top_k: int | None = None
    ) -> list[Hit]:
        return [Hit("doc_chunk_0000", 0.9, "Trẻ 20 kg", {"content_type": "table"})]

    def search_timed(
        self,
        question: str,
        branches: list[str],
        top_k: int | None = None,
        log_result: bool = True,
    ) -> SearchTiming:
        return SearchTiming(self.search(question, branches, top_k), 0.01, 0.02)

    def check_model(self) -> None:
        return None
