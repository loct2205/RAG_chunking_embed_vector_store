"""Test chỉ số chấm và đọc bộ câu hỏi."""

from __future__ import annotations

from math import log2
from pathlib import Path
from unittest.mock import Mock

import pytest

from rag.evaluate import (
    Evaluator,
    is_relevant,
    latency_percentiles,
    load_questions,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
    success_at_k,
)
from rag.schema import Chunk, EvalQuestion, Hit, SearchTiming


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


def test_latency_excludes_warmup_and_reports_all_stages(monkeypatch) -> None:
    search = Mock()
    search.search_timed.side_effect = [
        SearchTiming([], 0.01, 0.02),
        SearchTiming([], 0.03, 0.04),
    ]
    clock = iter([1.0, 1.04, 2.0, 2.08])
    monkeypatch.setattr("rag.evaluate.time.perf_counter", lambda: next(clock))
    result = Evaluator(search, [EvalQuestion("q", "?", ["a"], "text")]).measure_latency(
        ["dense"], top_k=10, warmup=3, runs=2
    )
    assert search.search.call_count == 3
    assert search.search_timed.call_count == 2
    assert result["encode_ms"]["p50"] == pytest.approx(20)
    assert result["query_ms"]["p50"] == pytest.approx(30)
    assert result["total_ms"]["p50"] == pytest.approx(60)
    assert result["total_ms"]["p95"] == pytest.approx(78)
    assert result["total_ms"]["p99"] == pytest.approx(79.6)
    assert result["warmup_runs"] == 3
    assert result["runs"] == 2


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
    assert "table_retrieval_recall_at_k" not in summary
    assert "table_ndcg_at_k" not in summary


def test_recall_and_ndcg_use_all_relevant_chunks() -> None:
    relevant = {"a", "b", "c"}
    assert recall_at_k(["wrong", "a", "b"], relevant, 2) == 1 / 3
    assert ndcg_at_k(["wrong", "a", "b"], relevant, 2) == pytest.approx(
        (1 / log2(3)) / (1 + 1 / log2(3))
    )
    assert recall_at_k(["a", "b"], relevant, 2) == 2 / 3
    assert ndcg_at_k(["a", "b"], relevant, 2) == 1.0


@pytest.mark.parametrize("metric", [recall_at_k, ndcg_at_k])
def test_new_metrics_empty_and_invalid_k(metric) -> None:
    assert metric([], {"a"}, 10) == 0.0
    assert metric(["a"], set(), 10) == 0.0
    assert metric(["wrong"], {"a"}, 10) == 0.0
    for k in (0, -1):
        with pytest.raises(ValueError):
            metric(["a"], {"a"}, k)


def test_duplicate_hits_do_not_inflate_new_metrics() -> None:
    assert recall_at_k(["a", "a", "b"], {"a", "b"}, 2) == 0.5
    assert ndcg_at_k(["a", "a", "b"], {"a", "b"}, 3) == pytest.approx(
        (1 + 1 / log2(4)) / (1 + 1 / log2(3))
    )


def test_evaluator_preserves_legacy_metrics_and_macro_averages_new_metrics() -> None:
    chunks = [
        Chunk(str(i), content, {})
        for i, content in enumerate(["apple", "apple", "apple", "pear", "table row"])
    ]
    questions = [
        EvalQuestion("q1", "apple?", ["apple"], "text"),
        EvalQuestion("q2", "pear?", ["pear"], "text"),
        EvalQuestion("q3", "row?", ["table row"], "table"),
        EvalQuestion("q4", "absent?", ["absent"], "table"),
    ]
    rankings = [
        [Hit(c.chunk_id, 1.0, c.content, {})] for c in [chunks[0], chunks[3], chunks[4]]
    ]
    rankings.append([])
    search = Mock()
    search.search.side_effect = rankings
    legacy = Evaluator(search, questions).score(["dense"], 1)
    search.search.side_effect = rankings
    report = Evaluator(search, questions, chunks).score(["dense"], 1)
    assert {key: report[key] for key in legacy} == legacy
    assert report["text_recall_at_k"] == 1.0
    assert report["text_retrieval_recall_at_k"] == pytest.approx((1 / 3 + 1) / 2)
    assert report["text_ndcg_at_k"] == 1.0
    assert report["table_retrieval_recall_at_k"] == 0.5
    assert report["table_ndcg_at_k"] == 0.5
    assert report["text_no_relevant_count"] == 0
    assert report["table_no_relevant_count"] == 1


@pytest.mark.parametrize(
    "hit", [Hit("other", 1, "apple", {}), Hit("a", 1, "stale", {})]
)
def test_evaluator_rejects_hits_from_a_different_corpus(hit) -> None:
    search = Mock()
    search.search.return_value = [hit]
    evaluator = Evaluator(
        search,
        [EvalQuestion("q", "apple?", ["apple"], "text")],
        [Chunk("a", "apple", {})],
    )
    with pytest.raises(ValueError, match="corpus"):
        evaluator.score(["dense"], 1)


def test_evaluator_rejects_duplicate_corpus_ids() -> None:
    chunk = Chunk("a", "apple", {})
    with pytest.raises(ValueError, match="chunk_id"):
        Evaluator(Mock(), [], [chunk, chunk])


def test_fused_rankings_use_the_same_corpus_denominator() -> None:
    chunks = [Chunk(key, "apple", {}) for key in ("a", "b", "c")]
    search = Mock()
    search.search.side_effect = [
        [Hit("a", 1, "apple", {}), Hit("b", 0.5, "apple", {})],
        [Hit("b", 1, "apple", {}), Hit("a", 0.5, "apple", {})],
    ]
    evaluator = Evaluator(
        search, [EvalQuestion("q", "apple?", ["apple"], "text")], chunks
    )
    report = evaluator.fuse_branches(["dense", "sparse"], 1, 10, 60)
    assert report["text_recall_at_k"] == 1.0
    assert report["text_retrieval_recall_at_k"] == 1 / 3
    assert report["text_ndcg_at_k"] == 1.0
    assert report["table_retrieval_recall_at_k"] == 0.0
    assert report["table_ndcg_at_k"] == 0.0


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
