"""Check that partial multi-document retrieval is not scored as complete evidence."""

from dataclasses import asdict
from unittest.mock import Mock

import pytest

from rag.schema import Chunk, Hit
from scripts.build_ground_truth import build_ground_truth, corpus_fingerprint
from scripts.score_ground_truth import evaluate_ground_truth


def fixture_data():
    chunks = [
        Chunk("a", "first fact", {"doc_id": "d1"}),
        Chunk("b", "first fact again", {"doc_id": "d1"}),
        Chunk("c", "second fact", {"doc_id": "d2"}),
    ]
    records = [asdict(c) for c in chunks]
    spec = {
        "corpus_sha256": corpus_fingerprint(records),
        "evidence_library": {
            "first": {
                "label": "first",
                "matches": [
                    {"chunk_id": "a", "contains": "first"},
                    {"chunk_id": "b", "contains": "first"},
                ],
            },
            "second": {
                "label": "second",
                "matches": [
                    {"chunk_id": "c", "contains": "second"},
                ],
            },
        },
        "questions": [
            {
                "qid": "q1",
                "question": "both facts?",
                "kind": "text",
                "reference_answer": "both",
                "evidence_keys": ["first", "second"],
            }
        ],
    }
    return chunks, records, spec


def test_partial_and_complete_evidence_with_alternative_chunks():
    chunks, records, spec = fixture_data()
    gt = build_ground_truth(records, spec)
    search = Mock()
    search.search.return_value = [
        Hit(c.chunk_id, 1.0, c.content, c.metadata) for c in [chunks[0], chunks[2]]
    ]
    result = evaluate_ground_truth(search, chunks, gt, ks=(1, 2))["by_k"]
    assert result["1"]["success_at_k"] == 1.0
    assert result["1"]["recall_at_k"] == pytest.approx(1 / 3)
    assert result["1"]["evidence_coverage_at_k"] == 0.5
    assert result["1"]["all_evidence_success_at_k"] == 0.0
    assert result["2"]["recall_at_k"] == pytest.approx(2 / 3)
    assert result["2"]["ndcg_at_k"] == 1.0
    assert result["2"]["all_evidence_success_at_k"] == 1.0


def test_changed_chunk_snapshot_is_rejected():
    _, records, spec = fixture_data()
    records[0]["content"] = "changed"
    with pytest.raises(ValueError, match="snapshot"):
        build_ground_truth(records, spec)


def test_missing_evidence_anchor_is_rejected():
    _, records, spec = fixture_data()
    spec["evidence_library"]["first"]["matches"][0]["contains"] = "absent"
    with pytest.raises(ValueError, match="anchor"):
        build_ground_truth(records, spec)


def test_sample_hits_are_rejected():
    chunks, records, spec = fixture_data()
    search = Mock()
    search.search.return_value = [Hit("sample", 1.0, "sample", {})]
    with pytest.raises(ValueError, match="another corpus"):
        evaluate_ground_truth(search, chunks, build_ground_truth(records, spec))


def test_success_mrr_and_coverage_respect_cutoff():
    chunks, records, spec = fixture_data()
    unrelated = Chunk("other", "unrelated fact", {"doc_id": "d3"})
    chunks.append(unrelated)
    records.append(asdict(unrelated))
    spec["corpus_sha256"] = corpus_fingerprint(records)
    search = Mock()
    search.search.return_value = [
        Hit(c.chunk_id, 1.0, c.content, c.metadata)
        for c in [unrelated, chunks[0], chunks[2]]
    ]
    result = evaluate_ground_truth(
        search, chunks, build_ground_truth(records, spec), ks=(1, 2, 3)
    )["by_k"]
    assert result["1"]["success_at_k"] == result["1"]["mrr_at_k"] == 0.0
    assert result["2"]["success_at_k"] == 1.0
    assert result["2"]["mrr_at_k"] == result["3"]["mrr_at_k"] == 0.5
    assert result["2"]["all_evidence_success_at_k"] == 0.0
    assert result["3"]["all_evidence_success_at_k"] == 1.0
