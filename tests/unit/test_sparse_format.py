"""Test định dạng vector thưa và gộp RRF dùng cho pgvector, Milvus."""

from __future__ import annotations

from rag.schema import Hit, SparseVec
from rag.store.milvus_rows import hit_from_milvus
from rag.store.ranking import fuse_hits
from rag.store.sparse_format import collapse_sparse, sparse_pairs, sparsevec_literal


def test_sparsevec_literal_is_one_based() -> None:
    vector = SparseVec([0, 5, 2], [0.5, 0.25, 1.5])
    assert sparsevec_literal(vector, 4) == "{1:0.5,2:0.25,3:1.5}/4"


def test_collapse_sparse_merges_indices_in_the_same_bucket() -> None:
    collapsed = collapse_sparse(SparseVec([0, 4], [0.5, 0.5]), 4)
    assert collapsed == SparseVec([0], [1.0])


def test_sparse_pairs_fills_an_empty_vector() -> None:
    assert sparse_pairs(SparseVec([], []), 8) == {0: 1e-12}


def test_fuse_hits_uses_rrf_score() -> None:
    first = Hit("b", 0.9, "một", {"content_type": "text"})
    second = Hit("a", 0.2, "hai", {})
    fused = fuse_hits([[first, second], [first]], rrf_k=60, top_k=1)
    assert fused[0].chunk_id == "b"
    assert fused[0].content == "một"
    assert fused[0].metadata == {"content_type": "text"}
    assert fused[0].score == 2 / 61


def test_hit_from_milvus_reads_entity() -> None:
    hit = hit_from_milvus(
        {
            "distance": 0.5,
            "entity": {
                "chunk_id": "DOC_001_chunk_0000",
                "content": "49,9",
                "metadata": {"content_type": "table"},
            },
        }
    )
    assert hit.chunk_id == "DOC_001_chunk_0000"
    assert hit.score == 0.5
    assert hit.content == "49,9"
    assert hit.metadata == {"content_type": "table"}
