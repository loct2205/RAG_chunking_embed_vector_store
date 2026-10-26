"""Test chuẩn hóa, header_context, build_embed_text và to_sparse."""

from __future__ import annotations

from rag.schema import SparseVec, to_sparse
from rag.text.normalizer import build_embed_text, header_context, normalize


def test_normalize_keeps_markdown_headings() -> None:
    raw = "# Chương 3\r\n\r\n## Sốt   xuất\thuyết  \n"
    assert normalize(raw) == "# Chương 3\n\n## Sốt xuất huyết\n"


def test_header_context_adds_own_heading() -> None:
    assert (
        header_context("/Chương 3/", "## Sốt xuất huyết") == "/Chương 3/Sốt xuất huyết/"
    )


def test_header_context_without_heading_keeps_path() -> None:
    assert header_context("/", "Không có tiêu đề") == "/"


def test_build_embed_text_prefixes_section_and_skips_empty_path() -> None:
    assert build_embed_text("nội dung", "/Mục dài/") == "/Mục dài/\nnội dung"
    assert build_embed_text("nội dung", "/") == "nội dung"


def test_to_sparse_casts_sorts_and_drops_zero() -> None:
    assert to_sparse({"42": 0.3, "7": 0.5, "9": 0.0}) == SparseVec([7, 42], [0.5, 0.3])
