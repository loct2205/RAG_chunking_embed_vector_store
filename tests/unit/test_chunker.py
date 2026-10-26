"""Test cắt chunk: bảng giữ nguyên hàng, mục sau vẫn còn tiêu đề."""

from __future__ import annotations

from rag.schema import Document
from rag.text.chunker import Chunker, split_table


def test_split_table_repeats_header_and_keeps_rows_whole() -> None:
    rows = "\n".join(f"| {weight} kg | {weight * 15} mg |" for weight in range(5, 45))
    table = "Bảng 2. Liều theo cân nặng\n| Cân nặng | Liều |\n|---|---|\n" + rows
    parts = split_table(
        table, max_tokens=60, count_tokens=lambda text: len(text.split())
    )
    assert len(parts) > 1
    for part in parts:
        lines = part.split("\n")
        assert (
            lines[0] == "Bảng 2. Liều theo cân nặng"
            and lines[1] == "| Cân nặng | Liều |"
        )
        assert all(line.startswith("|") and line.endswith("|") for line in lines[1:])


def test_later_chunks_keep_header_context() -> None:
    words = " ".join(f"tu{index}" for index in range(120))
    document = Document("sot", "sot.md", f"# Mục dài\n\n{words}\n")
    chunks = Chunker(64, 0, str.split).split_document(document)
    assert len(chunks) > 1
    assert chunks[0].chunk_id == "sot_chunk_0000"
    assert all(chunk.metadata["header_context"] == "/Mục dài/" for chunk in chunks)
    assert all(chunk.metadata["content_type"] == "text" for chunk in chunks)


def test_metadata_is_not_counted_against_chunk_size() -> None:
    words = " ".join(f"tu{index}" for index in range(40))
    document = Document("sot", "sot.md", f"# Mục dài\n\n{words}\n")
    chunks = Chunker(64, 0, str.split).split_document(document)
    assert len(chunks) == 1


def test_long_table_is_split_without_breaking_rows() -> None:
    rows = "\n".join(f"| {weight} kg | {weight * 60} mg |" for weight in range(5, 45))
    table = "Bảng 1. Liều theo cân nặng\n| Cân nặng | Liều |\n|---|---|\n" + rows
    document = Document("lieu", "lieu.md", f"# Liều\n\n{table}\n")
    chunks = Chunker(64, 0, str.split).split_document(document)
    table_chunks = [
        chunk for chunk in chunks if chunk.metadata["content_type"] == "table"
    ]
    assert len(table_chunks) > 1
    for chunk in table_chunks:
        lines = chunk.content.split("\n")
        assert lines[0] == "Bảng 1. Liều theo cân nặng"
        assert lines[1] == "| Cân nặng | Liều |"
        assert all(line.startswith("|") and line.endswith("|") for line in lines[1:])
    matched = [
        chunk
        for chunk in table_chunks
        if "20 kg" in chunk.content and "1200 mg" in chunk.content
    ]
    assert len(matched) == 1


def test_document_without_heading_still_chunks() -> None:
    document = Document(
        "note", "note.md", "Không có tiêu đề.\n\nBù nước khi nôn nhiều lần.\n"
    )
    chunks = Chunker(64, 0, str.split).split_document(document)
    assert len(chunks) == 1
    assert chunks[0].metadata["content_type"] == "text"
    assert "nôn nhiều lần" in chunks[0].content
