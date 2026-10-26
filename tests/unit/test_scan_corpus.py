"""Test các hàm thuần của script quét Markdown, không đọc mạng và không cần model."""

from __future__ import annotations

import io
from pathlib import Path

import pytest
from scan_corpus import (
    FileStats,
    build_warnings,
    count_words,
    estimate_table_tokens,
    find_tables,
    main,
    scan_file,
    use_utf8_stdio,
)


def _stats(**overrides: int | str) -> FileStats:
    values: dict[str, int | str | list[str]] = {
        "name": "a.md",
        "words": 10,
        "headings": 1,
        "max_depth": 1,
        "tables": 0,
        "max_table_rows": 0,
        "max_table_tokens": 0,
        "broken_tables": 0,
        "html_tables": 0,
        "images": 0,
    }
    values.update(overrides)
    return FileStats(**values)


def test_use_utf8_stdio_prints_vietnamese_on_cp1252() -> None:
    stdout = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", newline="\n")
    stderr = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", newline="\n")
    use_utf8_stdio(stdout, stderr)
    stdout.write("Âm tiết\n")
    stderr.write("Cảnh báo\n")
    stdout.flush()
    stderr.flush()
    assert stdout.buffer.getvalue().decode("utf-8") == "Âm tiết\n"
    assert stderr.buffer.getvalue().decode("utf-8") == "Cảnh báo\n"


def test_count_words_skips_markup() -> None:
    assert count_words("Sốt xuất huyết") == 3
    assert count_words("| # * 20 kg |") == 2


def test_estimate_table_tokens_adds_pipe_marks() -> None:
    lines = ["| Cân nặng | Liều |", "|---|---|", "| 20 kg | 1200 mg |"]
    text = "\n".join(lines)
    assert estimate_table_tokens(lines) == count_words(text) + text.count("|")
    assert estimate_table_tokens(lines) == 16


def test_find_tables_keeps_valid_blocks_and_counts_broken_ones() -> None:
    lines = [
        "Bảng 2. Liều",
        "| Cân nặng | Liều |",
        "|---|---|",
        "| 20 kg | 1200 mg |",
        "",
        "| :--- | ---: |",
        "|---|---|",
        "| 1 | 2 |",
        "văn thường",
        "| thiếu | phân cách |",
        "| 1 | 2 |",
    ]
    tables, broken = find_tables(lines)
    assert broken == 1
    assert len(tables) == 2
    assert tables[0] == [
        "| Cân nặng | Liều |",
        "|---|---|",
        "| 20 kg | 1200 mg |",
    ]
    assert tables[1][0] == "| :--- | ---: |"


def test_find_tables_on_text_without_pipes() -> None:
    assert find_tables(["# Mục", "không có bảng"]) == ([], 0)


def test_build_warnings_lists_each_problem() -> None:
    stats = _stats(
        headings=0,
        html_tables=2,
        broken_tables=1,
        max_table_tokens=800,
        images=3,
    )
    assert build_warnings(stats, chunk_size=512) == [
        "không có tiêu đề mục",
        "2 bảng HTML (chưa được xử lý)",
        "1 khối | thiếu dòng phân cách",
        "có bảng dài hơn một chunk (~800 token)",
        "3 ảnh (nội dung trong ảnh sẽ mất)",
    ]


def test_build_warnings_is_empty_when_article_is_plain() -> None:
    assert build_warnings(_stats(), chunk_size=512) == []


def test_scan_file_counts_heading_table_html_and_image(tmp_path: Path) -> None:
    article = tmp_path / "lieu.md"
    article.write_text(
        "# Liều dùng\n"
        "\n"
        "### Trẻ em\n"
        "\n"
        "<table><tr><td>html</td></tr></table>\n"
        "\n"
        "![biểu đồ](bieu-do.png)\n"
        "\n"
        "| Cân nặng | Liều |\n"
        "|---|---|\n"
        "| 20 kg | 1200 mg |\n",
        encoding="utf-8",
    )
    stats = scan_file(article, chunk_size=512)
    assert stats.name == "lieu.md"
    assert stats.headings == 2
    assert stats.max_depth == 3
    assert stats.tables == 1
    assert stats.max_table_rows == 1
    assert stats.html_tables == 1
    assert stats.images == 1
    assert stats.broken_tables == 0
    assert "không có tiêu đề mục" not in stats.warnings
    assert any("bảng HTML" in warning for warning in stats.warnings)
    assert any("ảnh" in warning for warning in stats.warnings)


def test_scan_file_warns_when_table_exceeds_chunk(tmp_path: Path) -> None:
    rows = [f"| {weight} kg | {weight * 15} mg |" for weight in range(5, 45)]
    table = "\n".join(["| Cân nặng | Liều |", "|---|---|", *rows])
    article = tmp_path / "bang-dai.md"
    article.write_text(f"# Liều\n\n{table}\n", encoding="utf-8")
    stats = scan_file(article, chunk_size=30)
    assert stats.tables == 1
    assert stats.max_table_rows == 40
    assert stats.max_table_tokens > 30
    assert any("dài hơn một chunk" in warning for warning in stats.warnings)


def test_main_prints_sorted_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "b.md").write_text("không có tiêu đề\n", encoding="utf-8")
    (tmp_path / "a.md").write_text("# Có tiêu đề\n\nMột câu ngắn.\n", encoding="utf-8")
    code = main([str(tmp_path), "--chunk-size", "512"])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.index("a.md") < captured.out.index("b.md")
    assert "không có tiêu đề mục" in captured.out
    assert "Số file: 2" in captured.out
    assert "File cần xem lại: 1" in captured.out
    assert captured.err == ""


def test_main_rejects_missing_directory(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["thu_muc_khong_co", "--chunk-size", "512"])
    captured = capsys.readouterr()
    assert code == 1
    assert "Không thấy thư mục" in captured.err
    assert captured.out == ""


def test_main_rejects_directory_without_markdown(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "ghi_chu.txt").write_text("không phải markdown", encoding="utf-8")
    code = main([str(tmp_path)])
    captured = capsys.readouterr()
    assert code == 1
    assert "Không tìm thấy file .md" in captured.err


def test_main_rejects_non_positive_chunk_size(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main([".", "--chunk-size", "0"])
    captured = capsys.readouterr()
    assert code == 1
    assert "--chunk-size" in captured.err
