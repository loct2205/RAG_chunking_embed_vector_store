"""Quét nhanh một thư mục Markdown trước khi cắt chunk.

Mỗi file in ra độ dài, số tiêu đề, số bảng và cảnh báo (thiếu tiêu đề, bảng HTML,
bảng thiếu dòng phân cách, bảng dài hơn một chunk, ảnh).
Chạy: python scripts/scan_corpus.py <thư_mục> [--chunk-size 512]. Chỉ dùng thư viện chuẩn;
số token là ước lượng, tokenizer BGE-M3 ở bước cắt chunk mới là số chính xác.
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

# Cùng mức chunking.chunk_size trong config.yaml. Script không đọc YAML
# vì bước quét chỉ được phụ thuộc thư viện chuẩn.
DEFAULT_CHUNK_SIZE = 512

_HEADING = re.compile(r"^(#{1,6})\s+\S")
_TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEPARATOR = re.compile(r"^\s*\|(\s*:?-{3,}:?\s*\|)+\s*$")
_HTML_TABLE = re.compile(r"<table\b", re.IGNORECASE)
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_WORD = re.compile(r"\w+")


@dataclass
class FileStats:
    """Thống kê của một file Markdown."""

    name: str
    words: int
    headings: int
    max_depth: int
    tables: int
    max_table_rows: int
    max_table_tokens: int
    broken_tables: int
    html_tables: int
    images: int
    warnings: list[str] = field(default_factory=list)


def use_utf8_stdio(stdout: TextIO, stderr: TextIO) -> None:
    """Đưa stdout và stderr về UTF-8 khi bảng mã hiện tại không in được tiếng Việt.

    Console Windows mặc định cp1252 không mã hóa được chữ như 'tiết' hay 'Cảnh'.
    print khi đó ném UnicodeEncodeError ngay dòng tiêu đề của báo cáo.
    """
    for stream in (stdout, stderr):
        encoding = stream.encoding or "utf-8"
        try:
            "tiết".encode(encoding)
        except (LookupError, UnicodeEncodeError):
            reconfigure = getattr(stream, "reconfigure", None)
            if reconfigure is not None:
                reconfigure(encoding="utf-8", errors="replace")


def count_words(text: str) -> int:
    """Đếm âm tiết hoặc từ, bỏ qua ký hiệu như | # *."""
    return len(_WORD.findall(text))


def estimate_table_tokens(table_lines: list[str]) -> int:
    """Ước lượng số token của bảng: chữ cộng dấu |, vì tokenizer tính cả dấu này."""
    text = "\n".join(table_lines)
    return count_words(text) + text.count("|")


def find_tables(lines: list[str]) -> tuple[list[list[str]], int]:
    """Trả về các bảng Markdown hợp lệ và số khối '| … |' thiếu dòng phân cách."""
    tables: list[list[str]] = []
    broken = 0
    index = 0
    while index < len(lines):
        if not _TABLE_LINE.match(lines[index]):
            index += 1
            continue
        end = index
        while end < len(lines) and _TABLE_LINE.match(lines[end]):
            end += 1
        block = lines[index:end]
        if len(block) >= 2 and _TABLE_SEPARATOR.match(block[1]):
            tables.append(block)
        else:
            broken += 1
        index = end
    return tables, broken


def build_warnings(stats: FileStats, chunk_size: int) -> list[str]:
    """Liệt kê những điểm cần xem lại trước khi cắt chunk."""
    warnings: list[str] = []
    if stats.headings == 0:
        warnings.append("không có tiêu đề mục")
    if stats.html_tables:
        warnings.append(f"{stats.html_tables} bảng HTML (chưa được xử lý)")
    if stats.broken_tables:
        warnings.append(f"{stats.broken_tables} khối | thiếu dòng phân cách")
    if stats.max_table_tokens > chunk_size:
        warnings.append(f"có bảng dài hơn một chunk (~{stats.max_table_tokens} token)")
    if stats.images:
        warnings.append(f"{stats.images} ảnh (nội dung trong ảnh sẽ mất)")
    return warnings


def scan_file(path: Path, chunk_size: int) -> FileStats:
    """Đọc một file và tính thống kê."""
    text = path.read_text(encoding="utf-8")
    lines = text.split("\n")
    depths = [len(match.group(1)) for line in lines if (match := _HEADING.match(line))]
    tables, broken = find_tables(lines)
    stats = FileStats(
        name=path.name,
        words=count_words(text),
        headings=len(depths),
        max_depth=max(depths, default=0),
        tables=len(tables),
        max_table_rows=max((len(table) - 2 for table in tables), default=0),
        max_table_tokens=max(
            (estimate_table_tokens(table) for table in tables), default=0
        ),
        broken_tables=broken,
        html_tables=len(_HTML_TABLE.findall(text)),
        images=len(_IMAGE.findall(text)),
    )
    stats.warnings = build_warnings(stats, chunk_size)
    return stats


def print_report(all_stats: list[FileStats], chunk_size: int) -> None:
    """In bảng thống kê từng file và phần tóm tắt.

    In thẳng ra stdout vì đây là báo cáo cho người chạy lệnh.
    """
    header = (
        f"{'File':<28} {'Âm tiết':>8} {'Tiêu đề':>8} {'Bảng':>5} "
        f"{'Hàng max':>9}  Cảnh báo"
    )
    print(header)
    print("-" * len(header))
    for stats in all_stats:
        heading_info = (
            f"{stats.headings} (h{stats.max_depth})" if stats.headings else "0"
        )
        warning_text = "; ".join(stats.warnings) or "-"
        print(
            f"{stats.name[:28]:<28} {stats.words:>8} {heading_info:>8} {stats.tables:>5} "
            f"{stats.max_table_rows:>9}  {warning_text}"
        )
    lengths = [stats.words for stats in all_stats]
    rough_chunks = sum(max(1, math.ceil(count / chunk_size)) for count in lengths)
    print()
    print(
        f"Số file: {len(all_stats)} | độ dài ngắn nhất/trung vị/dài nhất: "
        f"{min(lengths)}/{int(statistics.median(lengths))}/{max(lengths)} âm tiết"
    )
    print(
        f"File có bảng: {sum(1 for stats in all_stats if stats.tables)} | tổng số bảng: "
        f"{sum(stats.tables for stats in all_stats)} | ước lượng số chunk: ~{rough_chunks}"
    )
    print(f"File cần xem lại: {sum(1 for stats in all_stats if stats.warnings)}")


def main(argv: list[str] | None = None) -> int:
    """Đọc tham số dòng lệnh, quét các file .md và in báo cáo."""
    use_utf8_stdio(sys.stdout, sys.stderr)
    parser = argparse.ArgumentParser(
        description="Quét thư mục Markdown trước khi cắt chunk."
    )
    parser.add_argument("folder", type=Path, help="thư mục chứa các file .md")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="kích thước chunk dự kiến, dùng để cảnh báo bảng quá dài",
    )
    args = parser.parse_args(argv)

    if args.chunk_size < 1:
        print("`--chunk-size` phải là số nguyên dương, ví dụ 512.", file=sys.stderr)
        return 1
    if not args.folder.is_dir():
        print(
            f"Không thấy thư mục '{args.folder}'. "
            "Truyền đường dẫn tới thư mục chứa file .md, ví dụ data/raw.",
            file=sys.stderr,
        )
        return 1

    files = sorted(path for path in args.folder.glob("*.md") if path.is_file())
    if not files:
        print(f"Không tìm thấy file .md nào trong {args.folder}", file=sys.stderr)
        return 1
    print_report([scan_file(path, args.chunk_size) for path in files], args.chunk_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
