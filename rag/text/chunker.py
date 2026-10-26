"""Cắt Markdown thành chunk: tiêu đề trước, câu sau, bảng tách riêng.

Pipeline gọi sau normalize. Tokenizer được truyền vào, không nạp model tại đây.
"""

from __future__ import annotations

import logging
import re
import statistics
from collections.abc import Callable, Sequence

from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.schema import Document as LlamaDocument
from llama_index.core.schema import MetadataMode

from rag.schema import Chunk, Document
from rag.text.normalizer import header_context

log = logging.getLogger(__name__)

_TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEPARATOR = re.compile(r"^\s*\|(\s*:?-{3,}:?\s*\|)+\s*$")
_CAPTION = re.compile(r"^Bảng\s+\d+\b", re.IGNORECASE)
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Tách câu cục bộ để SentenceSplitter không tải dữ liệu punkt của NLTK."""
    parts = [part for part in _SENTENCE_BOUNDARY.split(text) if part]
    if parts:
        return parts
    return [text]


def split_table(
    table: str, max_tokens: int, count_tokens: Callable[[str], int]
) -> list[str]:
    """Chia bảng dài theo nhóm hàng; mỗi phần lặp lại chú thích và dòng tiêu đề cột.

    Không cắt giữa một hàng, nên mỗi giá trị luôn đi cùng tên cột của nó.
    Một hàng dài hơn max_tokens vẫn được giữ nguyên thành một phần.
    """
    lines = table.split("\n")
    try:
        separator = next(
            index for index, line in enumerate(lines) if _TABLE_SEPARATOR.match(line)
        )
    except StopIteration as exc:
        raise ValueError(
            "Bảng Markdown không có dòng phân cách |---|. Sửa bảng trước khi cắt chunk."
        ) from exc
    head, rows = lines[: separator + 1], lines[separator + 1 :]
    if not rows:
        return ["\n".join(head)]
    parts: list[str] = []
    current: list[str] = []
    for row in rows:
        candidate = "\n".join(head + current + [row])
        if current and count_tokens(candidate) > max_tokens:
            parts.append("\n".join(head + current))
            current = []
        current.append(row)
    if current:
        parts.append("\n".join(head + current))
    return parts


def split_prose_and_tables(text: str) -> list[tuple[str, str]]:
    """Tách bảng hợp lệ khỏi phần chữ. Dòng 'Bảng N. ...' ngay phía trên đi cùng bảng."""
    lines = text.split("\n")
    blocks: list[tuple[str, str]] = []
    prose: list[str] = []
    index = 0
    while index < len(lines):
        if not _TABLE_LINE.match(lines[index]):
            prose.append(lines[index])
            index += 1
            continue
        end = index
        while end < len(lines) and _TABLE_LINE.match(lines[end]):
            end += 1
        block = lines[index:end]
        if len(block) >= 2 and _TABLE_SEPARATOR.match(block[1]):
            caption = _pop_caption(prose)
            _flush_prose(blocks, prose)
            prose = []
            table_lines = ([caption] if caption else []) + block
            blocks.append(("table", "\n".join(table_lines)))
        else:
            prose.extend(block)
        index = end
    _flush_prose(blocks, prose)
    return blocks


class Chunker:
    """MarkdownNodeParser rồi SentenceSplitter. Bảng không đi qua SentenceSplitter."""

    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        tokenize: Callable[[str], Sequence[object]],
    ) -> None:
        if chunk_size < 1:
            raise ValueError(
                "chunk_size phải là số nguyên dương. Sửa chunking.chunk_size."
            )
        if chunk_overlap < 0 or chunk_overlap > chunk_size:
            raise ValueError(
                "chunk_overlap phải nằm trong khoảng 0..chunk_size. Sửa config.yaml."
            )
        self._chunk_size = chunk_size
        self._tokenize = tokenize
        self._markdown = MarkdownNodeParser()
        self._sentences = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tokenizer=tokenize,
            chunking_tokenizer_fn=split_sentences,
        )

    def split_document(self, document: Document) -> list[Chunk]:
        """Cắt một bài đã chuẩn hóa. chunk_id đánh số theo thứ tự đọc."""
        source = LlamaDocument(
            text=document.text,
            metadata={"doc_id": document.doc_id, "source": document.source},
        )
        sections = self._markdown.get_nodes_from_documents([source])
        pieces: list[tuple[str, dict[str, str]]] = []
        for section in sections:
            section_text = section.get_content(metadata_mode=MetadataMode.NONE)
            pieces.extend(self._pieces_from_section(section_text, section.metadata))
        if not pieces:
            raise ValueError(
                f"Tài liệu '{document.doc_id}' không ra chunk nào. Kiểm tra nội dung file."
            )
        chunks = [
            Chunk(
                chunk_id=f"{document.doc_id}_chunk_{index:04d}",
                content=content,
                metadata=metadata,
            )
            for index, (content, metadata) in enumerate(pieces)
        ]
        _log_stats(document.doc_id, chunks, self._tokenize)
        return chunks

    def _pieces_from_section(
        self, text: str, node_metadata: dict[str, object]
    ) -> list[tuple[str, dict[str, str]]]:
        first_line = text.split("\n", 1)[0]
        path = str(node_metadata.get("header_path", ""))
        metadata = {
            "doc_id": str(node_metadata.get("doc_id", "")),
            "source": str(node_metadata.get("source", "")),
            "header_path": path,
            "header_context": header_context(path, first_line),
        }
        pieces: list[tuple[str, dict[str, str]]] = []
        for kind, block in split_prose_and_tables(text):
            if kind == "table":
                pieces.extend(self._table_pieces(block, metadata))
            else:
                pieces.extend(self._prose_pieces(block, metadata))
        return pieces

    def _table_pieces(
        self, table: str, metadata: dict[str, str]
    ) -> list[tuple[str, dict[str, str]]]:
        typed = _with_type(metadata, "table")
        if _count_tokens(table, self._tokenize) <= self._chunk_size:
            return [(table, typed)]
        parts = split_table(
            table,
            self._chunk_size,
            lambda text: _count_tokens(text, self._tokenize),
        )
        return [(part, dict(typed)) for part in parts]

    def _prose_pieces(
        self, prose: str, metadata: dict[str, str]
    ) -> list[tuple[str, dict[str, str]]]:
        typed = _with_type(metadata, "text")
        node = LlamaDocument(text=prose, metadata=dict(typed))
        # LlamaIndex trừ độ dài metadata khỏi chunk_size. Loại hết key để chỉ đếm nội dung.
        keys = list(node.metadata.keys())
        node.excluded_embed_metadata_keys = keys
        node.excluded_llm_metadata_keys = keys
        nodes = self._sentences.get_nodes_from_documents([node])
        pieces: list[tuple[str, dict[str, str]]] = []
        for child in nodes:
            content = child.get_content(metadata_mode=MetadataMode.NONE).strip()
            if content:
                pieces.append((content, dict(typed)))
        return pieces


def _with_type(metadata: dict[str, str], content_type: str) -> dict[str, str]:
    copied = dict(metadata)
    copied["content_type"] = content_type
    return copied


def _count_tokens(text: str, tokenize: Callable[[str], Sequence[object]]) -> int:
    return len(tokenize(text))


def _pop_caption(prose: list[str]) -> str:
    while prose and not prose[-1].strip():
        prose.pop()
    if prose and _CAPTION.match(prose[-1].strip()):
        return prose.pop().strip()
    return ""


def _flush_prose(blocks: list[tuple[str, str]], prose: list[str]) -> None:
    text = "\n".join(prose).strip()
    prose.clear()
    if text:
        blocks.append(("text", text))


def _log_stats(
    doc_id: str, chunks: list[Chunk], tokenize: Callable[[str], Sequence[object]]
) -> None:
    lengths = sorted(_count_tokens(chunk.content, tokenize) for chunk in chunks)
    table_count = sum(
        1 for chunk in chunks if chunk.metadata.get("content_type") == "table"
    )
    middle = int(statistics.median(lengths))
    log.info(
        "Cắt %s: %d chunk, độ dài ngắn nhất/trung vị/dài nhất %d/%d/%d, chunk bảng %d",
        doc_id,
        len(chunks),
        lengths[0],
        middle,
        lengths[-1],
        table_count,
    )
