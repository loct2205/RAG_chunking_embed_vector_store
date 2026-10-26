"""Đọc các file Markdown trong một thư mục thành Document.

Pipeline gọi bước này trước khi chuẩn hóa và cắt chunk.
"""

from __future__ import annotations

from pathlib import Path

from rag.schema import Document


class DocumentLoader:
    """Đọc mọi file .md, sắp theo tên để lần chạy sau ra cùng thứ tự."""

    def load(self, folder: Path) -> list[Document]:
        if not folder.is_dir():
            raise ValueError(
                f"Không thấy thư mục dữ liệu '{folder}'. Sửa paths.data_dir trong config.yaml."
            )
        files = sorted(path for path in folder.glob("*.md") if path.is_file())
        if not files:
            raise ValueError(
                f"Không có file .md trong '{folder}'. Thêm tài liệu hoặc đổi paths.data_dir."
            )
        documents = [
            Document(
                doc_id=path.stem,
                source=path.name,
                text=path.read_text(encoding="utf-8"),
            )
            for path in files
        ]
        doc_ids = [document.doc_id for document in documents]
        if len(doc_ids) != len(set(doc_ids)):
            raise ValueError(
                "Hai file .md trùng tên. Đổi tên file để chunk_id không đè nhau."
            )
        return documents
