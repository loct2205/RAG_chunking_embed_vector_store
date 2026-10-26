"""Chuẩn hóa chữ dùng chung cho tài liệu, câu hỏi và đoạn đáp án.

Thuộc bước làm sạch trong luồng nạp dữ liệu; SearchService và Evaluator cũng gọi.
"""

from __future__ import annotations

import re
import unicodedata

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_INLINE_SPACES = re.compile(r"[ \t ]+")


def normalize(text: str) -> str:
    """Chuẩn NFC, bỏ ký tự điều khiển, gộp khoảng trắng trong dòng.

    Giữ nguyên xuống dòng vì MarkdownNodeParser cần dấu # ở đầu dòng.
    """
    text = unicodedata.normalize("NFC", text)
    text = _CONTROL_CHARS.sub("", text)
    lines = (_INLINE_SPACES.sub(" ", line).strip() for line in text.split("\n"))
    return "\n".join(lines)


def header_context(header_path: str, first_line: str) -> str:
    """Ghép đường dẫn tiêu đề cha với tiêu đề của chính mục.

    header_path của MarkdownNodeParser không chứa tiêu đề của chính mục.
    """
    own = first_line.lstrip("#").strip() if first_line.startswith("#") else ""
    return header_path + (f"{own}/" if own else "")


def build_embed_text(content: str, header_context_value: str) -> str:
    """Ghép đường dẫn tiêu đề vào nội dung để chunk con vẫn mang tên mục khi encode.

    Đường dẫn '/' là mục không có tiêu đề, nên không thêm vào câu embed.
    """
    if header_context_value in ("", "/"):
        return content
    return f"{header_context_value}\n{content}"
