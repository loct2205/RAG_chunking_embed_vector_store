"""Id điểm dùng chung cho mọi kho vector.

UUID5 từ chunk_id nên nạp lại không tạo điểm trùng.
"""

from __future__ import annotations

import uuid

_NAMESPACE = uuid.NAMESPACE_URL


def point_id(chunk_id: str) -> str:
    """Cùng chunk_id luôn ra cùng một id."""
    return str(uuid.uuid5(_NAMESPACE, chunk_id))
