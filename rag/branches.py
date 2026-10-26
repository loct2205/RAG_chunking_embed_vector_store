"""Tên nhánh tìm kiếm dùng chung cho config, CLI và SearchService."""

from __future__ import annotations

ALLOWED_BRANCHES = ("dense", "sparse", "bm25")


def validate_branches(branches: list[str]) -> list[str]:
    """Báo lỗi nếu danh sách nhánh rỗng, lặp, hoặc có tên lạ."""
    if not branches:
        raise ValueError("Cần ít nhất một nhánh: dense, sparse hoặc bm25.")
    unknown = [name for name in branches if name not in ALLOWED_BRANCHES]
    if unknown:
        raise ValueError(f"Nhánh {unknown} không hợp lệ. Chỉ dùng dense, sparse, bm25.")
    if len(branches) != len(set(branches)):
        raise ValueError(f"Nhánh bị lặp trong {branches}. Mỗi nhánh chỉ bật một lần.")
    return list(branches)


def parse_branch_list(text: str) -> list[str]:
    """Đọc cờ `--branches dense,sparse`."""
    parts = [part.strip() for part in text.split(",") if part.strip()]
    return validate_branches(parts)


def parse_eval_configs(text: str) -> list[list[str]]:
    """Đọc `--configs dense,sparse,bm25,dense+sparse` thành từng nhóm nhánh."""
    groups: list[list[str]] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split("+") if part.strip()]
        groups.append(validate_branches(parts))
    if not groups:
        raise ValueError(
            "`--configs` chưa có cấu hình. Ví dụ: dense,sparse,bm25,dense+sparse."
        )
    return groups
