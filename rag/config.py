"""Đọc config.yaml thành dataclass. Mọi tham số chạy đều đi qua file này."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from rag.branches import validate_branches


@dataclass(frozen=True)
class PathsConfig:
    data_dir: Path
    cache_dir: Path
    reports_dir: Path


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int
    chunk_overlap: int


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str
    model_path: Path
    max_length: int
    batch_size: int
    use_fp16: bool
    dense_model: str = "bge-m3"
    e5_model_path: Path = Path("models/multilingual-e5-base")
    e5_max_length: int = 512
    e5_batch_size: int = 16


@dataclass(frozen=True)
class Bm25Config:
    mode: str
    k1: float
    b: float
    vocab_path: Path


@dataclass(frozen=True)
class HnswConfig:
    m: int
    ef_construct: int
    full_scan_threshold: int


@dataclass(frozen=True)
class QdrantConfig:
    url: str
    collection: str
    upsert_batch_size: int
    hnsw: HnswConfig
    indexing_threshold: int
    hnsw_ef: int


@dataclass(frozen=True)
class SearchConfig:
    branches: list[str]
    prefetch_limit: int
    rrf_k: int
    top_k: int


@dataclass(frozen=True)
class EvalConfig:
    questions_path: Path
    warmup_runs: int
    latency_runs: int


@dataclass(frozen=True)
class PgVectorConfig:
    """Postgres + pgvector. DSN local của docker-compose, không phải mật khẩu thật."""

    dsn: str
    collection: str
    sparse_dimension: int


@dataclass(frozen=True)
class MilvusConfig:
    """Milvus standalone. URI là địa chỉ client, không gồm mật khẩu."""

    uri: str
    collection: str
    sparse_dimension: int


_STORES = ("qdrant", "pgvector", "milvus")


@dataclass(frozen=True)
class AppConfig:
    paths: PathsConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    bm25: Bm25Config
    qdrant: QdrantConfig
    search: SearchConfig
    eval: EvalConfig
    store: str
    pgvector: PgVectorConfig
    milvus: MilvusConfig


def load_config(path: Path) -> AppConfig:
    """Đọc YAML. Thiếu khóa thì báo tên khóa và file để sửa."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Không thấy file cấu hình '{path}'. Chạy lệnh với --config config.yaml."
        )
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError(f"'{path}' phải là một mapping YAML.")
    return AppConfig(
        paths=_paths(_section(loaded, "paths")),
        chunking=_chunking(_section(loaded, "chunking")),
        embedding=_embedding(_section(loaded, "embedding")),
        bm25=_bm25(_section(loaded, "bm25")),
        qdrant=_qdrant(_section(loaded, "qdrant")),
        search=_search(_section(loaded, "search")),
        eval=_eval(_section(loaded, "eval")),
        store=_store_name(loaded),
        pgvector=_pgvector(_section(loaded, "pgvector")),
        milvus=_milvus(_section(loaded, "milvus")),
    )


def _paths(section: dict[str, Any]) -> PathsConfig:
    return PathsConfig(
        data_dir=_as_path(section, "data_dir"),
        cache_dir=_as_path(section, "cache_dir"),
        reports_dir=_as_path(section, "reports_dir"),
    )


def _chunking(section: dict[str, Any]) -> ChunkingConfig:
    chunk_size = _positive_int(section, "chunk_size")
    chunk_overlap = _non_negative_int(section, "chunk_overlap")
    if chunk_overlap > chunk_size:
        raise ValueError(
            "chunking.chunk_overlap lớn hơn chunk_size. "
            "Giảm overlap hoặc tăng chunk_size trong config.yaml."
        )
    return ChunkingConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def _embedding(section: dict[str, Any]) -> EmbeddingConfig:
    dense_model = str(section.get("dense_model", "bge-m3"))
    if dense_model not in ("bge-m3", "e5-base"):
        raise ValueError("embedding.dense_model phải là bge-m3 hoặc e5-base.")
    e5 = {
        "e5_model_path": section.get("e5_model_path", "models/multilingual-e5-base"),
        "e5_max_length": section.get("e5_max_length", 512),
        "e5_batch_size": section.get("e5_batch_size", section.get("batch_size", 16)),
    }
    e5_max_length = _positive_int(e5, "e5_max_length")
    if e5_max_length > 512:
        raise ValueError("embedding.e5_max_length không được vượt quá 512 token.")
    return EmbeddingConfig(
        model_name=str(_require(section, "model_name")),
        model_path=_as_path(section, "model_path"),
        max_length=_positive_int(section, "max_length"),
        batch_size=_positive_int(section, "batch_size"),
        use_fp16=bool(_require(section, "use_fp16")),
        dense_model=dense_model,
        e5_model_path=_as_path(e5, "e5_model_path"),
        e5_max_length=e5_max_length,
        e5_batch_size=_positive_int(e5, "e5_batch_size"),
    )


def _bm25(section: dict[str, Any]) -> Bm25Config:
    mode = str(_require(section, "mode"))
    if mode not in ("syllable", "word"):
        raise ValueError("bm25.mode phải là syllable hoặc word trong config.yaml.")
    return Bm25Config(
        mode=mode,
        k1=float(_require(section, "k1")),
        b=float(_require(section, "b")),
        vocab_path=_as_path(section, "vocab_path"),
    )


def _qdrant(section: dict[str, Any]) -> QdrantConfig:
    hnsw = _section(section, "hnsw")
    return QdrantConfig(
        url=str(_require(section, "url")),
        collection=str(_require(section, "collection")),
        upsert_batch_size=_positive_int(section, "upsert_batch_size"),
        hnsw=HnswConfig(
            m=_positive_int(hnsw, "m"),
            ef_construct=_positive_int(hnsw, "ef_construct"),
            full_scan_threshold=_non_negative_int(hnsw, "full_scan_threshold"),
        ),
        indexing_threshold=_non_negative_int(section, "indexing_threshold"),
        hnsw_ef=_positive_int(section, "hnsw_ef"),
    )


def _search(section: dict[str, Any]) -> SearchConfig:
    branches = _require(section, "branches")
    if not isinstance(branches, list) or not all(
        isinstance(item, str) for item in branches
    ):
        raise ValueError(
            "search.branches phải là danh sách chuỗi, ví dụ [dense, sparse]."
        )
    return SearchConfig(
        branches=validate_branches([str(item) for item in branches]),
        prefetch_limit=_positive_int(section, "prefetch_limit"),
        rrf_k=_non_negative_int(section, "rrf_k"),
        top_k=_positive_int(section, "top_k"),
    )


def _eval(section: dict[str, Any]) -> EvalConfig:
    return EvalConfig(
        questions_path=_as_path(section, "questions_path"),
        warmup_runs=_non_negative_int(section, "warmup_runs"),
        latency_runs=_positive_int(section, "latency_runs"),
    )


def _store_name(loaded: dict[str, Any]) -> str:
    name = str(_require(loaded, "store"))
    if name not in _STORES:
        raise ValueError(
            "store phải là qdrant, pgvector hoặc milvus trong config.yaml."
        )
    return name


def _pgvector(section: dict[str, Any]) -> PgVectorConfig:
    return PgVectorConfig(
        dsn=str(_require(section, "dsn")),
        collection=str(_require(section, "collection")),
        sparse_dimension=_positive_int(section, "sparse_dimension"),
    )


def _milvus(section: dict[str, Any]) -> MilvusConfig:
    return MilvusConfig(
        uri=str(_require(section, "uri")),
        collection=str(_require(section, "collection")),
        sparse_dimension=_positive_int(section, "sparse_dimension"),
    )


def _section(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = _require(mapping, key)
    if not isinstance(value, dict):
        raise TypeError(f"'{key}' phải là một nhóm khóa trong config.yaml.")
    return value


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Thiếu khóa '{key}' trong config.yaml.")
    return mapping[key]


def _as_path(mapping: dict[str, Any], key: str) -> Path:
    value = _require(mapping, key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"'{key}' phải là đường dẫn trong config.yaml.")
    return Path(value)


def _positive_int(mapping: dict[str, Any], key: str) -> int:
    value = int(_require(mapping, key))
    if value < 1:
        raise ValueError(f"'{key}' phải là số nguyên dương trong config.yaml.")
    return value


def _non_negative_int(mapping: dict[str, Any], key: str) -> int:
    value = int(_require(mapping, key))
    if value < 0:
        raise ValueError(f"'{key}' phải là số nguyên không âm trong config.yaml.")
    return value
