"""Điểm vào duy nhất: ingest, search, eval, compare.

In kết quả ra stdout. Log tiến trình dùng logging.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import psycopg
from pymilvus import MilvusClient
from qdrant_client import QdrantClient

from rag.branches import parse_branch_list, parse_eval_configs
from rag.config import AppConfig, load_config
from rag.encode.base import Embedder
from rag.encode.bge_m3 import BgeSettings, load_bge_m3
from rag.encode.bm25 import Bm25Encoder
from rag.encode.cache import VectorCache
from rag.encode.fake import FakeEmbedder
from rag.evaluate import Evaluator, load_questions, report_payload, write_report
from rag.pipeline import IngestParts, IngestPipeline, configure_dense_embedder
from rag.schema import Hit
from rag.search import SearchService
from rag.store.base import VectorStore
from rag.store.milvus_store import MilvusSettings, MilvusStore
from rag.store.pgvector_store import PgSettings, PgVectorStore
from rag.store.qdrant_store import QdrantSettings, QdrantStore
from rag.text.chunker import Chunker
from rag.text.loader import DocumentLoader

log = logging.getLogger(__name__)
_PREVIEW_CHARS = 80


def main(argv: list[str] | None = None) -> int:
    """Đọc lệnh và chạy. Lỗi cấu hình in một dòng, không nuốt lỗi khác."""
    _use_utf8_stdio()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("pymilvus").setLevel(logging.WARNING)
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return _dispatch(args)
    except (FileNotFoundError, ValueError, TypeError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


def _use_utf8_stdio() -> None:
    """Console Windows cp1252 không in được chữ Việt trong kết quả tìm."""
    for stream in (sys.stdout, sys.stderr):
        encoding = stream.encoding or "utf-8"
        try:
            "tiết".encode(encoding)
        except (LookupError, UnicodeEncodeError):
            reconfigure = getattr(stream, "reconfigure", None)
            if reconfigure is not None:
                reconfigure(encoding="utf-8", errors="replace")


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=Path("config.yaml"))
    common.add_argument(
        "--fake-embedder",
        action="store_true",
        help="không nạp BGE-M3, dùng vector giả để chạy thử",
    )
    parser = argparse.ArgumentParser(prog="rag.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", parents=[common])
    ingest.add_argument(
        "--recreate", action="store_true", help="xóa collection rồi tạo lại"
    )

    search = sub.add_parser("search", parents=[common])
    search.add_argument("question")
    search.add_argument("--branches", default=None, help="ví dụ dense,sparse")

    evaluate = sub.add_parser("eval", parents=[common])
    evaluate.add_argument(
        "--configs", required=True, help="ví dụ dense,sparse,bm25,dense+sparse"
    )

    compare = sub.add_parser("compare", parents=[common])
    compare.add_argument("--configs", required=True, help="cùng dạng với eval")
    return parser


def _dispatch(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    if args.command == "ingest":
        return _ingest(config, args.fake_embedder, args.recreate)
    embedder = _embedder(config, args.fake_embedder)
    bm25 = Bm25Encoder(config.bm25.mode, config.bm25.k1, config.bm25.b)
    bm25.load(config.bm25.vocab_path)
    store = _store(config)
    service = SearchService(embedder, bm25, store, config.search.top_k)
    if args.command == "search":
        branches = (
            parse_branch_list(args.branches)
            if args.branches
            else list(config.search.branches)
        )
        _print_hits(service.search(args.question, branches))
        return 0
    questions = load_questions(config.eval.questions_path)
    cached_model, chunks, _ = VectorCache().load(config.paths.cache_dir)
    if cached_model != embedder.model_name or len(chunks) != store.count():
        raise ValueError("Cache đánh giá không khớp model/số chunk của collection.")
    evaluator = Evaluator(service, questions, chunks)
    groups = parse_eval_configs(args.configs)
    if args.command == "eval":
        return _eval(config, evaluator, groups, embedder.model_name)
    return _compare(config, evaluator, groups, embedder.model_name)


def _ingest(config: AppConfig, fake: bool, recreate: bool) -> int:
    embedder = _embedder(config, fake)
    parts = IngestParts(
        loader=DocumentLoader(),
        chunker=Chunker(
            config.chunking.chunk_size,
            config.chunking.chunk_overlap,
            embedder.tokenize,
        ),
        embedder=embedder,
        bm25=Bm25Encoder(config.bm25.mode, config.bm25.k1, config.bm25.b),
        store=_store(config),
        cache=VectorCache(),
    )
    count = IngestPipeline(config, parts).run(recreate)
    print(f"Đã nạp {count} chunk vào '{_collection_name(config)}'.")
    return 0


def _eval(
    config: AppConfig,
    evaluator: Evaluator,
    groups: list[list[str]],
    model_name: str,
) -> int:
    body: dict[str, object] = {}
    for branches in groups:
        label = "+".join(branches)
        quality = evaluator.score(branches, config.search.top_k)
        latency = evaluator.measure_latency(
            branches,
            config.search.top_k,
            config.eval.warmup_runs,
            config.eval.latency_runs,
        )
        body[label] = {"quality": quality, "latency": latency}
        _print_quality(label, quality, latency)
    payload = report_payload(config, body)
    payload["model_name"] = model_name
    path = write_report(
        config.paths.reports_dir, ",".join("+".join(group) for group in groups), payload
    )
    print(f"Đã ghi báo cáo {path}")
    return 0


def _compare(
    config: AppConfig,
    evaluator: Evaluator,
    groups: list[list[str]],
    model_name: str,
) -> int:
    body: dict[str, object] = {}
    for branches in groups:
        label = "+".join(branches)
        fused = evaluator.fuse_branches(
            branches,
            config.search.top_k,
            config.search.prefetch_limit,
            config.search.rrf_k,
        )
        body[label] = {"client_rrf": fused}
        print(
            f"{label} client-rrf  text {fused['text_recall_at_k']:.3f}  "
            f"table {fused['table_recall_at_k']:.3f}"
        )
        _print_retrieval_quality(fused)
    payload = report_payload(config, body)
    payload["model_name"] = model_name
    path = write_report(
        config.paths.reports_dir,
        "compare_" + ",".join("+".join(group) for group in groups),
        payload,
    )
    print(f"Đã ghi báo cáo {path}")
    return 0


def _embedder(config: AppConfig, fake: bool) -> Embedder:
    if fake:
        log.info(
            "Dùng FakeEmbedder, không nạp BGE-M3. Tokenizer tách theo khoảng trắng."
        )
        return FakeEmbedder()
    base = load_bge_m3(
        BgeSettings(
            model_name=config.embedding.model_name,
            model_path=config.embedding.model_path,
            max_length=config.embedding.max_length,
            batch_size=config.embedding.batch_size,
            use_fp16=config.embedding.use_fp16,
        )
    )
    return configure_dense_embedder(base, config.embedding)


def _store(config: AppConfig) -> VectorStore:
    if config.store == "pgvector":
        return _pgvector(config)
    if config.store == "milvus":
        return _milvus(config)
    return _qdrant(config)


def _qdrant(config: AppConfig) -> QdrantStore:
    qdrant = config.qdrant
    settings = QdrantSettings(
        collection=qdrant.collection,
        upsert_batch_size=qdrant.upsert_batch_size,
        prefetch_limit=config.search.prefetch_limit,
        rrf_k=config.search.rrf_k,
        hnsw_m=qdrant.hnsw.m,
        ef_construct=qdrant.hnsw.ef_construct,
        full_scan_threshold=qdrant.hnsw.full_scan_threshold,
        indexing_threshold=qdrant.indexing_threshold,
        hnsw_ef=qdrant.hnsw_ef,
    )
    return QdrantStore(QdrantClient(url=qdrant.url), settings)


def _pgvector(config: AppConfig) -> PgVectorStore:
    pgvector = config.pgvector
    qdrant = config.qdrant
    dsn = pgvector.dsn
    settings = PgSettings(
        collection=pgvector.collection,
        sparse_dimension=pgvector.sparse_dimension,
        upsert_batch_size=qdrant.upsert_batch_size,
        prefetch_limit=config.search.prefetch_limit,
        rrf_k=config.search.rrf_k,
        hnsw_m=qdrant.hnsw.m,
        ef_construct=qdrant.hnsw.ef_construct,
        hnsw_ef=qdrant.hnsw_ef,
    )
    return PgVectorStore(lambda: psycopg.connect(dsn), settings)


def _milvus(config: AppConfig) -> MilvusStore:
    milvus = config.milvus
    qdrant = config.qdrant
    uri = milvus.uri
    settings = MilvusSettings(
        collection=milvus.collection,
        sparse_dimension=milvus.sparse_dimension,
        upsert_batch_size=qdrant.upsert_batch_size,
        prefetch_limit=config.search.prefetch_limit,
        rrf_k=config.search.rrf_k,
        hnsw_m=qdrant.hnsw.m,
        ef_construct=qdrant.hnsw.ef_construct,
        hnsw_ef=qdrant.hnsw_ef,
    )
    return MilvusStore(lambda: MilvusClient(uri=uri), settings)


def _collection_name(config: AppConfig) -> str:
    if config.store == "pgvector":
        return config.pgvector.collection
    if config.store == "milvus":
        return config.milvus.collection
    return config.qdrant.collection


def _print_hits(hits: list[Hit]) -> None:
    if not hits:
        print("Không có kết quả.")
        return
    for rank, hit in enumerate(hits, start=1):
        preview = " ".join(hit.content.split())
        if len(preview) > _PREVIEW_CHARS:
            preview = preview[:_PREVIEW_CHARS] + "..."
        content_type = hit.metadata.get("content_type", "")
        print(f"{rank}. {hit.score:.4f}  {hit.chunk_id}  {content_type}")
        print(f"   {preview}")


def _print_quality(
    label: str, quality: dict[str, float | int], latency: dict[str, object]
) -> None:
    encode = latency["encode_ms"]
    query = latency["query_ms"]
    assert isinstance(encode, dict) and isinstance(query, dict)
    print(
        f"{label}  text {quality['text_recall_at_k']:.3f}  table {quality['table_recall_at_k']:.3f}  "
        f"encode p50 {encode['p50']:.1f} ms  query p50 {query['p50']:.1f} ms"
    )
    _print_retrieval_quality(quality)
    k = quality["top_k"]
    print(
        f"  Success@{k}: text {quality['text_success_at_k']:.3f}  "
        f"table {quality['table_success_at_k']:.3f}; "
        f"MRR@{k}: text {quality['text_mrr_at_k']:.3f}  "
        f"table {quality['table_mrr_at_k']:.3f}"
    )
    for stage in ("encode_ms", "query_ms", "total_ms"):
        stats = latency[stage]
        assert isinstance(stats, dict)
        print(
            f"  {stage}: p50 {stats['p50']:.1f}, "
            f"p95 {stats['p95']:.1f}, p99 {stats['p99']:.1f}"
        )


def _print_retrieval_quality(quality: dict[str, float | int]) -> None:
    if "text_retrieval_recall_at_k" not in quality:
        return
    k = quality["top_k"]
    print(
        f"  Recall@{k}: text {quality['text_retrieval_recall_at_k']:.3f}  "
        f"table {quality['table_retrieval_recall_at_k']:.3f}; "
        f"nDCG@{k}: text {quality['text_ndcg_at_k']:.3f}  "
        f"table {quality['table_ndcg_at_k']:.3f}"
    )


if __name__ == "__main__":
    sys.exit(main())
