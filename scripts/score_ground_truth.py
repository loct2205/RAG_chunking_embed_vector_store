"""Score annotated chunks and complete evidence coverage, without answer generation.

Run from the repository root with python -m scripts.score_ground_truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path

from rag.evaluate import ndcg_at_k, recall_at_k
from scripts.build_ground_truth import corpus_fingerprint


def evaluate_ground_truth(service, chunks, ground_truth, ks=(1, 5, 10), branches=None):
    """Retrieve once at max(K); one alternative per group covers that evidence."""
    if not ks or any(not isinstance(k, int) or k < 1 for k in ks):
        raise ValueError("ks must contain positive integers.")
    if not ground_truth:
        raise ValueError("Ground truth is empty.")
    corpus = {c.chunk_id: c for c in chunks}
    if len(corpus) != len(chunks):
        raise ValueError("Duplicate chunk IDs.")
    fingerprint = corpus_fingerprint([asdict(c) for c in chunks])
    seen = set()
    # Validate every annotation before issuing any searches.
    for q in ground_truth:
        if q["qid"] in seen:
            raise ValueError("Duplicate question ID.")
        seen.add(q["qid"])
        if q["corpus_sha256"] != fingerprint:
            raise ValueError("Ground truth belongs to another corpus snapshot.")
        relevant = set(q["relevant_chunk_ids"])
        groups = [set(g["chunk_ids"]) for g in q["required_evidence_groups"]]
        if not relevant or not relevant <= corpus.keys():
            raise ValueError("Ground truth references unknown or empty chunk IDs.")
        if not groups or any(not g or not g <= relevant for g in groups):
            raise ValueError("Invalid required evidence groups.")
        if set().union(*groups) != relevant:
            raise ValueError("Evidence groups must cover all relevant chunk IDs.")

    details = []
    for q in ground_truth:
        hits = service.search(q["question"], branches or ["dense", "sparse"], max(ks))
        for hit in hits:
            if hit.chunk_id not in corpus or (
                hit.content != corpus[hit.chunk_id].content
                or hit.metadata != corpus[hit.chunk_id].metadata
            ):
                raise ValueError("Search returned a chunk from another corpus.")
        ranked = [h.chunk_id for h in hits]
        relevant = set(q["relevant_chunk_ids"])
        groups = [set(g["chunk_ids"]) for g in q["required_evidence_groups"]]
        metrics = {}
        for k in sorted(set(ks)):
            retrieved = set(ranked[:k])
            covered = sum(bool(retrieved & g) for g in groups)
            metrics[str(k)] = {
                "success_at_k": float(bool(retrieved & relevant)),
                "recall_at_k": recall_at_k(ranked, relevant, k),
                "ndcg_at_k": ndcg_at_k(ranked, relevant, k),
                "mrr_at_k": next(
                    (1 / r for r, cid in enumerate(ranked[:k], 1) if cid in relevant),
                    0.0,
                ),
                "evidence_coverage_at_k": covered / len(groups),
                "all_evidence_success_at_k": float(covered == len(groups)),
            }
        details.append(
            {
                "qid": q["qid"],
                "question": q["question"],
                "kind": q["kind"],
                "question_type": q.get("question_type", "single_document"),
                "relevant_chunk_ids": q["relevant_chunk_ids"],
                "required_evidence_groups": q["required_evidence_groups"],
                "retrieved_chunk_ids": ranked,
                "first_relevant_rank": next(
                    (rank for rank, cid in enumerate(ranked, 1) if cid in relevant),
                    None,
                ),
                "missing_evidence_groups_by_k": {
                    str(k): [
                        group["label"]
                        for group in q["required_evidence_groups"]
                        if not set(ranked[:k]) & set(group["chunk_ids"])
                    ]
                    for k in sorted(set(ks))
                },
                "by_k": metrics,
            }
        )

    def summarize(rows):
        return {
            str(k): {
                "count": len(rows),
                **{
                    name: sum(row["by_k"][str(k)][name] for row in rows) / len(rows)
                    for name in details[0]["by_k"][str(k)]
                },
            }
            for k in sorted(set(ks))
        }

    return {
        "corpus_sha256": fingerprint,
        "question_count": len(details),
        "by_k": summarize(details),
        "by_kind": {
            kind: summarize([r for r in details if r["kind"] == kind])
            for kind in sorted({r["kind"] for r in details})
        },
        "by_question_type": {
            kind: summarize([r for r in details if r["question_type"] == kind])
            for kind in sorted({r["question_type"] for r in details})
        },
        "per_question": details,
    }


def main() -> None:
    from rag.branches import parse_eval_configs
    from rag.cli import _embedder, _store, _use_utf8_stdio
    from rag.config import load_config
    from rag.encode.bm25 import Bm25Encoder
    from rag.encode.cache import VectorCache
    from rag.evaluate import Evaluator, report_payload, write_report
    from rag.schema import EvalQuestion
    from rag.search import SearchService

    _use_utf8_stdio()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("config.corpus.ground-truth.yaml")
    )
    parser.add_argument(
        "--ground-truth", type=Path, help="Defaults to eval.questions_path in YAML."
    )
    parser.add_argument("--configs", default="dense,sparse,bm25,dense+sparse")
    parser.add_argument("--fake-embedder", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    ground_truth_path = args.ground_truth or config.eval.questions_path
    gt = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    model, chunks, _ = VectorCache().load(config.paths.cache_dir)
    embedder = _embedder(config, args.fake_embedder)
    store = _store(config)
    if model != embedder.model_name or len(chunks) != store.count():
        raise ValueError("Cache model/count differs from collection.")
    bm25 = Bm25Encoder(config.bm25.mode, config.bm25.k1, config.bm25.b)
    bm25.load(config.bm25.vocab_path)
    service = SearchService(embedder, bm25, store, config.search.top_k)
    # Used only by the existing latency timer; quality uses explicit chunk labels.
    questions = [
        EvalQuestion(q["qid"], q["question"], [q["reference_answer"]], q["kind"])
        for q in gt
    ]
    timer = Evaluator(service, questions)
    results = {}
    ks = tuple(sorted({1, min(5, config.search.top_k), config.search.top_k}))
    for branches in parse_eval_configs(args.configs):
        label = "+".join(branches)
        quality = evaluate_ground_truth(service, chunks, gt, ks, branches)
        latency = timer.measure_latency(
            branches,
            config.search.top_k,
            config.eval.warmup_runs,
            config.eval.latency_runs,
        )
        results[label] = {"quality": quality, "latency": latency}
        print(label, json.dumps(quality["by_k"]), flush=True)
        print("Latency (ms):", json.dumps(latency), flush=True)
        # Save each completed branch so a later failure does not discard results.
        payload = report_payload(config, results)
        payload.update(
            {
                "model_name": model,
                "ground_truth_path": str(ground_truth_path),
                "ground_truth_sha256": hashlib.sha256(
                    ground_truth_path.read_bytes()
                ).hexdigest(),
                "corpus_sha256": quality["corpus_sha256"],
                "chunk_count": len(chunks),
            }
        )
        path = write_report(config.paths.reports_dir, "ground_truth_" + label, payload)
        print(f"Report: {path}", flush=True)


if __name__ == "__main__":
    main()
