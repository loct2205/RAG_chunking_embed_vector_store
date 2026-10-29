"""Build explicit relevance labels against an immutable chunk snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def corpus_fingerprint(chunks: list[dict]) -> str:
    canonical = json.dumps(
        sorted(chunks, key=lambda c: c["chunk_id"]),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_ground_truth(chunks: list[dict], spec: dict) -> list[dict]:
    fingerprint = corpus_fingerprint(chunks)
    if fingerprint != spec["corpus_sha256"]:
        raise ValueError("Chunk snapshot differs from the annotated snapshot.")
    corpus = {c["chunk_id"]: c for c in chunks}
    if len(corpus) != len(chunks):
        raise ValueError("Duplicate chunk IDs.")
    records = []
    seen = set()
    for question in spec["questions"]:
        if question["qid"] in seen:
            raise ValueError("Duplicate question ID.")
        seen.add(question["qid"])
        groups = []
        evidence = {}
        for key in question["evidence_keys"]:
            entry = spec["evidence_library"][key]
            ids = []
            for match in entry["matches"]:
                cid, anchor = match["chunk_id"], match["contains"]
                if (
                    cid not in corpus
                    or not anchor
                    or anchor not in corpus[cid]["content"]
                ):
                    raise ValueError(f"Missing evidence anchor: {key}/{cid}.")
                chunk = corpus[cid]
                ids.append(cid)
                item = evidence.setdefault(
                    cid,
                    {
                        "chunk_id": cid,
                        "doc_id": chunk["metadata"]["doc_id"],
                        "content_type": chunk["metadata"].get("content_type", "text"),
                        "excerpts": [],
                    },
                )
                if anchor not in item["excerpts"]:
                    item["excerpts"].append(anchor)
            if not ids:
                raise ValueError(f"Empty evidence group: {key}.")
            groups.append({"label": entry["label"], "chunk_ids": sorted(set(ids))})
        if not groups:
            raise ValueError("A question needs at least one evidence group.")
        doc_ids = sorted({e["doc_id"] for e in evidence.values()})
        records.append(
            {
                "qid": question["qid"],
                "question": question["question"],
                "reference_answer": question["reference_answer"],
                "kind": question["kind"],
                "doc_ids": doc_ids,
                "question_type": "multi_document"
                if len(doc_ids) > 1
                else "single_document",
                "relevant_chunk_ids": sorted(evidence),
                "required_evidence_groups": groups,
                "evidence": [evidence[cid] for cid in sorted(evidence)],
                "corpus_sha256": fingerprint,
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=Path("extra/chunks.json"))
    parser.add_argument(
        "--spec", type=Path, default=Path("extra/ground_truth_spec.json")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = build_ground_truth(
        json.loads(args.chunks.read_text(encoding="utf-8")),
        json.loads(args.spec.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
