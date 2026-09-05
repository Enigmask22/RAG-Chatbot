"""Dựng fixture cho smoke eval của CI — chạy MỘT lần, ở máy có index thật.

`W5-09`. Cặp đôi nặng của `smoke.py`: ở đây có model, có index 15.814 chunk, có
GPU; bên kia không có gì cả.

## Fixture chứa gì, và vì sao đúng những thứ đó

* **30 câu hỏi + nhãn đã phân giải từ span.** Phân giải cần *mọi* chunk của
  tài liệu liên quan (~264 chunk/tài liệu ở corpus này), nên tính ở đây một lần
  thay vì kéo cả nghìn chunk vào fixture.
* **Chunk liên quan + chunk gây nhiễu.** Không có nhiễu thì `recall@10` luôn
  bằng 1 và cổng không đo gì.
* **Vector dense + sparse của cả hai.** Đó là toàn bộ lý do CI không cần model.

⚠️ Chunk gây nhiễu lấy **từ chính những tài liệu ấy**, không lấy ngẫu nhiên toàn
corpus: nhiễu cùng chủ đề mới ép thứ hạng phải đúng. Một tập nhiễu ngẫu nhiên
làm mọi cấu hình truy hồi trông như nhau.

## ⚠️ Vector lưu ở float16

Giảm một nửa dung lượng, và `pre-commit` chặn file > 1 MB. Sai số làm tròn có
thật, nhưng baseline được sinh **từ chính fixture đã làm tròn**, nên cổng vẫn
tự nhất quán. Cái mất là: con số smoke không so được với con số của index thật
— điều vốn đã đúng vì tập chunk khác nhau (xem docstring `smoke.py`).
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import random
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from rag_core.schemas import Chunk

from .golden import QueryCategory, load_golden_set
from .retrieval_eval import open_index
from .smoke import FIXTURE_VERSION

logger = logging.getLogger(__name__)

__all__ = ["build_fixture", "main"]

DEFAULT_N_QUERIES = 30
DEFAULT_DISTRACTORS_PER_DOC = 12


def _pack(rows: Sequence[tuple[str, dict[str, Any]]], dimension: int) -> dict[str, np.ndarray]:
    """Danh sách `(id, {dense, sparse})` → mảng phẳng dạng CSR cho `np.savez`.

    Một `.npz` với ba mảng thay vì một danh sách dict: JSON của 300×1024 float
    là ~7 MB text, còn `float16` nhị phân là 600 KB.
    """
    ids = np.array([key for key, _ in rows], dtype=object)
    dense = np.vstack([np.asarray(v["dense"], dtype=np.float32) for _, v in rows]).astype(
        np.float16
    )
    if dense.shape[1] != dimension:
        raise ValueError(f"vector {dense.shape[1]} chiều, config nói {dimension}")
    offsets = [0]
    indices: list[int] = []
    values: list[float] = []
    for _, vec in rows:
        sparse = vec["sparse"]
        indices.extend(int(i) for i in sparse["indices"])
        values.extend(float(v) for v in sparse["values"])
        offsets.append(len(indices))
    return {
        "ids": ids.astype("U"),
        "dense": dense,
        "sparse_offsets": np.asarray(offsets, dtype=np.int64),
        "sparse_indices": np.asarray(indices, dtype=np.int32),
        "sparse_values": np.asarray(values, dtype=np.float16),
    }


def _stored_to_plain(stored: dict[str, Any], dense_name: str, sparse_name: str) -> dict[str, Any]:
    """Named-vector của Qdrant → `{dense, sparse}` thuần."""
    sparse = stored.get(sparse_name)
    indices = getattr(sparse, "indices", None)
    values = getattr(sparse, "values", None)
    if indices is None and isinstance(sparse, dict):
        indices, values = sparse.get("indices"), sparse.get("values")
    if indices is None or values is None:
        raise ValueError("point không có sparse vector — collection sai schema?")
    return {
        "dense": list(stored[dense_name]),
        "sparse": {"indices": list(indices), "values": list(values)},
    }


def build_fixture(
    index_config: Path,
    out_meta: Path,
    *,
    n_queries: int = DEFAULT_N_QUERIES,
    distractors_per_doc: int = DEFAULT_DISTRACTORS_PER_DOC,
    seed: int = 20260905,
) -> dict[str, Any]:
    from rag_core.retrieval import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME

    session = open_index(index_config)
    golden = list(load_golden_set(Path("data/golden/golden_v1.jsonl")))
    resolved, resolution = session.resolve_labels(golden, 0.5)
    if resolution is not None:
        logger.info(
            "phân giải span: %d câu tính lại · %d giữ nhãn cũ · %d không khớp span nào",
            resolution.resolved,
            resolution.kept_chunk_ids,
            len(resolution.unmatched_queries),
        )

    # ⚠️ `unanswerable` bị loại: theo schema chúng KHÔNG có chunk liên quan, nên
    # mọi metric trả `None` và một phép trung bình sẽ nổ giữa CI. Chúng vẫn được
    # đo ở eval đêm, nơi có `refusal accuracy` để nói về chúng.
    usable = [
        q for q in resolved if q.relevant_chunk_ids and q.category is not QueryCategory.UNANSWERABLE
    ]
    rng = random.Random(seed)
    # Lấy mẫu **phân tầng theo nhóm**: `W5-04` đã đo được rằng 50 mẫu ngẫu nhiên
    # chỉ cho kỳ vọng 0,58 câu `NOT_FOUND`. Một smoke chỉ toàn `factoid` không
    # gác được nhánh multi-hop.
    by_category: dict[str, list[Any]] = {}
    for query in usable:
        by_category.setdefault(str(query.category), []).append(query)
    for bucket in by_category.values():
        rng.shuffle(bucket)
    picked: list[Any] = []
    order = sorted(by_category)
    while len(picked) < n_queries and any(by_category[c] for c in order):
        for category in order:
            if by_category[category] and len(picked) < n_queries:
                picked.append(by_category[category].pop())
    picked.sort(key=lambda q: q.query_id)
    logger.info(
        "chọn %d câu: %s",
        len(picked),
        {c: sum(1 for q in picked if str(q.category) == c) for c in order},
    )

    relevant_ids = {cid for q in picked for cid in q.relevant_chunk_ids}
    doc_ids = sorted({cid.split("::")[0] for cid in relevant_ids})
    distractors: list[str] = []
    for doc_id in doc_ids:
        pool = [
            chunk.chunk_id
            for chunk in session.store.fetch_doc_chunks([doc_id])
            if chunk.chunk_id not in relevant_ids
        ]
        rng.shuffle(pool)
        distractors.extend(pool[:distractors_per_doc])
    wanted = sorted(relevant_ids | set(distractors))
    logger.info(
        "chunk: %d liên quan + %d nhiễu = %d", len(relevant_ids), len(distractors), len(wanted)
    )

    chunks: dict[str, Chunk] = session.store.fetch_chunks(wanted)
    vectors = session.store.fetch_vectors(wanted)
    missing = [cid for cid in wanted if cid not in chunks or cid not in vectors]
    if missing:
        raise RuntimeError(f"index thiếu {len(missing)} chunk, ví dụ {missing[:3]}")

    # Vector câu hỏi: một lần gọi model thật, đúng đường `embed_query_hybrid` mà
    # `QdrantHybridRetriever` sẽ gọi ở CI — không phải `embed_documents`, vì
    # BGE-M3 là model bất đối xứng và hai đường cho hai vector khác nhau.
    query_rows: list[tuple[str, dict[str, Any]]] = []
    for query in picked:
        hybrid = session.embeddings.embed_query_hybrid(query.query)
        if hybrid is None:  # pragma: no cover - provider sai
            raise RuntimeError(
                f"{session.embeddings.name!r} không có `embed_query_hybrid` — "
                "smoke cần cả dense lẫn sparse của TRUY VẤN, không chỉ của chunk"
            )
        dense, sparse = hybrid
        query_rows.append(
            (
                query.query_id,
                {
                    "dense": np.asarray(dense, dtype=np.float32).reshape(-1).tolist(),
                    "sparse": {"indices": list(sparse.indices), "values": list(sparse.values)},
                },
            )
        )

    chunk_rows = [
        (cid, _stored_to_plain(vectors[cid], DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME))
        for cid in wanted
    ]

    dimension = session.embeddings.dimension
    packed_q = _pack(query_rows, dimension)
    packed_c = _pack(chunk_rows, dimension)

    out_meta.parent.mkdir(parents=True, exist_ok=True)
    npz_path = out_meta.with_name(out_meta.name.replace(".jsonl.gz", ".npz"))
    arrays: dict[str, np.ndarray] = {
        **{f"query_{k}": v for k, v in packed_q.items()},
        **{f"chunk_{k}": v for k, v in packed_c.items()},
    }
    np.savez_compressed(npz_path, **arrays)  # type: ignore[arg-type]

    built_at = datetime.now(UTC).isoformat()
    with gzip.open(out_meta, "wt", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                {
                    "kind": "header",
                    "version": FIXTURE_VERSION,
                    "built_at": built_at,
                    "source_collection": session.store.collection,
                    "embedding_model": session.embeddings.name,
                    "dimension": dimension,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        for query in picked:
            handle.write(
                json.dumps(
                    {
                        "kind": "query",
                        "query_id": query.query_id,
                        "query": query.query,
                        "category": str(query.category),
                        "lang": str(query.lang),
                        "relevant_chunk_ids": sorted(query.relevant_chunk_ids),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        for cid in wanted:
            chunk = chunks[cid]
            row = chunk.model_dump(mode="json", exclude_none=False)
            handle.write(json.dumps({"kind": "chunk", **row}, ensure_ascii=False) + "\n")

    sizes = {p.name: p.stat().st_size for p in (out_meta, npz_path)}
    logger.info("đã ghi %s", sizes)
    for name, size in sizes.items():
        if size > 1_000_000:
            logger.warning(
                "%s = %.0f KB — `pre-commit` chặn file > 1024 KB, giảm "
                "`--distractors-per-doc` hoặc `--n-queries`",
                name,
                size / 1024,
            )
    return {"queries": len(picked), "chunks": len(wanted), "sizes": sizes, "built_at": built_at}


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI mỏng
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(
        prog="python -m pipeline.eval.smoke_fixture",
        description="Dựng fixture đóng băng cho smoke eval của CI (W5-09).",
    )
    parser.add_argument(
        "--index-config", type=Path, default=Path("configs/indexing/bgem3-contextual.yaml")
    )
    parser.add_argument("--out", type=Path, default=Path("data/eval/smoke/smoke_v1.jsonl.gz"))
    parser.add_argument("--n-queries", type=int, default=DEFAULT_N_QUERIES)
    parser.add_argument("--distractors-per-doc", type=int, default=DEFAULT_DISTRACTORS_PER_DOC)
    args = parser.parse_args(argv)
    summary = build_fixture(
        args.index_config,
        args.out,
        n_queries=args.n_queries,
        distractors_per_doc=args.distractors_per_doc,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
