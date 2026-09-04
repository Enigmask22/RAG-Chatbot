"""Sinh câu trả lời cho cả golden set **qua HTTP**, ghi ra một artifact bất biến. `W5-01`.

## ⭐⭐ Vì sao đo qua HTTP chứ không dựng lại đường sinh trong pipeline

`tests/unit/test_architecture_boundaries.py` cấm `pipeline` import `serving`. Nên
có đúng hai cách lấy câu trả lời để chấm:

1. Dựng lại đường sinh bên trong `pipeline` bằng `rag_core` — prompt, bọc ngữ
   cảnh, luật trích nguồn, xác minh citation.
2. Gọi chính hệ thống đang chạy.

Cách 1 nghe gọn và **sai theo đúng kiểu tốn nhiều tháng để phát hiện**: đường
sinh thật gồm bộ định tuyến `W4-07`, bản viết lại câu hỏi, mốc nonce `W4-12`, và
`ChatTurn.prompt()` — tất cả đều sống ở `serving/`. Một bản dựng lại sẽ trôi khỏi
bản thật ngay lần sửa prompt đầu tiên, và lúc ấy báo cáo faithfulness nói về một
prompt không ai phục vụ. Đúng họ với lỗi mà phép kiểm danh tính bundle (`TD-38`)
sinh ra để chặn.

Nên: **eval nói chuyện với hệ thống qua đúng cái cổng người dùng dùng.** Giá phải
trả, nói thẳng: eval sinh giờ cần stack đang chạy — một phụ thuộc mà eval truy hồi
(`retrieval_eval.py`) không có.

## ⭐ Tách "chạy" khỏi "chấm", như `evaluate_run` đã tách

Module này **chỉ sinh và ghi**. Nó không tính một metric nào. Mọi phép chấm đọc
lại file JSONL nó ghi ra (`generation_metrics.py`), nên:

* chấm lại bằng rubric mới **không tốn một đồng** và không cần stack;
* con số trong báo cáo truy được về đúng dòng đã sinh ra nó;
* lần chạy tốn tiền xảy ra đúng một lần.

## ⚠️ Ba thứ lần chạy này KHÔNG đo được

* **Độ trễ.** Harness gọi song song, nên mọi con số thời gian ở đây là thời gian
  dưới tải tự tạo. Số p95 thật nằm ở `W4-13` (`probes/w4-13-e2e-latency.json`),
  đo tuần tự.
* **Cache trúng là dữ liệu, không phải lỗi.** `W4-10` có thể phục vụ một câu golden
  bằng câu trả lời của câu golden khác đủ giống. Ghi lại `cache.hit` + `similarity`
  + `matched_question` cho từng dòng; báo cáo phải công bố con số ấy chứ không
  lặng lẽ chấm một câu trả lời của câu hỏi khác.
* **Nhánh định tuyến.** Một câu golden bị `W4-07` xếp vào `clarify` thì **không hề
  đi qua truy hồi**. Đó là một chế độ hỏng thật và nó phải hiện ra trong bảng phân
  bố nhánh, không phải trốn trong một điểm faithfulness thấp.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .golden import GoldenQuery, load_golden_set

logger = logging.getLogger(__name__)

__all__ = [
    "AnswerRecord",
    "AnswerRun",
    "load_answer_run",
    "load_chunk_sidecar",
    "parse_sse",
    "run_answers",
    "write_answer_run",
    "write_chunk_sidecar",
]


@dataclass
class AnswerRecord:
    """Một lượt hỏi–đáp đã chạy thật, đủ để chấm lại mà không cần gọi lại."""

    query_id: str
    query: str
    category: str
    lang: str

    answer: str
    """Text người dùng thấy — block `CITATIONS:` đã bị `CitationHoldback` cắt."""

    route: str
    """`retrieve` · `no_retrieval` · `clarify`. Nhánh `clarify` không truy hồi gì."""

    rewritten: str | None
    prompt_spec: str | None
    bundle_version: str
    model: str | None
    finish_reason: str

    sources: list[dict[str, Any]] = field(default_factory=list)
    """Cái **đã đưa cho model**, theo đúng thứ tự `[n]` trong prompt."""

    citations: list[dict[str, Any]] = field(default_factory=list)
    """Cái model **tuyên bố** đã dùng, kèm `verified` của `W4-09`."""

    citation_block: str = "absent"
    invalid_ns: list[int] = field(default_factory=list)
    cache_hit: bool = False
    cache_similarity: float | None = None
    cache_matched_question: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    wall_ms: float = 0.0
    error: str | None = None

    @property
    def source_chunk_ids(self) -> list[str]:
        return [str(s["chunk_id"]) for s in self.sources]


@dataclass
class AnswerRun:
    """Artifact bất biến của một lần chạy. Header mang mọi thứ cần để đọc lại."""

    name: str
    created_at: str
    golden_path: str
    golden_sha256: str
    base_url: str
    bundle_versions: list[str]
    prompt_specs: list[str]
    models: list[str]
    n_queries: int
    cost_usd: float
    records: list[AnswerRecord]

    def header(self) -> dict[str, Any]:
        data = {k: v for k, v in asdict(self).items() if k != "records"}
        return data


# ------------------------------------------------------------------------ SSE


def parse_sse(raw: str) -> dict[str, Any]:
    """Gom các khung SSE của một lượt thành một dict.

    Viết tay thay vì dùng thư viện vì hợp đồng ở đây rất nhỏ và **cố định** —
    `meta` → `sources` → `delta`* → (`citations`) → (`done` | `error`) — và một
    thư viện SSE tổng quát sẽ che mất đúng thứ cần soi: khung `done` có mặt hay
    không. Một dòng `delta` dừng lại **không** nói được điều gì cả; nó giống hệt
    nhau khi model nói xong và khi kết nối đứt.
    """
    frames: dict[str, Any] = {"delta": [], "done": None, "error": None}
    event: str | None = None
    for line in raw.splitlines():
        if line.startswith("event: "):
            event = line[7:].strip()
        elif line.startswith("data: ") and event:
            payload = json.loads(line[6:])
            if event == "delta":
                frames["delta"].append(payload.get("text", ""))
            else:
                frames[event] = payload
            event = None
    return frames


def _record(query: GoldenQuery, frames: dict[str, Any], wall_ms: float) -> AnswerRecord:
    meta = frames.get("meta") or {}
    done = frames.get("done") or {}
    citations = frames.get("citations") or {}
    cache = meta.get("cache") or {}
    error = frames.get("error")
    return AnswerRecord(
        query_id=query.query_id,
        query=query.query,
        category=query.category.value,
        lang=query.lang.value,
        answer="".join(frames["delta"]),
        route=str(meta.get("route", "")),
        rewritten=meta.get("rewritten"),
        prompt_spec=meta.get("prompt"),
        bundle_version=str(meta.get("bundle_version", "")),
        model=meta.get("model"),
        # Không có khung `done` nghĩa là lượt ấy **chưa kết thúc**, không phải
        # kết thúc bình thường. Ghi rõ để phép chấm không coi một câu trả lời
        # bị cắt ngang là một câu trả lời ngắn.
        finish_reason=str(done.get("finish_reason", "no_done_frame")),
        sources=list((frames.get("sources") or {}).get("sources", [])),
        citations=list(citations.get("citations", [])),
        citation_block=str(citations.get("block", "absent")),
        invalid_ns=list(citations.get("invalid_ns", [])),
        cache_hit=bool(cache.get("hit", False)),
        cache_similarity=cache.get("similarity"),
        cache_matched_question=cache.get("matched_question"),
        usage=dict(done.get("usage") or {}),
        wall_ms=round(wall_ms, 1),
        error=str(error.get("detail")) if error else None,
    )


# ------------------------------------------------------------------- chạy thật


def _ask(
    client: Any,
    query: GoldenQuery,
    *,
    key: str,
    top_k: int,
    max_attempts: int = 4,
) -> AnswerRecord:
    """Một lượt. Thử lại **chỉ** khi bị chặn hạn mức, và theo đúng `Retry-After`.

    429 là câu trả lời hợp lệ của một hệ thống đang tự bảo vệ, không phải một
    lỗi — ngủ đúng số giây nó bảo rồi hỏi lại. Mọi mã lỗi khác thì ghi vào dòng
    và đi tiếp: một câu hỏng không được làm hỏng cả lần chạy 242 câu.
    """
    body = {"message": query.query, "top_k": top_k}
    headers = {"Authorization": f"Bearer {key}"}
    for attempt in range(max_attempts):
        started = time.perf_counter()
        response = client.post("/chat", headers=headers, json=body)
        wall_ms = (time.perf_counter() - started) * 1000.0
        if response.status_code == 429 and attempt < max_attempts - 1:
            wait = float(response.headers.get("Retry-After", "5"))
            logger.info("429 ở %s, đợi %.0fs", query.query_id, wait)
            time.sleep(wait + 0.5)
            continue
        if response.status_code != 200:
            return AnswerRecord(
                query_id=query.query_id,
                query=query.query,
                category=query.category.value,
                lang=query.lang.value,
                answer="",
                route="",
                rewritten=None,
                prompt_spec=None,
                bundle_version="",
                model=None,
                finish_reason="http_error",
                wall_ms=round(wall_ms, 1),
                error=f"HTTP {response.status_code}: {response.text[:200]}",
            )
        return _record(query, parse_sse(response.text), wall_ms)
    raise RuntimeError("không tới được đây")  # pragma: no cover


def run_answers(
    queries: Sequence[GoldenQuery],
    *,
    base_url: str,
    key: str,
    top_k: int = 5,
    concurrency: int = 3,
    timeout: float = 300.0,
    progress: bool = False,
) -> list[AnswerRecord]:
    """Chạy cả tập, **giữ nguyên thứ tự đầu vào**.

    `concurrency=3` là mặc định có lý do: khoá API mặc định cho 60 request/phút
    (`serving.core.auth`), và một lượt mất ~4 s — 3 luồng cho ~45 req/phút, dưới
    trần mà vẫn rút 242 câu từ ~20 phút xuống ~6 phút.
    """
    import httpx

    done_count = 0

    with httpx.Client(base_url=base_url, timeout=timeout) as client:

        def one(query: GoldenQuery) -> AnswerRecord:
            nonlocal done_count
            record = _ask(client, query, key=key, top_k=top_k)
            done_count += 1
            if progress:
                sys.stdout.write(
                    f"[{done_count:>3}/{len(queries)}] {record.query_id} "
                    f"{record.route or '-':<12} {record.finish_reason:<10} "
                    f"cite {len(record.citations):>2} "
                    f"{'CACHE' if record.cache_hit else ''}\n"
                )
                sys.stdout.flush()
            return record

        if concurrency == 1:
            return [one(q) for q in queries]
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            return list(pool.map(one, queries))


# ------------------------------------------------------------------- artifact


def write_answer_run(path: str | Path, run: AnswerRun) -> None:
    """JSONL: dòng đầu là header, mỗi dòng sau là một bản ghi.

    Cùng khuôn với `runs/*-per-query.jsonl` của `retrieval_eval`, và cùng lý do:
    đọc từng dòng được, `grep` được, và thêm field mới không phá file cũ.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps({"_header": run.header()}, ensure_ascii=False) + "\n")
        for record in run.records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def load_answer_run(path: str | Path) -> AnswerRun:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"{path}: file rỗng")
    header = json.loads(lines[0]).get("_header")
    if header is None:
        raise ValueError(f"{path}: dòng đầu phải là header `_header`")
    records = [AnswerRecord(**json.loads(line)) for line in lines[1:] if line.strip()]
    return AnswerRun(**header, records=records)


# ------------------------------------------------------- nội dung chunk (sidecar)


def sidecar_path(run_path: str | Path) -> Path:
    """`runs/x.jsonl` → `runs/x-chunks.jsonl`. Cùng prefix, cùng thư mục — quy
    tắc §3 của `reports/README.md`."""
    path = Path(run_path)
    return path.with_name(f"{path.stem}-chunks.jsonl")


def write_chunk_sidecar(
    run_path: str | Path,
    *,
    collection: str,
    qdrant_url: str,
    api_key: str | None = None,
) -> tuple[Path, int, list[str]]:
    """Kéo **nội dung** của mọi chunk mà lần chạy đã chạm, ghi thành file kề bên.

    ## ⭐ Vì sao phải có, và vì sao là file riêng

    Khung `sources` của SSE cố ý **không** mang nội dung chunk — gửi vài KB văn
    bản cho client ở mỗi request là lãng phí, và đó là quyết định đúng của
    `W4-06`. Nhưng judge cần đúng thứ đó để chấm faithfulness.

    Lấy từ index lúc chấm thì con số phụ thuộc vào một index **có thể đã đổi**,
    và lúc ấy "chấm lại miễn phí" là một lời nói dối. Nên nội dung được đông
    cứng ngay sau lần chạy, cùng lúc với câu trả lời.

    File **riêng** chứ không nhét vào từng dòng: một chunk hay được 5–10 truy
    vấn cùng dùng, nhân bản nó ra là làm artifact phình lên vô ích và mở đường
    cho hai bản khác nhau của cùng một chunk trong cùng một file.

    Trả `(đường dẫn, số chunk lấy được, danh sách id THIẾU)`. Id thiếu là tín
    hiệu thật: index đã đổi kể từ lần chạy, và mọi điểm faithfulness tính sau đó
    sẽ thiếu bằng chứng cho đúng những chunk ấy.
    """
    from rag_core.embedding import HashingEmbeddingProvider
    from rag_core.retrieval import QdrantDenseRetriever

    run = load_answer_run(run_path)
    wanted = sorted({cid for record in run.records for cid in record.source_chunk_ids})
    # `fetch_chunks` lấy theo id, **không** embed gì — nên provider ở đây chỉ để
    # thoả chữ ký constructor. Dùng bản hashing (không tải model 2,2 GB) thay vì
    # BGE-M3 là tiết kiệm ~40 giây và ~3 GB VRAM cho một tham số không được dùng.
    store = QdrantDenseRetriever(
        HashingEmbeddingProvider(dimension=8),
        collection=collection,
        url=qdrant_url,
        api_key=api_key,
    )
    found = store.fetch_chunks(wanted)
    missing = [cid for cid in wanted if cid not in found]

    target = sidecar_path(run_path)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                {
                    "_header": {
                        "run": run.name,
                        "collection": collection,
                        "n_requested": len(wanted),
                        "n_found": len(found),
                        "missing": missing,
                    }
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        for chunk_id in wanted:
            chunk = found.get(chunk_id)
            if chunk is None:
                continue
            handle.write(
                json.dumps(
                    {
                        "chunk_id": chunk_id,
                        "doc_id": chunk.doc_id,
                        "content": chunk.content,
                        "section_path": list(chunk.section_path),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return target, len(found), missing


def load_chunk_sidecar(run_path: str | Path) -> dict[str, str]:
    """`chunk_id` → nội dung. Rỗng nếu chưa có sidecar."""
    path = sidecar_path(run_path)
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        if line.strip():
            row = json.loads(line)
            out[row["chunk_id"]] = row["content"]
    return out


def summarise(records: Sequence[AnswerRecord]) -> dict[str, Any]:
    """Những con số phải nhìn **trước** khi tin bất kỳ metric nào tính từ file này."""
    routes: dict[str, int] = {}
    finishes: dict[str, int] = {}
    for record in records:
        routes[record.route or "?"] = routes.get(record.route or "?", 0) + 1
        finishes[record.finish_reason] = finishes.get(record.finish_reason, 0) + 1
    return {
        "n": len(records),
        "routes": dict(sorted(routes.items())),
        "finish_reasons": dict(sorted(finishes.items())),
        "cache_hits": sum(1 for r in records if r.cache_hit),
        "errors": sum(1 for r in records if r.error),
        "empty_answers": sum(1 for r in records if not r.answer.strip()),
        "no_citation_block": sum(1 for r in records if r.citation_block == "absent"),
        "invalid_citation_block": sum(1 for r in records if r.citation_block == "invalid"),
        "unverified_citations": sum(
            1 for r in records for c in r.citations if not c.get("verified")
        ),
        "total_citations": sum(len(r.citations) for r in records),
        "cost_usd": round(sum(float(r.usage.get("cost_usd", 0.0)) for r in records), 6),
    }


def main(argv: Sequence[str] | None = None) -> int:
    import hashlib

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", type=Path, default=Path("data/golden/golden_v1.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--name", default="")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="0 = cả tập")
    parser.add_argument(
        "--chunks-collection",
        default="",
        help="Kéo nội dung chunk từ collection này ra file sidecar sau khi chạy xong",
    )
    parser.add_argument("--qdrant-url", default="http://127.0.0.1:6333")
    parser.add_argument(
        "--only-chunks",
        action="store_true",
        help="Bỏ qua bước sinh, chỉ dựng lại sidecar cho file --out đã có",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.only_chunks:
        if not args.chunks_collection:
            parser.error("--only-chunks cần --chunks-collection")
        path, found, missing = write_chunk_sidecar(
            args.out, collection=args.chunks_collection, qdrant_url=args.qdrant_url
        )
        sys.stdout.write(f"{path}: {found} chunk, thiếu {len(missing)}\n")
        if missing:
            sys.stdout.write(f"THIẾU (index đã đổi?): {missing[:10]}\n")
        return 0

    queries = load_golden_set(args.golden)
    if args.limit:
        queries = queries[: args.limit]
    key = args.key_file.read_text(encoding="utf-8").strip()

    records = run_answers(
        queries,
        base_url=args.base_url,
        key=key,
        top_k=args.top_k,
        concurrency=args.concurrency,
        progress=True,
    )
    run = AnswerRun(
        name=args.name or args.out.stem,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        golden_path=str(args.golden),
        golden_sha256=hashlib.sha256(args.golden.read_bytes()).hexdigest(),
        base_url=args.base_url,
        bundle_versions=sorted({r.bundle_version for r in records if r.bundle_version}),
        prompt_specs=sorted({r.prompt_spec for r in records if r.prompt_spec}),
        models=sorted({r.model for r in records if r.model}),
        n_queries=len(records),
        cost_usd=round(sum(float(r.usage.get("cost_usd", 0.0)) for r in records), 6),
        records=records,
    )
    write_answer_run(args.out, run)
    sys.stdout.write(json.dumps(summarise(records), ensure_ascii=False, indent=2) + "\n")
    sys.stdout.write(f"đã ghi {args.out}\n")

    if args.chunks_collection:
        path, found, missing = write_chunk_sidecar(
            args.out, collection=args.chunks_collection, qdrant_url=args.qdrant_url
        )
        sys.stdout.write(f"đã ghi {path}: {found} chunk, thiếu {len(missing)}\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
