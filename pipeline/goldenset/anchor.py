"""Neo nhãn nháp golden set vào văn bản gốc, thay vì vào `chunk_id`.

Chạy một lần để chuyển `relevant_chunk_ids` (chỉ đúng với một cấu hình chunking)
thành `relevant_spans` (bền qua mọi cấu hình) — xem `TD-12`.

## Không cần Qdrant

Span được tính bằng cách **chunk lại corpus** với đúng config đã sinh nháp, chứ
không đọc từ index. Làm được vì `chunk_id` là xác định (`f"{doc_id}::{index:05d}"`)
và nội dung chunk cũng xác định — có test canh trên corpus thật rằng refactor
`W1-11` không đổi một byte nào. Nên bảng `chunk_id → span` dựng lại được bất cứ
lúc nào từ corpus, và corpus thì có sha256 trong manifest.

## Thu span về đúng câu trích dẫn

Nháp giữ `supporting_quotes` đã được đối chiếu với chunk. Nếu tìm được câu trích
dẫn đó trong văn bản gốc thì span thu về đúng nó (~150 ký tự) thay vì cả chunk
(~1000 ký tự). Chính xác hơn hẳn: một span rộng bằng cả chunk sẽ khớp cả những
chunk mới chỉ chứa phần *không* liên quan của chunk cũ.

Tìm kiếm phải **bỏ qua khác biệt khoảng trắng**: trích dẫn được đối chiếu với
`chunk.content`, mà content đã bị splitter đổi khoảng trắng (nối bằng `"\\n"` hoặc
`" "` bất kể nguyên bản). So thô bằng `str.find` sẽ trượt phần lớn.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from pipeline.eval.golden import QueryCategory
from pipeline.goldenset.schema import GoldenDraft
from rag_core.schemas import Chunk, TextSpan

__all__ = [
    "AnchorReport",
    "anchor_drafts",
    "build_chunk_index",
    "find_ignoring_whitespace",
]

logger = logging.getLogger(__name__)

_WS = re.compile(r"\s+")
_MIN_QUOTE_CHARS = 16
"""Trích dẫn ngắn hơn thì không thu span theo nó.

Một chuỗi vài ký tự kiểu `"GDP"` xuất hiện hàng chục lần trong cùng một chunk;
thu span về lần khớp đầu tiên sẽ neo bằng chứng vào câu sai. Thà giữ span rộng
bằng cả chunk — rộng quá thì khớp thừa, còn sai chỗ thì khớp vào đoạn không liên
quan.

16 chứ không phải một con số lớn hơn: `"GDP đạt 6,7% năm 2017."` chỉ 22 ký tự
nhưng rất đặc trưng, và trích dẫn số liệu ngắn như thế rất phổ biến trong corpus
báo cáo kinh tế. Ngưỡng cao sẽ đẩy phần lớn chúng về span rộng bằng cả chunk.
Rủi ro còn lại có chặn: việc tìm kiếm bị giới hạn trong **vùng của chunk**, nên
kể cả khớp sai câu thì span vẫn nằm trong chunk gốc.
"""


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Gom khoảng trắng, đồng thời trả bảng tra offset về văn bản gốc.

    `mapping[i]` là chỉ số trong `text` của ký tự thứ `i` của chuỗi đã chuẩn hoá.
    Không có bảng này thì tìm được vị trí trong bản chuẩn hoá cũng vô dụng — span
    phải tính theo văn bản gốc, vì đó mới là thứ bất biến.
    """
    out: list[str] = []
    mapping: list[int] = []
    prev_space = False
    for i, ch in enumerate(text):
        if ch.isspace():
            if not prev_space and out:
                out.append(" ")
                mapping.append(i)
            prev_space = True
            continue
        prev_space = False
        out.append(ch)
        mapping.append(i)
    return "".join(out), mapping


def find_ignoring_whitespace(
    haystack: str, needle: str, *, region: tuple[int, int] | None = None
) -> tuple[int, int] | None:
    """Tìm `needle` trong `haystack`, coi mọi chuỗi khoảng trắng là tương đương.

    Args:
        haystack: văn bản gốc.
        needle: chuỗi cần tìm.
        region: chỉ tìm trong `[start, end)`. Truyền vùng của chunk vào để trích
            dẫn không khớp vào một chỗ trùng chữ ở tài liệu khác chỗ.

    Returns:
        `(start, end)` theo offset của `haystack`, hoặc `None` nếu không thấy.
    """
    lo, hi = region if region is not None else (0, len(haystack))
    lo = max(0, lo)
    hi = min(len(haystack), hi)
    if lo >= hi:
        return None

    flat_needle = _WS.sub(" ", needle).strip()
    if not flat_needle:
        return None

    window = haystack[lo:hi]
    flat, mapping = _normalize_with_map(window)
    at = flat.find(flat_needle)
    if at < 0:
        return None

    start = lo + mapping[at]
    last = at + len(flat_needle) - 1
    end = lo + mapping[last] + 1
    return (start, end)


def build_chunk_index(chunks: Sequence[Chunk]) -> dict[str, Chunk]:
    """Bảng `chunk_id → Chunk`. Trùng id là lỗi, không phải chuyện bỏ qua được."""
    out: dict[str, Chunk] = {}
    for chunk in chunks:
        if chunk.chunk_id in out:
            raise ValueError(
                f"`{chunk.chunk_id}` xuất hiện hai lần — chunk_id phải là duy nhất "
                "trong một lượt chunk, nếu không thì mọi ánh xạ nhãn đều mơ hồ"
            )
        out[chunk.chunk_id] = chunk
    return out


class AnchorReport(BaseModel):
    """Kết quả neo, đủ chi tiết để biết chỗ nào còn yếu."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    drafts_total: int
    answerable: int
    anchored: int
    """Số câu có ít nhất một span."""
    spans_total: int
    narrowed_by_quote: int
    """Span thu về đúng câu trích dẫn — chính xác hơn span rộng bằng cả chunk."""
    widened_to_chunk: int
    """Span phải lấy cả chunk vì không tìm được trích dẫn trong văn bản gốc."""
    missing_chunk_ids: list[str] = Field(default_factory=list)
    unanchored_query_ids: list[str] = Field(default_factory=list)

    def log_summary(self) -> None:
        log = logging.getLogger("anchor")
        log.info(
            "Neo %d/%d câu trả lời được · %d span (%d thu theo trích dẫn · %d rộng bằng chunk)",
            self.anchored,
            self.answerable,
            self.spans_total,
            self.narrowed_by_quote,
            self.widened_to_chunk,
        )
        if self.missing_chunk_ids:
            log.warning(
                "%d chunk_id không dựng lại được từ corpus: %s",
                len(self.missing_chunk_ids),
                ", ".join(self.missing_chunk_ids[:5]),
            )
        if self.unanchored_query_ids:
            log.warning(
                "%d câu KHÔNG neo được span nào, vẫn phải dựa vào chunk_id: %s",
                len(self.unanchored_query_ids),
                ", ".join(self.unanchored_query_ids[:5]),
            )


def anchor_drafts(
    drafts: Sequence[GoldenDraft],
    chunk_index: Mapping[str, Chunk],
    doc_texts: Mapping[str, str],
) -> tuple[list[GoldenDraft], AnchorReport]:
    """Thêm `relevant_spans` vào từng nháp trả lời được.

    Args:
        drafts: tập nháp cần neo.
        chunk_index: `chunk_id → Chunk` của **đúng cấu hình đã sinh nháp**.
        doc_texts: `doc_id → Document.content`.

    Returns:
        Danh sách nháp mới (thứ tự giữ nguyên) và bản báo cáo. `relevant_chunk_ids`
        được **giữ nguyên**, không xoá: nó vẫn là nhãn đúng cho index hiện tại, và
        xoá đi thì không còn cách nào đối chiếu ánh xạ span có ra đúng tập cũ hay
        không — phép kiểm đó chính là bằng chứng cho `TD-12`.
    """
    out: list[GoldenDraft] = []
    missing: list[str] = []
    unanchored: list[str] = []
    n_narrow = n_wide = n_spans = n_answerable = n_anchored = 0

    for draft in drafts:
        q = draft.query
        if q.category is QueryCategory.UNANSWERABLE:
            out.append(draft)
            continue

        n_answerable += 1
        quotes = list(draft.provenance.supporting_quotes)
        spans: list[TextSpan] = []
        seen: set[tuple[str, int, int]] = set()

        for chunk_id in q.relevant_chunk_ids:
            chunk = chunk_index.get(chunk_id)
            if chunk is None:
                missing.append(chunk_id)
                continue
            base = chunk.span
            if base is None:
                missing.append(chunk_id)
                continue

            text = doc_texts.get(chunk.doc_id)
            found: tuple[int, int] | None = None
            if text is not None:
                for quote in quotes:
                    if len(_WS.sub(" ", quote).strip()) < _MIN_QUOTE_CHARS:
                        continue
                    found = find_ignoring_whitespace(text, quote, region=(base.start, base.end))
                    if found is not None:
                        break

            if found is not None:
                span = TextSpan(doc_id=chunk.doc_id, start=found[0], end=found[1])
                n_narrow += 1
            else:
                span = base
                n_wide += 1

            key = (span.doc_id, span.start, span.end)
            if key not in seen:
                seen.add(key)
                spans.append(span)

        n_spans += len(spans)
        if spans:
            n_anchored += 1
        else:
            unanchored.append(q.query_id)

        out.append(
            draft.model_copy(update={"query": q.model_copy(update={"relevant_spans": spans})})
        )

    report = AnchorReport(
        drafts_total=len(drafts),
        answerable=n_answerable,
        anchored=n_anchored,
        spans_total=n_spans,
        narrowed_by_quote=n_narrow,
        widened_to_chunk=n_wide,
        missing_chunk_ids=sorted(set(missing)),
        unanchored_query_ids=unanchored,
    )
    return out, report


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: `python -m pipeline.goldenset.anchor`. Ghi tại chỗ trừ khi có `--out`."""
    import argparse
    from pathlib import Path

    from pipeline.goldenset.schema import load_drafts, write_drafts
    from pipeline.indexing.config import load_index_config
    from pipeline.indexing.corpus_loader import load_documents

    parser = argparse.ArgumentParser(
        description="Neo nhãn nháp golden set vào văn bản gốc (TD-12).",
    )
    parser.add_argument("--drafts", type=Path, default=Path("data/golden/draft_v1.jsonl"))
    parser.add_argument("--index-config", type=Path, default=Path("configs/indexing/baseline.yaml"))
    parser.add_argument("--out", type=Path, default=None, help="Mặc định: ghi đè --drafts")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    drafts = load_drafts(args.drafts)
    logger.info("Đọc %d nháp", len(drafts))

    cfg = load_index_config(args.index_config)
    docs = load_documents(cfg.manifest_path, cfg.corpus_dir, verify_hash=True)
    logger.info("Chunk lại %d tài liệu bằng %s", len(docs), args.index_config)

    chunker = cfg.build_chunker(cfg.build_embeddings())
    chunker.prepare(len(docs))
    chunks = chunker.chunk(docs)
    logger.info("%d chunk", len(chunks))

    anchored, report = anchor_drafts(
        drafts,
        build_chunk_index(chunks),
        {d.doc_id: d.content for d in docs},
    )
    report.log_summary()

    target = args.out or args.drafts
    write_drafts(target, anchored)
    logger.info("Đã ghi %s", target)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(report.model_dump_json(indent=2), encoding="utf-8")
        logger.info("Đã ghi %s", args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
