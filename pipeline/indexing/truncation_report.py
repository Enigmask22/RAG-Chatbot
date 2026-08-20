"""Đo `TD-11` trên corpus thật, và hiệu chuẩn `chunk_size` từ số đo đó.

Chạy: `make truncation BUNDLE=baseline` (không cần Qdrant, không cần build lại
index — chỉ chunk rồi đếm token, và chunking đã có cache).

Hai câu hỏi mà script này trả lời:

1. **Cấu hình hiện tại mất bao nhiêu nội dung?** `% chunk bị cắt` và, quan trọng
   hơn, `% token không tới được vector`.
2. **`chunk_size` nào thì không mất gì?** `chunk_size` tính bằng **ký tự** còn
   giới hạn model tính bằng **token**, và tỉ lệ quy đổi phụ thuộc ngôn ngữ +
   tokenizer. Nên nó phải được **đo**, không đoán. Đây là cùng một nguyên tắc
   với ngưỡng triage 0,5797 ở `W1-11`: hằng số gõ tay là hằng số của một corpus
   khác.

Chia theo ngôn ngữ vì tokenizer của baseline là PhoBERT — nó học trên tiếng
Việt, nên xé chữ tiếng Anh thành nhiều mảnh hơn. Nếu đúng thế thì tài liệu EN
bị cắt nặng hơn VI, và điều đó **cộng dồn** với việc nhóm `cross_lingual` đang
có recall@5 = 0: câu hỏi tiếng Anh, tài liệu tiếng Việt, và bên tiếng Anh còn
mất thêm phần đuôi.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

from rag_core.embedding.truncation import TruncationStats, token_stats
from rag_core.schemas import Chunk, Document

from .config import IndexConfig, load_index_config
from .corpus_loader import load_documents

__all__ = ["TruncationReport", "main", "measure_config"]

logger = logging.getLogger("pipeline.indexing.truncation")

#: Số token mà `[CLS]`/`[SEP]` chiếm sẵn trong cửa sổ.
SPECIAL_TOKENS = 2


class DocumentLoss(NamedTuple):
    """Một dòng của bảng "tài liệu mất nhiều nhất".

    Là NamedTuple chứ không phải `dict[str, object]`: mypy strict không kiểm
    được kiểu bên trong dict như thế, và ở `W1-11` chính chỗ đó đã cho một
    `float(...)` trên `object` lọt qua review.
    """

    doc_id: str
    lang: str
    n_chunks: int
    tokens_lost_ratio: float
    token_max: int


@dataclass
class TruncationReport:
    config_name: str
    embedding_model: str
    limit: int
    chunk_size: int
    neighbor_context_chars: int
    overall: dict[str, float]
    by_language: dict[str, dict[str, float]] = field(default_factory=dict)
    worst_documents: list[DocumentLoss] = field(default_factory=list)
    suggested_chunk_size: dict[str, int] = field(default_factory=dict)
    created_at: str = ""

    def to_json(self) -> str:
        payload = {
            "config_name": self.config_name,
            "embedding_model": self.embedding_model,
            "limit": self.limit,
            "chunk_size": self.chunk_size,
            "neighbor_context_chars": self.neighbor_context_chars,
            "overall": self.overall,
            "by_language": self.by_language,
            "worst_documents": [row._asdict() for row in self.worst_documents],
            "suggested_chunk_size": self.suggested_chunk_size,
            "created_at": self.created_at,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def log_summary(self) -> None:
        logger.info("─" * 68)
        logger.info("TD-11 · config `%s` · model %s", self.config_name, self.embedding_model)
        logger.info(
            "  cửa sổ model     %d token (còn %d cho nội dung sau special token)",
            self.limit,
            self.limit - SPECIAL_TOKENS,
        )
        logger.info(
            "  chunking         chunk_size %d ký tự · neighbor_context %d",
            self.chunk_size,
            self.neighbor_context_chars,
        )
        o = self.overall
        logger.info(
            "  TỔNG             %d/%d chunk bị cắt (%.1f%%) · mất %.1f%% token",
            int(o["n_truncated"]),
            int(o["n_texts"]),
            100 * o["truncated_ratio"],
            100 * o["tokens_lost_ratio"],
        )
        logger.info(
            "  token/chunk      p50 %d · p95 %d · max %d",
            int(o["token_p50"]),
            int(o["token_p95"]),
            int(o["token_max"]),
        )
        for lang, stats in sorted(self.by_language.items()):
            logger.info(
                "  %-4s             %d chunk · cắt %.1f%% · mất %.1f%% token · %.3f token/ký tự",
                lang,
                int(stats["n_texts"]),
                100 * stats["truncated_ratio"],
                100 * stats["tokens_lost_ratio"],
                stats["tokens_per_char"],
            )
        if self.suggested_chunk_size:
            logger.info("  chunk_size gợi ý (để ~95%% chunk nằm trong cửa sổ):")
            for key, value in sorted(self.suggested_chunk_size.items()):
                logger.info("    %-22s %d ký tự", key, value)
        if self.worst_documents:
            logger.info("  tài liệu mất nhiều token nhất:")
            for row in self.worst_documents:
                logger.info(
                    "    %-42s %s · mất %.1f%% · dài nhất %d token",
                    row.doc_id[:42],
                    row.lang,
                    100 * row.tokens_lost_ratio,
                    row.token_max,
                )
        logger.info("─" * 68)


def _suggest_chunk_size(stats: TruncationStats, *, neighbor_context_chars: int) -> int:
    """`chunk_size` để ~95% chunk nằm trong cửa sổ, theo mật độ đã đo.

    Hai chỗ dễ sai, và lần đầu viết hàm này tôi sai chỗ thứ nhất:

    1. **Phải dùng phân vị thấp của mật độ, không phải trung bình.** Trung bình
       trả lời "chunk *trung bình* vừa khít cửa sổ" — tức ngưỡng mà một nửa số
       chunk vẫn bị cắt. Trên corpus này nó cho ra 946 ký tự, trong khi ở 1000
       ký tự đã có 56,9% chunk bị cắt. Con số vô lý nhưng trông hợp lý.
    2. **`neighbor_context_chars` phải trừ hai lần** — ngữ cảnh dán cả trước và
       sau, nên một chunk `chunk_size` ký tự đi vào model với tối đa
       `chunk_size + 2*neighbor` ký tự. Ở baseline là 200 ký tự mỗi chunk.

    Và kể cả đúng cả hai, đây vẫn chỉ là **điểm khởi đầu để thử**: quan hệ ký
    tự↔token không tuyến tính hoàn toàn. Câu trả lời thật là build lại rồi đo lại.
    """
    budget = stats.char_budget(special_tokens=SPECIAL_TOKENS)
    return max(0, budget - 2 * neighbor_context_chars)


def measure_config(
    config: IndexConfig,
    *,
    verify_hash: bool = True,
    top_documents: int = 5,
) -> TruncationReport:
    documents = load_documents(
        config.manifest_path,
        config.corpus_dir,
        languages=config.languages,
        doc_types=config.doc_types,
        max_documents=config.max_documents,
        verify_hash=verify_hash,
    )
    embeddings = config.build_embeddings()
    limit = embeddings.max_sequence_tokens
    if limit is None:
        raise RuntimeError(
            f"Provider {embeddings.name!r} không cho biết `max_sequence_tokens`, "
            "không đo được TD-11. Đừng đọc điều này thành 'không bị cắt'."
        )

    chunker = config.build_chunker(embeddings)
    chunker.prepare(len(documents))
    logger.info(
        "Đo %d tài liệu · model %s · giới hạn %d token", len(documents), embeddings.name, limit
    )

    all_counts: list[int] = []
    all_lengths: list[int] = []
    per_lang: dict[str, list[int]] = defaultdict(list)
    per_lang_chars: dict[str, list[int]] = defaultdict(list)
    per_doc: list[tuple[str, str, TruncationStats]] = []

    for doc in documents:
        chunks: Sequence[Chunk] = chunker.chunk([doc])
        if not chunks:
            continue
        texts = [c.content for c in chunks]
        counts = embeddings.count_tokens(texts)
        if counts is None:  # pragma: no cover - provider có limit mà không đếm được
            raise RuntimeError(f"Provider {embeddings.name!r} có giới hạn nhưng không đếm token")
        lengths = [len(t) for t in texts]
        lang = _document_language(doc)

        all_counts.extend(counts)
        all_lengths.extend(lengths)
        per_lang[lang].extend(counts)
        per_lang_chars[lang].extend(lengths)
        per_doc.append((doc.doc_id, lang, token_stats(counts, limit=limit, chars=lengths)))

    overall = token_stats(all_counts, limit=limit, chars=all_lengths)
    per_lang_stats = {
        lang: token_stats(counts, limit=limit, chars=per_lang_chars[lang])
        for lang, counts in per_lang.items()
    }
    by_language = {lang: stats.as_dict() for lang, stats in per_lang_stats.items()}

    neighbor = config.chunking.neighbor_context_chars
    suggested = {"overall": _suggest_chunk_size(overall, neighbor_context_chars=neighbor)}
    for lang, stats in per_lang_stats.items():
        suggested[lang] = _suggest_chunk_size(stats, neighbor_context_chars=neighbor)
    # Cấu hình dùng cho cả corpus phải chịu được ngôn ngữ tệ nhất, không phải
    # trung bình — lấy trung bình là bảo đảm ngôn ngữ kia vẫn bị cắt.
    per_language_values = [v for k, v in suggested.items() if k != "overall"]
    if per_language_values:
        suggested["safe_for_all_languages"] = min(per_language_values)

    worst = sorted(per_doc, key=lambda row: row[2].tokens_lost_ratio, reverse=True)[:top_documents]

    report = TruncationReport(
        config_name=config.name,
        embedding_model=config.embedding_model,
        limit=limit,
        chunk_size=config.chunking.chunk_size,
        neighbor_context_chars=neighbor,
        overall=overall.as_dict(),
        by_language=by_language,
        worst_documents=[
            DocumentLoss(
                doc_id=doc_id,
                lang=lang,
                n_chunks=stats.n_texts,
                tokens_lost_ratio=round(stats.tokens_lost_ratio, 4),
                token_max=stats.token_max,
            )
            for doc_id, lang, stats in worst
        ],
        suggested_chunk_size=suggested,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    report.log_summary()
    return report


def _document_language(doc: Document) -> str:
    return str(doc.metadata.lang.value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Đo phần text bị model embedding cắt (TD-11)")
    parser.add_argument("--config", type=Path, default=Path("configs/indexing/baseline.yaml"))
    parser.add_argument("--report", type=Path, help="Ghi báo cáo JSON ra file")
    parser.add_argument("--no-verify-hash", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    config = load_index_config(args.config)
    report = measure_config(config, verify_hash=not args.no_verify_hash)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report.to_json(), encoding="utf-8")
        logger.info("Đã ghi báo cáo %s", args.report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
