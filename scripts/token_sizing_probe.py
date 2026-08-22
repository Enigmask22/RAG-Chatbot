"""Đo `chunk_size` tính bằng **ký tự** so với tính bằng **token**, trên corpus thật.

Ba câu hỏi:

1. Một chunk `chunk_size=1000` ký tự là bao nhiêu token — và con số đó có giống
   nhau giữa hai model, giữa hai ngôn ngữ không?
2. Với model đang dùng, có chunk nào vượt cửa sổ không? (DoD `W3-06`)
3. Bật `size_unit="tokens"` thì bộ chunk đổi thế nào, và tốn thêm bao nhiêu?

Chỉ nạp **tokenizer**, không nạp trọng số: `count_tokens` của
`HuggingFaceEmbeddingProvider` cũng chỉ dùng tokenizer, nên con số giống hệt mà
không tốn 2,2 GB và không cần GPU.

Dùng:

    uv run python scripts/token_sizing_probe.py
    uv run python scripts/token_sizing_probe.py --model BAAI/bge-m3 --tokens 256
"""

from __future__ import annotations

import argparse
import csv
import logging
import statistics
import sys
import time
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))

from rag_core.chunking import ChunkingConfig, ChunkingStrategy, build_chunker
from rag_core.schemas import Chunk, Document, DocumentMetadata

logger = logging.getLogger("token_sizing_probe")

MANIFEST = Path("data/corpus_manifest.csv")
CORPUS = Path("data/corpus")

MODELS = {
    "BAAI/bge-m3": 8192,
    "bkai-foundation-models/vietnamese-bi-encoder": 256,
}

# Khối `chunking` của `configs/indexing/baseline.yaml` và `bgem3.yaml` — hai file
# đó dùng chung đúng bộ số này, nên đây là bộ chunk mà `W1-13` và `W2-01` đã index.
BASELINE = {
    "strategy": ChunkingStrategy.HYBRID,
    "chunk_size": 1000,
    "chunk_overlap": 100,
    "min_chunk_size": 200,
    "max_chunk_size": 1500,
    "neighbor_context_chars": 100,
}


class TokenizerCounter:
    """Bọc `AutoTokenizer` thành `TokenCounter`, không nạp trọng số model."""

    def __init__(self, name: str, limit: int) -> None:
        from transformers import AutoTokenizer

        self.name = name
        self._limit = limit
        self._tokenizer = AutoTokenizer.from_pretrained(name)

    @property
    def max_sequence_tokens(self) -> int | None:
        return self._limit

    def count_tokens(self, texts: Sequence[str]) -> list[int] | None:
        if not texts:
            return []
        encoded = self._tokenizer(list(texts), truncation=False)["input_ids"]
        return [len(ids) for ids in encoded]


def load_corpus() -> list[tuple[str, Document]]:
    """Đọc **byte** rồi decode, đúng như `pipeline/indexing/corpus_loader.py`.

    ⚠️ `Path.read_text()` trên Windows chuẩn hoá CRLF → LF, tức đo trên một
    corpus **khác** cái mà pipeline index: lệch 208.941 ký tự (1,5%) và 18.038
    chunk thay vì 15.814. Một phép đo trông rất bình thường trên một corpus
    không tồn tại.
    """
    rows = list(csv.DictReader(MANIFEST.open(encoding="utf-8")))
    out: list[tuple[str, Document]] = []
    for row in rows:
        text = (CORPUS / row["relative_path"]).read_bytes().decode("utf-8")
        out.append(
            (
                row["lang"],
                Document(
                    doc_id=row["doc_id"],
                    content=text,
                    metadata=DocumentMetadata(source_url=row["source_url"], license=row["license"]),
                ),
            )
        )
    return out


def chunk_all(
    config: ChunkingConfig, docs: Sequence[tuple[str, Document]], counter: object
) -> dict[str, list[Chunk]]:
    chunker = build_chunker(config, counter)  # type: ignore[arg-type]
    chunker.prepare(len(docs))
    by_lang: dict[str, list[Chunk]] = {}
    for lang, doc in docs:
        by_lang.setdefault(lang, []).extend(chunker.chunk([doc]))
    return by_lang


def report(counter: TokenizerCounter, by_lang: dict[str, list[Chunk]], limit: int) -> None:
    logger.info(
        "%6s %7s %11s %9s %9s %9s %7s",
        "lang",
        "chunk",
        "ký tự/tok",
        "tok p50",
        "tok p95",
        "tok max",
        "vượt",
    )
    everything: list[int] = []
    for lang, chunks in sorted(by_lang.items()):
        texts = [c.content for c in chunks]
        counts = counter.count_tokens(texts) or []
        everything += counts
        ordered = sorted(counts)
        logger.info(
            "%6s %7d %11.2f %9d %9d %9d %7d",
            lang,
            len(chunks),
            sum(len(t) for t in texts) / sum(counts),
            ordered[len(ordered) // 2],
            ordered[int(0.95 * len(ordered))],
            ordered[-1],
            sum(1 for c in counts if c > limit),
        )
    ordered = sorted(everything)
    logger.info(
        "%6s %7d %11s %9d %9d %9d %7d",
        "tổng",
        len(everything),
        "",
        int(statistics.median(everything)),
        ordered[int(0.95 * len(ordered))],
        ordered[-1],
        sum(1 for c in everything if c > limit),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="chỉ đo một model (mặc định: cả hai)")
    parser.add_argument(
        "--tokens",
        type=int,
        default=0,
        help="chạy thêm một lượt với size_unit='tokens' và chunk_size bằng ngần này",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    docs = load_corpus()
    logger.info("%d tài liệu · %d ký tự", len(docs), sum(len(d.content) for _, d in docs))

    models = {args.model: MODELS[args.model]} if args.model else MODELS
    for name, limit in models.items():
        counter = TokenizerCounter(name, limit)

        started = time.perf_counter()
        by_lang = chunk_all(ChunkingConfig(**BASELINE), docs, counter)  # type: ignore[arg-type]
        chars_seconds = time.perf_counter() - started
        logger.info(
            "\n### %s · giới hạn %d token · size_unit=chars (%.1fs)", name, limit, chars_seconds
        )
        report(counter, by_lang, limit)

        if not args.tokens:
            continue

        # ⚠️ So `size_unit=tokens` với baseline nguyên bản là so **hai thứ khác
        # nhau hai chỗ**: đơn vị, và ngữ cảnh hàng xóm (baseline bật 100 ký tự,
        # chế độ token buộc phải tắt vì hai thứ loại trừ nhau). Padding 100 ký tự
        # mỗi bên cộng tới 200 ký tự vào MỌI chunk, tức gần gấp đôi một chunk 200
        # ký tự — đủ để một mình nó giải thích cả chênh lệch. Nên mốc so sánh
        # phải là baseline đã TẮT padding.
        reference = dict(BASELINE, neighbor_context_chars=0)
        by_lang = chunk_all(ChunkingConfig(**reference), docs, counter)  # type: ignore[arg-type]
        logger.info("\n### %s · size_unit=chars, KHÔNG ngữ cảnh hàng xóm (mốc so sánh)", name)
        report(counter, by_lang, limit)

        config = ChunkingConfig(
            strategy=ChunkingStrategy.HYBRID,
            size_unit="tokens",
            chunk_size=args.tokens,
            chunk_overlap=max(1, args.tokens // 10),
            min_chunk_size=max(1, args.tokens // 5),
            max_chunk_size=args.tokens * 3 // 2,
            neighbor_context_chars=0,  # loại trừ nhau với trần token
        )
        started = time.perf_counter()
        by_lang = chunk_all(config, docs, counter)
        token_seconds = time.perf_counter() - started
        logger.info(
            "\n### %s · size_unit=tokens chunk_size=%d max=%d (%.1fs, chậm hơn %.1f×)",
            name,
            config.chunk_size,
            config.max_chunk_size,
            token_seconds,
            token_seconds / chars_seconds if chars_seconds else 0.0,
        )
        report(counter, by_lang, min(limit, config.max_chunk_size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
