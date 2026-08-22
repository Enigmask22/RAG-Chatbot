"""Đo `StructureChunker` trên tài liệu thật: cấu trúc thu được và cái giá phải trả.

Ba câu hỏi mà fixture không trả lời được:

1. Tài liệu thật cho ra **mấy cấp** heading? (Đáp án đo được: PDF → đúng một
   cấp. Phân cấp chỉ tồn tại ở định dạng nào tự mang nó, như DOCX/HTML/MD.)
2. Cắt theo heading làm phân bố kích thước chunk lệch đi bao nhiêu so với cắt
   theo ký tự?
3. Bao nhiêu chunk phải **hạ** `section_path` xuống tổ tiên chung vì gộp qua
   ranh giới section — tức bao nhiêu chunk sẽ nói dối nếu không có luật ấy?

Dùng:

    uv run python scripts/structure_probe.py <file.pdf|docx|md|...> [...]
    uv run python scripts/structure_probe.py --corpus
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "packages"))

from rag_core.chunking import ChunkingConfig, ChunkingStrategy, FixedSizeChunker
from rag_core.chunking.structure import StructureChunker
from rag_core.loaders import load_document
from rag_core.loaders.base import LoadedDocument
from rag_core.schemas import Chunk, Document, DocumentMetadata

logger = logging.getLogger("structure_probe")

METADATA = DocumentMetadata(source_url="https://example.org/probe", license="probe")
CORPUS = Path("data/corpus")


@dataclass(frozen=True)
class SizeStats:
    count: int
    minimum: int
    median: int
    p95: int
    maximum: int

    @classmethod
    def of(cls, chunks: list[Chunk]) -> SizeStats:
        sizes = sorted(len(c.content) for c in chunks)
        return cls(
            count=len(sizes),
            minimum=sizes[0],
            median=int(statistics.median(sizes)),
            p95=sizes[min(len(sizes) - 1, int(0.95 * len(sizes)))],
            maximum=sizes[-1],
        )

    def row(self) -> str:
        return (
            f"n={self.count:>5} min={self.minimum:>5} median={self.median:>5} "
            f"p95={self.p95:>5} max={self.maximum:>5}"
        )


def _chunk(structure: LoadedDocument, doc_id: str, **overrides: object) -> list[Chunk]:
    config = ChunkingConfig(strategy=ChunkingStrategy.STRUCTURE, **overrides)  # type: ignore[arg-type]
    chunker = StructureChunker(config)
    chunker.bind(doc_id, structure)
    document = Document(doc_id=doc_id, content=structure.text, metadata=METADATA)
    return chunker.chunk([document])


def _truthful(chunk: Chunk, structure: LoadedDocument) -> bool:
    assert chunk.end_char is not None
    truth = list(structure.section_path_at(chunk.end_char - 1))
    return truth[: len(chunk.section_path)] == chunk.section_path


def probe_file(path: Path) -> None:
    structure = load_document(path)
    located = [h for h in structure.headings if h.located]
    depths = sorted({h.depth for h in located})

    logger.info("===== %s =====", path.name)
    logger.info(
        "%d ký tự · %d heading (%d định vị được) · cấp %s · %d bảng",
        len(structure.text),
        len(structure.headings),
        len(located),
        depths or "—",
        structure.table_count,
    )

    merged = _chunk(structure, path.name)
    split = _chunk(structure, path.name, structure_merge_short_sections=False)
    fixed = FixedSizeChunker(ChunkingConfig()).chunk(
        [Document(doc_id=path.name, content=structure.text, metadata=METADATA)]
    )

    degraded = sum(
        1
        for c in merged
        if len(c.section_path) < len(structure.section_path_at(c.end_char - 1 if c.end_char else 0))
    )
    lying = sum(1 for c in merged if not _truthful(c, structure))

    logger.info("structure (gộp)    %s", SizeStats.of(merged).row())
    logger.info("structure (không)  %s", SizeStats.of(split).row())
    logger.info("fixed              %s", SizeStats.of(fixed).row())
    logger.info(
        "có section_path: %d/%d (%.1f%%) · hạ xuống tổ tiên chung: %d · nói dối: %d",
        sum(1 for c in merged if c.section_path),
        len(merged),
        100 * sum(1 for c in merged if c.section_path) / len(merged),
        degraded,
        lying,
    )


def probe_corpus() -> None:
    """Corpus hiện tại có bao nhiêu cấu trúc để mà dùng?"""
    files = sorted(CORPUS.glob("*.txt"))
    if not files:
        logger.error("Không thấy %s — chạy `dvc pull` trước.", CORPUS)
        return
    with_headings = 0
    for path in files:
        if load_document(path).headings:
            with_headings += 1
    logger.info(
        "corpus: %d tài liệu · có heading máy đọc được: %d (%.0f%%)",
        len(files),
        with_headings,
        100 * with_headings / len(files),
    )
    logger.info(
        "0%s là kết quả đúng chứ không phải hỏng: `.txt` đi qua `load_plain`, và text "
        "thuần không mang cấu trúc. Xem `reports/tasks/w3-03-structure-chunker.md` §3.",
        "%",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument("--corpus", action="store_true", help="quét toàn bộ data/corpus")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.corpus:
        probe_corpus()
    for path in args.files:
        probe_file(path)
    if not args.corpus and not args.files:
        parser.error("cần ít nhất một file hoặc --corpus")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
