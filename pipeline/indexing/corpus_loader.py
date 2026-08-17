"""Manifest corpus → `Document`, có kiểm tra toàn vẹn.

Điểm quan trọng nhất ở đây là **đối chiếu sha256**. Manifest ghi hash của từng
file lúc tải về; nếu file trên đĩa đã khác đi (sửa tay, tải lại từ nguồn đã đổi
nội dung, đứt giữa lúc ghi) thì phải dừng chứ không được index tiếp. Lý do:
golden set ở `W1-11` trỏ tới `chunk_id` cụ thể, mà `chunk_id` là
`{doc_id}::{index}` — nội dung đổi thì cùng một `chunk_id` sẽ chỉ sang đoạn văn
khác, và mọi con số recall sau đó đều sai mà không có gì báo.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path

from rag_core.schemas import DocType, Document, DocumentMetadata, Language

from ..corpus.manifest import CorpusEntry, load_manifest

__all__ = ["CorpusIntegrityError", "iter_documents", "load_documents", "select_entries"]

logger = logging.getLogger(__name__)


class CorpusIntegrityError(RuntimeError):
    """File trên đĩa không khớp manifest, hoặc thiếu file."""


def select_entries(
    entries: Sequence[CorpusEntry],
    *,
    languages: Sequence[str] = (),
    doc_types: Sequence[str] = (),
    max_documents: int | None = None,
) -> list[CorpusEntry]:
    """Lọc theo ngôn ngữ/loại tài liệu rồi cắt bớt, **theo thứ tự ổn định**.

    Sắp xếp theo `doc_id` trước khi cắt để `max_documents=10` luôn chọn đúng 10
    tài liệu đó, bất kể thứ tự dòng trong manifest. Một lần chạy thử mà mỗi lần
    lấy một tập khác nhau thì không so sánh được với lần trước.
    """
    lang_set = {lang.lower() for lang in languages}
    type_set = {dt.lower() for dt in doc_types}
    chosen = [
        entry
        for entry in sorted(entries, key=lambda e: e.doc_id)
        if (not lang_set or entry.lang.value in lang_set)
        and (not type_set or entry.doc_type.value in type_set)
    ]
    return chosen[:max_documents] if max_documents is not None else chosen


def _parse_published_at(raw: str) -> datetime | None:
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        logger.warning("published_at không đọc được: %r", raw)
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _to_document(entry: CorpusEntry, text: str) -> Document:
    return Document(
        doc_id=entry.doc_id,
        content=text,
        metadata=DocumentMetadata(
            source_url=entry.source_url,
            license=entry.license,
            source_path=entry.relative_path,
            title=entry.title or None,
            lang=Language(entry.lang),
            doc_type=DocType(entry.doc_type),
            published_at=_parse_published_at(entry.published_at),
            extra={
                "corpus_source": entry.source,
                "landing_url": entry.landing_url,
                "license_url": entry.license_url,
                "manifest_sha256": entry.sha256,
            },
        ),
    )


def iter_documents(
    entries: Sequence[CorpusEntry],
    corpus_dir: str | Path,
    *,
    verify_hash: bool = True,
) -> Iterator[Document]:
    """Sinh `Document` từng cái một — corpus lớn không cần nằm hết trong RAM."""
    root = Path(corpus_dir)
    for entry in entries:
        path = root / entry.relative_path
        if not path.exists():
            raise CorpusIntegrityError(
                f"{entry.doc_id}: thiếu file {path}. "
                "Chạy `scripts/fetch_corpus.py` để tải lại, hoặc `dvc pull` nếu đã có W1-09."
            )
        payload = path.read_bytes()
        if verify_hash:
            digest = hashlib.sha256(payload).hexdigest()
            if digest != entry.sha256:
                raise CorpusIntegrityError(
                    f"{entry.doc_id}: nội dung khác manifest "
                    f"(đĩa {digest[:12]}, manifest {entry.sha256[:12]}). "
                    "File đã bị sửa sau khi tải. Index tiếp sẽ làm golden set trỏ sai chunk."
                )
        text = payload.decode("utf-8")
        if not text.strip():
            raise CorpusIntegrityError(f"{entry.doc_id}: file rỗng sau khi decode")
        yield _to_document(entry, text)


def load_documents(
    manifest_path: str | Path,
    corpus_dir: str | Path,
    *,
    languages: Sequence[str] = (),
    doc_types: Sequence[str] = (),
    max_documents: int | None = None,
    verify_hash: bool = True,
) -> list[Document]:
    entries = load_manifest(manifest_path)
    if not entries:
        raise CorpusIntegrityError(
            f"Manifest {manifest_path} rỗng hoặc không tồn tại. "
            "Chạy `scripts/fetch_corpus.py --config configs/corpus/worldbank_vietnam.yaml` trước."
        )
    selected = select_entries(
        entries, languages=languages, doc_types=doc_types, max_documents=max_documents
    )
    if not selected:
        raise CorpusIntegrityError(
            f"Bộ lọc không khớp tài liệu nào trong {len(entries)} dòng manifest "
            f"(languages={list(languages)}, doc_types={list(doc_types)})"
        )
    return list(iter_documents(selected, corpus_dir, verify_hash=verify_hash))
