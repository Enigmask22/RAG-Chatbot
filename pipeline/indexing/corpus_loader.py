"""Manifest corpus → `Document`, có kiểm tra toàn vẹn.

Điểm quan trọng nhất ở đây là **đối chiếu sha256**. Manifest ghi hash của từng
file lúc tải về; nếu file trên đĩa đã khác đi (sửa tay, tải lại từ nguồn đã đổi
nội dung, đứt giữa lúc ghi) thì phải dừng chứ không được index tiếp. Lý do:
golden set ở `W1-11` trỏ tới `chunk_id` cụ thể, mà `chunk_id` là
`{doc_id}::{index}` — nội dung đổi thì cùng một `chunk_id` sẽ chỉ sang đoạn văn
khác, và mọi con số recall sau đó đều sai mà không có gì báo.

## `TD-22`: hash byte thôi thì chuỗi toàn vẹn bị hở ở giữa

Tới hết `W2`, phép biến đổi `byte → Document.content` là **hàm đồng nhất**
(`payload.decode("utf-8")`), nên ghim `sha256` của byte là ghim luôn văn bản. Đo
được: **60/60** tài liệu corpus có `text_sha256` trùng khít `sha256`.

Từ `W3-01` có parser đứng giữa, và lúc đó
`content = parse(byte, phiên bản parser, tuỳ chọn parse)` — manifest ghim đúng
**một trong ba** đầu vào. ⚠️ Chế độ hỏng là loại tệ nhất: `sha256` vẫn khớp,
`iter_documents` vẫn xanh, mọi `chunk_id` vẫn tồn tại, không test nào đỏ, và mọi
`TextSpan` của golden set trỏ lệch.

Nên module này **parse qua đúng loader mà cả hệ thống dùng** rồi đối chiếu hai
cột mới của manifest: `text_sha256` và `parse_fingerprint`.

Luật khi manifest chưa ghim (`text_sha256` rỗng) **không** phải "bỏ qua", mà
theo loader:

* `plain` (`.txt`) — chấp nhận. `bytes.decode("utf-8").encode("utf-8")` trả lại
  đúng byte cũ, nên `sha256` **đã** ghim văn bản; không có gì thêm để kiểm.
* mọi loader khác — **lỗi**. Đó chính xác là chỗ hở, và nó phải chặn ngay ở đây
  chứ không phải được phát hiện ba tuần sau qua một con số recall lạ.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path

from rag_core.loaders import LoadedDocument, LoaderError, OcrMode, load_document, loader_for
from rag_core.schemas import DocType, Document, DocumentMetadata, Language

from ..corpus.manifest import CorpusEntry, load_manifest

__all__ = [
    "CorpusIntegrityError",
    "ParsePinError",
    "iter_documents",
    "load_documents",
    "select_entries",
]

logger = logging.getLogger(__name__)


class CorpusIntegrityError(RuntimeError):
    """File trên đĩa không khớp manifest, hoặc thiếu file."""


class ParsePinError(CorpusIntegrityError):
    """Văn bản parse ra không khớp cái manifest đã ghim, hoặc manifest chưa ghim.

    Tách khỏi `CorpusIntegrityError` vì cách xử lý khác hẳn: hash byte lệch nghĩa
    là **file** đã đổi (tải lại, hoặc điều tra); vân tay parse lệch nghĩa là
    **môi trường** đã đổi trong khi file y nguyên, và lối ra là ghim lại manifest
    *sau khi* đã xác nhận văn bản mới vẫn dùng được (`make corpus-pin`).
    """


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


def _to_document(entry: CorpusEntry, loaded: LoadedDocument) -> Document:
    """`CorpusEntry` + kết quả parse → `Document`.

    `loaded.as_metadata()` đi vào `extra`: parser nào, version nào, vân tay nào,
    `text_sha256` nào. Từ một `chunk_id` bất kỳ truy ngược được lần parse đã sinh
    ra nó — thứ mà `TD-22` cần và trước đây không có ai gọi (hàm này được dựng ở
    `W3-01` rồi nằm không, chỉ một test đụng tới).
    """
    return Document(
        doc_id=entry.doc_id,
        content=loaded.text,
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
                **loaded.as_metadata(),
            },
        ),
    )


def _ocr_language(entry: CorpusEntry) -> str | None:
    """Ngôn ngữ khai trong manifest, đưa xuống cổng OCR.

    Không phải gợi ý mà là một cái chốt: `require_ocr_support` **từ chối**
    `vi` vì máy OCR hiện có trả rác cho tiếng Việt (`TD-23`). Manifest đã biết
    ngôn ngữ của từng tài liệu, nên không có lý do gì để cổng đó phải đoán.
    """
    return None if entry.lang is Language.UNKNOWN else entry.lang.value


def _fingerprint_delta(pinned: str, actual: str) -> str:
    """Chỗ **khác nhau** giữa hai vân tay, không phải cả hai chuỗi.

    Lý do có hàm này thay vì in hai chuỗi: vân tay của một PDF có 9 phần, và
    "gói nào đã dịch chuyển" là toàn bộ thông tin người đọc cần. In cả hai bắt
    họ tự dò bằng mắt đúng lúc đang hoảng.
    """
    was, now = pinned.split("|"), actual.split("|")
    added = [part for part in now if part not in was]
    removed = [part for part in was if part not in now]
    bits = []
    if removed:
        bits.append("mất: " + ", ".join(removed))
    if added:
        bits.append("thêm: " + ", ".join(added))
    return " · ".join(bits) or "không khác gì (cùng phần, khác thứ tự)"


def _verify_parse(entry: CorpusEntry, loaded: LoadedDocument, kind: str) -> None:
    """Đối chiếu văn bản parse ra với cái manifest đã ghim. Xem docstring module."""
    actual = loaded.fingerprint.canonical
    if not entry.text_sha256:
        if kind == "plain":
            # Hàm đồng nhất: `sha256` của byte đã ghim luôn văn bản.
            return
        raise ParsePinError(
            f"{entry.doc_id}: manifest chưa ghim `text_sha256` mà tài liệu này đi qua "
            f"loader `{kind}` (không phải hàm đồng nhất). Hash byte không ghim được "
            f"văn bản khi có parser đứng giữa — xem TD-22. "
            f"Chạy `make corpus-pin` để ghim, vân tay hiện tại: {actual}"
        )

    if loaded.text_sha256 == entry.text_sha256:
        if entry.parse_fingerprint and entry.parse_fingerprint != actual:
            # Văn bản y nguyên nhưng môi trường đã đổi. Không phải lỗi — nhưng
            # phải nói ra, vì lần đổi TIẾP THEO có thể không may như lần này.
            logger.warning(
                "%s: parser đã đổi mà văn bản không đổi (%s). Ghim lại bằng "
                "`make corpus-pin` để lần lệch thật sau này chỉ đúng thủ phạm.",
                entry.doc_id,
                _fingerprint_delta(entry.parse_fingerprint, actual),
            )
        return

    delta = (
        _fingerprint_delta(entry.parse_fingerprint, actual)
        if entry.parse_fingerprint
        else "manifest không ghim vân tay nên không nói được cái gì đã đổi"
    )
    raise ParsePinError(
        f"{entry.doc_id}: văn bản parse ra khác manifest "
        f"(đĩa {loaded.text_sha256[:12]}, manifest {entry.text_sha256[:12]}) "
        f"trong khi byte thì y nguyên. Nghĩa là **môi trường parse** đã đổi: {delta}. "
        "Mọi TextSpan của golden set neo vào văn bản cũ, nên index tiếp sẽ cho "
        "recall sai mà không có gì đỏ. Xác nhận văn bản mới rồi `make corpus-pin`."
    )


def iter_documents(
    entries: Sequence[CorpusEntry],
    corpus_dir: str | Path,
    *,
    verify_hash: bool = True,
    ocr: OcrMode = "auto",
) -> Iterator[Document]:
    """Sinh `Document` từng cái một — corpus lớn không cần nằm hết trong RAM.

    Parse đi qua `rag_core.loaders.load_document`, tức **đúng** đường mà mọi chỗ
    khác dùng. Bản trước tự decode utf-8 tại chỗ; với `.txt` thì hai đường cho
    kết quả trùng khít, nhưng chúng là hai đường, và hai đường thì sớm muộn lệch.
    """
    root = Path(corpus_dir)
    for entry in entries:
        path = root / entry.relative_path
        if not path.exists():
            raise CorpusIntegrityError(
                f"{entry.doc_id}: thiếu file {path}. "
                "Chạy `scripts/fetch_corpus.py` để tải lại, hoặc `dvc pull` nếu đã có W1-09."
            )
        try:
            kind = loader_for(path)
            loaded = load_document(path, ocr=ocr, language=_ocr_language(entry))
        except LoaderError as exc:
            # Bọc lại để `doc_id` đi kèm. `LoaderError` nói "d-1.txt rỗng"; trong
            # một lượt 60 tài liệu thì tên file không đủ để biết dòng manifest nào
            # phải sửa, và người gọi vốn chỉ bắt `CorpusIntegrityError`.
            raise CorpusIntegrityError(f"{entry.doc_id}: {exc}") from exc
        if verify_hash and loaded.source_sha256 != entry.sha256:
            raise CorpusIntegrityError(
                f"{entry.doc_id}: nội dung khác manifest "
                f"(đĩa {loaded.source_sha256[:12]}, manifest {entry.sha256[:12]}). "
                "File đã bị sửa sau khi tải. Index tiếp sẽ làm golden set trỏ sai chunk."
            )
        _verify_parse(entry, loaded, kind)
        yield _to_document(entry, loaded)


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
