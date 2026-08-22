"""Backend docling — PDF, DOCX, PPTX, XLSX, HTML, MD → markdown + heading.

Tên module là `docling_backend` chứ không phải `docling` để đọc code không phải
tự hỏi `import docling` bên trong đang trỏ vào đâu.

**OCR mặc định TẮT, và đó là một quyết định có số đo.** Pipeline PDF của docling
bật OCR sẵn: lần chạy đầu tải ~30 MB trọng số RapidOCR rồi chạy detect + recog
trên từng trang. Đo trên fixture một trang của `W3-01`: **70,56 s** khi bật,
**0,12–0,77 s** khi tắt — chênh khoảng hai bậc độ lớn cho một trang PDF
born-digital vốn đã có sẵn text layer, tức toàn bộ chi phí ấy mua về đúng thứ
đã có. Phát hiện scan rồi mới gọi OCR là việc của `W3-02`; ở đây tắt để `W3-01`
không âm thầm trả giá đó cho mọi tài liệu.

**Độ sâu heading phải chuẩn hoá, vì cùng một tài liệu ra ba kiểu đánh số.** Đo
trên bộ fixture 6 định dạng (cùng nội dung logic: h1 → h2 → h3):

| nguồn | h1 | h2 | h3 |
|---|---|---|---|
| `.docx` | `section_header` **level 1** | level 2 | level 3 |
| `.md` / `.html` | **`title`** (không có level) | `section_header` **level 1** | level 2 |
| `.pptx` | `title` | `title` | `list_item` |
| `.xlsx` | — | — | — |

Tin thẳng `item.level` thì cùng một heading là **cấp 3** khi tới từ DOCX và
**cấp 2** khi tới từ HTML, tức `section_path` mà `W3-03` dựng sẽ khác nhau tuỳ
định dạng nguồn của cùng một nội dung. Quy tắc dùng ở đây: `title` chiếm cấp 1,
và `section_header level L` nằm ở cấp `L + 1` **nếu** đã gặp `title` trước đó,
ngược lại là cấp `L`. Quy tắc này khớp cả bốn cột trên.

⚠️ Giới hạn của quy tắc: nó không phân biệt được `title` là *tiêu đề tài liệu*
với `title` là *một heading ngang hàng*. PPTX rơi đúng vào chỗ đó — mỗi slide
một `title`, nên mọi slide thành cấp 1 và không có cây phân cấp nào. Không phải
lỗi của quy tắc mà là của định dạng: PPTX không mang phân cấp heading.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from .base import Heading, LoadedDocument, LoaderError, ParseFingerprint

__all__ = [
    "DOCLING_FORMATS",
    "component_versions",
    "docling_version",
    "load_with_docling",
    "model_revisions",
]

logger = logging.getLogger(__name__)

DOCLING_FORMATS = frozenset(
    {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm", ".md", ".markdown"}
)

_TITLE_LABELS = frozenset({"title"})
_HEADER_LABELS = frozenset({"section_header"})
_TABLE_LABELS = frozenset({"table"})


_ALWAYS = ("docling-core",)
"""Gói serialise `DoclingDocument` → markdown. Mọi định dạng đều đi qua."""

_PDF_ONLY = ("docling-parse", "pypdfium2", "docling-ibm-models")
"""Đọc text layer + model layout/table. Chỉ PDF chạm tới."""

_OCR_ONLY = ("rapidocr",)


def _dist_version(name: str) -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return "absent"


_PDF_MODELS: tuple[tuple[str, str], ...] = (
    # (repo HF, revision mà docling yêu cầu)
    ("docling-project/docling-layout-heron", "main"),
    ("docling-project/docling-models", "v2.3.0"),
)
"""Trọng số mà pipeline PDF nạp. Xem `model_revisions` về vì sao phải ghim.

Nguồn: `docling/datamodel/stage_model_specs.py:997` (layout, `revision="main"`) và
`docling/models/stages/table_structure/table_structure_model.py:105` (bảng,
`revision="v2.3.0"`).
"""


@lru_cache(maxsize=1)
def model_revisions() -> tuple[str, ...]:
    """Commit SHA **thực tế** của trọng số PDF đang nằm trong cache HF.

    ## Vì sao ghim version gói vẫn chưa đủ

    `components` đóng được lỗ "gói phụ dịch chuyển", nhưng pipeline PDF của
    docling còn một đầu vào nữa mà **không** con số version nào chạm tới: trọng
    số model tải từ Hugging Face. Và hai model được đối xử khác nhau:

    | model | repo | revision docling yêu cầu |
    |---|---|---|
    | bảng | `docling-project/docling-models` | **`v2.3.0`** — tag cố định |
    | bố cục | `docling-project/docling-layout-heron` | **`main`** — nhánh di động |

    Model bố cục quyết định thứ tự đọc và cách chia khối của trang PDF, tức
    quyết định thứ tự các đoạn trong markdown xuất ra. Một lượt push lên `main`
    của repo ấy đổi văn bản parse ra **mà không một con số version nào trên máy
    này nhúc nhích** — không `docling`, không `docling-core`, không gì cả. Đây là
    tầng sâu hơn của `TD-22` và là tầng mà chỉ `text_sha256` bắt được.

    Nên ghim vào vân tay **commit SHA đã phân giải** đọc từ cache HF (`refs/`),
    không phải chuỗi `"main"`. Chỉ đọc đĩa, không gọi mạng — nếu trọng số chưa
    được tải thì trả `repo@absent` thay vì đi tải về.
    """
    from huggingface_hub.constants import HF_HUB_CACHE
    from huggingface_hub.file_download import repo_folder_name

    out: list[str] = []
    for repo_id, revision in _PDF_MODELS:
        ref = (
            Path(HF_HUB_CACHE)
            / repo_folder_name(repo_id=repo_id, repo_type="model")
            / "refs"
            / revision
        )
        try:
            resolved = ref.read_text(encoding="utf-8").strip()[:12] or "absent"
        except OSError:
            resolved = "absent"
        out.append(f"{repo_id.split('/')[-1]}@{resolved}")
    return tuple(out)


@lru_cache(maxsize=8)
def component_versions(suffix: str, ocr: bool) -> tuple[str, ...]:
    """`("tên=version", ...)` của các gói mà **đúng đường parse này** đi qua.

    Chia theo đường đi thay vì ghi tất: xem `ParseFingerprint` về vì sao ghi
    thừa cũng có giá. `lru_cache` vì `importlib.metadata.version` đọc đĩa và
    hàm này bị gọi một lần cho mỗi tài liệu.
    """
    names = list(_ALWAYS)
    pins: tuple[str, ...] = ()
    if suffix == ".pdf":
        names += list(_PDF_ONLY)
        pins = model_revisions()
    if ocr:
        names += list(_OCR_ONLY)
    return tuple(f"{name}={_dist_version(name)}" for name in sorted(names)) + pins


def docling_version() -> str:
    """Version của docling, hoặc `"absent"` khi chưa cài.

    ⚠️ Một mình con số này **không** ghim được văn bản xuất ra — xem
    `ParseFingerprint.components` và `model_revisions`. Nó là tên gói ô dù,
    không phải gói làm việc.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("docling")
    except PackageNotFoundError:  # pragma: no cover - chỉ xảy ra khi thiếu extra
        return "absent"


@lru_cache(maxsize=2)
def _converter(ocr: bool) -> Any:
    """`DocumentConverter` dùng lại giữa các lần gọi.

    ⚠️ `maxsize=2` cố ý nhỏ. Pipeline PDF nạp model bố cục lên GPU, nên mỗi
    converter sống thêm là thêm VRAM — cùng loại ngân sách mà `W0-06` đang đếm
    cho ba `lru_cache` bên `rag_core`. Hai chỗ ở đây là `ocr=False` (mặc định)
    và `ocr=True` (`W3-02`), không cần hơn.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions()
    options.do_ocr = ocr
    options.do_table_structure = True
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )


def _normalise_depth(label: str, level: int | None, title_seen: bool) -> int:
    if label in _TITLE_LABELS:
        return 1
    base = level if level is not None and level >= 1 else 1
    return base + 1 if title_seen else base


def _collect(document: Any, text: str) -> tuple[tuple[Heading, ...], int]:
    """Đi qua các item của docling, dựng danh sách heading + đếm bảng.

    Vị trí heading tìm bằng `str.find` **tiến dần** (`cursor`) chứ không tìm từ
    đầu: hai heading trùng chữ trong một tài liệu là chuyện bình thường (`Khoản
    1` lặp ở mọi điều), và tìm từ đầu sẽ gán cả hai về cùng một offset.
    """
    headings: list[Heading] = []
    tables = 0
    title_seen = False
    cursor = 0

    for item, _ in document.iterate_items():
        label = str(getattr(item, "label", "") or "")
        if label in _TABLE_LABELS:
            tables += 1
            continue
        if label not in _TITLE_LABELS and label not in _HEADER_LABELS:
            continue

        content = (getattr(item, "text", "") or "").strip()
        if not content:
            continue

        depth = _normalise_depth(label, getattr(item, "level", None), title_seen)
        if label in _TITLE_LABELS:
            title_seen = True

        position = text.find(content, cursor)
        if position < 0:
            logger.debug("Không định vị được heading %r trong markdown xuất ra", content[:60])
        else:
            cursor = position + len(content)
        headings.append(Heading(depth=depth, text=content, start_char=position))

    return tuple(headings), tables


def load_with_docling(
    path: str | Path,
    *,
    source_sha256: str,
    ocr: bool = False,
) -> LoadedDocument:
    """Parse một file bằng docling và trả `LoadedDocument`.

    Nhận **đường dẫn** chứ không nhận bytes: backend PDF của docling đọc theo
    kiểu random-access qua pypdfium2, và gói bytes lại thành file tạm chỉ để
    docling mở ra là thêm một chỗ hỏng mà không mua được gì.
    """
    try:
        converter = _converter(ocr)
    except ImportError as exc:  # pragma: no cover - phụ thuộc môi trường
        raise LoaderError(
            "Thiếu docling. Cài extra `ingestion`: `uv sync --extra ingestion`."
        ) from exc

    try:
        result = converter.convert(str(path))
    except Exception as exc:  # docling ném nhiều kiểu lỗi tuỳ backend
        raise LoaderError(f"{Path(path).name}: docling không parse được ({exc})") from exc

    document = result.document
    text = document.export_to_markdown()
    if not text.strip():
        raise LoaderError(f"{Path(path).name}: docling trả về văn bản rỗng")

    headings, tables = _collect(document, text)
    return LoadedDocument(
        text=text,
        source_sha256=source_sha256,
        fingerprint=ParseFingerprint(
            loader="docling",
            library="docling",
            library_version=docling_version(),
            options=(f"ocr={str(ocr).lower()}", "table_structure=true"),
            components=component_versions(Path(path).suffix.lower(), ocr),
        ),
        headings=headings,
        table_count=tables,
    )
