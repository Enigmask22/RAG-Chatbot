"""Chính sách OCR: máy nào, cho ngôn ngữ nào, và bao nhiêu job một lúc. `W3-02` + `TD-23`.

⚠️⚠️ **Phép đo đầu tiên của module này (2026-08-28) KHÔNG HỢP LỆ, và bảng kết
luận cũ đã bị thay (2026-09-04).** Fixture sinh bằng font mặc định của Pillow
(Aileron) — font **không có glyph tiếng Việt**, dấu bị render thành ô ☒ ngay
trong ảnh. "Không OCR nào đọc được tiếng Việt" thật ra là "không OCR nào đọc
được ký tự ☒": cả hai máy bị chấm trên một đề bài hỏng. Sau khi sửa fixture
sang DejaVu Sans (commit trong repo) và đo lại **cùng ảnh, cùng lần chạy**:

| máy | tiếng Anh | tiếng Việt |
|---|---|---|
| `rapidocr` PP-OCRv6 (mặc định docling) | ✅ đủ chữ | ❌ **vứt 3/5 dòng**, sai dấu (`mức`→`múc`) |
| `easyocr` latin-g2, lang `[vi,en]` | ✅ đủ chữ | ✅ **dấu sống 8/8**, char-acc 0,91–0,97 |

Lỗi nặng nhất của EasyOCR là `phần`→`phẩn`; RapidOCR trên ảnh mới còn xáo trật
tự một dòng tiếng Anh (tầng layout của docling, không phải máy OCR).

Đo: `plans/reports/probes/td-23-easyocr.json`. Nên chính sách là **chọn máy theo
ngôn ngữ**: `vi` đi EasyOCR, `en` giữ RapidOCR (đường cũ, vân tay cũ). Ngôn ngữ
chưa đo (fr, zh, …) vẫn bị **từ chối** — lý do giữ nguyên: OCR sai sinh ra văn
bản *trông như nội dung*, đi thẳng vào embedding → index → citation, và không
phép kiểm nào ở tầng sau phân biệt được. Rác nguy hiểm hơn rỗng.

⚠️ Giới hạn đã biết của đường `vi`: tầng xếp thứ tự đọc của **docling** (không
phải của máy OCR) xáo một phần trật tự từ trên trang nhiều khối — nội dung và
dấu vào đủ, trật tự trong một dòng có thể lệch. Ảnh hưởng citation nguyên văn,
ít ảnh hưởng truy hồi (embedding/BM25 nhìn túi từ nhiều hơn nhìn thứ tự).

VLM qua OpenRouter (khi có key) và Tesseract + `vie` trong Dockerfile (`W4-13`)
vẫn là hai lối nâng cấp tiếp theo của `TD-23` nếu chất lượng EasyOCR không đủ
cho corpus scan thật — quyết định đó cần tài liệu scan thật, chưa có trong corpus.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from .base import LoaderError

__all__ = [
    "DEFAULT_GATE",
    "DEFAULT_OCR_ENGINE",
    "OCR_ENGINE",
    "OCR_ENGINES",
    "OCR_VERIFIED_LANGUAGES",
    "SECONDS_PER_PAGE",
    "OcrBudgetError",
    "OcrGate",
    "OcrLanguageError",
    "engine_for",
    "ocr_supports",
    "require_ocr_support",
]

logger = logging.getLogger(__name__)

OCR_ENGINES: dict[str, str] = {
    "rapidocr": "rapidocr:PP-OCRv6-ch",
    "easyocr": "easyocr:latin-g2",
}
"""Máy OCR khả dụng → nhãn đầy đủ (kèm tên model, vì model quyết định ngôn ngữ)."""

DEFAULT_OCR_ENGINE = "rapidocr"
"""Máy khi không biết ngôn ngữ: đường cũ, vân tay cũ, hành vi `W3-02` giữ nguyên."""

OCR_ENGINE = OCR_ENGINES[DEFAULT_OCR_ENGINE]
"""Nhãn máy mặc định. Tên cũ giữ lại vì `W3-02` export nó ra ngoài."""

OCR_VERIFIED_LANGUAGES: dict[str, frozenset[str]] = {
    "rapidocr": frozenset({"en"}),
    "easyocr": frozenset({"en", "vi"}),
}
"""Ngôn ngữ đã **đo** là đọc được, theo từng máy. Không phải danh sách nhà sản
xuất tuyên bố — EasyOCR tuyên bố 80+ ngôn ngữ, ở đây chỉ ghi hai thứ đã chấm
trên fixture (xem bảng ở docstring module)."""

SECONDS_PER_PAGE = 0.35
"""Chi phí OCR **cận biên**, đo trên 4 lần chạy lại trong cùng tiến trình.

⚠️ **Và con số này đính chính một con số của `W3-01`.** Ở đó tôi ghi "OCR đắt hai
bậc độ lớn: 70,56 s vs 0,12–0,77 s" và dùng nó làm lý do phải phát hiện scan. Đo
lại 5 lượt liên tiếp: **12,67 · 0,34 · 0,34 · 0,35 · 0,36** giây. Tức phần lớn cái
70 s ấy là **nạp model một lần cho cả tiến trình** (lần đầu còn kèm tải ~30 MB
trọng số), không phải giá mỗi trang. Giá mỗi trang thật là ~0,35 s so với ~0,12 s
— **gấp ~3 lần, không phải gấp 500**.

Nên lý do `W3-02` đáng làm **không phải chi phí**. Nó là:

1. **Đúng/sai** — không phát hiện scan thì tài liệu ảnh trả về rỗng và `W3-01`
   ném `LoaderError`, tức tài liệu bị bỏ im lặng khỏi index.
2. **Chốt ngôn ngữ** — bật OCR bừa cho corpus tiếng Việt sinh ra rác *trông như
   nội dung*. Đây mới là thứ đắt.
3. Chi phí cố định ~12 s/tiến trình vẫn tránh được cho tài liệu born-digital.
"""


class OcrLanguageError(LoaderError):
    """Máy OCR hiện có không đọc được ngôn ngữ của tài liệu."""


class OcrBudgetError(LoaderError):
    """Tài liệu vượt trần số trang cho một lần OCR."""


def engine_for(language: str | None) -> str:
    """Chọn máy OCR theo ngôn ngữ. Trả khoá trong `OCR_ENGINES`.

    `None` = chưa biết ngôn ngữ → máy mặc định (chỗ gọi phải cảnh báo). Ngôn ngữ
    chưa đo → `OcrLanguageError`. Duyệt máy mặc định trước để `en` không âm thầm
    đổi máy — đổi máy là đổi văn bản xuất ra, tức đổi vân tay parse.
    """
    if language is None:
        return DEFAULT_OCR_ENGINE
    lang = language.strip().lower()
    for engine in (DEFAULT_OCR_ENGINE, *sorted(OCR_VERIFIED_LANGUAGES)):
        if lang in OCR_VERIFIED_LANGUAGES[engine]:
            return engine
    verified = {key: sorted(value) for key, value in sorted(OCR_VERIFIED_LANGUAGES.items())}
    raise OcrLanguageError(
        f"chưa máy OCR nào ở đây được ĐO với ngôn ngữ {language!r}. Đã đo: {verified}. "
        "OCR sai không trả lỗi — nó trả văn bản trông như nội dung (phép đo 2026-09-04: "
        "RapidOCR vứt 3/5 dòng tiếng Việt trên ảnh hợp lệ và không báo gì). "
        "Rác đi vào index thì không tầng nào phía sau nhận ra. Xem TD-23."
    )


def ocr_supports(language: str | None) -> bool:
    """`None` = chưa biết ngôn ngữ → cho qua, nhưng chỗ gọi phải cảnh báo."""
    if language is None:
        return True
    lang = language.strip().lower()
    return any(lang in verified for verified in OCR_VERIFIED_LANGUAGES.values())


def require_ocr_support(language: str | None, *, name: str = "tài liệu") -> str:
    """Chọn máy cho ngôn ngữ này, hoặc ném lỗi có kèm số đo. Trả khoá máy.

    Thông điệp lỗi cố ý mang theo **một số đo thật**: người đọc log cần thấy
    ngay vì sao đây không phải chuyện "thử bật cờ xem sao".
    """
    if language is None:
        logger.warning(
            "%s: OCR chạy khi chưa biết ngôn ngữ. Máy mặc định %s mới chỉ kiểm với %s — "
            "văn bản tiếng Việt phải khai `language='vi'` mới được đưa sang EasyOCR.",
            name,
            OCR_ENGINE,
            sorted(OCR_VERIFIED_LANGUAGES[DEFAULT_OCR_ENGINE]),
        )
        return DEFAULT_OCR_ENGINE
    try:
        return engine_for(language)
    except OcrLanguageError as exc:
        raise OcrLanguageError(f"{name}: {exc}") from None


@dataclass
class OcrGate:
    """Hàng đợi cho OCR: chặn song song, và chặn tài liệu quá dài.

    Hai thứ khác nhau, cùng một cổng:

    * `max_concurrent` — OCR chạy model detect + recog **trên GPU**, và ngân sách
      VRAM của card 8 GB đã kín gần một nửa vì embedding + reranker (`W2-07` đo
      3.900/8.188 MiB). Hai job OCR song song là cách nhanh nhất để OOM giữa một
      job index hai tiếng. Mặc định **1**, và đó cũng là giới hạn thật: converter
      của docling nằm sau `lru_cache`, dùng chung từ nhiều luồng không an toàn.
    * `max_pages` — chặn đầu vào bệnh lý, **không** chặn báo cáo bình thường.
      Xem `SECONDS_PER_PAGE`: một bản scan 129 trang chỉ tốn ~45 s, còn 500 trang
      là ~3 phút. Trần đặt ở 500 để một file 5.000 trang phải được nói ra tường
      minh chứ không âm thầm chiếm nửa tiếng của job index.
    """

    max_concurrent: int = 1
    max_pages: int = 500
    _semaphore: threading.BoundedSemaphore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_concurrent < 1:
            raise ValueError("max_concurrent phải ≥ 1")
        if self.max_pages < 1:
            raise ValueError("max_pages phải ≥ 1")
        self._semaphore = threading.BoundedSemaphore(self.max_concurrent)

    def check_budget(self, pages: int, *, name: str = "tài liệu") -> None:
        if pages > self.max_pages:
            estimate = pages * SECONDS_PER_PAGE
            raise OcrBudgetError(
                f"{name}: {pages} trang vượt trần OCR {self.max_pages} trang "
                f"(~{estimate / 60:.1f} phút ở {SECONDS_PER_PAGE:.2f} s/trang). "
                "Nâng `OcrGate(max_pages=…)` nếu thật sự muốn."
            )

    @contextmanager
    def reserve(self, pages: int, *, name: str = "tài liệu") -> Iterator[None]:
        self.check_budget(pages, name=name)
        acquired = self._semaphore.acquire(blocking=False)
        if not acquired:
            logger.info("%s: đợi khe OCR (tối đa %d job song song)", name, self.max_concurrent)
            self._semaphore.acquire()
        try:
            yield
        finally:
            self._semaphore.release()


DEFAULT_GATE = OcrGate()
"""Cổng dùng chung. Một tiến trình một cổng, nếu không thì trần song song vô nghĩa."""
