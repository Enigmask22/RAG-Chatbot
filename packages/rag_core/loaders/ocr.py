"""Chính sách OCR: máy nào, cho ngôn ngữ nào, và bao nhiêu job một lúc. `W3-02`.

**Kết quả quyết định mọi thứ trong module này: máy OCR đi kèm docling đọc được
tiếng Anh và KHÔNG đọc được tiếng Việt.** Đo trên fixture một trang có sẵn cả hai
đoạn, cùng ảnh, cùng độ phân giải, cùng một lần chạy:

| model rec | tiếng Anh | tiếng Việt |
|---|---|---|
| `ch` PP-OCRv6 (mặc định của docling) | ✅ nguyên văn | `Tāng trng t 7,09 phān trām nām 2024` |
| `latin` PP-OCRv3 | ✅ nguyên văn | `Tng trXXng XXt 7,09 phn trm nm 2024` |

Bản gốc: `Tăng trưởng đạt 7,09 phần trăm năm 2024`. Model `ch` bịa ra dấu sai,
model `latin` **bỏ hẳn dấu** và chèn `XX` cho ký tự ngoài bộ chữ. Con số thì cả
hai đều đọc đúng (7,09 · 5,05 · 405,5 · 3,63) — hỏng nằm ở dấu và ở hình dạng
chữ cái tiếng Việt, không ở chữ số.

⚠️ **Nên với corpus tiếng Việt, OCR bật lên còn tệ hơn OCR tắt.** Tắt thì tài
liệu rỗng và có người nhận ra; bật thì nó sinh ra văn bản *trông như nội dung*,
đi thẳng vào embedding, vào index, vào citation — và không phép kiểm nào ở tầng
sau phân biệt được. Module này vì thế **từ chối** thay vì trả về rác.

**Vì sao không dùng VLM như plan viết.** Dòng `W3-02` trong plan ghi *"Qwen2.5-VL
/ Gemini Vision"*, và phép đo trên đúng là lý do nên làm thế. Nhưng môi trường
hiện tại **không chạy được đường đó**: không có `OPENROUTER_API_KEY`, còn
DeepSeek — key duy nhất đang có — không có model thị giác. Viết một đường gọi API
trả phí mà không chạy thử được lần nào là đúng chế độ hỏng `W2-07` đã ghi lại
("chạy xong, đúng số ô, không có dữ liệu"). Để lại `TD-23` kèm điều kiện mở khoá.

Tesseract (`vie` traineddata đọc tiếng Việt tốt) cũng là một lối ra, nhưng nó là
**binary hệ điều hành** chứ không phải wheel — thuộc Dockerfile, không thuộc
commit này. Đã kiểm: máy hiện tại không có `tesseract`, không có `onnxruntime`.
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
    "OCR_ENGINE",
    "OCR_VERIFIED_LANGUAGES",
    "SECONDS_PER_PAGE",
    "OcrBudgetError",
    "OcrGate",
    "OcrLanguageError",
    "ocr_supports",
    "require_ocr_support",
]

logger = logging.getLogger(__name__)

OCR_ENGINE = "rapidocr:PP-OCRv6-ch"
"""Máy OCR docling dùng mặc định. Ghi cả tên model vì đó là thứ quyết định ngôn ngữ."""

OCR_VERIFIED_LANGUAGES = frozenset({"en"})
"""Ngôn ngữ đã **đo** là đọc được. Không phải ngôn ngữ nhà sản xuất tuyên bố."""

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


def ocr_supports(language: str | None) -> bool:
    """`None` = chưa biết ngôn ngữ → cho qua, nhưng chỗ gọi phải cảnh báo."""
    if language is None:
        return True
    return language.strip().lower() in OCR_VERIFIED_LANGUAGES


def require_ocr_support(language: str | None, *, name: str = "tài liệu") -> None:
    """Ném lỗi có kèm số đo nếu ngôn ngữ nằm ngoài tập đã kiểm.

    Thông điệp lỗi cố ý mang theo **một ví dụ output hỏng**: người đọc log cần
    thấy ngay vì sao đây không phải chuyện "thử bật cờ xem sao".
    """
    if ocr_supports(language):
        if language is None:
            logger.warning(
                "%s: OCR chạy khi chưa biết ngôn ngữ. Máy %s mới chỉ kiểm với %s — "
                "văn bản tiếng Việt sẽ ra rác mà không có gì báo.",
                name,
                OCR_ENGINE,
                sorted(OCR_VERIFIED_LANGUAGES),
            )
        return
    raise OcrLanguageError(
        f"{name}: OCR bị từ chối cho ngôn ngữ {language!r}. Máy hiện có ({OCR_ENGINE}) "
        f"mới chỉ kiểm được {sorted(OCR_VERIFIED_LANGUAGES)}; với tiếng Việt nó trả về "
        "'Tāng trng t 7,09 phān trām nām 2024' cho 'Tăng trưởng đạt 7,09 phần trăm "
        "năm 2024'. Rác đi vào index thì không tầng nào phía sau nhận ra. Xem TD-23."
    )


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
