"""Structured citations: parse block `CITATIONS:` + xác minh quote theo chunk. `W4-09`.

Ba mảnh, cùng một hợp đồng văn bản:

1. `split_citation_block` — cắt block ra khỏi câu trả lời đầy đủ, validate JSON
   bằng pydantic. Dùng ở cuối stream (serving) và trong eval (`W5-02`).
2. `verify_citations` — đối chiếu từng `quote` với đúng chunk mà `n` chỉ vào.
   Không match → `Citation(verified=False)`. **Trả về, không vứt**: một citation
   bịa phải hiện ra trong khung SSE, không được im lặng biến mất.
3. `CitationHoldback` — bộ đệm cho đường stream: block không được rò vào các
   khung `delta`, kể cả khi marker bị cắt đôi giữa hai delta.

Vì sao `n` chứ không phải `chunk_id`: model đã thấy ngữ cảnh đánh số `[1]`,
`[2]`… trong prompt (`ChatTurn.prompt`), và số nguồn là thứ nó vốn phải chép vào
câu trả lời từ `W4-06`. Bắt nó chép lại một UUID 36 ký tự là thêm một chỗ chép
sai mà không mua được gì — ánh xạ `n → chunk_id` nằm ở phía mình, tất định.

Vì sao match sau **chuẩn hoá whitespace, giữ nguyên hoa thường**: stream trả
markdown nên xuống dòng/khoảng trắng không ổn định, còn hoa thường đổi là *sửa
chữ* — đúng loại "sửa nhẹ" mà xác minh tồn tại để bắt. Nới lỏng thêm quy tắc nào
phải kèm một phép đo cho thấy quy tắc hiện tại từ chối oan.

Một nới lỏng đã qua được điều kiện ấy: **dấu lược** (`NEW-08`/`TD-64`, đo ở
`W5-02`: 19/67 lỗi cấp quote là `...` nối hai mẩu nguyên văn). Xem
`_quote_matches` — các mảnh quanh dấu lược phải khớp nguyên văn, đúng thứ tự.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from rag_core.schemas import Chunk, Citation

__all__ = [
    "MARKER",
    "CitationClaim",
    "CitationHoldback",
    "CitationReport",
    "ParsedAnswer",
    "split_citation_block",
    "verify_citations",
]

MARKER = "CITATIONS:"
"""Phải đứng ở **đầu dòng**. Xuất hiện giữa dòng là văn bản thường, không phải block."""


class CitationClaim(BaseModel):
    """Một phần tử trong block — lời *tuyên bố* của model, chưa xác minh."""

    model_config = ConfigDict(extra="forbid")

    n: int = Field(ge=1)
    quote: str = Field(min_length=1)


_CLAIMS = TypeAdapter(list[CitationClaim])


@dataclass(frozen=True)
class ParsedAnswer:
    """Kết quả cắt block khỏi câu trả lời đầy đủ.

    `block`: `"ok"` = có và parse được · `"absent"` = không có block (model bỏ
    qua chỉ dẫn — với câu từ chối thì đây là nhánh thường gặp, xem P3 của báo
    cáo) · `"invalid"` = có marker mà JSON hỏng, chi tiết ở `error`.
    """

    text: str
    claims: tuple[CitationClaim, ...]
    block: Literal["ok", "absent", "invalid"]
    error: str | None = None


def _find_marker(text: str) -> int | None:
    """Vị trí bắt đầu **vùng cắt** — gồm đúng một `\\n` đứng ngay trước marker.

    Nuốt một newline để văn bản nhìn thấy không kết thúc bằng dòng trống; chỉ
    một, để chuỗi delta đã stream và `text` sau cắt là **cùng một chuỗi** —
    `CitationHoldback` giữ lại đúng phần này và không hơn.
    """
    start = 0
    while True:
        pos = text.find(MARKER, start)
        if pos < 0:
            return None
        if pos == 0:
            return 0
        if text[pos - 1] == "\n":
            return pos - 1
        start = pos + len(MARKER)


def split_citation_block(full_text: str) -> ParsedAnswer:
    """Cắt block khỏi câu trả lời. JSON validate bằng pydantic (DoD)."""
    cut = _find_marker(full_text)
    if cut is None:
        return ParsedAnswer(text=full_text, claims=(), block="absent")
    marker_pos = cut + 1 if full_text[cut] == "\n" else cut
    text = full_text[:cut]
    raw = full_text[marker_pos + len(MARKER) :].strip()
    if not raw:
        return ParsedAnswer(text=text, claims=(), block="invalid", error="block rỗng sau marker")
    try:
        claims = _CLAIMS.validate_json(raw)
    except ValidationError as exc:
        # Một lỗi JSON *không* trả lại block vào văn bản nhìn thấy: nửa JSON hỏng
        # trên màn hình người dùng tệ hơn một câu trả lời thiếu đuôi.
        return ParsedAnswer(
            text=text,
            claims=(),
            block="invalid",
            error=f"{exc.error_count()} lỗi validate: {exc.errors()[0].get('msg', '?')}",
        )
    return ParsedAnswer(text=text, claims=tuple(claims), block="ok")


def _normalise(value: str) -> str:
    """Chỉ whitespace. Hoa thường giữ nguyên — xem docstring module."""
    return " ".join(value.split())


_ELLIPSIS = re.compile(r"\.{3,}|…|\[\.\.\.\]|\[…\]")
"""Các dạng dấu lược model dùng thật: `...`, `…`, `[...]`, `[…]`."""


def _quote_matches(quote: str, content: str) -> bool:
    """Quote khớp chunk — dấu lược được hiểu là "bỏ một quãng", không phải chữ.

    ## `NEW-08`/`TD-64`: phép đo "từ chối oan" mà docstring module đòi

    `W5-02` đo citation accuracy cấp quote 0,8308 < ngưỡng 0,85, và **19/67**
    lỗi là model chép hai mẩu nguyên văn nối bằng `...` — một cách trích dẫn
    hợp lệ trong văn viết, bị matcher chuỗi-con từ chối như thể quote bịa.
    Đây đúng là phép đo mà điều kiện "nới lỏng phải kèm số" chờ.

    Luật: tách quote theo dấu lược; **mọi** mảnh phải xuất hiện nguyên văn
    (sau chuẩn hoá whitespace) trong chunk, **theo đúng thứ tự**, không chồng
    lấn — `find` tiếp tục từ cuối mảnh trước. Thứ tự là phần giữ độ chặt:
    hai mẩu có thật nhưng đảo chiều là một câu chunk không nói.

    ⚠️ Quote chỉ toàn dấu lược (không còn mảnh nào) trả `False` — trước đây
    `"..." in content` có thể `True` và đó là một quote không nói gì cả.
    """
    haystack = _normalise(content)
    segments = [s for s in (_normalise(part) for part in _ELLIPSIS.split(quote)) if s]
    if not segments:
        return False
    position = 0
    for segment in segments:
        found = haystack.find(segment, position)
        if found < 0:
            return False
        position = found + len(segment)
    return True


@dataclass(frozen=True)
class CitationReport:
    """Đầu ra xác minh, sẵn cho một khung SSE.

    `invalid_ns` tách khỏi `citations` vì một `n` ngoài phạm vi **không có chunk
    nào để gắn** — dựng một `Citation` với `chunk_id` bịa để nhét nó vào danh
    sách là tự làm điều mình đang bắt model.
    """

    citations: tuple[Citation, ...]
    invalid_ns: tuple[int, ...]
    block: Literal["ok", "absent", "invalid"]
    error: str | None = None

    @property
    def verified_count(self) -> int:
        return sum(1 for c in self.citations if c.verified)

    def as_frame(self) -> dict[str, object]:
        return {
            "citations": [c.model_dump(mode="json") for c in self.citations],
            "invalid_ns": list(self.invalid_ns),
            "block": self.block,
            "error": self.error,
            "verified": self.verified_count,
            "total": len(self.citations) + len(self.invalid_ns),
        }


def verify_citations(parsed: ParsedAnswer, chunks: Sequence[Chunk]) -> CitationReport:
    """Đối chiếu từng claim với đúng chunk mà `n` của nó chỉ vào.

    `chunks` theo **đúng thứ tự đã đánh số trong prompt** (`[1]` = `chunks[0]`).
    Quote phải là chuỗi con (sau chuẩn hoá whitespace) của **chính chunk đó** —
    quote đúng nguyên văn nhưng nằm ở chunk khác vẫn là `verified=False`: số
    nguồn sai dẫn người đọc tới nhầm tài liệu, tệ ngang quote bịa.
    """
    citations: list[Citation] = []
    invalid: list[int] = []
    for claim in parsed.claims:
        if claim.n > len(chunks):
            invalid.append(claim.n)
            continue
        chunk = chunks[claim.n - 1]
        meta = chunk.metadata
        citations.append(
            Citation(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                quote=claim.quote,
                verified=_quote_matches(claim.quote, chunk.content),
                source_url=str(meta.source_url) if meta and meta.source_url else None,
                section_path=list(chunk.section_path),
            )
        )
    return CitationReport(
        citations=tuple(citations),
        invalid_ns=tuple(invalid),
        block=parsed.block,
        error=parsed.error,
    )


class CitationHoldback:
    """Giữ block khỏi các khung `delta` khi marker có thể bị cắt đôi giữa stream.

    Hợp đồng: nối mọi chuỗi `feed()` trả ra rồi cộng `flush()` = phần văn bản
    **trước** vùng cắt của `_find_marker` — tức khớp từng byte với
    `split_citation_block(full_text).text`. Test ghim bất biến này bằng cách
    cắt cùng một văn bản ở mọi vị trí.

    Cách giữ: sau mỗi delta, phần đuôi buffer còn *có thể* trở thành
    `"\\n" + MARKER` (hoặc `MARKER` ở byte 0 của stream) chưa được phát ra.
    Thấy marker trọn vẹn ở đầu dòng → từ đó về sau nuốt hết (block chỉ dùng để
    parse ở cuối, không bao giờ phát).
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._capturing = False
        self._at_line_start = True  # đầu stream tính là đầu dòng

    def feed(self, delta: str) -> str:
        if self._capturing:
            return ""
        self._buffer += delta
        probe = ("\n" if self._at_line_start else "") + self._buffer
        cut = _find_marker(probe)
        if cut is not None:
            emit = probe[:cut]
            self._capturing = True
            self._buffer = ""
            return emit[1:] if self._at_line_start else emit
        # Giữ lại đuôi dài nhất còn có thể thành "\n" + MARKER (marker phải đứng
        # sau một newline, hoặc ở byte 0 — trường hợp đó `probe` đã thêm "\n").
        target = "\n" + MARKER
        keep = 0
        limit = min(len(probe), len(target) - 1)
        for length in range(limit, 0, -1):
            if target.startswith(probe[-length:]):
                keep = length
                break
        emit_end = len(probe) - keep
        emit = probe[:emit_end]
        self._buffer = probe[emit_end:]
        if self._at_line_start:
            # Bỏ "\n" mồi đã thêm vào probe — nó có thể nằm ở phần phát ra
            # hoặc phần giữ lại, tuỳ vị trí cắt.
            if emit:
                emit = emit[1:]
            else:
                self._buffer = self._buffer[1:]
        if emit:
            self._at_line_start = emit.endswith("\n")
        return emit

    def flush(self) -> str:
        """Hết stream mà marker chưa trọn vẹn → phần giữ lại là văn bản thường."""
        if self._capturing:
            return ""
        tail, self._buffer = self._buffer, ""
        return tail
