"""Kiểu dữ liệu chung cho mọi loader, và chỗ để đặt **vân tay parse**.

Phần đáng đọc kỹ nhất ở module này không phải kiểu dữ liệu mà là lý do
`ParseFingerprint` tồn tại.

**Chuỗi toàn vẹn của dự án cho tới hết `W2` dựa vào một sự trùng hợp.**
`pipeline/indexing/corpus_loader.py` đối chiếu `sha256` của **byte** trên đĩa
với manifest, rồi gán thẳng `Document.content = payload.decode("utf-8")`. Vì
phép biến đổi giữa hai đầu là **hàm đồng nhất**, ghim byte cũng chính là ghim
nội dung — nên `TextSpan` của golden set (neo theo offset ký tự trong
`Document.content`) an toàn tuyệt đối.

`W3-01` chèn một bộ parse vào giữa. Từ lúc đó:

```
content = parse(bytes, phiên_bản_parser, tuỳ_chọn_parse)
```

Manifest ghim **một** trong ba đầu vào. Nâng docling từ 2.121 lên 2.122 có thể
đổi một khoảng trắng trong markdown xuất ra, và thế là mọi offset sau chỗ đó
lệch đi — trong khi `sha256` của file vẫn khớp, không phép kiểm nào đỏ, và mọi
con số recall sau đó sai một cách im lặng. Đây đúng là khuôn của lỗi mà
`TD-12`/`chunk_id` đã tốn một tuần để tìm ra, chỉ khác trục.

Nên hợp đồng ở đây là: loader phải khai báo **cả ba** đầu vào (`loader`,
`library_version`, `options`) và trả kèm `text_sha256` của kết quả. Ghim được
hay chưa là chuyện của manifest (`TD-22`); *biết* mà không ghim còn hơn không
biết.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "Heading",
    "LoadedDocument",
    "LoaderError",
    "ParseFingerprint",
    "UnsupportedFormatError",
    "detect_format",
]


class LoaderError(RuntimeError):
    """Không đọc được tài liệu."""


class UnsupportedFormatError(LoaderError):
    """Phần mở rộng không có loader nào nhận."""


def detect_format(path: str | Path) -> str:
    """Phần mở rộng đã chuẩn hoá, ví dụ `.pdf`.

    Cố ý **chỉ** dựa vào tên file chứ không sniff magic bytes: `W3-08` sẽ nhận
    file qua HTTP upload, nơi phần mở rộng là thứ người gửi khai báo, và một
    loader chọn backend theo nội dung thật của file là một mặt tấn công (gửi
    file `.txt` mà ruột là zip). Chọn theo tên rồi để backend tự từ chối khi
    ruột không khớp thì hành vi đoán được hơn.
    """
    return Path(path).suffix.lower()


@dataclass(frozen=True, slots=True)
class ParseFingerprint:
    """Mọi thứ mà đổi cái nào cũng đổi văn bản xuất ra.

    ## Vì sao có `components`, và vì sao thiếu nó là một lỗ hổng thật

    Bản `W3-01` ghi đúng **một** số version: của `docling`. Nhưng gói phát tán
    tên `docling` không phải gói làm ra văn bản. Đo trên môi trường này
    (`TD-22`): `docling==2.121.0` yêu cầu

    ```
    docling-core       >=2.91.0,<3.0.0     ← export_to_markdown() nằm ở ĐÂY
    docling-ibm-models >=3.13.0,<4         ← model layout/table
    docling-parse      >=7.12.0,<8.0.0     ← đọc text layer PDF
    pypdfium2          >=4.30.0,<6.0.0     ← hai major version
    rapidocr           >=3.9.1,<4.0.0      ← chỉ khi bật OCR
    ```

    `LoadedDocument.text` là giá trị trả về của `DoclingDocument.export_to_markdown`,
    và hàm đó sống trong **`docling-core`**. Nên `docling-core` đi từ 2.91 lên
    2.99 là markdown có thể đổi trong khi vân tay **không đổi một ký tự** — đúng
    chế độ hỏng mà `TD-22` mô tả: hash byte vẫn khớp, vân tay vẫn khớp, không
    test nào đỏ, và mọi offset span lệch đi.

    `components` chỉ ghi những gói **đường parse này thật sự đi qua** (PDF khác
    DOCX, OCR khác không OCR). Ghi thừa cũng có giá: một lượt nâng `rapidocr`
    làm mọi tài liệu DOCX báo "parser đã đổi" trong khi chúng chưa từng chạm
    OCR, và một cảnh báo kêu suốt là một cảnh báo bị tắt.
    """

    loader: str
    library: str
    library_version: str
    options: tuple[str, ...] = ()
    components: tuple[str, ...] = ()
    """`"tên=version"` của các gói phụ mà đường parse này đi qua."""

    @property
    def canonical(self) -> str:
        """Dạng chuỗi ổn định — `options`/`components` được sắp để không phụ thuộc lời gọi."""
        parts = [
            self.loader,
            self.library,
            self.library_version,
            *sorted(self.options),
            *sorted(self.components),
        ]
        return "|".join(parts)

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class Heading:
    """Một heading, kèm vị trí của nó trong `LoadedDocument.text`.

    `depth` bắt đầu từ 1 và **đã chuẩn hoá giữa các backend** — xem
    `docling_backend._normalise_depth`, vì cùng một tài liệu ba cấp heading cho
    ra ba cách đánh số khác nhau tuỳ định dạng nguồn.

    `start_char = -1` nghĩa là không định vị được heading trong văn bản xuất ra.
    Giữ heading lại thay vì bỏ đi: mất đường dẫn section còn hơn mất cả heading.
    """

    depth: int
    text: str
    start_char: int = -1

    @property
    def located(self) -> bool:
        return self.start_char >= 0


@dataclass(frozen=True, slots=True)
class LoadedDocument:
    """Kết quả của một lần parse.

    `text` là markdown với mọi định dạng đi qua docling, và là **nguyên văn**
    với `.txt`. Sự khác nhau đó có chủ đích và là điều kiện để mọi con số của
    `W2` còn giá trị — xem `loaders/__init__.py`.
    """

    text: str
    source_sha256: str
    fingerprint: ParseFingerprint
    headings: tuple[Heading, ...] = ()
    table_count: int = 0
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def text_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def max_depth(self) -> int:
        return max((h.depth for h in self.headings), default=0)

    def section_path_at(self, offset: int) -> tuple[str, ...]:
        """Đường dẫn heading có hiệu lực tại vị trí `offset` của `text`.

        Đây là thứ `W3-03` cần để gán `section_path` cho từng chunk mà không
        phải parse lại markdown. Heading không định vị được thì **bỏ qua** — nó
        không có vị trí thì không nói được nó có hiệu lực ở đâu.

        ⚠️ Bản `W3-01` viết `break` cho cả hai điều kiện, tức **một** heading
        không định vị được sẽ làm mù toàn bộ phần còn lại của tài liệu: mọi
        heading sau nó biến mất khỏi đường dẫn, im lặng. `_collect` sinh ra đúng
        tình huống ấy (`text.find` thất bại thì `start_char = -1` nhưng `cursor`
        không đổi, nên heading kế tiếp vẫn định vị được). `W3-03` là chỗ tiêu thụ
        đầu tiên nên cũng là chỗ phát hiện ra.
        """
        stack: list[Heading] = []
        for heading in self.headings:
            if not heading.located:
                continue
            if heading.start_char > offset:
                break
            while stack and stack[-1].depth >= heading.depth:
                stack.pop()
            stack.append(heading)
        return tuple(h.text for h in stack)

    def as_metadata(self) -> dict[str, str]:
        """Phần đi vào `DocumentMetadata.extra` — đủ để truy lại lần parse này."""
        return {
            "parser": self.fingerprint.loader,
            "parser_library": self.fingerprint.library,
            "parser_version": self.fingerprint.library_version,
            "parse_fingerprint": self.fingerprint.digest,
            "text_sha256": self.text_sha256,
            **self.extra,
        }
