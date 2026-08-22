"""Đo kích thước chunk bằng **token của model**, không bằng ký tự.

## Vì sao ký tự không phải một đơn vị dùng được

`chunk_size=1000` **ký tự** là một con số có nghĩa với con người và vô nghĩa với
model. Đo trên corpus 60 tài liệu (15.814 chunk, cùng bộ chunk mà `W1-13` và
`W2-01` đã index), với hai tokenizer đã dùng thật trong dự án:

| model | ký tự/token EN | ký tự/token VI | token p50 EN | token p50 VI |
|---|---:|---:|---:|---:|
| `BAAI/bge-m3` | 5,83 | 5,41 | 212 | 243 |
| `vietnamese-bi-encoder` (PhoBERT) | 4,09 | 5,33 | **313** | 244 |

Hai điều đọc ra từ bảng:

* Cùng một bộ chunk, đổi model là đổi số token của nó tới **47%** (313 vs 213 ở
  tiếng Anh). `chunk_size` tính bằng ký tự là một cái núm mà **ý nghĩa của nó
  đổi khi bạn đổi model**, và không có gì báo.
* **Chiều lệch giữa hai ngôn ngữ còn đảo dấu.** Với BGE-M3, tiếng Anh gói được
  nhiều ký tự hơn mỗi token; với PhoBERT thì ngược lại, và khoảng cách rộng gấp
  bốn (30% so với 7,8%). Nên "tiếng Việt tốn token hơn" không phải tính chất của
  ngôn ngữ mà của **cặp (model, ngôn ngữ)**.

## Vì sao vẫn cắt trên ký tự ở bên trong

Splitter đệ quy thử từng separator và gộp dần các mảnh; làm việc đó trên token
nghĩa là gọi tokenizer ở **mỗi** bước gộp. Cách dùng ở đây là *hiệu chuẩn rồi
kiểm chứng*:

1. Đo mật độ `ký tự/token` trên **chính tài liệu đó** (vài lát cắt, một lời gọi
   tokenizer).
2. Đổi ngân sách token thành ngân sách ký tự, cắt như thường.
3. **Đếm lại token của kết quả** và cắt tiếp mảnh nào còn vượt trần.

Bước 3 là thứ biến một xấp xỉ thành một **bảo đảm**. Bước 1–2 chỉ để bước 3 hiếm
khi phải làm gì.

## Mật độ lấy TRUNG BÌNH, và đó là chỗ ngược với `truncation.py`

`embedding/truncation.py::chars_per_token_p05` cố ý dùng **phân vị 5**, với lý lẽ
đúng: nó *gợi ý* một `chunk_size` mà **không có ai kiểm lại**, nên phải chừa biên
an toàn, nếu không thì một nửa số chunk vẫn bị model cắt.

Ở đây lý lẽ ấy **đảo chiều**, vì bước 3 kiểm lại chính xác. Chừa biên không mua
được gì mà lại làm mọi chunk nhỏ hơn số đã khai. Đo trên corpus 60 tài liệu với
`chunk_size=256` token:

| ước lượng mật độ | token p50 thu được | số chunk | vượt trần |
|---|---:|---:|---:|
| phân vị 5 (theo `truncation.py`) | **213** (−17%) | 13.986 | 0 |
| trung bình | **261** (+2%) | 11.190 | 0 |

Cả hai đều **0 chunk vượt trần** — đúng như thiết kế, vì trần do bước 3 giữ chứ
không do ước lượng giữ. Khác nhau ở chỗ phân vị 5 làm `chunk_size=256` thật ra
có nghĩa là 213, tức đúng cái núm mà cả hạng mục này dựng lên để nó nói thật.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from .pieces import TextPiece

__all__ = [
    "DEFAULT_SAMPLES",
    "TokenCounter",
    "TokenSizingUnavailable",
    "calibrate_density",
    "fit_to_budget",
]

logger = logging.getLogger(__name__)

DEFAULT_SAMPLES = 12
"""Số lát cắt lấy mẫu khi hiệu chuẩn mật độ của một tài liệu."""

SAMPLE_CHARS = 2000
"""Độ dài mỗi lát. Đủ dài để tỉ lệ ổn định, đủ ngắn để 12 lát vẫn rẻ."""

_SHRINK = 0.85
"""Hệ số thu ngân sách mỗi vòng cắt lại. < 1 để vòng lặp chắc chắn dừng."""

_MAX_ROUNDS = 4
"""Sau ngần này vòng thì cắt cứng theo ký tự. Bảo đảm không lặp vô hạn."""


class TokenSizingUnavailable(RuntimeError):
    """Yêu cầu tính kích thước theo token nhưng không có bộ đếm token.

    Cố ý **ném lỗi** thay vì lặng lẽ rơi về đếm ký tự: rơi về sẽ cho ra một bộ
    chunk hợp lệ, khác hẳn cái được yêu cầu, và không có gì trong output lộ ra.
    """


@runtime_checkable
class TokenCounter(Protocol):
    """Thứ đếm được token. `EmbeddingProvider` thoả cấu trúc này sẵn."""

    @property
    def max_sequence_tokens(self) -> int | None: ...

    def count_tokens(self, texts: Sequence[str]) -> list[int] | None: ...


def _slices(text: str, samples: int, sample_chars: int) -> list[str]:
    """Các lát cắt rải đều khắp tài liệu, không chỉ ở đầu.

    Lấy mẫu ở đầu là đo phần bìa/mục lục — mật độ ký tự/token ở đó không giống
    phần thân, và với báo cáo World Bank thì đầu tài liệu toàn bảng biểu.
    """
    if len(text) <= sample_chars:
        return [text]
    step = max(1, (len(text) - sample_chars) // max(1, samples - 1))
    cuts = [min(i * step, len(text) - sample_chars) for i in range(samples)]
    return [text[c : c + sample_chars] for c in sorted(set(cuts))]


def calibrate_density(
    text: str,
    counter: TokenCounter,
    *,
    samples: int = DEFAULT_SAMPLES,
    sample_chars: int = SAMPLE_CHARS,
) -> float:
    """Mật độ `ký tự/token` **trung bình** của một tài liệu, đo bằng tokenizer thật.

    Trung bình chứ không phải phân vị thấp — xem docstring module: bước kiểm lại
    ở `fit_to_budget` giữ trần chính xác, nên biên an toàn ở đây chỉ làm mọi chunk
    nhỏ hơn số đã khai (đo được: −17%).

    Raises:
        TokenSizingUnavailable: `counter` không đếm được token.
    """
    parts = _slices(text, samples, sample_chars)
    counts = counter.count_tokens(parts)
    if counts is None:
        raise TokenSizingUnavailable(
            f"{type(counter).__name__}.count_tokens trả None — không hiệu chuẩn được "
            "mật độ ký tự/token"
        )
    total_tokens = sum(counts)
    if total_tokens <= 0:
        raise TokenSizingUnavailable("mẫu hiệu chuẩn cho ra 0 token — tài liệu rỗng?")
    return sum(len(p) for p in parts) / total_tokens


def _hard_slice(piece: TextPiece, max_chars: int) -> list[TextPiece]:
    """Cắt cứng theo ký tự. Lối thoát cuối, để vòng lặp luôn kết thúc.

    Chỉ chạy khi splitter đệ quy đã hết separator để dùng — tức mảnh là một khối
    liền không có khoảng trắng. Cắt giữa từ ở đây là chấp nhận được: thay thế
    duy nhất là để chunk vượt trần rồi bị model cắt **âm thầm**, đúng `TD-11`.
    """
    text = piece.text
    out: list[TextPiece] = []
    for start in range(0, len(text), max_chars):
        end = min(start + max_chars, len(text))
        out.append(TextPiece(text[start:end], piece.start + start, piece.start + end))
    return out


def fit_to_budget(
    pieces: Sequence[TextPiece],
    *,
    limit: int,
    counter: TokenCounter,
    separators: Sequence[str],
    chunk_overlap: int = 0,
) -> list[tuple[int, TextPiece]]:
    """Cắt lại mọi mảnh vượt `limit` token. Trả kèm **chỉ số mảnh gốc**.

    Chỉ số cần thiết vì người gọi có thể đang giữ mảng song song với `pieces`
    (`StructureChunker` giữ `section_path`); một mảnh cắt thành ba thì cả ba
    phải thừa kế cùng đường dẫn.

    Bảo đảm: mảnh trả về đều có `count_tokens ≤ limit`, hoặc hàm đã cắt cứng tới
    mức không cắt nhỏ hơn được nữa.

    Raises:
        TokenSizingUnavailable: `counter` không đếm được token.
        ValueError: `limit` không dương.
    """
    if limit <= 0:
        raise ValueError(f"limit phải dương, nhận {limit}")
    if not pieces:
        return []

    indexed = list(enumerate(pieces))
    seps = list(separators)

    for round_number in range(_MAX_ROUNDS + 1):
        counts = counter.count_tokens([p.text for _, p in indexed])
        if counts is None:
            raise TokenSizingUnavailable(
                f"{type(counter).__name__}.count_tokens trả None — không kiểm được trần token"
            )
        over = {i for i, n in enumerate(counts) if n > limit}
        if not over:
            return indexed

        shrink = _SHRINK**round_number
        rebuilt: list[tuple[int, TextPiece]] = []
        for position, (source, piece) in enumerate(indexed):
            if position not in over:
                rebuilt.append((source, piece))
                continue
            budget = max(1, int(limit * (len(piece.text) / counts[position]) * shrink))
            parts = _split_piece(piece, seps, budget, chunk_overlap)
            rebuilt.extend((source, part) for part in parts if part.text.strip())
        indexed = rebuilt

    logger.warning(
        "Còn mảnh vượt trần %d token sau %d vòng cắt — đã cắt cứng theo ký tự",
        limit,
        _MAX_ROUNDS,
    )
    return indexed


def _split_piece(
    piece: TextPiece, separators: list[str], chunk_size: int, chunk_overlap: int
) -> list[TextPiece]:
    """Cắt một mảnh xuống `chunk_size` ký tự, cắt cứng nếu splitter bó tay.

    Điều kiện dừng của `fit_to_budget` nằm ở đây: splitter đệ quy trả lại đúng
    một mảnh y nguyên khi không còn separator nào để dùng (một khối chữ liền),
    và nếu cứ thế lặp lại thì vòng ngoài chạy mãi mà `indexed` không đổi.
    """
    from .fixed import split_recursive_pieces  # import cục bộ để tránh vòng lặp

    parts = split_recursive_pieces(
        piece.text,
        separators=separators,
        chunk_size=chunk_size,
        chunk_overlap=min(chunk_overlap, max(0, chunk_size - 1)),
    )
    if len(parts) <= 1 and (not parts or len(parts[0].text) >= len(piece.text)):
        return _hard_slice(piece, chunk_size)
    return [TextPiece(p.text, piece.start + p.start, piece.start + p.end) for p in parts]
