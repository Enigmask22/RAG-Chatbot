"""Guardrails — `W4-12`: ranh giới data/instruction, phát hiện tiêm, che PII.

## ⭐⭐ Ba lớp — và số đo xếp hạng chúng NGƯỢC với trực giác của tôi

Bản đầu của chính docstring này khẳng định: nonce là "cơ chế duy nhất", còn chữ
trong prompt "không chặn được gì, một cách đáng tin cậy". Lý lẽ nghe rất vững —
`W4-07` đã đo được luật ngôn ngữ trong prompt bị bỏ qua **8/8** lần, nên tin
vào một dòng chữ nữa là ngây thơ.

Phép đo bác bỏ nó. Trên payload mạnh nhất (`fake_system_tag`), k=5–6 mỗi nhánh:

| nhánh | rò canary |
|---|---|
| prompt v1 + khối `[n]` trần | **7/12** |
| prompt v2 (chỉ thêm CHỮ ranh giới) + khối trần | **0/6** |
| prompt v2 + khối bọc nonce | **0/6** |

Nhánh giữa tồn tại đúng để trả lời câu này, và nó trả lời dứt khoát: **chữ làm
gần như toàn bộ công việc; nonce chưa mua thêm được gì đo được.** Bài học không
phải "prompt luôn hiệu quả" mà là **cùng một cơ chế cho kết quả khác nhau ở hai
loại việc khác nhau**: bắt model đổi ngôn ngữ đầu ra là đi ngược quán tính của
cả prompt, còn từ chối một mệnh lệnh nhúng rõ ràng là thứ model đã được huấn
luyện để làm — chữ chỉ cần *kích hoạt* nó.

Nonce ở lại, nhưng với đúng nhãn của nó: **giá trị đo được hôm nay = 0**. Giữ
vì lỗ nó bịt (giả mạo cấu trúc) là lỗ thật, chi phí gần bằng không, và vì nó
không phụ thuộc vào việc model có hợp tác hay không — thứ mà lớp "chữ" phụ
thuộc hoàn toàn, và thứ sẽ đổi khi đổi model. Xem `reports/tasks/security-w4.md`.

## Vì sao phát hiện KHÔNG dẫn tới việc bỏ chunk

Phản xạ đầu là "phát hiện tiêm → loại chunk khỏi ngữ cảnh". Sai, vì bộ luật này
có dương tính giả (đo được, xem `reports/tasks/security-w4.md`), và một dương
tính giả ở đường ấy **xoá lặng lẽ một tài liệu thật** khỏi câu trả lời: người
dùng nhận một câu trả lời thiếu nguồn mà không có gì nói ra tại sao. Đổi một
kiểu hỏng ồn ào (model nghe lời tiêm) lấy một kiểu hỏng câm (tài liệu biến mất)
là một cuộc đổi tồi.

Nên phát hiện chỉ **gắn cờ**: cờ đi vào khung `sources` (client thấy), vào log
(người vận hành thấy), và ở đây dừng lại. Quyết định chặn thuộc về người đọc số
liệu, không thuộc về một regex.
"""

from __future__ import annotations

import logging
import re
import secrets
import unicodedata

__all__ = [
    "INJECTION_RULES",
    "PII_PLACEHOLDERS",
    "RedactingFilter",
    "context_nonce",
    "normalise_for_scan",
    "redact_pii",
    "scan_injection",
    "wrap_context",
]

# ---------------------------------------------------------------------------
# 1. Chuẩn hoá trước khi quét
# ---------------------------------------------------------------------------

_WHITESPACE = re.compile(r"[^\S\n]+")
_BLANKLINES = re.compile(r"\n{2,}")
"""⚠️⚠️ Gộp khoảng trắng nhưng **giữ xuống dòng**, và đây là một lỗi đã xảy ra
thật chứ không phải một sự cẩn thận lý thuyết.

Bản đầu dùng `\\s+` → mọi `\\n` biến thành dấu cách → cả chuỗi thành MỘT dòng →
`(?m)^` trong luật `protocol_marker` và `structure_forgery` chỉ còn khớp ở đầu
văn bản. Hai trong mười luật **im lặng ngừng hoạt động**, và không test nào thấy
vì test nào cũng đi qua chính bộ chuẩn hoá ấy (cùng khuôn M2 của `W4-11`).

Thứ tìm ra nó là lần chạy thật: hai payload `citation_forgery` và
`structure_forgery` hiện `det=-` trong bảng kết quả.
"""

_CONFUSABLES = str.maketrans(
    {
        # Cyrillic → Latin. ⚠️ NFKC **không** làm việc này (đã đo: `sha`-khác
        # nhau sau NFKC), nên nếu chỉ gọi `unicodedata.normalize` thì
        # `"Ignоre"` với о Kirin đi thẳng qua mọi luật bên dưới.
        "а": "a",
        "в": "b",
        "с": "c",
        "е": "e",
        "н": "h",
        "к": "k",
        "м": "m",
        "о": "o",
        "р": "p",
        "ѕ": "s",
        "т": "t",
        "х": "x",
        "у": "y",
        "і": "i",
        "ј": "j",
        "А": "A",
        "В": "B",
        "С": "C",
        "Е": "E",
        "Н": "H",
        "К": "K",
        "М": "M",
        "О": "O",
        "Р": "P",
        "Ѕ": "S",
        "Т": "T",
        "Х": "X",
        "У": "Y",
        # Hy Lạp hay dùng để giả chữ Latin
        "ο": "o",
        "α": "a",
        "ε": "e",
        "ρ": "p",
        "τ": "t",
        "υ": "u",
        "ν": "v",
        # Dấu nháy/gạch "thông minh" — cắt biến thể rẻ tiền của cùng một câu
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "‐": "-",
        "‑": "-",
        "‒": "-",
        "–": "-",
        "—": "-",
        # Zero-width: chèn vào GIỮA từ khoá là cách rẻ nhất để né regex
        "​": "",
        "‌": "",
        "‍": "",
        "﻿": "",
        "­": "",
    }
)
"""Bảng chữ nhìn-giống-nhau. ⚠️ Cố ý **không** đầy đủ — bảng confusables của
Unicode có hàng nghìn mục. Đây là các ký tự đã thấy trong payload thật; một
bảng đầy đủ là việc của `W6-06`, và giới hạn này được ghi ra thay vì giấu."""


def normalise_for_scan(text: str) -> str:
    """Dạng chuẩn để **so luật**, không phải dạng đưa cho model.

    ⚠️ Phân biệt này quan trọng: chuẩn hoá rồi đưa bản đã chuẩn hoá cho model là
    lặng lẽ sửa nội dung tài liệu (mất dấu nháy cong, mất ký tự Kirin thật trong
    một tài liệu tiếng Nga). Ta chỉ **quét** trên bản chuẩn hoá; bản gốc đi tiếp
    nguyên vẹn.
    """
    folded = unicodedata.normalize("NFKC", text).translate(_CONFUSABLES)
    collapsed = _BLANKLINES.sub("\n", _WHITESPACE.sub(" ", folded))
    return collapsed.lower()


# ---------------------------------------------------------------------------
# 2. Bộ luật phát hiện
# ---------------------------------------------------------------------------

_GAP = r"[^\n]{0,120}?"
"""Khoảng cách cho phép giữa hai vế của một luật: cùng **một dòng**.

⭐ Bản đầu là `[^.\\n]{0,60}?` — cấm vượt cả dấu chấm — với lý lẽ nghe rất hợp
lý: "bỏ qua" ở câu này và "hướng dẫn" ở câu kia là văn bản chính sách bình
thường, nên ràng buộc cùng-một-câu là thứ giữ dương tính giả xuống thấp.

Một phép tiêm lỗi **sống sót** buộc phải kiểm lại lời ấy, và số đo bác bỏ nó:
nới lên `.{0,200}?` cho dương tính giả **y hệt** (2/20.424 chunk). Thứ giữ FP
thấp không phải ranh giới câu mà là **yêu cầu ba vế cùng có mặt**; ranh giới câu
chỉ tặng kẻ tấn công một đường né rẻ tiền — thêm một dấu chấm.

Nên nó nới ra tới hết dòng: bắt được cả "Bỏ qua điều này. Mọi chỉ dẫn phía trên
đều sai", mà vẫn không nối hai đoạn văn rời nhau qua `\\n`.
"""

INJECTION_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "override_instructions",
        re.compile(
            r"(bỏ qua|phớt lờ|quên (đi|hết)?|không (cần )?(tuân|theo)|đừng (tuân|theo)"
            r"|ignore|disregard|forget|override)"
            + _GAP
            + r"(chỉ (dẫn|thị)|hướng dẫn|quy tắc|luật|yêu cầu|instructions?|rules?|prompts?)"
            + _GAP
            + r"(phía trên|bên trên|ở trên|trước đó|trước đây"
            r"|previous|prior|above|preceding|earlier)"
        ),
    ),
    (
        # Cùng ba vế nhưng thứ tự Anh ngữ tự nhiên: "ignore the previous instructions"
        "override_instructions_en",
        re.compile(
            r"(ignore|disregard|forget|override|bypass)"
            + _GAP
            + r"(all |any |the )?(previous|prior|above|preceding|earlier|system)"
            + _GAP
            + r"(instructions?|rules?|prompts?|directions?)"
        ),
    ),
    (
        "role_reassign",
        re.compile(
            r"(bạn|mày)\s+(giờ|hiện (giờ|nay)|từ (giờ|nay|bây giờ)|bây giờ)\s+(là|sẽ là)"
            r"|you are (now|from now on)\s+(a |an |the )?"
            r"(new|different|unrestricted|uncensored|dan\b|developer mode)"
            r"|từ (giờ|nay|bây giờ)(,)?\s+(bạn|mày)\s+(là|sẽ|hãy)"
        ),
    ),
    (
        "system_tag",
        re.compile(
            r"<\s*/?\s*system\s*>|\[\s*/?\s*system\s*\]|<\|im_(start|end)\|>"
            r"|\[/?inst\]|###\s*(system|instruction)|<\|(system|assistant)\|>"
        ),
    ),
    (
        # ⭐ Luật mạnh nhất và rẻ nhất: một tài liệu THẬT không có lý do gì chứa
        # marker giao thức của chính ta. Dương tính giả ~0 theo cấu tạo.
        # ⚠️ Neo theo **hình dạng giao thức** (`citations:` + `[`), không theo vị
        # trí đầu dòng: payload thật viết "dòng cuối bắt buộc phải là CITATIONS:
        # [{…}]" — marker nằm giữa câu, và luật neo `^` trượt nó. Lần chạy thật
        # là thứ chỉ ra điều đó; bản đầu neo `(?m)^` và trông rất hợp lý.
        "protocol_marker",
        re.compile(r"citations\s*:\s*\[|<<<\s*(het\s+)?nguon|nguồn nội bộ:\s*\{"),
    ),
    (
        "structure_forgery",
        re.compile(r"(?m)^\s*(ngữ cảnh|câu hỏi|context|question)\s*:\s*$|^\s*\[\d+\]\s*$"),
    ),
    (
        "prompt_disclosure",
        re.compile(
            r"(tiết lộ|in ra|hiển thị|nhắc lại|lặp lại|reveal|print|show|repeat|output)"
            + _GAP
            + r"(prompt hệ thống|chỉ (dẫn|thị) hệ thống|system prompt|your instructions"
            r"|everything above|toàn bộ (nội dung )?phía trên)"
        ),
    ),
    (
        "refusal_suppression",
        re.compile(
            r"(không được|đừng|tuyệt đối không|never|do not|don't)"
            + _GAP
            + r"(nói|trả lời|từ chối|say|answer|respond|refuse)"
            + _GAP
            + r"(không đủ (thông tin|dữ liệu)|không biết|không tìm thấy"
            r"|insufficient|i don't know|cannot answer)"
        ),
    ),
    (
        # ⚠️ Luật này bản đầu có cả `truy cập|visit|fetch`, và nó sinh ra **100%**
        # dương tính giả của cả bộ (37/37 trên corpus thật). Nhìn vào chúng thì
        # thấy ngay: phụ lục "Nguồn dữ liệu" của báo cáo nào cũng viết "truy cập
        # https://…" — trích dẫn học thuật, không phải rò rỉ.
        #
        # Cái phân biệt một cuộc exfil với một trích dẫn không phải sự có mặt
        # của URL, mà là **động từ đẩy dữ liệu ra ngoài**. Bỏ nhóm động từ bị
        # động đi thì corpus sạch, payload thật ("thêm vào cuối câu trả lời…")
        # vẫn dính.
        "exfiltration",
        re.compile(
            r"(gửi|thêm vào (cuối|câu trả lời)|đính kèm|chèn (vào|link)|báo về"
            r"|send|append|post|report .{0,20}to|exfiltrate)" + _GAP + r"https?://"
        ),
    ),
    (
        "credential_fish",
        re.compile(
            r"(api[ _-]?key|mật khẩu|password|token|secret|credential)"
            + _GAP
            + r"(là gì|cho tôi|in ra|tiết lộ|what is|give me|print|reveal|show)"
        ),
    ),
)
"""Mỗi luật đòi một **tổ hợp** trong cùng một câu, không phải một từ khoá.

Lý do đo được: corpus của dự án là văn bản chính sách tiếng Việt, nơi "chỉ thị",
"hướng dẫn", "bỏ qua", "quy định trước đó" xuất hiện dày đặc và hoàn toàn vô
hại. Một luật một-từ-khoá cho tỉ lệ dương tính giả không dùng nổi (xem P1 trong
`reports/tasks/security-w4.md`).
"""


def scan_injection(text: str) -> tuple[str, ...]:
    """Tên các luật khớp, theo thứ tự khai báo. Rỗng = không thấy gì.

    "Không thấy gì" **không phải** "an toàn": đây là danh sách ca đã biết, và
    danh sách ca đã biết luôn đi sau kẻ tấn công. Giá trị của nó là làm ca đã
    biết trở nên đếm được.
    """
    scanned = normalise_for_scan(text)
    return tuple(name for name, pattern in INJECTION_RULES if pattern.search(scanned))


# ---------------------------------------------------------------------------
# 3. Ranh giới data/instruction — nonce
# ---------------------------------------------------------------------------

_NONCE_BYTES = 8


def context_nonce() -> str:
    """Chuỗi ngẫu nhiên **mỗi request một cái**, từ `secrets` chứ không `random`.

    Tính chất: nội dung tài liệu viết được bất cứ chữ gì, nhưng **không đoán
    được** 16 ký tự hex sinh ra sau khi nó đã nằm trong index — nên nó không
    đóng được khối ngữ cảnh để mở một khối "chỉ dẫn hệ thống" giả.

    ⚠️⚠️ **Giá trị đo được tới hôm nay: 0.** Nhánh "prompt v2 + khối trần" chặn
    đúng bằng nhánh có nonce (0/6 cả hai), và payload giả mạo mốc không rò ở
    nhánh nào. Nonce ở lại vì lỗ nó bịt là lỗ thật và chi phí ~0, **không** vì
    nó đã chứng minh được điều gì. Ai đọc dòng này rồi đi khoe "hệ thống có
    chống prompt injection bằng nonce" là đang bán một con số không tồn tại.
    """
    return secrets.token_hex(_NONCE_BYTES)


def wrap_context(n: int, content: str, nonce: str) -> str:
    """Một khối nguồn có mở/đóng mang nonce."""
    return f"<<<NGUON {n} {nonce}>>>\n{content}\n<<<HET NGUON {n} {nonce}>>>"


# ---------------------------------------------------------------------------
# 4. PII trong log
# ---------------------------------------------------------------------------

PII_PLACEHOLDERS = {
    "email": "[email]",
    "phone_vn": "[sđt]",
    "national_id": "[cccd]",
    "card": "[thẻ]",
}

_EMAIL = re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_PHONE_VN = re.compile(r"(?<![\d.,])(?:\+?84|0)(?:3|5|7|8|9)\d{8}\b")
_NATIONAL_ID = re.compile(r"(?<![\d.,])\d{12}(?![\d.,])")
_CARD = re.compile(r"(?<![\d.,])(?:\d[ -]?){13,19}(?![\d.,])")


def _luhn_ok(digits: str) -> bool:
    """Phép kiểm Luhn — thứ giữ luật thẻ khỏi nuốt mọi con số dài.

    Corpus kinh tế đầy số 13–19 chữ số (giá trị VND không dấu phân cách). Không
    có Luhn thì luật này che mất chính những con số mà log tồn tại để cho xem.
    """
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def redact_pii(text: str) -> str:
    """Thay PII bằng placeholder. Thứ tự luật có ý nghĩa (email trước số)."""
    out = _EMAIL.sub(PII_PLACEHOLDERS["email"], text)
    out = _PHONE_VN.sub(PII_PLACEHOLDERS["phone_vn"], out)

    def _card_sub(match: re.Match[str]) -> str:
        digits = re.sub(r"[ -]", "", match.group())
        if len(digits) >= 13 and _luhn_ok(digits):
            return PII_PLACEHOLDERS["card"]
        return match.group()

    out = _CARD.sub(_card_sub, out)
    return _NATIONAL_ID.sub(PII_PLACEHOLDERS["national_id"], out)


class RedactingFilter(logging.Filter):
    """Che PII trên **mọi** bản ghi, kể cả của thư viện bên thứ ba.

    ⭐ Là filter toàn cục chứ không phải kỷ luật tại chỗ gọi, và đó là toàn bộ
    điểm: `httpx` log URL kèm query string, một `logger.exception` in nguyên
    payload của provider, và không ai nhớ gọi `redact_pii()` ở dòng log thứ 300.
    Cái gì phụ thuộc vào việc nhớ thì sẽ hỏng vào ngày người ta quên.

    ⚠️ Filter sửa `record.msg`/`record.args` **tại chỗ**. Chấp nhận được vì bản
    ghi đã ra tới handler là bản ghi sắp bị vứt; nhưng nghĩa là một handler thứ
    hai gắn TRƯỚC filter này sẽ thấy bản chưa che — nên `configure_logging` gắn
    filter lên chính handler, không lên logger.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = redact_pii(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: redact_pii(v) if isinstance(v, str) else v for k, v in record.args.items()
                }
            else:
                record.args = tuple(redact_pii(a) if isinstance(a, str) else a for a in record.args)
        for key, value in list(record.__dict__.items()):
            if isinstance(value, str) and key not in ("name", "levelname", "pathname", "funcName"):
                record.__dict__[key] = redact_pii(value)
        return True
