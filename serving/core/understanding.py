"""Hiểu câu hỏi trước khi đi tìm tài liệu — `W4-07`.

Ba việc, và chúng khác nhau về **bản chất** chứ không chỉ về nội dung:

| việc | tín hiệu | cái giá khi sai |
|---|---|---|
| định tuyến `NO_RETRIEVAL`/`CLARIFY`/`RETRIEVE` | luật, miễn phí | bỏ truy hồi một câu hỏi thật |
| viết lại câu hỏi đa lượt | một lượt gọi LLM | truy hồi bằng một câu không ai hỏi |
| phát hiện ngôn ngữ | luật, miễn phí | ép model trả lời sai ngôn ngữ |

## ⭐⭐ Vì sao bộ phân loại này **bất đối xứng có chủ đích**

Hai hướng sai không bằng giá nhau, và chênh nhau rất xa:

* **Truy hồi thừa** (đáng lẽ `NO_RETRIEVAL` nhưng vẫn đi tìm): tốn ~800 ms và
  vài chunk vô hại nhét vào prompt. Người dùng không thấy gì khác.
* **Truy hồi thiếu** (một câu hỏi thật bị xếp là chào hỏi): model trả lời bằng
  kiến thức nội tại của nó, **không có nguồn nào**, và giọng vẫn tự tin y hệt.
  Đó chính xác là thứ mà cả hệ thống RAG này tồn tại để không xảy ra.

Nên luật không phải "điểm vượt ngưỡng" mà là **"mọi token đều nằm trong một từ
vựng đóng, và ít nhất một token là lời chào thật"**. Hình dạng ấy không có núm
nào để vặn, và một lời chào lạ không có trong từ vựng thì chỉ đi truy hồi thừa.

⚠️⚠️ Vế thứ hai được thêm vào **sau** một phép tiêm lỗi, và bản đầu của chính
đoạn văn này khẳng định sai. Nó viết rằng "một câu hỏi không bao giờ lọt qua
được vì nó luôn chứa ít nhất một từ ngoài từ vựng" — nhưng từ vựng khi ấy có cả
từ xưng hô (`bạn`, `anh`, `chị`, `em`, `thầy`, `cô`), mà chúng **cũng là danh từ
nội dung**. `"thầy cô"` và `"chị em"` do đó rơi thẳng vào nhánh đắt tiền. Bài
học không phải về tiếng Việt: **một từ vựng đóng chỉ an toàn khi mọi từ trong nó
chỉ có một vai.**

⚠️ Đây cũng là lý do luật **không** phải "câu có chứa lời chào". `"hello, what is
the poverty line?"` chứa một lời chào và là một câu hỏi thật; `"Chào mừng đầu tư
nước ngoài..."` chứa `chào` như tiền tố của một từ khác. Cả hai đều có case ghim.

## ⭐ Vì sao không dùng thư viện phát hiện ngôn ngữ

`langdetect` / `fasttext` trả về một nhãn kèm điểm tin cậy cho **mọi** đầu vào,
kể cả một câu ba chữ mà không ai đoán nổi — và `langdetect` còn không tất định
nếu không gieo hạt, thứ mà quy tắc "mọi cái trên đường eval phải tái lập được"
đã loại từ đầu.

Bài học của `TD-37` đúng nguyên văn ở đây: một tín hiệu **tự tin và sai** tệ hơn
một tín hiệu **thiếu**. Nên hàm ở đây trả `"unknown"` khi không chắc, và
`"unknown"` có hệ quả cụ thể: **không sinh chỉ thị ngôn ngữ nào cả**. Ép sai
ngôn ngữ tệ hơn không ép gì.

Dấu thanh tiếng Việt là tín hiệu gần như không thể nhầm — có dấu thì là tiếng
Việt. Chiều ngược lại thì không đúng (tiếng Việt viết không dấu tồn tại), và
hàm này trả `"unknown"` cho nó thay vì đoán.

## ⭐⭐ Chỉ thị ngôn ngữ: đo được, và kết quả 8/8 → 0/8

Lần chạy thật của `W4-06` thấy model đáp tiếng Việt cho một câu hỏi tiếng Anh,
dù luật 4 của `SYSTEM_PROMPT` nói ngược lại. Dự đoán của tôi trước khi đo là một
dòng chỉ thị thêm vào sẽ *giảm* tỉ lệ ấy chứ không xoá được nó.

Đo thật (`reports/probes/w4-07-language-directive.json`, 8 câu hỏi tiếng Anh,
cùng ngữ cảnh đã truy hồi, `deepseek-v4-flash`, `temp=0`):

| | trả lời sai ngôn ngữ |
|---|---|
| không có chỉ thị (= hành vi `W4-06`) | **8/8** |
| có `"Answer in English."` cuối lượt người dùng | **0/8** |

Hai điều bất ngờ, và cả hai đều ngược dự đoán. Tỉ lệ nền không phải "thỉnh
thoảng" mà là **tất cả** — chuyện của `W4-06` không phải xui, nó tất định. Và
một dòng chữ vá được **toàn bộ** trên mẫu này.

💡 Cơ chế nhiều khả năng không phải "model nghe lời": `SYSTEM_PROMPT` viết hoàn
toàn bằng tiếng Việt và phần lớn chunk cũng tiếng Việt, nên mặc định của prompt
kéo về tiếng Việt; chỉ thị nằm ở **cuối lượt người dùng** thắng vì nó gần nhất
và tường minh nhất. Đó là lý do vị trí của nó là một quyết định, không phải một
chi tiết — xem `QueryPlan.directive`.

⚠️ 8 câu, một model, một prompt. Đây là bằng chứng đủ để **bật** nó, không phải
một phát biểu về việc model tuân chỉ dẫn nói chung. Nên phép đo ở lại: có
`detect_language()` thì ngôn ngữ câu trả lời cũng đo được, và chênh lệch thành
một con số trong khung `done` thay vì một giai thoại — hôm nay 0/8, và nếu ngày
mai đổi model thì con số ấy tự nói ra.

Chỗ duy nhất ngôn ngữ hoạt động như một **cơ chế** là câu hỏi lại của
`CLARIFY`: text ấy do mã chọn từ một bảng, không do model sinh, nên nó luôn đúng
ngôn ngữ đã phát hiện — hoặc song ngữ khi không phát hiện được.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from rag_core.generation import default_registry
from rag_core.llm import ChatMessage, LLMError, LLMProvider

__all__ = [
    "CLARIFY_FALLBACK",
    "CLARIFY_TEXT",
    "LANGUAGE_DIRECTIVE",
    "QUERY_REWRITE",
    "REWRITE_SYSTEM_PROMPT",
    "Language",
    "QueryPlan",
    "QueryUnderstanding",
    "Route",
    "classify",
    "detect_language",
]

logger = logging.getLogger(__name__)

Route = Literal["retrieve", "no_retrieval", "clarify"]
Language = Literal["vi", "en", "unknown"]


# ------------------------------------------------------------------ từ vựng


def _words(block: str) -> frozenset[str]:
    """Một khối chữ → tập từ.

    Từ vựng ở dưới viết thành khối chứ không thành list literal: chúng là **danh
    sách từ**, và một danh sách từ đọc được, sửa được, so diff được khi nó trông
    như văn bản. Bọc qua hàm này thay vì rải bốn `# noqa: SIM905`.
    """
    return frozenset(block.split())


_VIETNAMESE_MARKS = set("àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ")
"""Chữ cái chỉ tiếng Việt mới có. Một chữ trong đây là đủ để kết luận."""

_COMMON_ENGLISH = _words(
    """
    the a an and or but if of in on at to for from by with about into over after
    is are was were be been being do does did done have has had can could will
    would should may might must
    i you he she it we they me him her us them my your his its our their this
    that these those there here what which who whom whose when where why how
    not no yes all any some more most much many few
    hello hi hey good morning afternoon evening thanks thank bye goodbye please
    sorry ok okay cheers welcome
    """
)
"""Từ tiếng Anh **thường gặp**, không phải một danh sách từ chức năng thuần.

Có cả `hello`/`thanks` vì đường này phải nhận ra được đúng những câu ngắn nhất
mà endpoint gặp nhiều nhất. **Không** có `per`, `via`, `versus`… — chúng là giới
từ nhưng không nằm trong nhóm phổ biến nhất, và thêm chúng vào chỉ để một case
xanh thì danh sách này thôi không còn là một danh sách nữa mà thành một tập
tham số vặn theo bộ test.
"""

_SOCIAL_CORE = _words(
    """
    hello hi hey yo greetings morning afternoon evening night
    thanks thank bye goodbye cheers ok okay okey welcome
    chào chao cảm cám ơn tạm biệt hẹn
    """
)
"""Từ **chỉ** dùng để chào, cảm ơn, chào tạm biệt. Không từ nào trong đây là một
chủ đề mà người ta đi tra tài liệu."""

_SOCIAL_PARTICLES = _words(
    """
    good you see ya k all thats that everything else nice
    xin nhé nha ạ à ừ thôi rồi gặp lại
    """
)
"""Tiểu từ thuần: chúng không mang nội dung ở bất kỳ câu nào."""

_SOCIAL_ADDRESS = _words("bạn anh chị em mình thầy cô")
"""⚠️ Từ xưng hô — và chúng **cũng là danh từ nội dung**.

`"thầy cô"`, `"chị em"` là chủ đề người ta tra thật. Nên chúng đứng cạnh một lời
chào thì bị nuốt (`"chào bạn"`), nhưng đứng một mình thì vẫn **đếm là nội dung**
và câu đi truy hồi. Chúng cố ý **không** nằm trong `_STOPWORDS`.
"""

_SOCIAL_FILLER = _SOCIAL_PARTICLES | _SOCIAL_ADDRESS
"""⭐⭐ Từ đi **kèm** một lời chào, nhưng tự chúng không phải lời chào.

Đây là chỗ một phép tiêm lỗi phơi ra lỗ thật. Bản đầu gộp cả hai nhóm vào một
từ vựng và hỏi "mọi token có nằm trong đó không" — nên `"thầy cô"` và
`"chị em"`, hai **danh từ nội dung** hoàn toàn bình thường của tiếng Việt, bị
xếp là chào hỏi và **không được truy hồi**. Đúng hướng hỏng đắt mà docstring
module thề là không thể xảy ra: một câu hỏi thật được trả lời không nguồn.

Nên luật thành hai vế: mọi token phải nằm trong từ vựng, **và** ít nhất một
token phải là `_SOCIAL_CORE`. `"bạn"` một mình đi truy hồi; `"chào bạn"` thì
không.

⚠️ Cố ý **không** có `vậy`, `thế`, `đó` ở cả hai nhóm — chúng là từ chỉ trỏ, và
một câu chỉ gồm chúng (`"thế ạ?"`) là chuyện của `CLARIFY`.
"""

_SOCIAL_TOKENS = _SOCIAL_CORE | _SOCIAL_FILLER

_REFERENCE_WORDS = _words(
    """
    đó này nầy ấy đấy kia nó họ chúng vậy thế
    it its that this they them those these he she him her there
    """
)
"""Từ **chỉ trỏ**: chúng trỏ ra ngoài câu, nên câu chứa chúng có thể không tự đủ nghĩa."""

_CONTINUATION = re.compile(
    r"^\s*(còn|thế còn|vậy còn|còn về|thế thì|vậy thì|what about|how about|and|or)\b",
    re.IGNORECASE,
)
"""Tỉnh lược **danh từ**: `"còn Lào?"`, `"what about 2020?"`.

⭐ Không có mẫu này thì cả một họ câu hỏi đa lượt lọt lưới, vì chúng không chứa
đại từ nào để bắt — chỗ trống nằm ở vị trí danh từ, và tiếng Việt bỏ hẳn nó đi.
Mẫu chỉ khớp ở **đầu câu**: `còn` giữa câu là "vẫn còn", một nghĩa khác hẳn.
"""

_VIETNAMESE_FUNCTION = _words(
    """
    là của có không ở và với cho một các những được bao nhiêu nào gì ai khi đâu
    sao thì mà nhưng hoặc cũng đã đang sẽ rất quá hơn nhất về từ đến tới trong
    ngoài trên dưới cái điều việc con người ta chứ ư nhỉ vẫn còn nữa lại chỉ
    """
)

_STOPWORDS = (
    _COMMON_ENGLISH | _SOCIAL_CORE | _SOCIAL_PARTICLES | _REFERENCE_WORDS | _VIETNAMESE_FUNCTION
)
"""Từ **không** tính là nội dung khi đếm "câu này có gì để truy hồi không".

⭐ `_SOCIAL_ADDRESS` cố ý vắng mặt: xem docstring của nó.
"""

_TOKEN_SPLIT = re.compile(r"[^\w]+", re.UNICODE)

_APOSTROPHES = str.maketrans("", "", "'’ʼ")
"""⭐ Dấu lược nằm **trong** từ, không phải giữa hai từ.

Tách theo `[^\\w]+` một cách ngây thơ thì `"that's"` thành `that` + `s`, và cái
`s` mồ côi ấy không thuộc từ vựng nào cả — nó biến `"thanks, that's all"` thành
một câu có "từ nội dung", tức thành một câu hỏi. Đúng một lỗi ấy là ca duy nhất
sai ở lần chạy đầu trên bộ 18 case, và nó không phải lỗ hổng từ vựng: nó chạm
**mọi** dạng rút gọn tiếng Anh (`don't`, `what's`, `it's`).
"""


def _tokens(text: str) -> list[str]:
    stripped = text.lower().translate(_APOSTROPHES)
    return [tok for tok in _TOKEN_SPLIT.split(stripped) if tok]


# ------------------------------------------------------------------ ngôn ngữ


def detect_language(text: str) -> Language:
    """`"vi"` / `"en"` / `"unknown"` — và `"unknown"` là một câu trả lời hợp lệ.

    Xem §"Vì sao không dùng thư viện" ở docstring module: hàm này **từ chối
    đoán**, vì hệ quả duy nhất của nhãn ngôn ngữ là một chỉ thị bắt model trả
    lời bằng ngôn ngữ ấy — và ép sai tệ hơn không ép.
    """
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return "unknown"
    non_latin = sum(1 for ch in letters if not unicodedata.name(ch, "").startswith("LATIN"))
    if non_latin * 2 > len(letters):
        # Chữ Hán, Nhật, Ả Rập, Thái… — ngoài hai ngôn ngữ đã biết. Trả
        # `"unknown"` chứ không im lặng chọn `"en"` vì nó không viết bằng chữ
        # Latin: đó là đoán, và nó đoán sai với đúng nhóm người dùng mà phép
        # kiểm này lẽ ra phải bảo vệ.
        return "unknown"
    lowered = text.lower()
    if any(ch in _VIETNAMESE_MARKS for ch in lowered):
        return "vi"
    if any(tok in _COMMON_ENGLISH for tok in _tokens(text)):
        return "en"
    # Tiếng Việt viết không dấu rơi vào đây, và `"unknown"` là mô tả đúng: chuỗi
    # `"ti le ngheo"` không phân biệt được với một mã sản phẩm.
    return "unknown"


# ------------------------------------------------------------------ định tuyến


def _is_social(tokens: Sequence[str]) -> bool:
    """Mọi token trong từ vựng xã giao, **và** ít nhất một token là lời chào thật.

    Vế thứ hai không phải để cho chặt hơn: thiếu nó thì `"thầy cô"` — hai danh từ
    nội dung — là "toàn từ xã giao", vì từ xưng hô nằm trong cùng một túi với lời
    chào. Xem `_SOCIAL_FILLER`.

    ⚠️ Không có trần độ dài. Bản đầu có (`<= 6` token) và một phép tiêm đổi nó
    thành 60 **không làm đỏ test nào** — viết được một test cho nó nghĩa là phải
    bịa ra một đầu vào không ai gửi. Cái trần ấy mã hoá một phỏng đoán, còn lỗ
    thật nằm ở từ vựng chứ không ở độ dài.
    """
    return (
        bool(tokens)
        and all(tok in _SOCIAL_TOKENS for tok in tokens)
        and any(tok in _SOCIAL_CORE for tok in tokens)
    )


def _has_reference(text: str, tokens: Sequence[str]) -> bool:
    return bool(_CONTINUATION.match(text)) or any(tok in _REFERENCE_WORDS for tok in tokens)


def _content_words(tokens: Sequence[str]) -> list[str]:
    return [tok for tok in tokens if tok not in _STOPWORDS]


def classify(question: str, *, has_history: bool) -> tuple[Route, bool, str]:
    """→ `(route, needs_rewrite, reason)`. Thuần luật, không chạm mạng, tất định.

    Thứ tự các nhánh **là** đặc tả, vì các điều kiện chồng lên nhau:

    1. Toàn từ xã giao → `NO_RETRIEVAL`.
    2. Không còn từ nội dung nào, và cũng không viết lại được → `CLARIFY`.
    3. Còn lại → `RETRIEVE`; viết lại **chỉ khi** có cả từ chỉ trỏ lẫn lịch sử.

    ⭐ Nhánh 3 là chỗ luật này khác với "cứ có lịch sử thì viết lại". Phần lớn
    lượt thứ hai trở đi vẫn là câu tự đủ nghĩa, và viết lại chúng là trả tiền
    cho một lượt LLM để nhận về gần đúng chuỗi cũ — kèm rủi ro nhận về một chuỗi
    *khác*.
    """
    tokens = _tokens(question)
    if _is_social(tokens):
        return "no_retrieval", False, "toàn từ xã giao"

    dependent = _has_reference(question, tokens)
    if not _content_words(tokens) and not (dependent and has_history):
        # Không có gì để truy hồi, và cũng không có lịch sử để giải nghĩa chỗ
        # trỏ. Hỏi lại là việc duy nhất trung thực; đi truy hồi bằng `"???"` trả
        # về 5 chunk ngẫu nhiên và model sẽ viết một đoạn văn về chúng.
        return "clarify", False, "không có từ nội dung nào để truy hồi"

    if dependent and has_history:
        return "retrieve", True, "có từ chỉ trỏ và có lịch sử để giải nghĩa"
    if dependent:
        return "retrieve", False, "có từ chỉ trỏ nhưng không có lịch sử"
    return "retrieve", False, "câu tự đủ nghĩa"


# ------------------------------------------------------------------ viết lại

QUERY_REWRITE = default_registry().get("query-rewrite")
"""`W4-11`: nạp từ registry YAML — cùng cơ chế version + hash với hai prompt
của `chat.py`. Nội dung giữ nguyên byte so với hằng số cũ của `W4-07`."""

REWRITE_SYSTEM_PROMPT = QUERY_REWRITE.text
"""⚠️ Luật 4 của template tồn tại vì kiểu hỏng đắt nhất của bước này là viết lại
**thừa**: một câu hỏi bị nhét thêm chi tiết từ lượt trước vẫn truy hồi ra chunk
trông hợp lý, nên nó không biểu hiện thành lỗi ở đâu cả — nó chỉ trả lời một
câu hỏi khác câu người dùng vừa gõ. Rewrite thiếu thì ngược lại: `sources` lệch
chủ đề và người dùng thấy ngay."""

LANGUAGE_DIRECTIVE: dict[str, str] = {
    "vi": "Trả lời bằng tiếng Việt.",
    "en": "Answer in English.",
}
"""Không có khoá `"unknown"`, và đó là toàn bộ ý: không biết thì **không** chỉ thị."""

CLARIFY_TEXT: dict[str, str] = {
    "vi": (
        "Câu hỏi chưa đủ rõ để tìm tài liệu. Bạn cho biết cụ thể bạn đang hỏi về nội dung nào nhé?"
    ),
    "en": (
        "I need a little more to go on. Could you say which topic or document you are asking about?"
    ),
}

CLARIFY_FALLBACK = "{vi}\n\n{en}".format(**CLARIFY_TEXT)
"""Không phát hiện được ngôn ngữ thì trả lời **song ngữ**, chứ không chọn bừa một bên."""


@dataclass(frozen=True)
class QueryPlan:
    """Kết quả của bước hiểu câu hỏi — đủ để `ChatService` không phải suy luận thêm."""

    route: Route
    question: str
    """Chuỗi **thực sự** đưa vào truy hồi. Bằng `original` nếu không viết lại."""

    original: str
    language: Language
    rewritten: bool
    reason: str
    rewrite_ms: float | None = None
    rewrite_cost_usd: float | None = None
    rewrite_model: str | None = None

    @property
    def retrieves(self) -> bool:
        return self.route == "retrieve"

    def directive(self) -> str:
        """Dòng ép ngôn ngữ nối vào lượt người dùng, hoặc `""` khi không biết.

        ⭐ Nằm ở **lượt người dùng**, không ở `SYSTEM_PROMPT`. Hai lý do, và cả
        hai đều thật: nội dung ở cuối được tuân theo nhiều hơn, và — quan trọng
        hơn — prompt hệ thống sắp có version + hash ở `W4-11`, nên nhét một
        chuỗi *đổi theo từng lượt* vào đó sẽ chẻ hash thành một bản cho mỗi ngôn
        ngữ và làm hỏng đúng thứ registry ấy sinh ra để đo.
        """
        line = LANGUAGE_DIRECTIVE.get(self.language, "")
        return f"\n\n{line}" if line else ""

    def clarify_text(self) -> str:
        return CLARIFY_TEXT.get(self.language, CLARIFY_FALLBACK)

    def as_meta(self) -> dict[str, Any]:
        """Phần công khai cho khung SSE `meta`.

        ⭐ `rewrite_ms` ở đây chứ không ở khung `done`, dù nó là một số đo: bước
        viết lại nằm **trước** truy hồi, nên nó là phần TTFB mà người vận hành
        không quy được cho ai nếu chỉ nhìn `done.ttfb_ms`. `None` = không viết
        lại, và đó là phần lớn lượt.
        """
        return {
            "route": self.route,
            "language": self.language,
            "rewritten": self.rewritten,
            "question": self.question,
            "rewrite_ms": round(self.rewrite_ms, 1) if self.rewrite_ms is not None else None,
        }


@dataclass
class QueryUnderstanding:
    """Luật trước, LLM sau — và LLM chỉ được gọi khi luật nói là cần.

    `llm` khai kiểu `LLMProvider` (không stream): bước này cần **một chuỗi trọn
    vẹn** trước khi truy hồi chạy được, nên stream không giúp gì. `W4-08` cắm
    router vào đây y như đã cắm vào `ChatService`.
    """

    llm: LLMProvider | None = None
    max_rewrite_tokens: int = 96
    timeout_s: float = 6.0
    """⚠️ Bước này nằm **trước** truy hồi, nên mỗi mili giây ở đây cộng thẳng vào
    TTFB. Thà mất một lần viết lại (rơi về câu gốc, vẫn truy hồi được) còn hơn
    treo người dùng chờ một provider chậm."""

    extra_body: Mapping[str, Any] | None = None
    max_history_turns: int = 6

    async def plan(self, question: str, history: Sequence[ChatMessage]) -> QueryPlan:
        original = question.strip()
        language = detect_language(original)
        route, needs_rewrite, reason = classify(original, has_history=bool(history))

        plan = QueryPlan(
            route=route,
            question=original,
            original=original,
            language=language,
            rewritten=False,
            reason=reason,
        )
        if not needs_rewrite or self.llm is None:
            if needs_rewrite:
                logger.warning("cần viết lại câu hỏi nhưng chưa cấu hình LLM — dùng câu gốc")
            return plan

        rewritten, elapsed_ms, cost, model = await self._rewrite(original, history)
        if rewritten is None:
            return plan
        return QueryPlan(
            route=route,
            question=rewritten,
            original=original,
            # Ngôn ngữ đọc từ câu **gốc**: câu viết lại do model sinh ra, và nếu
            # nó đổi ngôn ngữ thì cái cần ép vẫn là ngôn ngữ người dùng đã dùng.
            language=language,
            rewritten=True,
            reason=reason,
            rewrite_ms=elapsed_ms,
            rewrite_cost_usd=cost,
            rewrite_model=model,
        )

    async def _rewrite(
        self, question: str, history: Sequence[ChatMessage]
    ) -> tuple[str | None, float, float, str | None]:
        """Trả `(câu mới | None, ms, usd, model)`. `None` = giữ nguyên câu gốc.

        ⚠️ **Không** ném ra ngoài. Viết lại là bước *cải thiện*; hỏng nó thì câu
        gốc vẫn truy hồi được và người dùng vẫn có câu trả lời. Biến nó thành
        một lỗi 503 là đổi một hạng mục tăng chất lượng thành một điểm chết mới.
        """
        assert self.llm is not None
        recent = list(history)[-self.max_history_turns :]
        transcript = "\n".join(f"{msg.role.upper()}: {msg.content}" for msg in recent)
        messages = [
            ChatMessage(role="system", content=REWRITE_SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=f"HỘI THOẠI:\n{transcript}\n\nCÂU HỎI CẦN VIẾT LẠI: {question}",
            ),
        ]
        started = time.perf_counter()
        try:
            response = await asyncio.wait_for(
                # `complete()` là **đồng bộ** (httpx blocking). Gọi thẳng ở đây
                # sẽ chặn vòng lặp sự kiện suốt cả lượt gọi mạng — cùng lý lẽ đã
                # đưa `retrieve()` vào `to_thread` ở `W4-06`.
                #
                # ⚠️ `wait_for` huỷ được cái *chờ*, không huỷ được cái *thread*:
                # quá hạn thì request đi tiếp bằng câu gốc trong khi thread kia
                # vẫn chạy nốt và vẫn bị tính tiền. Đúng đánh đổi ở đây (người
                # dùng không phải chờ), nhưng nó là một khoản chi không xuất
                # hiện trong `usage` của lượt này.
                asyncio.to_thread(
                    self.llm.complete,
                    messages,
                    temperature=0.0,
                    max_tokens=self.max_rewrite_tokens,
                    extra_body=self.extra_body,
                ),
                timeout=self.timeout_s,
            )
        except TimeoutError:
            logger.warning("viết lại câu hỏi quá %.1fs — dùng câu gốc", self.timeout_s)
            return None, (time.perf_counter() - started) * 1000.0, 0.0, None
        except LLMError as exc:
            logger.warning("viết lại câu hỏi thất bại (%s) — dùng câu gốc", exc)
            return None, (time.perf_counter() - started) * 1000.0, 0.0, None

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        candidate = _clean_rewrite(response.text, question)
        if candidate is None:
            return None, elapsed_ms, response.usage.cost_usd, response.model
        return candidate, elapsed_ms, response.usage.cost_usd, response.model


_MAX_GROWTH = 4.0
_MAX_EXTRA_CHARS = 120


def _clean_rewrite(raw: str, original: str) -> str | None:
    """Lọc đầu ra của model. `None` = không dùng được, giữ câu gốc.

    ⭐ Ba phép kiểm này là chỗ duy nhất đứng giữa `sources` và một câu hỏi mà
    **không ai gõ ra**. Model ở đây được yêu cầu in đúng một dòng, nhưng "được
    yêu cầu" và "sẽ làm" là hai chuyện khác nhau — cùng bài học của luật 4 trong
    `SYSTEM_PROMPT` mà `W4-06` đã đo thấy thất bại.
    """
    text = raw.strip().strip('"').strip()
    if not text:
        return None
    # Model hay thêm một dòng giải thích phía sau. Dòng đầu là câu hỏi.
    text = text.splitlines()[0].strip().strip('"').strip()
    if not text:
        return None
    if len(text) > len(original) * _MAX_GROWTH + _MAX_EXTRA_CHARS:
        logger.warning(
            "bỏ câu viết lại vì dài bất thường (%d ký tự từ %d) — dùng câu gốc",
            len(text),
            len(original),
        )
        return None
    if text == original:
        return None
    return text
