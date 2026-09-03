"""`RagBundle` — hợp đồng duy nhất giữa Pipeline Plane và Serving Plane.

Cả kiến trúc hai plane đứng trên đúng một câu: *serving không import pipeline,
hai bên chỉ gặp nhau ở một artifact bất biến có version*. Câu ấy đúng hay sai
không do sơ đồ quyết định, mà do **file này** quyết định — cụ thể là do việc mỗi
trường ở đây có mặc định hay không.

## Vì sao mặc định là thứ nguy hiểm nhất trong file này

Một trường có mặc định trong bundle schema là một hằng số cấu hình **sống trong
mã serving** thay vì trong artifact. Khi đó:

* Bundle không còn mô tả đủ hệ thống. Hai lần deploy cùng một bundle trên hai
  phiên bản image khác nhau chạy ra hai hệ thống khác nhau, và không có gì đỏ.
* Gate ở `W5` so hai bundle với nhau sẽ so thiếu đúng phần đã bị mặc định nuốt.
* Rollback về bundle cũ không rollback được phần đó.

Nên luật ở đây: **trường nào ảnh hưởng kết quả truy hồi hoặc sinh thì bắt buộc**,
kể cả khi giá trị của nó gần như luôn giống nhau. Mặc định chỉ dành cho trường
mô tả (`snapshot`, `notes`) và cho `rerank = None`, nơi `None` mang nghĩa **tắt**
chứ không mang nghĩa "dùng mặc định" — xem `RagBundle.rerank`.

## Checksum này chứng nhận cái gì

⚠️ Nó băm **dạng chuẩn hoá của model đã validate**, không băm byte của file.
Hệ quả phải nói rõ vì hai chiều đều quan trọng:

* Format lại JSON (đổi thụt lề, đổi thứ tự khoá, đổi cách escape unicode)
  **không** làm hỏng checksum. Đó là hành vi đúng cho một file cấu hình đi qua
  git, CI, và `docker cp`.
* Nhưng nó **không** chứng nhận rằng collection Qdrant mà bundle trỏ tới đúng là
  collection đã được đo. Checksum bảo vệ *manifest*, không bảo vệ *chỉ mục*.
  Chỗ khớp hai thứ đó là `components.index.fingerprint`, và người kiểm là
  serving lúc load (`W4-02`/`W4-03`) — không phải hàm này.

Nói cách khác: checksum bắt được **sửa tay vào manifest**, và chỉ thế thôi.

⚠️ **Hệ quả chưa xử lý — thêm trường vào schema làm hỏng chữ ký của mọi bundle
cũ.** Băm trên model *đã validate* nghĩa là một trường mới có mặc định sẽ được
pydantic lấp vào khi đọc manifest cũ, payload đem băm đổi, và bundle cũ thành
"sai chữ ký" dù không ai chạm vào nó. Đây là mặt trái trực tiếp của tính chất
"xoá một trường optional cũng bị bắt" — không có cách nào giữ cả hai.

Lối ra **không** phải là ký lại bundle cũ: chữ ký khi ấy chứng nhận một nội dung
mà pipeline chưa bao giờ sinh ra. Lối ra là sinh lại bundle từ artifact nguồn.
Ghi ở `TD-36`, cần giải quyết trước khi có bundle thứ hai được promote.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "BundleChecksumError",
    "BundleValidationError",
    "ChunkingComponent",
    "EmbeddingComponent",
    "EvalReport",
    "GateRecord",
    "GateStatus",
    "GenerationComponent",
    "IndexComponent",
    "JudgeSpec",
    "PromptComponent",
    "RagBundle",
    "RerankComponent",
    "RetrievalComponent",
    "canonical_blob",
    "compute_checksum",
    "parse_semver",
]

NonEmptyStr = Annotated[str, Field(min_length=1)]

CHECKSUM_PREFIX = "sha256:"

#: Regex semver chính thức (semver.org, mục "Backus–Naur Form Grammar").
#: Dùng nguyên bản thay vì viết lại cho gọn: bản rút gọn thường nhận
#: `1.2.3.4` hoặc `01.2.3`, và một version nhận sai là một bundle sắp xếp sai
#: thứ tự — tức rollback về nhầm bản.
_SEMVER = re.compile(
    r"^(?P<major>0|[1-9]\d*)"
    r"\.(?P<minor>0|[1-9]\d*)"
    r"\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<prerelease>(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
    r"(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(?:\+(?P<build>[0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)

_GIT_SHA = re.compile(r"^[0-9a-f]{7,40}$")


class BundleValidationError(ValueError):
    """Bundle không hợp lệ về mặt nội dung (thiếu trường, sai định dạng)."""


class BundleChecksumError(BundleValidationError):
    """Checksum không khớp — manifest đã bị sửa sau khi ký.

    Tách riêng khỏi `BundleValidationError` vì hai lỗi này đòi hai phản ứng khác
    nhau: thiếu trường là bug của bên **sinh** bundle, sai checksum là bundle đã
    bị **sửa** sau khi sinh.
    """


def parse_semver(version: str) -> tuple[int, int, int, tuple[object, ...]]:
    """Khoá sắp xếp theo đúng luật ưu tiên của semver.

    Cần cho rollback (`W4-02`) và cho leaderboard (`W5`): "bản trước đó" phải là
    một câu có nghĩa. So chuỗi thì `1.10.0 < 1.9.0`, và đó là kiểu lỗi chỉ lộ ra
    ở lần release thứ mười.

    Prerelease xếp **trước** bản chính thức cùng số (`1.0.0-rc1 < 1.0.0`), và
    `build` bị bỏ qua hoàn toàn — semver quy định nó không tham gia so sánh.
    """
    match = _SEMVER.match(version)
    if match is None:
        raise BundleValidationError(
            f"`bundle_version` phải là semver: {version!r}. Ví dụ hợp lệ: '1.4.0', '2.0.0-rc.1'."
        )
    core = (int(match["major"]), int(match["minor"]), int(match["patch"]))
    prerelease = match["prerelease"]
    if prerelease is None:
        # Không có prerelease ⇒ ưu tiên CAO hơn mọi prerelease cùng core.
        return (*core, (1,))
    parts: list[object] = [0]
    for piece in prerelease.split("."):
        # Số so với số theo giá trị; chuỗi so với chuỗi theo ASCII; số luôn nhỏ
        # hơn chuỗi. Bọc thành tuple `(hạng, giá trị)` để tuple so sánh được mà
        # không đụng `int < str`.
        parts.append((0, int(piece)) if piece.isdigit() else (1, piece))
    return (*core, tuple(parts))


# ---------------------------------------------------------------------------
# Components — mỗi lớp là một tầng của pipeline runtime
# ---------------------------------------------------------------------------


class _Component(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ChunkingComponent(_Component):
    """Serving **không** chunk gì cả — vậy tại sao bundle vẫn phải mang khối này?

    Vì hai lý do đều thuộc về vận hành, không thuộc về runtime:

    1. Đường ingest online (`W4`) chunk tài liệu mới, và nó **phải** chunk giống
       hệt lúc build index, nếu không thì tài liệu mới nằm trên một phân bố khác
       phần còn lại của collection.
    2. `chunking_fingerprint` là thứ nói được "artifact ngữ cảnh này thuộc về bộ
       chunk nào" — xem `IndexConfig.chunking_fingerprint`.
    """

    strategy: NonEmptyStr
    chunk_size: int = Field(gt=0)
    chunk_overlap: int = Field(ge=0)
    contextual: bool
    """Chunk đã được dán câu định vị trước khi embed hay chưa (`W3-04`).

    Bắt buộc chứ không mặc định `False`: nó đổi **nội dung** của mọi vector, và
    một bundle im lặng về nó là một bundle không mô tả được index của chính nó.
    """
    chunking_fingerprint: NonEmptyStr

    @model_validator(mode="after")
    def _overlap_below_size(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"`chunk_overlap` ({self.chunk_overlap}) phải nhỏ hơn "
                f"`chunk_size` ({self.chunk_size}) — bằng hoặc lớn hơn thì "
                "chunker không tiến được và sinh vòng lặp vô hạn."
            )
        return self


class EmbeddingComponent(_Component):
    """`dim` và `normalize` là hai trường serving **không được** đoán.

    Sai `dim` thì Qdrant từ chối ngay — lỗi ồn, vô hại. Sai `normalize` thì
    **không ai từ chối gì cả**: truy vấn vẫn chạy, điểm số vẫn ra, thứ hạng chỉ
    đơn giản là sai. Đó là lý do nó bắt buộc.
    """

    model: NonEmptyStr
    dim: int = Field(gt=0)
    normalize: bool
    revision: str | None = None
    """Commit của repo HuggingFace. `None` = chưa ghim.

    ⚠️ `None` được phép nhưng là **nợ**, không phải trạng thái bình thường: model
    trên Hub sửa được tại chỗ, nên bundle không ghim revision thì không tái lập
    được. `W5` nên từ chối promote bundle có `revision = None`.
    """


class IndexComponent(_Component):
    """Chỗ nối duy nhất giữa manifest và dữ liệu thật.

    `fingerprint` là `IndexConfig.fingerprint` — băm đúng những trường quyết định
    vector. Serving so nó với vân tay ghi trong collection lúc load; lệch nghĩa
    là bundle và index không thuộc về nhau, và đó là lỗi phải chặn ở `/ready`
    chứ không phải phát hiện qua metric tụt vài tuần sau.
    """

    backend: NonEmptyStr
    collection: NonEmptyStr
    fingerprint: NonEmptyStr
    n_chunks: int = Field(ge=0)
    n_documents: int = Field(ge=0)
    snapshot: str | None = None
    """URI của snapshot để dựng lại index. `None` = chỉ trỏ tới collection đang sống.

    Là một trong hai trường được phép mặc định, vì nó không đổi hành vi truy hồi
    — nó chỉ đổi việc khôi phục có tự động được hay không.
    """


class RetrievalComponent(_Component):
    """Cấu hình nhánh nền. `options` cố tình để mở, và đây là một đánh đổi.

    Mỗi nhánh có tham số riêng (`hybrid` cần `k`/`candidate_k`, `dense` không cần
    gì). Ghim thành các trường tường minh thì mỗi lần thêm nhánh phải sửa schema
    và mọi bundle cũ hết đọc được. Để `dict` thì đánh mất `extra="forbid"` ở
    trong đó — nên phần kiểm chuyển sang `build_branch`, nơi truyền sai tham số
    đã **nổ** từ `W2-07` chứ không bị bỏ qua.
    """

    mode: NonEmptyStr
    top_k: int = Field(gt=0)
    options: dict[str, Any] = Field(default_factory=dict)


class RerankComponent(_Component):
    max_length: int = Field(gt=0)
    """Cửa sổ của cross-encoder. Bắt buộc vì `W3-04` đo được nó là **ràng buộc
    thật**: dán ngữ cảnh đẩy p95 token từ 352 lên 453 trên trần 512."""
    model: NonEmptyStr
    candidates: int = Field(gt=0)
    top_n: int = Field(gt=0)

    @model_validator(mode="after")
    def _top_n_below_candidates(self) -> Self:
        if self.top_n > self.candidates:
            raise ValueError(
                f"`top_n` ({self.top_n}) > `candidates` ({self.candidates}): "
                "reranker không thể trả về nhiều hơn số ứng viên nó nhận."
            )
        return self


class PromptComponent(_Component):
    """`hash` chứ không chỉ `version`.

    Số version của prompt do người gõ tay; nội dung prompt do người sửa tay. Hai
    thứ đó lệch nhau là chuyện thường, và khi lệch thì bundle nói dối về thứ đã
    được đo. `W4-11` (prompt registry) tính hash từ nội dung thật.
    """

    id: NonEmptyStr
    version: int = Field(ge=1)
    hash: NonEmptyStr


class GenerationComponent(_Component):
    primary: NonEmptyStr
    """Slug tường minh, **không** phải preset. Quy tắc cứng #1 của dự án: một
    preset OpenRouter đổi model dưới chân mình mà không đổi tên, nên mọi con số
    eval gắn với nó là số của một hệ thống không xác định."""
    fallback: str | None = None
    max_tokens: int = Field(gt=0)
    temperature: float = Field(ge=0.0, le=2.0)


class BundleComponents(_Component):
    chunking: ChunkingComponent
    embedding: EmbeddingComponent
    index: IndexComponent
    retrieval: RetrievalComponent
    rerank: RerankComponent | None = None
    """`None` nghĩa là **không rerank**, không phải "dùng reranker mặc định".

    Đây là chỗ dễ hỏng nhất của cả schema. Nếu serving đọc `None` rồi rơi về một
    reranker mặc định trong mã, thì bundle mô tả một hệ thống mà bundle không
    điều khiển — và mọi lời hứa về gate, rollback, tái lập đều rỗng. Ràng buộc
    ấy không ép được bằng type; nó được ghim bằng
    `test_bundle.py::test_absent_rerank_means_disabled_not_default`.
    """
    prompt: PromptComponent | None = None
    generation: GenerationComponent | None = None
    """⚠️ `None` ở hai trường này **không** giống `rerank=None`.

    `rerank=None` nghĩa là *tắt rerank* — một cấu hình hoàn chỉnh. Còn thiếu
    `generation` nghĩa là bundle **không mô tả tầng sinh**, tức nó là một bundle
    *retrieval-only*: đo được, so được, gate được trên metric truy hồi, nhưng
    `/chat` không nạp nó được.

    Tồn tại vì hôm nay (`W4-01`) tầng sinh chưa được dựng (`W4-08`/`W4-11`). Lối
    thoát đúng cho tình huống ấy là để bundle **nói thật rằng nó thiếu**, chứ
    không phải nhét một `prompt.hash = "todo"` vào rồi ký lên đó — một artifact
    bịa mà có chữ ký hợp lệ tệ hơn một artifact khai thiếu.

    Ràng buộc giữ cho lối thoát không thành lỗ hổng nằm ở
    `RagBundle._generation_metrics_need_a_generation_stack`.
    """


# ---------------------------------------------------------------------------
# Eval + gate
# ---------------------------------------------------------------------------


class JudgeSpec(_Component):
    model: NonEmptyStr
    temperature: float = Field(ge=0.0, le=2.0)
    kappa_vs_human: float | None = Field(default=None, ge=-1.0, le=1.0)
    """Mức đồng thuận giữa judge và người. `None` = **chưa đo**, và một judge
    chưa đối chiếu với người là một thước đo chưa được hiệu chuẩn."""


class EvalReport(_Component):
    """Không có khối này thì không có bundle. Đó là toàn bộ luận điểm của dự án.

    "Eval trước tối ưu" chỉ là khẩu hiệu cho tới khi có một chỗ mà **thiếu số đo
    là lỗi cứng**. Chỗ đó là đây: `RagBundle.eval` không có mặc định, nên một
    cấu hình chưa từng được đo không đóng gói được, nên nó không deploy được.
    """

    golden_set: NonEmptyStr
    n_queries: int = Field(gt=0)
    evaluated_with_generator: NonEmptyStr
    """⭐ Trường bắt buộc mà cả `W5-07` lẫn `G5` dựa vào.

    Chất lượng end-to-end là **hàm của cả retrieval lẫn generator**. Đo bundle A
    bằng Qwen3-8B rồi đo bundle B bằng DeepSeek và so hai con số với nhau là so
    hai hệ thống khác nhau ở hai chỗ, rồi quy toàn bộ chênh lệch cho retrieval.
    Gate chỉ được so **like-for-like**, và nó chỉ làm được thế nếu trường này
    luôn có mặt — nên nó không có mặc định, kể cả chuỗi rỗng.
    """
    judge: JudgeSpec | None = None
    retrieval_metrics: dict[str, float] = Field(default_factory=dict)
    generation_metrics: dict[str, float] = Field(default_factory=dict)
    p95_latency_ms: float | None = Field(default=None, ge=0.0)
    cost_per_query_usd: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def _some_metric_present(self) -> Self:
        if not self.retrieval_metrics and not self.generation_metrics:
            raise ValueError(
                "`eval` phải mang ít nhất một metric. Một khối eval rỗng đi qua "
                "được mọi phép kiểm hình thức mà không chứng nhận điều gì."
            )
        return self


class GateStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_RUN = "NOT_RUN"
    """Cố ý có mặt và cố ý **không** phải mặc định của `RagBundle.gate`.

    "Chưa chạy gate" là một trạng thái hợp lệ của một bundle release candidate.
    Nó khác hẳn "gate đã chạy và PASS", và trộn hai thứ đó lại — bằng cách để
    `gate` là optional và coi vắng mặt là ổn — chính là cách một bundle chưa
    kiểm đi ra production.
    """


class GateRecord(_Component):
    status: GateStatus
    champion_compared: str | None = None
    report: str | None = None

    @model_validator(mode="after")
    def _decided_gate_names_its_champion(self) -> Self:
        if self.status is not GateStatus.NOT_RUN and not self.champion_compared:
            raise ValueError(
                f"gate `{self.status.value}` phải nêu `champion_compared`: "
                "PASS/FAIL là một phán quyết **so với** một bản cụ thể, không "
                "phải một thuộc tính tự thân của bundle."
            )
        return self


# ---------------------------------------------------------------------------
# Bundle
# ---------------------------------------------------------------------------


def canonical_blob(payload: dict[str, Any]) -> bytes:
    """Dạng chuẩn hoá dùng để băm. Ba lựa chọn, mỗi lựa chọn có lý do.

    * `sort_keys=True` — thứ tự khoá trong JSON không mang nghĩa, nên nó không
      được mang vào hash.
    * `ensure_ascii=False` + encode UTF-8 — corpus song ngữ, và `ensure_ascii`
      đổi hash theo cách nhìn không ra khi so hai file bằng mắt.
    * `separators` không khoảng trắng — thụt lề là chuyện hiển thị.
    """
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )


def compute_checksum(payload: dict[str, Any]) -> str:
    """`payload` phải là manifest **đã bỏ khoá `checksum`**.

    Không tự bỏ hộ ở đây: một hàm im lặng xoá khoá của người gọi sẽ khiến
    `compute_checksum(bundle.model_dump())` và `bundle.checksum` khớp nhau kể cả
    khi luồng ghi và luồng đọc bất đồng về việc cái gì được băm.
    """
    if "checksum" in payload:
        raise BundleValidationError(
            "`checksum` không được nằm trong dữ liệu đem băm — nó là kết quả, không phải đầu vào."
        )
    return CHECKSUM_PREFIX + hashlib.sha256(canonical_blob(payload)).hexdigest()


class RagBundle(BaseModel):
    """Artifact bất biến, có version, nối hai plane.

    `frozen=True` không phải để cho đẹp: `verify_checksum` chứng nhận nội dung
    tại một thời điểm, và một object sửa được sau khi verify làm phép chứng nhận
    ấy hết giá trị ngay dòng sau.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    bundle_version: NonEmptyStr
    created_at: datetime
    git_sha: NonEmptyStr
    """SHA của commit đã sinh ra bundle. Bắt buộc vì nó là đường duy nhất đi
    ngược từ một artifact đang chạy về mã đã tạo ra nó."""
    components: BundleComponents
    eval: EvalReport
    gate: GateRecord
    notes: str | None = None
    checksum: str | None = None
    """`None` = **chưa ký**. Ký bằng `signed()`, kiểm bằng `verify_checksum()`.

    Optional để `RagBundle(...)` dựng được từ mã mà không phải tự tính hash, chứ
    không phải để checksum thành tuỳ chọn: `load_bundle` từ chối bundle chưa ký.
    """

    @field_validator("bundle_version")
    @classmethod
    def _semver(cls, value: str) -> str:
        parse_semver(value)
        return value

    @field_validator("git_sha")
    @classmethod
    def _looks_like_sha(cls, value: str) -> str:
        if not _GIT_SHA.match(value):
            raise ValueError(
                f"`git_sha` không giống SHA git: {value!r} (7–40 ký tự hex, chữ thường)."
            )
        return value

    @field_validator("created_at")
    @classmethod
    def _timezone_aware(cls, value: datetime) -> datetime:
        """Không nhận datetime naive.

        Bundle được sinh trên GPU thuê và đọc trên máy khác múi giờ; một mốc thời
        gian không mang offset là một mốc không so được với mốc khác, và thứ tự
        thời gian là thứ `W4-02` dùng để chọn bản rollback.
        """
        if value.tzinfo is None:
            raise ValueError("`created_at` phải có timezone (dùng UTC).")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _generation_metrics_need_a_generation_stack(self) -> Self:
        """Đo tầng sinh mà không mô tả tầng sinh là con số không quy được về đâu.

        Đây là thứ giữ cho `prompt=None`/`generation=None` là một **khai báo
        thiếu** chứ không phải một lỗ hổng: bundle nào có `faithfulness` hay
        `answer_relevancy` trong `eval` thì buộc phải nói nó đo bằng prompt nào
        và model nào — nếu không thì `W5` so hai bundle và quy chênh lệch cho
        retrieval trong khi nó có thể đến từ một prompt đã đổi.
        """
        if self.eval.generation_metrics and (
            self.components.generation is None or self.components.prompt is None
        ):
            missing = [
                name
                for name, value in (
                    ("prompt", self.components.prompt),
                    ("generation", self.components.generation),
                )
                if value is None
            ]
            raise ValueError(
                f"`eval.generation_metrics` có số đo nhưng `components` thiếu "
                f"{', '.join(missing)}. Một metric của tầng sinh mà không nêu "
                "prompt/model đã sinh ra nó thì không so được với bundle khác."
            )
        return self

    # ---------------------------------------------------------------- ký/kiểm

    @property
    def serves_generation(self) -> bool:
        """Bundle này nạp được vào `/chat` hay chỉ vào đường truy hồi.

        Serving hỏi câu này lúc load (`W4-02`/`W4-03`) thay vì lúc có request
        đầu tiên: một bundle retrieval-only phải làm `/ready` trả 503 cho tuyến
        chat, không phải trả 500 giữa một luồng SSE đang chạy.
        """
        return self.components.generation is not None and self.components.prompt is not None

    @property
    def version_key(self) -> tuple[int, int, int, tuple[object, ...]]:
        return parse_semver(self.bundle_version)

    def unsigned_payload(self) -> dict[str, Any]:
        """Manifest bỏ `checksum`, dạng JSON thuần — đầu vào của phép băm."""
        payload: dict[str, Any] = json.loads(self.model_dump_json())
        payload.pop("checksum", None)
        return payload

    def signed(self) -> RagBundle:
        """Bản sao có `checksum`. Ký lại một bundle đã ký cũng cho cùng kết quả."""
        return self.model_copy(update={"checksum": compute_checksum(self.unsigned_payload())})

    def verify_checksum(self) -> None:
        if not self.checksum:
            raise BundleChecksumError(
                f"bundle {self.bundle_version} chưa ký: `checksum` rỗng. "
                "Bundle chưa ký không được nạp — dùng `.signed()` lúc sinh."
            )
        expected = compute_checksum(self.unsigned_payload())
        if self.checksum != expected:
            raise BundleChecksumError(
                f"checksum không khớp cho bundle {self.bundle_version}:\n"
                f"  ghi trong manifest: {self.checksum}\n"
                f"  tính lại từ nội dung: {expected}\n"
                "Manifest đã bị sửa sau khi ký. Sinh lại bundle từ pipeline "
                "thay vì sửa tay rồi ký lại — nội dung mới chưa được đo."
            )
