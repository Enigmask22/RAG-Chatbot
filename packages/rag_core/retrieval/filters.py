"""Filter metadata có kiểu cho tầng truy hồi. `W2-06`.

Trước hạng mục này, `filters` là `dict[str, Any]` không kiểm gì. Vấn đề không
phải thẩm mỹ — nó là **chế độ hỏng im lặng đúng loại tệ nhất**:

`filters={"tenant": "acme"}` (thiếu `_id`) đi thẳng xuống Qdrant thành điều kiện
trên một field không tồn tại, và Qdrant trả **0 kết quả**. Không lỗi, không cảnh
báo. Từ ngoài nhìn vào nó giống "tenant này chưa có tài liệu" — một câu trả lời
hoàn toàn hợp lý và hoàn toàn sai. Cùng loại bẫy với `TD-11` (giả định không đo)
và với `ensure_collection` ở `W2-02` (nửa index không được dùng mà số vẫn trông
bình thường).

`MetadataFilter` có `extra="forbid"`, nên khoá gõ sai **nổ ngay ở chỗ gọi** kèm
danh sách khoá hợp lệ.

**Hướng của chế độ hỏng là điều đáng nghĩ nhất ở đây.** Với filter thường (lọc
theo `lang` để đo breakdown) thì hỏng-thành-rỗng chỉ gây nhầm lẫn. Với
`tenant_id` thì hai hướng hỏng **không** đối xứng:

* Filter quá chặt → thiếu kết quả. Người dùng thấy, báo lại, sửa được.
* Filter quá lỏng → **dữ liệu tenant khác lọt ra**. Không ai thấy, kể cả người
  bị rò.

Nên mọi mặc định ở module này nghiêng về hướng thứ nhất: thiếu `tenant_id` trong
payload thì point **không** khớp bất kỳ filter tenant nào (đó cũng là hành vi
`MatchValue` của Qdrant, và có test ghim nó chứ không chỉ tin vào tài liệu).

⚠️ **Cái module này KHÔNG làm được, và đó là giới hạn thật:** nó không ép người
gọi *phải* truyền `tenant_id`. `retrieve(query)` không có filter vẫn thấy tất cả,
và không có chỗ nào trong `rag_core` biết được như vậy là đúng (eval chạy trên
toàn corpus) hay là một lỗ rò (serving quên truyền tenant). Chỗ ép được là tầng
serving, nơi tenant đến từ token đã xác thực — `W4-04`. Ghi ở đây để lúc đó
không ai tưởng `W2-06` đã đóng chuyện đó.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, model_validator

from ..schemas import DocType, Language

if TYPE_CHECKING:
    from qdrant_client import models

__all__ = [
    "FILTER_FIELDS",
    "FilterSpec",
    "MetadataFilter",
    "build_filter",
]

#: Field payload **phẳng** mà `QdrantDenseRetriever._payload` ghi ở cấp cao nhất,
#: và là những field duy nhất lọc được mà không cần nested path. Danh sách này
#: phải khớp cả `_payload` lẫn các payload index dựng trong `ensure_collection` —
#: có test canh cả hai chiều, vì lệch một chiều nào cũng là hỏng im lặng: field
#: có trong payload mà không có index thì quét toàn bộ collection, còn có index
#: mà không có trong payload thì mọi filter trên nó trả rỗng.
FILTER_FIELDS = frozenset({"chunk_id", "doc_id", "lang", "doc_type", "tenant_id", "published_at"})


class MetadataFilter(BaseModel):
    """Điều kiện lọc trên payload phẳng, kiểm ở chỗ gọi.

    Mọi field đều `None` = không lọc gì. Truyền list cho field khớp-chính-xác thì
    thành phép hợp (`MatchAny`), tức "một trong những giá trị này".

    Khoảng thời gian dùng **hai field riêng** thay vì một tuple, vì gần như mọi
    lần dùng thật chỉ cần một đầu ("tài liệu từ 2020 trở lại đây"), và một tuple
    `(None, x)` đọc khó hơn `published_before=x`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    chunk_id: str | list[str] | None = None
    doc_id: str | list[str] | None = None
    tenant_id: str | list[str] | None = None
    doc_type: DocType | list[DocType] | None = None
    lang: Language | list[Language] | None = None

    published_after: datetime | None = None
    """Bao gồm mốc: `published_at >= published_after`."""

    published_before: datetime | None = None
    """Bao gồm mốc: `published_at <= published_before`.

    Cố ý bao gồm cả hai đầu. Nửa mở (`< before`) đúng hơn về mặt toán khi ghép
    nhiều khoảng liền nhau, nhưng ở đây đầu vào của người dùng là ngày, và
    `published_before=2024-12-31` mà loại chính ngày 31/12 là thứ không ai đoán
    được. Ghi vào docstring vì đây là loại lựa chọn mà đọc code không suy ra.
    """

    @model_validator(mode="after")
    def _reject_empty_window(self) -> MetadataFilter:
        """Khoảng thời gian rỗng là lỗi, không phải kết quả rỗng.

        `published_after > published_before` không thể khớp point nào. Để nó chạy
        thì nó trả 0 kết quả và trông y như "không có tài liệu nào trong khoảng"
        — lại đúng cái chế độ hỏng im lặng mà module này tồn tại để chặn.
        """
        after, before = self.published_after, self.published_before
        if after is not None and before is not None and after > before:
            raise ValueError(
                f"published_after ({after.isoformat()}) muộn hơn published_before "
                f"({before.isoformat()}) — khoảng rỗng, không point nào khớp được. "
                "Nếu đúng ý là 'ngoài khoảng này' thì đó là hai truy vấn, không phải một filter."
            )
        return self

    @model_validator(mode="after")
    def _reject_empty_list(self) -> MetadataFilter:
        """`[]` là lỗi, vì `MatchAny(any=[])` khớp **không gì cả**.

        `doc_type=[]` đọc rất tự nhiên như "không lọc theo doc_type" nhưng Qdrant
        hiểu là "khớp một trong không giá trị nào" = rỗng. Hai cách đọc trái
        ngược nhau về cùng một dòng code, nên bắt buộc phải nổ.
        """
        for name in ("chunk_id", "doc_id", "tenant_id", "doc_type", "lang"):
            value = getattr(self, name)
            if isinstance(value, list) and not value:
                raise ValueError(
                    f"{name}=[] khớp không giá trị nào (Qdrant `MatchAny(any=[])`), "
                    f"không phải 'bỏ qua field này'. Dùng {name}=None nếu không muốn lọc."
                )
        return self

    def is_empty(self) -> bool:
        """Không có điều kiện nào — `to_qdrant()` sẽ trả `None`."""
        return all(value is None for value in self.__dict__.values())

    def to_qdrant(self) -> models.Filter | None:
        """Dựng `models.Filter`. Trả `None` khi rỗng, vì Qdrant coi đó là không lọc."""
        from qdrant_client import models

        # Chú thích kiểu tường minh: `list` là invariant nên `list[FieldCondition]`
        # không khớp `list[Condition]` mà `Filter.must` mong đợi.
        conditions: list[models.Condition] = []
        for field in ("chunk_id", "doc_id", "tenant_id", "doc_type", "lang"):
            value = getattr(self, field)
            if value is None:
                continue
            if isinstance(value, list):
                conditions.append(
                    models.FieldCondition(
                        key=field, match=models.MatchAny(any=[str(v) for v in value])
                    )
                )
            else:
                conditions.append(
                    models.FieldCondition(key=field, match=models.MatchValue(value=str(value)))
                )
        if self.published_after is not None or self.published_before is not None:
            # `DatetimeRange` chứ không phải `Range`: `Range` so số, và
            # `published_at` được ghi dưới dạng chuỗi RFC3339 nên so số sẽ so
            # theo thứ tự từ điển — đúng một cách tình cờ với cùng định dạng và
            # sai ngay khi có offset múi giờ khác nhau.
            conditions.append(
                models.FieldCondition(
                    key="published_at",
                    range=models.DatetimeRange(
                        gte=self.published_after,
                        lte=self.published_before,
                    ),
                )
            )
        if not conditions:
            return None
        return models.Filter(must=conditions)


#: Cái mà mọi `retrieve`/`fetch_*` nhận. `dict` có mặt để mọi chỗ gọi cũ và mọi
#: nguồn ngoài (YAML config, query string) dùng được mà không phải import gì — nó
#: vẫn đi qua `MetadataFilter.model_validate` nên vẫn được kiểm khoá.
#: `MetadataFilter` là đường nên dùng ở code mới: type checker bắt được tên field
#: sai, còn `dict` thì chỉ nổ lúc chạy.
type FilterSpec = MetadataFilter | dict[str, Any] | None


def build_filter(filters: FilterSpec) -> models.Filter | None:
    """Đổi filter thành `models.Filter` của Qdrant, **kiểm khoá trước**.

    Hàm module chứ không phải method: `W2-04` (RRF) dựng hai truy vấn trong **một**
    request nên nó cần filter mà không đi qua một instance store nào.

    Nhận `dict` để mọi chỗ gọi cũ chạy nguyên vẹn, nhưng `dict` giờ đi qua
    `MetadataFilter` nên khoá lạ **nổ** thay vì trả rỗng. Đó là toàn bộ điểm của
    `W2-06` — xem docstring module.
    """
    if filters is None:
        return None
    if isinstance(filters, MetadataFilter):
        return filters.to_qdrant()
    try:
        parsed = MetadataFilter.model_validate(filters)
    except Exception as exc:
        unknown = sorted(set(filters) - set(MetadataFilter.model_fields))
        if unknown:
            raise ValueError(
                f"Khoá filter không hợp lệ: {unknown}. Hợp lệ: "
                f"{sorted(MetadataFilter.model_fields)}. "
                "Một khoá gõ sai đi xuống Qdrant sẽ trả 0 kết quả mà không báo lỗi, "
                "và điều đó không phân biệt được với 'không có dữ liệu'."
            ) from exc
        raise
    return parsed.to_qdrant()
