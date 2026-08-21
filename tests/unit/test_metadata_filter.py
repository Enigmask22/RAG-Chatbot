"""`W2-06` — filter metadata có kiểu.

Trọng tâm của file này **không** phải "filter dựng đúng `models.Filter`" — đó là
phần dễ. Nó là: mọi cách viết filter sai đều **nổ**, chứ không trả về danh sách
rỗng trông y như "không có dữ liệu".
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import pytest
from pydantic import ValidationError

from rag_core.retrieval import FILTER_FIELDS, PAYLOAD_INDEXES, MetadataFilter, build_filter
from rag_core.schemas import DocType, Language

if TYPE_CHECKING:
    from qdrant_client import models


def conditions(built: models.Filter | None) -> list[Any]:
    """`Filter.must` sau khi đã kiểm là không rỗng.

    `must` của qdrant-client khai kiểu là hợp 8 nhánh (`IsEmptyCondition`,
    `NestedCondition`, `Filter` lồng…). `MetadataFilter` chỉ dựng đúng một nhánh
    (`FieldCondition`), nên thu hẹp **một lần ở đây** thay vì rải `type: ignore`
    khắp file — mỗi `type: ignore` là một chỗ che mất lỗi thật về sau.
    """
    assert built is not None
    assert built.must is not None
    return list(built.must)


class TestRejectsSilentEmptiness:
    """Mỗi test ở đây ứng với một cách viết filter cho 0 kết quả mà không báo lỗi."""

    def test_unknown_key_raises_and_names_the_valid_ones(self) -> None:
        with pytest.raises(ValueError) as err:
            build_filter({"tenant": "acme"})
        message = str(err.value)
        assert "tenant" in message
        # Thông báo phải liệt kê khoá hợp lệ: người gõ sai `tenant` cần biết là
        # `tenant_id`, và bắt họ đi đọc source để tìm ra là một lần hỏng nữa.
        assert "tenant_id" in message
        assert "0 kết quả" in message

    def test_several_unknown_keys_are_all_named(self) -> None:
        with pytest.raises(ValueError) as err:
            build_filter({"tenant": "a", "language": "vi"})
        assert "tenant" in str(err.value)
        assert "language" in str(err.value)

    def test_empty_list_raises_because_match_any_of_nothing_matches_nothing(self) -> None:
        """`doc_type=[]` đọc như 'không lọc' nhưng Qdrant hiểu là 'khớp rỗng'."""
        with pytest.raises(ValidationError) as err:
            MetadataFilter(doc_type=[])
        assert "không giá trị nào" in str(err.value)

    @pytest.mark.parametrize("field", ["chunk_id", "doc_id", "tenant_id", "lang"])
    def test_empty_list_raises_for_every_multi_value_field(self, field: str) -> None:
        with pytest.raises(ValidationError):
            MetadataFilter.model_validate({field: []})

    def test_reversed_window_raises_instead_of_matching_nothing(self) -> None:
        with pytest.raises(ValidationError) as err:
            MetadataFilter(
                published_after=datetime(2024, 1, 1, tzinfo=UTC),
                published_before=datetime(2020, 1, 1, tzinfo=UTC),
            )
        assert "khoảng rỗng" in str(err.value)

    def test_equal_bounds_is_allowed_because_a_single_day_is_a_real_query(self) -> None:
        """`after == before` khớp đúng một mốc — hợp lệ, không phải khoảng rỗng."""
        moment = datetime(2024, 6, 1, tzinfo=UTC)
        flt = MetadataFilter(published_after=moment, published_before=moment)
        assert flt.to_qdrant() is not None


class TestBuildsWhatQdrantExpects:
    def test_single_value_becomes_match_value(self) -> None:
        (only,) = conditions(build_filter({"tenant_id": "t1"}))
        assert only.key == "tenant_id"
        assert only.match.value == "t1"

    def test_list_becomes_match_any(self) -> None:
        (only,) = conditions(build_filter({"doc_type": ["dev_report", "legal"]}))
        assert only.match.any == ["dev_report", "legal"]

    def test_enum_is_serialised_to_its_value_not_its_repr(self) -> None:
        """`DocType.LEGAL` phải xuống Qdrant thành `"legal"`.

        `StrEnum` nên `str(DocType.LEGAL) == "legal"`; test này ghim tính chất đó
        vì đổi sang `Enum` thường sẽ cho `"DocType.LEGAL"` và filter im lặng
        không khớp gì.
        """
        found = conditions(build_filter(MetadataFilter(doc_type=DocType.LEGAL, lang=Language.VI)))
        assert {c.key: c.match.value for c in found} == {"doc_type": "legal", "lang": "vi"}

    def test_date_window_uses_datetime_range_not_numeric_range(self) -> None:
        """`Range` so số; `published_at` là chuỗi RFC3339 nên phải là `DatetimeRange`."""
        from qdrant_client import models

        after = datetime(2020, 1, 1, tzinfo=UTC)
        (only,) = conditions(build_filter({"published_after": after}))
        assert only.key == "published_at"
        assert isinstance(only.range, models.DatetimeRange)

    def test_conditions_are_flat_must_so_qdrant_can_use_payload_indexes(self) -> None:
        found = conditions(
            build_filter(
                {
                    "tenant_id": "t1",
                    "lang": "vi",
                    "published_after": datetime(2020, 1, 1, tzinfo=UTC),
                }
            )
        )
        assert len(found) == 3
        # Không lồng Filter trong Filter — must phẳng là thứ Qdrant tối ưu được.
        assert all(not hasattr(c, "must") for c in found)

    @pytest.mark.parametrize("empty", [None, {}, MetadataFilter()])
    def test_nothing_to_filter_gives_none_not_an_empty_filter(self, empty: object) -> None:
        """`Filter(must=[])` và `None` **không** giống nhau với mọi phiên bản Qdrant."""
        assert build_filter(cast("MetadataFilter | None", empty)) is None

    def test_metadata_filter_instance_passes_through(self) -> None:
        assert build_filter(MetadataFilter(tenant_id="t1")) is not None


class TestFieldsStayInSyncWithTheIndex:
    """Hai chiều, vì lệch chiều nào cũng là một kiểu hỏng im lặng khác nhau."""

    def test_every_indexed_field_is_filterable(self) -> None:
        """Có index mà không lọc được = index chết, tốn chỗ và gây hiểu sai."""
        assert {field for field, _ in PAYLOAD_INDEXES} <= FILTER_FIELDS

    def test_every_filterable_field_has_an_index(self) -> None:
        """Lọc được mà không có index = Qdrant quét toàn bộ collection.

        Kết quả vẫn **đúng**, chỉ chậm — nên không test nào khác phát hiện được.
        """
        assert {field for field, _ in PAYLOAD_INDEXES} >= FILTER_FIELDS

    def test_published_at_is_indexed_as_datetime_not_keyword(self) -> None:
        """Index `keyword` trên `published_at` dựng được và làm `DatetimeRange` quét.

        Đúng loại hỏng chỉ về hiệu năng: kết quả không sai nên không lộ ra.
        """
        schema = dict(PAYLOAD_INDEXES)
        assert schema["published_at"] == "datetime"

    def test_filter_fields_are_flat_payload_keys_not_nested_paths(self) -> None:
        """Nested path (`chunk.metadata.lang`) cũng lọc được nhưng chậm hơn.

        `_payload` cố ý làm phẳng để tránh chúng — ghim ở đây để không ai thêm
        một field lọc theo kiểu lồng vào mà không đổi luôn `_payload`.
        """
        assert not any("." in field for field in FILTER_FIELDS)

    def test_filter_model_covers_every_filter_field(self) -> None:
        """Mỗi field lọc được phải có đường vào qua `MetadataFilter`.

        `published_at` xuất hiện dưới dạng **hai** field (`_after`/`_before`) nên
        nó được xử lý riêng — phần còn lại phải khớp một-một.
        """
        names = set(MetadataFilter.model_fields)
        assert {"published_after", "published_before"} <= names
        assert FILTER_FIELDS - {"published_at"} <= names


class TestFrozen:
    def test_filter_is_frozen_so_it_cannot_be_widened_after_a_check(self) -> None:
        """Điểm bảo mật, không phải thẩm mỹ.

        Nếu tầng serving kiểm `filters.tenant_id == token.tenant` rồi truyền tiếp,
        một filter đổi được cho phép nới nó ra **sau** khi đã kiểm.
        """
        flt = MetadataFilter(tenant_id="t1")
        with pytest.raises(ValidationError):
            flt.tenant_id = "t2"
