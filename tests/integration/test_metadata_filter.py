"""`W2-06` — filter metadata trên Qdrant thật, và cách ly tenant.

Ba nhóm, và nhóm thứ hai mới là phần đáng nhất của hạng mục:

* `TestSearchPath` — filter trên `retrieve()` của cả **bốn** nhánh. Phần này gần
  như đã có từ `W1-07`/`W2-05`; ở đây chỉ đóng lại cho đủ và thêm khoảng thời
  gian.
* `TestFetchPathWasTheHole` — `fetch_chunks()`/`fetch_doc_chunks()` **bỏ qua
  filter hoàn toàn** cho tới hạng mục này. Đó là đường mà `W4` sẽ dùng để giải
  citation và mở rộng ngữ cảnh, tức đường mà một `chunk_id` của tenant khác trả
  về nội dung đầy đủ dù mọi truy vấn vector đều lọc đúng.
* `TestFilterIsAppliedAtQdrant` — chứng minh filter áp **ở server**, không phải
  lọc lại ở client. Đó là chữ đầu tiên của DoD, và nó không suy ra được từ việc
  kết quả trông đúng: lọc sau cũng cho kết quả đúng, chỉ là dữ liệu đã bị đọc.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import pytest

from rag_core.embedding import HashingEmbeddingProvider
from rag_core.retrieval import (
    PAYLOAD_INDEXES,
    MetadataFilter,
    QdrantDenseRetriever,
    build_branch,
)
from rag_core.schemas import Chunk, DocType, DocumentMetadata, Language

pytest.importorskip("qdrant_client", reason="cần extra `qdrant`: uv sync --extra qdrant")
pytestmark = pytest.mark.integration

QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")

_QUERY = "ngân sách đầu tư công hạ tầng giao thông"

#: `(key, tenant, lang, doc_type, published_at)`. Nội dung **cố ý gần giống nhau**
#: giữa hai tenant: nếu chunk của `t2` khác chủ đề thì truy vấn vector sẽ không
#: trả nó về dù không có filter, và test sẽ xanh mà không chứng minh được gì.
_ROWS: tuple[tuple[str, str, Language, DocType, str], ...] = (
    ("a", "t1", Language.VI, DocType.DEV_REPORT, "2019-06-15"),
    ("b", "t1", Language.VI, DocType.LEGAL, "2021-03-01"),
    ("c", "t1", Language.EN, DocType.DEV_REPORT, "2023-11-30"),
    ("d", "t1", Language.EN, DocType.ANNUAL_REPORT, "2024-12-31"),
    # Bốn chunk của tenant khác, phủ hết mọi ô của `t1` để không filter nào loại
    # chúng một cách tình cờ nhờ `lang`/`doc_type`/ngày.
    ("e", "t2", Language.VI, DocType.DEV_REPORT, "2019-06-15"),
    ("f", "t2", Language.VI, DocType.LEGAL, "2021-03-01"),
    ("g", "t2", Language.EN, DocType.DEV_REPORT, "2023-11-30"),
    ("h", "t2", Language.EN, DocType.ANNUAL_REPORT, "2024-12-31"),
    # Không tenant, không ngày: hai ô "thiếu field" mà `MatchValue` và
    # `DatetimeRange` xử lý khác nhau — xem `TestMissingFields`.
    ("x", "", Language.VI, DocType.OTHER, ""),
)

_CONTENT = "Ngân sách nhà nước đầu tư công hạ tầng giao thông thống kê theo kế hoạch trung hạn."


def _chunk(key: str, tenant: str, lang: Language, doc_type: DocType, published: str) -> Chunk:
    return Chunk(
        chunk_id=f"doc-{key}::00000",
        doc_id=f"doc-{key}",
        content=f"{_CONTENT} Mã {key.upper()}.",
        chunk_index=0,
        metadata=DocumentMetadata(
            source_url=f"https://example.org/{key}",
            license="CC BY 4.0",
            lang=lang,
            doc_type=doc_type,
            published_at=(
                datetime.fromisoformat(published).replace(tzinfo=UTC) if published else None
            ),
        ),
        extra=({"tenant_id": tenant} if tenant else {}),
    )


CHUNKS = [_chunk(*row) for row in _ROWS]
T1_IDS = {f"doc-{k}::00000" for k, tenant, *_ in _ROWS if tenant == "t1"}
T2_IDS = {f"doc-{k}::00000" for k, tenant, *_ in _ROWS if tenant == "t2"}
ALL_MODES = ("dense", "sparse", "hybrid", "reranked")


@pytest.fixture
def store() -> Iterator[QdrantDenseRetriever]:
    collection = f"test_filter_{uuid.uuid4().hex[:8]}"
    retriever = QdrantDenseRetriever(
        HashingEmbeddingProvider(dimension=256, sparse=True),
        collection=collection,
        url=QDRANT_URL,
    )
    try:
        retriever.ensure_collection(recreate=True)
    except Exception as exc:  # pragma: no cover - phụ thuộc môi trường
        pytest.fail(f"Không kết nối được Qdrant tại {QDRANT_URL}: {exc}. Chạy `make up` trước.")
    try:
        retriever.upsert(CHUNKS)
        yield retriever
    finally:
        retriever.client.delete_collection(collection)


def _ids(hits: Any) -> set[str]:
    return {hit.chunk.chunk_id for hit in hits}


class TestSearchPath:
    def test_without_a_filter_both_tenants_are_visible(self, store: QdrantDenseRetriever) -> None:
        """Điểm neo của cả file: không có nó thì mọi test cách ly đều vô nghĩa.

        Nếu truy vấn vốn đã không trả chunk của `t2` (vì nội dung khác chủ đề)
        thì "có filter → không thấy `t2`" là xanh vì lý do sai.
        """
        found = _ids(store.retrieve(_QUERY, top_k=9))
        assert found & T1_IDS
        assert found & T2_IDS, "corpus test phải để hai tenant cùng cạnh tranh trên cùng truy vấn"

    @pytest.mark.parametrize("mode", ALL_MODES)
    def test_no_cross_tenant_leak_on_any_branch(
        self, store: QdrantDenseRetriever, mode: str
    ) -> None:
        """DoD của `W2-06`. Chạy trên cả bốn nhánh vì mỗi nhánh có đường filter riêng.

        `dense` dùng `query_points`; `hybrid` dùng `query_batch_points` với filter
        truyền vào **từng** `QueryRequest`; `reranked` chuyển tiếp xuống nhánh nền.
        Một nhánh quên truyền là một lỗ, và ba nhánh còn lại xanh không nói gì về
        nhánh thứ tư.
        """
        branch = build_branch(
            store, mode, **({"rerank_device": "cpu"} if mode == "reranked" else {})
        )
        if mode == "reranked":
            # Reranker thật không liên quan ở đây; điều cần biết là ứng viên nào
            # tới được nó. Bản giả cho điểm theo thứ tự để khỏi nạp model 2,2GB.
            branch.reranker = _OrderReranker()  # type: ignore[attr-defined]
        found = _ids(branch.retrieve(_QUERY, top_k=9, filters={"tenant_id": "t1"}))
        assert found, f"nhánh {mode} không trả gì — filter chặt quá, test mất nghĩa"
        assert not found & T2_IDS, f"nhánh {mode} rò dữ liệu tenant t2"
        assert found <= T1_IDS

    def test_lang_filter(self, store: QdrantDenseRetriever) -> None:
        found = _ids(store.retrieve(_QUERY, top_k=9, filters={"lang": "en"}))
        assert found == {"doc-c::00000", "doc-d::00000", "doc-g::00000", "doc-h::00000"}

    def test_doc_type_list_is_a_union(self, store: QdrantDenseRetriever) -> None:
        found = _ids(store.retrieve(_QUERY, top_k=9, filters={"doc_type": ["legal", "other"]}))
        assert found == {"doc-b::00000", "doc-f::00000", "doc-x::00000"}

    def test_date_window_is_inclusive_at_both_ends(self, store: QdrantDenseRetriever) -> None:
        """Ghim lựa chọn đã ghi trong docstring của `MetadataFilter`.

        Hai mốc là **đúng** ngày của `b` và `c`. Nếu ai đổi sang nửa mở thì hai
        chunk đó rơi ra và test này đỏ — đó là điểm của nó.
        """
        flt = MetadataFilter(
            published_after=datetime(2021, 3, 1, tzinfo=UTC),
            published_before=datetime(2023, 11, 30, tzinfo=UTC),
        )
        found = _ids(store.retrieve(_QUERY, top_k=9, filters=flt))
        assert found == {"doc-b::00000", "doc-c::00000", "doc-f::00000", "doc-g::00000"}

    def test_one_sided_window(self, store: QdrantDenseRetriever) -> None:
        flt = MetadataFilter(published_after=datetime(2024, 1, 1, tzinfo=UTC))
        assert _ids(store.retrieve(_QUERY, top_k=9, filters=flt)) == {
            "doc-d::00000",
            "doc-h::00000",
        }

    def test_tenant_and_date_compose_as_and(self, store: QdrantDenseRetriever) -> None:
        flt = MetadataFilter(tenant_id="t1", published_after=datetime(2022, 1, 1, tzinfo=UTC))
        assert _ids(store.retrieve(_QUERY, top_k=9, filters=flt)) == {
            "doc-c::00000",
            "doc-d::00000",
        }

    def test_unknown_key_raises_before_touching_qdrant(self, store: QdrantDenseRetriever) -> None:
        with pytest.raises(ValueError, match="tenant_id"):
            store.retrieve(_QUERY, top_k=5, filters={"tenant": "t1"})


class TestMissingFields:
    """Point thiếu field phải **rơi ra**, không phải lọt vào. Hướng hỏng là điểm."""

    def test_a_chunk_without_tenant_matches_no_tenant_filter(
        self, store: QdrantDenseRetriever
    ) -> None:
        """`doc-x` không có `tenant_id`. Nó không được thuộc về tenant nào.

        Ghim hành vi `MatchValue` của Qdrant chứ không chỉ tin tài liệu: nếu một
        phiên bản sau coi field thiếu là khớp mọi giá trị thì đây là lỗ rò.
        """
        for tenant in ("t1", "t2", ""):
            found = _ids(store.retrieve(_QUERY, top_k=9, filters={"tenant_id": tenant or "t3"}))
            assert "doc-x::00000" not in found

    def test_a_chunk_without_a_date_falls_out_of_every_window(
        self, store: QdrantDenseRetriever
    ) -> None:
        """`doc-x` không có `published_at` → không khớp `DatetimeRange` nào.

        Đây là lý do `backfill_flat_payload` phải chạy trên collection cũ: point
        thiếu field trông y như point ngoài khoảng.
        """
        wide = MetadataFilter(
            published_after=datetime(1900, 1, 1, tzinfo=UTC),
            published_before=datetime(2100, 1, 1, tzinfo=UTC),
        )
        found = _ids(store.retrieve(_QUERY, top_k=9, filters=wide))
        assert "doc-x::00000" not in found
        assert len(found) == 8, "tám chunk còn lại đều có ngày nên phải nằm trong khoảng rộng"


class TestFetchPathWasTheHole:
    """Đường mà `W4` sẽ dùng để giải citation, và nó không hề đi qua filter."""

    def test_fetch_chunks_without_a_filter_still_returns_everything(
        self, store: QdrantDenseRetriever
    ) -> None:
        """Ghim hành vi **hiện tại**, không phải hành vi mong muốn.

        `fetch_chunks` không filter là đúng cho eval (chạy trên toàn corpus) và là
        lỗ rò cho serving. `rag_core` không phân biệt được hai ngữ cảnh đó, nên
        chỗ ép là `W4-04`. Test này tồn tại để lúc ấy không ai tưởng `W2-06` đã
        đóng chuyện này.
        """
        got = store.fetch_chunks(["doc-a::00000", "doc-e::00000"])
        assert set(got) == {"doc-a::00000", "doc-e::00000"}

    def test_fetch_chunks_with_a_tenant_filter_drops_the_other_tenant(
        self, store: QdrantDenseRetriever
    ) -> None:
        got = store.fetch_chunks(["doc-a::00000", "doc-e::00000"], filters={"tenant_id": "t1"})
        assert set(got) == {"doc-a::00000"}

    def test_fetch_chunks_filter_survives_ids_from_another_tenant_only(
        self, store: QdrantDenseRetriever
    ) -> None:
        """Ca tấn công thật: mọi id đều thuộc tenant khác → phải trả rỗng."""
        assert store.fetch_chunks(list(T2_IDS), filters={"tenant_id": "t1"}) == {}

    def test_fetch_doc_chunks_with_a_tenant_filter(self, store: QdrantDenseRetriever) -> None:
        got = store.fetch_doc_chunks(["doc-a", "doc-e"], filters={"tenant_id": "t1"})
        assert [c.chunk_id for c in got] == ["doc-a::00000"]

    def test_fetch_doc_chunks_unfiltered_is_unchanged(self, store: QdrantDenseRetriever) -> None:
        """Đường của `pipeline/eval/spans.py` — không được đổi hành vi."""
        got = store.fetch_doc_chunks(["doc-a", "doc-e"])
        assert {c.chunk_id for c in got} == {"doc-a::00000", "doc-e::00000"}

    def test_fetch_paths_agree_with_the_search_path(self, store: QdrantDenseRetriever) -> None:
        """Cùng filter, cùng tập chunk — dù ba đường code hoàn toàn khác nhau.

        `retrieve` dùng `query_points`, `fetch_chunks` dùng `scroll` theo
        `chunk_id`, `fetch_doc_chunks` dùng `scroll` theo `doc_id`. Ba cách dựng
        filter khác nhau cho cùng một câu hỏi là ba chỗ để lệch nhau.
        """
        flt = MetadataFilter(tenant_id="t1", lang=Language.EN)
        searched = _ids(store.retrieve(_QUERY, top_k=9, filters=flt))
        fetched = set(store.fetch_chunks([f"doc-{k}::00000" for k, *_ in _ROWS], filters=flt))
        by_doc = {
            c.chunk_id for c in store.fetch_doc_chunks([f"doc-{k}" for k, *_ in _ROWS], filters=flt)
        }
        assert searched == fetched == by_doc == {"doc-c::00000", "doc-d::00000"}


class TestFilterIsAppliedAtQdrant:
    def test_filter_travels_in_the_request_not_applied_after(
        self, store: QdrantDenseRetriever, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Chữ đầu của DoD: "áp ở tầng Qdrant (không post-filter)".

        Không suy ra được từ kết quả — lọc sau cũng cho kết quả đúng. Nên phải
        theo dõi chính request.
        """
        seen: list[Any] = []
        original = store.client.query_points

        def spy(*args: Any, **kwargs: Any) -> Any:
            seen.append(kwargs.get("query_filter"))
            return original(*args, **kwargs)

        monkeypatch.setattr(store.client, "query_points", spy)
        store.retrieve(_QUERY, top_k=5, filters={"tenant_id": "t1"})
        assert len(seen) == 1
        assert seen[0] is not None, "filter không tới được request — đang lọc ở client"
        keys = {c.key for c in seen[0].must}
        assert keys == {"tenant_id"}

    def test_hybrid_puts_the_filter_on_both_sub_requests(
        self, store: QdrantDenseRetriever, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Nhánh hybrid gửi hai `QueryRequest`; quên một cái là rò một nửa.

        Và nửa bị rò sẽ **không** hiện ra trong kết quả cuối nếu RRF không đẩy nó
        lên top-k — tức lỗ có thể tồn tại rất lâu mà mọi test kết quả vẫn xanh.
        """
        seen: list[Any] = []
        original = store.client.query_batch_points

        def spy(*args: Any, **kwargs: Any) -> Any:
            seen.extend(r.filter for r in kwargs["requests"])
            return original(*args, **kwargs)

        monkeypatch.setattr(store.client, "query_batch_points", spy)
        build_branch(store, "hybrid").retrieve(_QUERY, top_k=5, filters={"tenant_id": "t1"})
        assert len(seen) == 2
        assert all(f is not None for f in seen), "một nhánh con không mang filter"

    def test_every_filter_field_has_a_live_payload_index(self, store: QdrantDenseRetriever) -> None:
        """`ensure_collection` phải dựng đủ index — kiểm trên Qdrant thật.

        Test đơn vị chỉ so hai hằng số trong Python. Cái này hỏi chính server, nên
        nó bắt được cả trường hợp `create_payload_index` lặng lẽ không nhận một
        `field_schema` nào đó.
        """
        live = store.client.get_collection(store.collection).payload_schema or {}
        assert {field for field, _ in PAYLOAD_INDEXES} <= set(live)


class _OrderReranker:
    """Reranker giả: điểm giảm dần theo thứ tự nhánh nền đưa vào.

    Giữ thứ tự của nhánh nền để test filter không phải quan tâm tới thứ hạng —
    ở đây câu hỏi duy nhất là ứng viên **nào** tới được reranker.
    """

    name = "order-fake"

    def score(self, query: str, texts: Any) -> list[float]:
        return [float(len(texts) - i) for i in range(len(texts))]
