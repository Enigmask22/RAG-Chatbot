"""`TD-38` — runtime dựng ra có **là** hệ thống đã được đo không.

Ba phép kiểm sẵn có của `QdrantRuntimeBuilder` (số chiều, schema collection, số
điểm) đều nói về **index**. Chúng xanh hết khi bundle chạy đúng index nhưng với
`rrf_k` khác, `candidates` khác, hay reranker chạy fp32 thay vì fp16 — tức đúng
lúc mọi số trong `bundle.eval` nói về một hệ thống khác hệ thống đang phục vụ.

Không cần Qdrant/GPU: phép kiểm danh tính là một phép so chuỗi, tách được khỏi
phần dựng.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any, cast

import pytest

from rag_core.bundle import (
    BundleComponents,
    ChunkingComponent,
    EmbeddingComponent,
    EvalReport,
    GateRecord,
    GateStatus,
    IndexComponent,
    RagBundle,
    RerankComponent,
    RetrievalComponent,
)
from rag_core.retrieval.base import Retriever
from serving.core.runtime import BundleRuntimeError, QdrantRuntimeBuilder, drift_of

MEASURED = (
    "reranked[qdrant-hybrid:rag_bgem3_ctx:rrf1-c20]:BAAI/bge-reranker-v2-m3@cuda:L512:float16:n50"
)
#: Cùng bundle, cùng index, cùng model — chỉ khác **độ chính xác số**. `W2-05` đo
#: được fp16 và fp32 trùng top-1 98,3%, tức 1,7% câu trả lời khác nhau.
ON_CPU = MEASURED.replace("@cuda:L512:float16", "@cpu:L512")


@dataclass
class FakeRetriever:
    name: str


def make_bundle(version: str = "1.0.0", *, retriever_name: str | None = MEASURED) -> RagBundle:
    return RagBundle(
        bundle_version=version,
        created_at=datetime(2026, 9, 3, tzinfo=UTC),
        git_sha="29718f8",
        components=BundleComponents(
            chunking=ChunkingComponent(
                strategy="hybrid",
                chunk_size=1000,
                chunk_overlap=100,
                contextual=True,
                chunking_fingerprint="c7ca3e6fc4da29a5",
            ),
            embedding=EmbeddingComponent(model="BAAI/bge-m3", dim=1024, normalize=True),
            index=IndexComponent(
                backend="qdrant",
                collection="rag_bgem3_ctx",
                fingerprint="f" * 64,
                n_chunks=15814,
                n_documents=60,
            ),
            retrieval=RetrievalComponent(mode="hybrid", top_k=20, options={"k": 1}),
            rerank=RerankComponent(
                model="BAAI/bge-reranker-v2-m3", candidates=50, top_n=6, max_length=512
            ),
            retriever_name=retriever_name,
        ),
        eval=EvalReport(
            golden_set="golden_v1",
            n_queries=209,
            evaluated_with_generator="deepseek-chat@2026-09",
            retrieval_metrics={"ndcg@10": 0.6888},
        ),
        gate=GateRecord(status=GateStatus.NOT_RUN),
    )


def check(builder: QdrantRuntimeBuilder, name: str, bundle: RagBundle) -> None:
    builder._check_identity(cast("Retriever", FakeRetriever(name)), bundle)


def test_the_same_system_passes() -> None:
    check(QdrantRuntimeBuilder(url="x"), MEASURED, make_bundle())


def test_serving_at_a_different_numeric_precision_is_refused() -> None:
    """⭐⭐ Chính là lỗ hổng `TD-38` mô tả, và nó là lỗ hổng **im lặng**.

    Cùng bundle, cùng index, cùng model, cùng `max_length` — chỉ khác `dtype`.
    Cả ba phép kiểm kia xanh: số chiều đúng, schema đúng, số điểm đúng. Và 1,7%
    câu trả lời khác với những gì đã được đo.
    """
    with pytest.raises(BundleRuntimeError) as excinfo:
        check(QdrantRuntimeBuilder(url="x"), ON_CPU, make_bundle())
    assert "float16" in str(excinfo.value)
    assert "cpu" in str(excinfo.value)


@pytest.mark.parametrize(
    "actual",
    [
        MEASURED.replace("rrf1", "rrf60"),  # `W2-04` đo được k=60 là giá trị TỆ NHẤT
        MEASURED.replace("n50", "n20"),  # pool nông hơn pool đã đo
        MEASURED.replace("L512", "L256"),  # cắt ngắn nhiều hơn
        MEASURED.replace("qdrant-hybrid", "qdrant-dense"),  # bỏ hẳn một nhánh
    ],
)
def test_every_knob_that_changes_the_numbers_is_caught(actual: str) -> None:
    """Bốn cần điều khiển, bốn hệ thống khác nhau — và **một** phép so chuỗi bắt
    hết, vì quy ước đặt tên của `rag_core` đã gom sẵn đúng chúng."""
    with pytest.raises(BundleRuntimeError):
        check(QdrantRuntimeBuilder(url="x"), actual, make_bundle())


def test_a_harmless_difference_is_not_in_the_name_so_it_does_not_trip() -> None:
    """Mặt kia của cùng một quyết định: `batch_size` **không** nằm trong `name` vì
    nó không đổi điểm. Nếu nó có mặt thì phép kiểm này sẽ đỏ mỗi lần đổi máy, và
    người ta sẽ tắt nó — mất luôn cả bốn ca ở trên."""
    check(QdrantRuntimeBuilder(url="x", batch_size=8), MEASURED, make_bundle())


def test_an_old_bundle_without_the_field_warns_instead_of_refusing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Bundle sinh trước `TD-38` không có trường này. Từ chối chúng nghĩa là bản
    sửa lỗi tự làm hỏng mọi artifact đã phát hành — nhưng im lặng cho qua thì
    người đọc log tưởng phép kiểm đã chạy."""
    with caplog.at_level(logging.WARNING):
        check(QdrantRuntimeBuilder(url="x"), ON_CPU, make_bundle(retriever_name=None))
    assert "TD-38" in caplog.text


def test_the_escape_hatch_is_off_by_default_and_loud_when_on(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """⭐ Máy dev không có GPU vẫn phải chạy thử được bundle đã eval trên
    `cuda:float16` — cái giá của *không chạy thử được* lớn hơn. Nhưng cửa thoát
    phải kêu: WARNING mỗi lần kích hoạt, **và** một trường trong
    `GET /admin/bundle`, vì một cửa thoát im lặng mới là cửa thoát nguy hiểm."""
    builder = QdrantRuntimeBuilder(url="x", allow_runtime_drift=True)
    with caplog.at_level(logging.WARNING):
        check(builder, ON_CPU, make_bundle(version="9.9.9"))
    assert "BUNDLE_ALLOW_RUNTIME_DRIFT" in caplog.text
    assert drift_of("9.9.9") == {"expected": MEASURED, "actual": ON_CPU}


def test_no_drift_is_recorded_for_a_matching_runtime() -> None:
    check(QdrantRuntimeBuilder(url="x", allow_runtime_drift=True), MEASURED, make_bundle("9.9.8"))
    assert drift_of("9.9.8") is None


# ---------------------------------------------------------------------------
# ⭐⭐ Đường `__call__` — phần dễ đứt nhất là chỗ NỐI, không phải từng hàm
# ---------------------------------------------------------------------------


class FakeQdrant:
    """Chỉ trả lời đúng hai câu mà builder hỏi: schema collection và số điểm."""

    def __init__(self, *, dim: int = 8, points: int = 15814) -> None:
        self.dim, self.points = dim, points

    def get_collection(self, name: str) -> Any:
        vectors = {"dense": SimpleNamespace(size=self.dim)}
        # Không khai sparse: `hashing:8` là provider dense-only, và
        # `schema_problems` cố ý coi "collection có sparse mà provider không sinh"
        # là **lỗi** — nửa index không được dùng mà số vẫn trông bình thường (`W2-02`).
        params = SimpleNamespace(vectors=vectors, sparse_vectors={})
        return SimpleNamespace(config=SimpleNamespace(params=params))

    def count(self, collection_name: str, exact: bool = True) -> Any:
        return SimpleNamespace(count=self.points)


def hashing_bundle(**kwargs: Any) -> RagBundle:
    """Bundle dùng provider `hashing:` — sinh cả dense lẫn sparse, không cần GPU
    và không tải model nào (`W2-02` dựng nó đúng cho việc này)."""
    bundle = make_bundle(**kwargs)
    return bundle.model_copy(
        update={
            "components": bundle.components.model_copy(
                update={
                    "embedding": EmbeddingComponent(model="hashing:8", dim=8, normalize=True),
                    "retrieval": RetrievalComponent(mode="dense", top_k=20),
                    "rerank": None,
                }
            )
        }
    )


DENSE_NAME = "qdrant-dense:rag_bgem3_ctx"


def test_the_identity_check_is_actually_wired_into_the_build() -> None:
    """⭐⭐ Test bắt được thứ mà 10 test ở trên bỏ lọt.

    Chúng gọi thẳng `_check_identity`, nên **xoá hẳn dòng gọi nó khỏi
    `__call__`** vẫn xanh hết — phép kiểm còn nguyên và không bao giờ chạy. Lần
    tiêm lỗi đầu tiên cho đúng kết quả đó, và đó là lý do có `client` tiêm được.
    """
    builder = QdrantRuntimeBuilder(url="x", client=FakeQdrant())
    with pytest.raises(BundleRuntimeError, match="hệ thống khác"):
        builder(hashing_bundle(retriever_name="qdrant-dense:collection-nào-đó"))


def test_a_matching_runtime_builds_all_the_way_through() -> None:
    retriever, reranker = QdrantRuntimeBuilder(url="x", client=FakeQdrant())(
        hashing_bundle(retriever_name=DENSE_NAME)
    )
    assert retriever.name == DENSE_NAME
    assert reranker is None


def test_a_collection_that_grew_since_the_eval_is_refused() -> None:
    """Phép kiểm **danh tính**, không phải toàn vẹn: các số trong `bundle.eval`
    được đo trên một index có đúng bằng ấy chunk."""
    builder = QdrantRuntimeBuilder(url="x", client=FakeQdrant(points=15815))
    with pytest.raises(BundleRuntimeError, match="15,815"):
        builder(hashing_bundle(retriever_name=DENSE_NAME))


def test_a_model_that_does_not_match_the_declared_dim_is_refused() -> None:
    bundle = hashing_bundle(retriever_name=DENSE_NAME)
    lying = bundle.model_copy(
        update={
            "components": bundle.components.model_copy(
                update={
                    "embedding": EmbeddingComponent(model="hashing:8", dim=1024, normalize=True)
                }
            )
        }
    )
    with pytest.raises(BundleRuntimeError, match="chiều"):
        QdrantRuntimeBuilder(url="x", client=FakeQdrant())(lying)
