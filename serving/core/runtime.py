"""Dựng runtime thật từ một `RagBundle` — `W4-03`, trả món nợ của `W4-02`.

`BundleRegistry` nhận `RuntimeBuilder` từ ngoài nên nó không biết gì về Qdrant
hay torch. Đây là bản cài đặt thật của cái Protocol ấy: **manifest vào, retriever
đang chạy ra**.

## Nó là chỗ duy nhất bắt được "bundle nói một đằng, hệ thống chạy một nẻo"

Checksum của `W4-01` bảo vệ manifest khỏi bị sửa sau khi ký. Nó không nói gì về
việc collection mà manifest trỏ tới **có còn là collection đã được eval hay
không** — Qdrant nằm ngoài chữ ký. Nên chỗ này hỏi lại ba câu mà bundle có nêu
và hệ thống trả lời được:

* **số chiều** — model nạp lên sinh bao nhiêu chiều, so với `embedding.dim`;
* **schema collection** — có đúng named vector mà nhánh này cần không
  (`verify_schema`, dùng lại phép kiểm của `W2-02`);
* **số điểm** — `index.n_chunks` là số chunk mà eval đã đo trên đó.

⚠️ **Câu thứ tư hỏi không được, và đó là lỗ hổng thật.** `index.fingerprint` là
vân tay của `IndexConfig`, và nó chỉ tồn tại trong file state của Pipeline Plane
(`IndexState`, cạnh `build_index`). Serving Plane không có file đó và **không
được phép** có — nó thuộc plane kia. Hệ quả: một collection bị build lại với
chunking khác nhưng **ra đúng bằng ấy chunk** sẽ qua được cả ba phép kiểm trên.
Số điểm bắt được ca thường gặp, không bắt được ca này. Xem `TD-38`.

## Cache model: đã có sẵn, nhưng khoá của nó rộng hơn ta tưởng

`W4-02` ghi nợ "cache model theo danh tính" vì giữ bundle trước để rollback là
giữ **hai** runtime. Món nợ ấy **phần lớn đã được trả từ trước**:
`rag_core.reranking.cross_encoder._load_cross_encoder` là `lru_cache(maxsize=2)`,
và `rag_core.embedding` cũng cache tương tự. Hai bundle chỉ khác `top_k` hay
`rrf_k` dùng chung đúng một bản trọng số.

⚠️ Nhưng khoá cache của reranker là bộ **bốn** `(model, device, max_length,
dtype)`, mà `max_length` **nằm trong** `RerankComponent`. Nên hai bundle khác
nhau *chỉ ở* `rerank.max_length` là **hai** model trong cache — 2,2 GB nữa trên
GPU 8 GB đã có 3,3 GB của bge-m3. Đổi bundle theo hướng đó lúc đang chạy là cách
thật để OOM một tiến trình đang phục vụ, và phiên pytest của `W2-05` đã OOM đúng
theo tổ hợp đó.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol, cast

from rag_core.bundle import RagBundle
from rag_core.reranking.base import Reranker
from rag_core.retrieval.base import Retriever

__all__ = ["BundleRuntimeError", "QdrantRuntimeBuilder", "drift_of"]

_DRIFT: dict[str, dict[str, str]] = {}
"""Lệch tên retriever đã bị bỏ qua, theo version bundle.

Ở tầng module chứ không trong builder: người cần đọc nó là route
`GET /admin/bundle`, và route đó cầm `ActiveBundle` chứ không cầm builder.
"""


def drift_of(version: str) -> dict[str, str] | None:
    return _DRIFT.get(version)


logger = logging.getLogger(__name__)


class _SchemaAware(Protocol):
    def verify_schema(self) -> None: ...


class BundleRuntimeError(RuntimeError):
    """Bundle hợp lệ về hình thức nhưng không dựng được trên hệ thống này.

    Tách khỏi `BundleValidationError` (sai schema/chữ ký) vì hai lỗi này đòi hai
    hành động khác nhau: một cái sửa manifest, một cái sửa **hạ tầng**.
    """


@dataclass
class QdrantRuntimeBuilder:
    """`RuntimeBuilder` thật. Mọi tham số ở đây là thứ bundle **không** nêu.

    Ranh giới đó là có chủ đích: bundle mô tả *hệ thống đã được đo*, còn URL của
    Qdrant hay việc máy này có GPU hay không là *chỗ nó chạy*. Trộn hai loại vào
    một file thì cùng một bundle không đem sang môi trường khác được nữa.

    ⭐ `device` và `dtype` của reranker rơi vào **khe hở** giữa hai loại ấy: bản
    fp16 và bản fp32 của cross-encoder là hai model khác nhau *về số* (`W2-05` đo
    được trùng top-1 98,3%), nên chúng đổi kết quả — tức thuộc về bundle — nhưng
    chúng cũng là thuộc tính của phần cứng. `TD-38` đóng khe hở ấy bằng
    `components.retriever_name`, và phép so ở `_check_identity` bắt được **cả**
    trường hợp phục vụ trên CPU một bundle đã eval trên GPU.
    """

    url: str
    api_key: str | None = None
    device: str = "auto"
    batch_size: int = 32
    allow_runtime_drift: bool = False
    """Cho phép chạy một bundle mà runtime dựng ra **không khớp** tên đã đo.

    Tồn tại cho đúng một tình huống: máy dev không có GPU muốn chạy thử một
    bundle đã eval trên `cuda:float16`. Refuse cứng ở đó thì không ai chạy được
    hệ thống trên laptop, và cái giá của việc *không chạy thử được* lớn hơn.

    ⚠️ Tắt theo mặc định, phải bật tường minh cho từng môi trường, và khi bật thì
    **mỗi lần kích hoạt** ghi một dòng WARNING kèm cả hai tên — đồng thời
    `GET /admin/bundle` trả `runtime_drift`, nên nó không chỉ nằm trong log mà
    còn nằm ở chỗ người vận hành nhìn. Một cửa thoát im lặng mới là cửa thoát
    nguy hiểm.
    """
    #: Ngắn hơn hẳn `ReadinessProbes.timeout_s` để luồng của probe tự quay về
    #: trước khi `wait_for` bỏ chờ — nếu ngược lại thì mỗi chu kỳ probe rò một
    #: luồng đang chờ socket.
    qdrant_timeout_s: float = 1.5
    client: Any = None
    """Client Qdrant dựng sẵn, thay cho việc tự mở kết nối từ `url`.

    Có mặt để **đường `__call__` chạy được trong test**. Không có nó thì mọi test
    của builder buộc phải gọi thẳng từng hàm con, và phép **nối** chúng lại — thứ
    dễ đứt nhất — không được kiểm: bỏ hẳn dòng gọi `_check_identity` khỏi
    `__call__` vẫn xanh toàn bộ. Đó đúng là kết quả của lần tiêm lỗi đầu tiên.
    """

    def __call__(self, bundle: RagBundle) -> tuple[Retriever, Reranker | None]:
        from rag_core.embedding import build_embedding_provider
        from rag_core.retrieval import build_branch
        from rag_core.retrieval.qdrant_store import QdrantDenseRetriever

        components = bundle.components
        spec = components.embedding

        embeddings = build_embedding_provider(
            spec.model,
            device=self.device,
            batch_size=self.batch_size,
            normalize=spec.normalize,
        )
        if embeddings.dimension != spec.dim:
            raise BundleRuntimeError(
                f"bundle {bundle.bundle_version} khai embedding {spec.dim} chiều nhưng "
                f"{spec.model!r} nạp lên sinh {embeddings.dimension} chiều — "
                "manifest và trọng số không phải cùng một model."
            )

        store = QdrantDenseRetriever(
            embeddings,
            collection=components.index.collection,
            url=self.url,
            api_key=self.api_key,
            timeout=self.qdrant_timeout_s,
            client=self.client,
        )

        retriever = build_branch(
            store,
            components.retrieval.mode,
            **_branch_options(components.retrieval.options),
        )
        reranker: Reranker | None = None
        if components.rerank is not None:
            reranker = _build_reranker(components.rerank, device=self.device)
            retriever = _wrap(retriever, reranker, components.rerank)

        # Sau khi dựng, TRƯỚC khi `BundleRegistry` đổi sang: cả hai phép kiểm dưới
        # đây chạm mạng, và luật 1 của `W4-02` nói mọi thứ hỏng được phải xảy ra
        # trước phép gán.
        # ⚠️ `verify_schema` có ở **mọi** retriever của `rag_core` nhưng ABC
        # `Retriever` không khai nó, nên người cầm một `Retriever` không được
        # hứa gì. Sửa ABC là việc của `rag_core`, ngoài phạm vi `W4-03`; ở đây
        # `cast` để chỗ này đọc ra là một lỗ hổng hợp đồng chứ không phải một
        # phép gọi bình thường.
        cast("_SchemaAware", retriever).verify_schema()
        _check_size(store, bundle)
        self._check_identity(retriever, bundle)

        # Log `retriever.name` chứ không log lại từng trường: quy ước đặt tên của
        # `rag_core` gom **đúng những cần điều khiển làm đổi kết quả** vào một
        # chuỗi, nên đây là dòng duy nhất so được bằng mắt với `retriever` trong
        # báo cáo eval mà bundle này đóng gói.
        logger.info(
            "runtime bundle %s: %s",
            bundle.bundle_version,
            retriever.name,
            extra={"bundle_version": bundle.bundle_version, "retriever": retriever.name},
        )
        return retriever, reranker

    def _check_identity(self, retriever: Retriever, bundle: RagBundle) -> None:
        """⭐⭐ `TD-38`: runtime vừa dựng có **là** hệ thống đã được đo không.

        Đây là phép kiểm mà ba phép kiểm kia không thay được. Số chiều, schema
        collection và số điểm nói về **index**; chúng đều xanh khi bundle chạy
        đúng index nhưng với `rrf_k` khác, `candidates` khác, hay reranker chạy
        fp32 thay vì fp16 — tức khi mọi số trong `bundle.eval` nói về một hệ
        thống khác hệ thống đang phục vụ.
        """
        expected = bundle.components.retriever_name
        if expected is None:
            logger.warning(
                "bundle %s không khai `retriever_name` (sinh trước TD-38) — "
                "không kiểm được runtime có khớp hệ thống đã eval hay không",
                bundle.bundle_version,
            )
            return
        if retriever.name == expected:
            return
        message = (
            f"bundle {bundle.bundle_version} được eval trên\n  {expected}\n"
            f"nhưng máy này dựng ra\n  {retriever.name}\n"
            "— mọi metric trong manifest nói về một hệ thống khác."
        )
        if not self.allow_runtime_drift:
            raise BundleRuntimeError(message)
        logger.warning("BUNDLE_ALLOW_RUNTIME_DRIFT đang bật: %s", message)
        _DRIFT[bundle.bundle_version] = {"expected": expected, "actual": retriever.name}


def _branch_options(options: dict[str, Any]) -> dict[str, Any]:
    """JSON không có tuple. `weights` đi ra khỏi manifest dưới dạng list, và
    `QdrantHybridRetriever` đưa nó vào `name` — nên không đổi lại thì hai lần
    chạy cùng cấu hình có hai cái tên khác nhau."""
    resolved = dict(options)
    weights = resolved.get("weights")
    if isinstance(weights, list):
        resolved["weights"] = tuple(weights)
    return resolved


def _build_reranker(spec: Any, *, device: str) -> Reranker:
    from rag_core.reranking import CrossEncoderReranker

    return CrossEncoderReranker(spec.model, device=device, max_length=spec.max_length)


def _wrap(base: Retriever, reranker: Reranker, spec: Any) -> Retriever:
    from rag_core.retrieval.reranked import RerankedRetriever

    return RerankedRetriever(base, reranker, candidates=spec.candidates, top_n=spec.top_n)


def _check_size(store: Any, bundle: RagBundle) -> None:
    """Số điểm thật vs `index.n_chunks`.

    Không phải phép kiểm tính toàn vẹn — nó là phép kiểm **danh tính**: các số
    trong `bundle.eval` được đo trên một index có đúng bằng ấy chunk. Lệch nghĩa
    là index đã bị build lại hoặc bị ghi thêm sau khi eval chạy, và mọi con số
    trong manifest nói về một thứ không còn tồn tại.
    """
    actual = store.count()
    if actual != bundle.components.index.n_chunks:
        raise BundleRuntimeError(
            f"collection {bundle.components.index.collection!r} có {actual:,} điểm nhưng "
            f"bundle {bundle.bundle_version} được eval trên {bundle.components.index.n_chunks:,}. "
            "Index đã đổi sau khi đóng gói — mọi metric trong manifest nói về một index khác."
        )
