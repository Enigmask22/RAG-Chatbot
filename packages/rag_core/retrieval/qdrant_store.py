"""Lưu trữ và truy hồi trên Qdrant — dense và sparse trong **một** collection.

Ba quyết định của tầng này:

* **Dùng named vector (`dense`) ngay cả khi mới có một loại vector.** Quyết định
  từ `W1-07`, và `W2-02` là chỗ nó trả nợ: collection tạo bằng vector vô danh
  không thể thêm named vector mà không build lại toàn bộ index.
* **Point ID là UUIDv5 sinh từ `chunk_id`.** Qdrant chỉ nhận UUID hoặc số nguyên
  làm ID. Sinh xác định từ `chunk_id` khiến việc upsert lại cùng một chunk ghi đè
  đúng point cũ thay vì tạo bản trùng — đây chính là tính idempotent mà `W1-08`
  yêu cầu, và nó phải nằm ở tầng này chứ không phải ở script build index.
* **`ensure_collection` KIỂM TRA schema khi collection đã tồn tại** (`W2-02`).
  Trước đây nó thấy tồn tại là trả về ngay, nên chạy provider sinh sparse lên
  collection dense-only sẽ chết ở *giữa* một job 15.000 chunk — hoặc tệ hơn là
  ghi thành công phần dense và im lặng bỏ phần sparse. Giờ nó chết ở giây đầu và
  in ra đúng lệnh phải chạy.

`retrieve()` là **dense**; sparse có đường riêng `retrieve_sparse()`. Cố ý không
gộp: `W2-04` (RRF) cần hai danh sách xếp hạng *độc lập* để hợp nhất, và một hàm
trả "hybrid" ngay từ tầng store sẽ không tách được đóng góp của mỗi nhánh —
thứ mà `RetrievedChunk.dense_score`/`sparse_score` tồn tại để giữ.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Collection, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from ..embedding.base import EmbeddingProvider
from ..embedding.sparse import SparseVector
from ..schemas import Chunk, RetrievalMode, RetrievedChunk
from .base import Retriever
from .filters import FILTER_FIELDS, FilterSpec, MetadataFilter, build_filter

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

__all__ = [
    "DENSE_VECTOR_NAME",
    "FILTER_FIELDS",
    "PAYLOAD_INDEXES",
    "SPARSE_VECTOR_NAME",
    "MetadataFilter",
    "QdrantDenseRetriever",
    "build_filter",
    "chunk_point_id",
    "points_to_chunks",
    "schema_problems",
]

logger = logging.getLogger(__name__)

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"

#: (field payload, kiểu index Qdrant) cho mọi field lọc được. Phải phủ đúng
#: `FILTER_FIELDS` — có test canh hai chiều, vì lệch chiều nào cũng hỏng im lặng:
#: field lọc được mà không có index thì quét toàn bộ collection; có index mà
#: `_payload` không ghi thì mọi filter trên nó trả rỗng.
PAYLOAD_INDEXES: tuple[tuple[str, str], ...] = (
    ("chunk_id", "keyword"),
    ("doc_id", "keyword"),
    ("lang", "keyword"),
    ("doc_type", "keyword"),
    ("tenant_id", "keyword"),
    ("published_at", "datetime"),
)
_ID_NAMESPACE = uuid.UUID("6f4f2f7a-6f0c-5d3f-9a1e-0c7d9b2a4e11")


def chunk_point_id(chunk_id: str) -> str:
    """ID xác định cho một chunk — cùng `chunk_id` luôn cho cùng point ID."""
    return str(uuid.uuid5(_ID_NAMESPACE, chunk_id))


def schema_problems(
    *,
    dense_sizes: Mapping[str, int],
    sparse_names: Collection[str],
    want_dimension: int,
    want_sparse: bool,
) -> list[str]:
    """So schema thật của collection với schema mà provider hiện tại cần.

    Hàm thuần, không chạm Qdrant — để test được mọi ca lệch mà không cần server.
    Trả danh sách vấn đề bằng lời; rỗng nghĩa là khớp.

    Ba ca lệch, mỗi ca hỏng theo một kiểu khác nhau:

    * **Thiếu named vector `dense`** — collection của phiên bản cũ dùng vector vô
      danh. Mọi truy vấn `using="dense"` sẽ lỗi.
    * **Số chiều khác** — Qdrant từ chối upsert, nhưng chỉ *sau khi* đã nạp model
      và chunk xong. Đây là ca xảy ra mỗi lần đổi model embedding.
    * **Thiếu/thừa named vector `sparse`** — ca nguy hiểm nhất vì nó là ca *mới*:
      provider sinh sparse mà collection không có chỗ chứa.

    Cố ý **không** coi "collection có sparse mà provider không sinh" là lỗi im
    lặng: nó nghĩa là đang eval bằng provider dense-only trên index hybrid, và
    con số sẽ trông bình thường trong khi một nửa index không được dùng.
    """
    problems: list[str] = []

    if DENSE_VECTOR_NAME not in dense_sizes:
        found = sorted(dense_sizes) or ["(vector vô danh)"]
        problems.append(f"thiếu named vector {DENSE_VECTOR_NAME!r}; collection đang có {found}")
    elif dense_sizes[DENSE_VECTOR_NAME] != want_dimension:
        problems.append(
            f"số chiều dense là {dense_sizes[DENSE_VECTOR_NAME]} nhưng provider "
            f"sinh {want_dimension} chiều"
        )

    has_sparse = SPARSE_VECTOR_NAME in sparse_names
    if want_sparse and not has_sparse:
        problems.append(
            f"provider sinh sparse nhưng collection không có named vector {SPARSE_VECTOR_NAME!r}"
        )
    elif has_sparse and not want_sparse:
        problems.append(
            f"collection có named vector {SPARSE_VECTOR_NAME!r} nhưng provider "
            f"hiện tại chỉ sinh dense — nửa index sẽ không được dùng tới"
        )
    return problems


def points_to_chunks(points: Sequence[Any], *, mode: RetrievalMode) -> list[RetrievedChunk]:
    """Dựng `RetrievedChunk` từ point của Qdrant, gán điểm vào đúng nhánh.

    `rank` liên tục từ 1 **sau khi** đã bỏ point lỗi — nếu đánh số trước rồi bỏ
    thì dãy rank có lỗ, và nDCG/MRR đọc rank như vị trí thật nên sẽ tính sai một
    cách âm thầm. `W2-04` phụ thuộc vào tính chất này chặt hơn nữa: RRF lấy
    **chính** thứ hạng này làm đầu vào.

    Hàm module vì `W2-04` đọc kết quả của hai nhánh từ một response batch, không
    qua `retrieve()`/`retrieve_sparse()` — gọi hai method đó sẽ embed truy vấn
    **hai lần** (12,6 ms mỗi lần, đo ở `W2-03`), tức trả gấp đôi tiền forward pass
    cho đúng một kết quả. Đây là phiên bản phía truy vấn của quyết định "một
    forward pass" ở `W2-01`.
    """
    results: list[RetrievedChunk] = []
    for point in points:
        raw_chunk = (point.payload or {}).get("chunk")
        if raw_chunk is None:  # pragma: no cover - point ghi bởi phiên bản cũ
            logger.warning("Point %s thiếu payload `chunk`, bỏ qua", point.id)
            continue
        score = float(point.score)
        results.append(
            RetrievedChunk(
                chunk=Chunk.model_validate(raw_chunk),
                score=score,
                rank=len(results) + 1,
                mode=mode,
                dense_score=score if mode is RetrievalMode.DENSE else None,
                sparse_score=score if mode is RetrievalMode.SPARSE else None,
            )
        )
    return results


class QdrantDenseRetriever(Retriever):
    def __init__(
        self,
        embeddings: EmbeddingProvider,
        *,
        collection: str = "rag_chunks",
        url: str = "http://127.0.0.1:6333",
        api_key: str | None = None,
        client: QdrantClient | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.embeddings = embeddings
        self.collection = collection
        self.name = f"qdrant-dense:{collection}"
        self._client = client
        self._url = url
        self._api_key = api_key
        self._timeout = timeout

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            from qdrant_client import QdrantClient

            self._client = QdrantClient(
                url=self._url, api_key=self._api_key, timeout=int(self._timeout)
            )
        return self._client

    # ------------------------------------------------------------ collection

    @property
    def writes_sparse(self) -> bool:
        """Provider hiện tại có sinh sparse vector hay không.

        Đọc từ `sparse_vocab_size` chứ không phải từ một cờ cấu hình: năng lực
        này thuộc về provider, và một cờ riêng cho phép tồn tại cấu hình "model
        sinh sparse, nhưng đừng ghi" vừa hợp lệ vừa vô nghĩa.
        """
        return self.embeddings.sparse_vocab_size is not None

    def live_schema(self) -> tuple[dict[str, int], frozenset[str]]:
        """Schema thật của collection: `({tên dense: số chiều}, {tên sparse})`.

        Collection dùng vector vô danh trả dict rỗng — đúng, vì `schema_problems`
        cần phân biệt "không có `dense`" với "có `dense` sai chiều".
        """
        params = self.client.get_collection(self.collection).config.params
        vectors = params.vectors
        dense_sizes: dict[str, int] = {}
        if isinstance(vectors, dict):
            dense_sizes = {name: int(cfg.size) for name, cfg in vectors.items()}
        sparse = params.sparse_vectors or {}
        return dense_sizes, frozenset(sparse)

    def verify_schema(self) -> None:
        """Chết ngay nếu schema của collection không khớp provider hiện tại.

        Chi tiết quan trọng: gọi **trước** khi chunk và embed. Không có bước này
        thì lỗi lệch schema xuất hiện ở lần upsert đầu — tức sau khi đã nạp 2,2GB
        trọng số và chunk xong tài liệu đầu tiên — và thông báo của Qdrant không
        nói phải làm gì để sửa.
        """
        dense_sizes, sparse_names = self.live_schema()
        problems = schema_problems(
            dense_sizes=dense_sizes,
            sparse_names=sparse_names,
            want_dimension=self.embeddings.dimension,
            want_sparse=self.writes_sparse,
        )
        if not problems:
            return
        raise RuntimeError(
            f"Collection {self.collection!r} không khớp cấu hình hiện tại:\n"
            + "\n".join(f"  · {p}" for p in problems)
            + "\nQdrant không sửa được schema tại chỗ. Build lại bằng `--recreate`,"
            "\nhoặc đổi `collection` trong config để giữ lại index cũ."
        )

    def ensure_collection(self, *, recreate: bool = False) -> None:
        from qdrant_client import models

        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False
        if exists:
            self.verify_schema()
            return

        # `SparseVectorParams` KHÔNG dùng `modifier=Modifier.IDF`. Trọng số của
        # BGE-M3 là **đã học** — model tự quyết token nào quan trọng. Chồng thêm
        # IDF của Qdrant lên là nhân đôi phép hạ bậc từ phổ biến, và hỏng theo
        # kiểu im lặng: điểm vẫn ra số, chỉ là sai. IDF dành cho nhánh BM25 thô
        # ở `W2-03`, nơi giá trị đầu vào là tần suất chứ không phải trọng số.
        sparse_config = (
            {SPARSE_VECTOR_NAME: models.SparseVectorParams()} if self.writes_sparse else None
        )
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=self.embeddings.dimension,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config=sparse_config,
        )
        # Index payload cho các field lọc được (`W2-06`). Tạo sớm thì rẻ; tạo sau
        # khi đã có vài trăm nghìn point thì Qdrant phải quét lại toàn bộ.
        #
        # `published_at` là `datetime`, không phải `keyword`: nó được ghi dưới
        # dạng chuỗi RFC3339 nên index keyword *cũng* dựng được và mọi truy vấn
        # khớp-chính-xác vẫn chạy — rồi `DatetimeRange` sẽ không dùng được index
        # đó và Qdrant lùi về quét. Hỏng về hiệu năng, không về kết quả, tức đúng
        # loại không ai phát hiện.
        for field, schema in PAYLOAD_INDEXES:
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field,
                field_schema=schema,
            )

    def ensure_payload_indexes(self) -> list[str]:
        """Dựng payload index còn thiếu trên collection **đã tồn tại**.

        Tách khỏi `ensure_collection` vì hai việc khác nhau về thời điểm:
        `ensure_collection` chạy khi build index mới, còn method này là đường
        **migrate** — `W2-06` thêm `published_at` vào `PAYLOAD_INDEXES` và các
        collection dựng trước đó (`rag_bgem3`, 15.814 point) không có nó.

        Không có nó thì `DatetimeRange` vẫn cho **kết quả đúng** — Qdrant lùi về
        quét toàn bộ. Đó là lý do phải gọi tường minh: một filter chậm 100 lần mà
        đúng kết quả sẽ không bao giờ tự lộ ra trong test.
        """
        existing = set(self.client.get_collection(self.collection).payload_schema or {})
        created: list[str] = []
        for field, schema in PAYLOAD_INDEXES:
            if field in existing:
                continue
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=field,
                field_schema=schema,
            )
            created.append(field)
        return created

    def backfill_flat_payload(self, *, batch: int = 512) -> int:
        """Dựng lại field payload phẳng từ object `chunk` lồng bên trong.

        Trả số point đã cập nhật. **Không** embed lại gì: vector không đổi nên
        mọi con số eval đã công bố vẫn đúng — khác hẳn việc build lại index.

        Vì sao cần: `_payload` là chỗ duy nhất biết field nào được làm phẳng, và
        khi nó thêm một field (`W2-06` thêm `published_at`) thì mọi point ghi
        trước đó thiếu field ấy. Point thiếu field **không khớp** `DatetimeRange`,
        nên trước khi backfill thì `published_after=2020` sẽ trả 0 kết quả trên
        toàn bộ corpus — đúng chế độ hỏng im lặng mà `W2-06` tồn tại để chặn, chỉ
        là lần này do dữ liệu cũ chứ không do code.

        So payload hiện có với payload đúng rồi chỉ ghi phần lệch: chạy lại lần
        thứ hai là no-op, nên nó an toàn khi gọi trong script build.
        """
        updated = 0
        offset: Any = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                limit=batch,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                raw_chunk = payload.get("chunk")
                if raw_chunk is None:  # pragma: no cover - point ghi bởi phiên bản cũ
                    continue
                want = self._payload(Chunk.model_validate(raw_chunk))
                missing = {
                    key: value
                    for key, value in want.items()
                    if key != "chunk" and payload.get(key) != value
                }
                if not missing:
                    continue
                self.client.set_payload(
                    collection_name=self.collection,
                    payload=missing,
                    points=[point.id],
                    wait=True,
                )
                updated += 1
            if offset is None:
                break
        return updated

    def count(self) -> int:
        return int(self.client.count(self.collection, exact=True).count)

    def fetch_chunks(
        self,
        chunk_ids: Sequence[str],
        *,
        filters: FilterSpec = None,
    ) -> dict[str, Chunk]:
        """Lấy chunk theo `chunk_id`, bỏ qua id không có trong collection.

        Cố ý trả dict thay vì list: người gọi hầu như luôn cần biết id **nào**
        thiếu, và một list ngắn hơn đầu vào chỉ nói được rằng có thiếu. Việc
        `set(chunk_ids) - result.keys()` là chỗ duy nhất phát hiện được golden set
        đang trỏ tới chunk không còn tồn tại — chuyện xảy ra ngay khi index được
        build lại với cấu hình chunking khác.

        ⚠️ **`filters` có mặt ở đây vì đây là một đường vòng qua filter (`W2-06`).**
        Tầng search (`retrieve`) đã áp filter ở Qdrant từ `W1-07`, nhưng lấy chunk
        **theo id** thì không đi qua đó chút nào. Ở `W4` tầng serving sẽ gọi đúng
        method này để giải citation và mở rộng ngữ cảnh (`parent_chunk_id`) — và
        một `chunk_id` đoán được hoặc lấy từ câu trả lời cũ của tenant khác sẽ
        trả về nội dung đầy đủ, dù mọi truy vấn vector đều đã lọc đúng.

        Hai đường code là **có chủ ý**: `client.retrieve` không nhận filter
        (Qdrant không hỗ trợ), nên có filter thì phải chuyển sang `scroll`. Giữ
        `retrieve` cho đường không filter vì nó là đường nóng của việc phân giải
        nhãn golden set và nó lấy đúng point theo id, không phân trang.
        """
        if not chunk_ids:
            return {}
        wanted = list(dict.fromkeys(chunk_ids))
        query_filter = build_filter(filters)
        if query_filter is None:
            points: Sequence[Any] = self.client.retrieve(
                collection_name=self.collection,
                ids=[chunk_point_id(cid) for cid in wanted],
                with_payload=True,
            )
        else:
            points = self._scroll_filtered("chunk_id", wanted, query_filter)
        found: dict[str, Chunk] = {}
        for point in points:
            raw_chunk = (point.payload or {}).get("chunk")
            if raw_chunk is None:  # pragma: no cover - point ghi bởi phiên bản cũ
                continue
            chunk = Chunk.model_validate(raw_chunk)
            found[chunk.chunk_id] = chunk
        return found

    # ------------------------------------------------------------ ghi

    def _embed_batch(self, texts: Sequence[str]) -> tuple[Any, list[SparseVector] | None]:
        """Embed một lô, lấy sparse cùng lúc nếu provider có.

        **Một** lần gọi provider, không hai. Với BGE-M3 thì sparse chỉ là một
        `Linear(1024 → 1)` đặt lên cùng `last_hidden_state` đã dùng cho dense,
        nên gọi `embed_documents` rồi gọi tiếp một hàm sparse là trả gấp đôi tiền
        forward pass cho đúng một kết quả — trên 15.000 chunk là ~380 giây.
        """
        if self.writes_sparse:
            hybrid = self.embeddings.embed_documents_hybrid(texts)
            if hybrid is not None:
                return hybrid.dense, hybrid.sparse
            # Provider khai có `sparse_vocab_size` mà không cài `embed_documents_hybrid`
            # là bug của provider, không phải trạng thái hợp lệ: schema collection
            # đã tạo chỗ cho sparse, ghi thiếu thì nhánh sparse im lặng trả rỗng.
            raise RuntimeError(
                f"Provider {self.embeddings.name!r} khai sparse_vocab_size="
                f"{self.embeddings.sparse_vocab_size} nhưng embed_documents_hybrid trả None"
            )
        return self.embeddings.embed_documents(texts), None

    def upsert(self, chunks: Sequence[Chunk], *, batch_size: int = 128) -> int:
        """Ghi chunk vào collection. Gọi lại với cùng chunk thì không sinh bản trùng."""
        from qdrant_client import models

        if not chunks:
            return 0

        total = 0
        for start in range(0, len(chunks), batch_size):
            batch = list(chunks[start : start + batch_size])
            dense, sparse = self._embed_batch([c.content for c in batch])
            # Thay cho `zip(strict=True)` của bản dense-only: lệch độ dài nghĩa là
            # provider trả không đủ hàng, và indexing theo offset sẽ gán embedding
            # cho sai chunk thay vì báo lỗi.
            if len(dense) != len(batch) or (sparse is not None and len(sparse) != len(batch)):
                raise RuntimeError(
                    f"Provider trả {len(dense)} dense và "
                    f"{'-' if sparse is None else len(sparse)} sparse cho {len(batch)} chunk"
                )
            points: list[models.PointStruct] = []
            for offset, chunk in enumerate(batch):
                vector: dict[str, Any] = {
                    DENSE_VECTOR_NAME: np.asarray(dense[offset], dtype=np.float32).tolist()
                }
                if sparse is not None:
                    vector[SPARSE_VECTOR_NAME] = models.SparseVector(**sparse[offset].as_qdrant())
                points.append(
                    models.PointStruct(
                        id=chunk_point_id(chunk.chunk_id),
                        vector=vector,
                        payload=self._payload(chunk),
                    )
                )
            self.client.upsert(collection_name=self.collection, points=points, wait=True)
            total += len(points)
        return total

    @staticmethod
    def _payload(chunk: Chunk) -> dict[str, Any]:
        # Lưu nguyên chunk để dựng lại được object đầy đủ lúc truy hồi, cộng vài
        # field phẳng ở cấp trên cùng để Qdrant lọc được mà không cần nested path.
        payload: dict[str, Any] = {"chunk": chunk.model_dump(mode="json")}
        payload["chunk_id"] = chunk.chunk_id
        payload["doc_id"] = chunk.doc_id
        payload["content_hash"] = chunk.content_hash
        if chunk.metadata is not None:
            payload["lang"] = chunk.metadata.lang.value
            payload["doc_type"] = chunk.metadata.doc_type.value
            if chunk.metadata.published_at is not None:
                # RFC3339 là định dạng `DatetimeRange` của Qdrant nhận. Chỉ ghi
                # khi có giá trị: ghi `None` sẽ tạo ra một field tồn-tại-mà-rỗng,
                # và khi đó `DatetimeRange` bỏ qua point đó *im lặng* thay vì để
                # nó rơi vào nhóm "không có ngày" mà người gọi kiểm được.
                payload["published_at"] = chunk.metadata.published_at.isoformat()
        if "tenant_id" in chunk.extra:
            payload["tenant_id"] = chunk.extra["tenant_id"]
        return payload

    def fetch_doc_chunks(
        self,
        doc_ids: Sequence[str],
        *,
        batch: int = 512,
        filters: FilterSpec = None,
    ) -> list[Chunk]:
        """Lấy **mọi** chunk của các tài liệu được nêu, không qua truy vấn vector.

        Cần cho việc ánh xạ nhãn golden set theo span (`pipeline/eval/spans.py`):
        muốn biết chunk nào của index hiện tại chứa một đoạn bằng chứng thì phải
        xem hết chunk của tài liệu đó, chứ không phải top-k của một truy vấn.

        Dùng `scroll` với filter `doc_id` — trường này đã có payload index từ
        `ensure_collection`, nên đây là phép quét theo khoá chứ không phải quét
        toàn bộ collection.

        ⚠️ `filters` (`W2-06`): đường vòng thứ hai qua filter, và rộng hơn
        `fetch_chunks` — một `doc_id` trả về **toàn bộ** chunk của tài liệu đó.
        Điều kiện được **gộp vào cùng một `must`** với `doc_id`, không lọc sau,
        nên Qdrant không bao giờ trả point của tenant khác về tiến trình này.
        """
        if not doc_ids:
            return []
        wanted = list(dict.fromkeys(doc_ids))
        points = self._scroll_filtered("doc_id", wanted, build_filter(filters), batch=batch)
        out: list[Chunk] = []
        for point in points:
            raw_chunk = (point.payload or {}).get("chunk")
            if raw_chunk is None:  # pragma: no cover - point ghi bởi phiên bản cũ
                continue
            out.append(Chunk.model_validate(raw_chunk))
        return out

    def _scroll_filtered(
        self,
        key: str,
        values: Sequence[str],
        extra: Any,
        *,
        batch: int = 512,
    ) -> list[Any]:
        """Quét mọi point có `payload[key]` thuộc `values`, gộp thêm `extra`.

        Gộp bằng cách **nối `must`** chứ không lồng `Filter` vào `Filter`: hai
        cách cho cùng kết quả nhưng một `must` phẳng là thứ Qdrant tối ưu được
        bằng payload index, còn filter lồng thì tuỳ phiên bản.

        Trả point thô để người gọi tự dựng `Chunk` — `fetch_chunks` cần dict theo
        `chunk_id` còn `fetch_doc_chunks` cần list, và nhồi cả hai vào đây sẽ
        thành một hàm có hai chế độ.
        """
        from qdrant_client import models

        conditions: list[models.Condition] = [
            models.FieldCondition(key=key, match=models.MatchAny(any=list(values)))
        ]
        if extra is not None:
            conditions.extend(extra.must or [])
        flt = models.Filter(must=conditions)
        out: list[Any] = []
        offset: Any = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                scroll_filter=flt,
                limit=batch,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            out.extend(points)
            if offset is None:
                break
        return out

    def delete_points(self, point_ids: Sequence[str]) -> int:
        """Xoá đúng các point được liệt kê. Trả về số point đã yêu cầu xoá.

        Cần tách khỏi `delete_by_doc` cho trường hợp một tài liệu bị chunk lại
        thành **ít chunk hơn**: chỉ phần đuôi thừa mới phải đi, phần đầu đã bị
        upsert ghi đè rồi.
        """
        from qdrant_client import models

        if not point_ids:
            return 0
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.PointIdsList(points=list(point_ids)),
            wait=True,
        )
        return len(point_ids)

    def delete_by_doc(self, doc_id: str) -> None:
        from qdrant_client import models

        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))
                    ]
                )
            ),
            wait=True,
        )

    # ------------------------------------------------------------ đọc

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        *,
        filters: FilterSpec = None,
    ) -> list[RetrievedChunk]:
        vector = np.asarray(self.embeddings.embed_query(query), dtype=np.float32).tolist()
        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            using=DENSE_VECTOR_NAME,
            limit=top_k,
            query_filter=self._build_filter(filters),
            with_payload=True,
        )

        return self._to_chunks(response.points, mode=RetrievalMode.DENSE)

    def retrieve_sparse(
        self,
        query: str,
        top_k: int = 10,
        *,
        filters: FilterSpec = None,
    ) -> list[RetrievedChunk]:
        """Truy hồi bằng **chỉ** nhánh sparse — trọng số lexical của BGE-M3.

        Tách khỏi `retrieve()` chứ không gộp thành một hàm "hybrid": `W2-04` (RRF)
        cần hai danh sách xếp hạng độc lập để hợp nhất, và một hàm trả sẵn hybrid
        thì không tách được đóng góp của mỗi nhánh. Đó cũng là lý do
        `RetrievedChunk` giữ `dense_score` và `sparse_score` riêng.

        Sparse dùng **dot product**, không phải cosine: trọng số đã không âm và
        độ dài vector mang thông tin (chunk dài khớp nhiều token hơn thì đáng
        điểm cao hơn). Đây là mặc định của Qdrant cho sparse, không cấu hình được
        — và cũng là lý do `SparseVector` cấm entry bằng 0.
        """
        if not self.writes_sparse:
            raise RuntimeError(
                f"Provider {self.embeddings.name!r} không sinh sparse vector "
                "(sparse_vocab_size is None) — không có gì để truy vấn"
            )
        from qdrant_client import models

        hybrid = self.embeddings.embed_query_hybrid(query)
        if hybrid is None:  # pragma: no cover - bug của provider, xem `_embed_batch`
            raise RuntimeError(f"Provider {self.embeddings.name!r} không trả sparse cho truy vấn")
        response = self.client.query_points(
            collection_name=self.collection,
            query=models.SparseVector(**hybrid[1].as_qdrant()),
            using=SPARSE_VECTOR_NAME,
            limit=top_k,
            query_filter=self._build_filter(filters),
            with_payload=True,
        )
        return self._to_chunks(response.points, mode=RetrievalMode.SPARSE)

    def _to_chunks(self, points: Sequence[Any], *, mode: RetrievalMode) -> list[RetrievedChunk]:
        return points_to_chunks(points, mode=mode)

    def _build_filter(self, filters: FilterSpec) -> Any:
        return build_filter(filters)
