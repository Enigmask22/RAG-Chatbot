"""Nhánh reranked — xếp lại pool ứng viên bằng cross-encoder. `W2-05`.

Đây là chỗ kết luận kiến trúc của `W2-04` được cài thành code: nhánh nền lo
**vùng phủ**, tầng này lo **thứ hạng**. Nên nó bọc một `Retriever` bất kỳ, không
bọc `QdrantDenseRetriever` — `--rerank-base` chọn được dense, sparse hay hybrid,
và đó là một chiều của bảng ablation `W2-08`.

Ba quyết định:

**`candidates` sâu hơn `top_k`, và nó là trần cứng.** Reranker chỉ xếp lại những
gì nhánh nền đưa cho; chunk đúng không nằm trong pool thì không có phép xếp nào
cứu được. Nên `hit_rate@1` sau rerank bị chặn trên bởi `hit_rate@candidates` của
nhánh nền — con số đó phải đo **trước** khi kết luận reranker tốt hay tệ.

**`top_n` là trần của lượt trả về, KHÔNG phải của phép đo.** `top_n=6` (ngân sách
context lúc phục vụ) mà đi chấm `recall@20` thì `recall@20` bị chặn ở 6 chunk và
mọi metric @10/@20 mất nghĩa. Mặc định `None` = không chặn thêm, và lúc chặn thì
`min(top_k, top_n)`. Có test canh cái bẫy đó.

**Điểm bằng nhau thì giữ thứ tự của nhánh nền.** `sorted` của Python là ổn định
và khoá sắp xếp mang cả chỉ số gốc, nên khi cross-encoder không phân biệt được
hai ứng viên thì tiên nghiệm của bộ sinh ứng viên được giữ lại. Ties là chuyện
thật, không phải giả thiết: `W2-04` đã gặp đúng vấn đề này với điểm RRF.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ..schemas import RetrievalMode, RetrievedChunk
from .base import Retriever
from .filters import FilterSpec

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..reranking import Reranker
    from ..schemas import Chunk

__all__ = ["DEFAULT_RERANK_CANDIDATES", "RerankedRetriever"]

#: Số ứng viên lấy từ nhánh nền để đưa cho cross-encoder. Đây là con số của DoD
#: `W2-05` ("rerank 50 → 6 trong < 400ms") và cũng là trần vùng phủ của nhánh.
DEFAULT_RERANK_CANDIDATES = 50


class RerankedRetriever(Retriever):
    """Lấy pool từ `base` rồi xếp lại bằng `reranker`."""

    def __init__(
        self,
        base: Retriever,
        reranker: Reranker,
        *,
        candidates: int | None = None,
        top_n: int | None = None,
    ) -> None:
        if candidates is not None and candidates < 1:
            raise ValueError(f"candidates phải ≥ 1, nhận {candidates}")
        if top_n is not None and top_n < 1:
            raise ValueError(f"top_n phải ≥ 1, nhận {top_n}")
        self.base = base
        self.reranker = reranker
        self.candidates = candidates
        self.top_n = top_n
        # Nhãn mang cả nhánh nền: "reranked" một mình không nói được nó xếp lại
        # cái gì, mà `W2-08` sẽ có ít nhất ba dòng reranked khác nhau ở đúng chỗ đó.
        suffix = f"n{candidates or DEFAULT_RERANK_CANDIDATES}"
        if top_n is not None:
            suffix += f"-top{top_n}"
        self.name = f"reranked[{base.name}]:{reranker.name}:{suffix}"

    def _depth(self, top_k: int) -> int:
        """Số ứng viên lấy từ nhánh nền. Không bao giờ nông hơn `top_k`.

        ⚠️ Với nhánh nền hybrid, `depth` đi vào `top_k` của nó và do đó vào
        `_depth` của nó — tức `candidates=50` **làm sâu luôn pool hợp nhất** dù
        `candidate_k=20`. `W2-04` đo được `candidate_k` không có tác dụng ở `k=1`
        nên trên cấu hình thắng thì điều đó vô hại, nhưng nó không vô hại ở `k`
        lớn và phải nói ra.
        """
        return max(top_k, self.candidates or DEFAULT_RERANK_CANDIDATES)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        *,
        filters: FilterSpec = None,
    ) -> list[RetrievedChunk]:
        pool = self.base.retrieve(query, self._depth(top_k), filters=filters)
        if not pool:
            return []
        scores = self.reranker.score(query, [hit.chunk.content for hit in pool])
        if len(scores) != len(pool):
            # Hợp đồng của `Reranker.score` là "đúng `len(texts)` điểm, theo thứ
            # tự đầu vào". Vi phạm ở đây nghĩa là điểm bị gán lệch cho chunk khác,
            # và kết quả vẫn trông hợp lệ — nên phải chết chứ không được zip ngắn.
            raise RuntimeError(
                f"Reranker {self.reranker.name!r} trả {len(scores)} điểm cho "
                f"{len(pool)} ứng viên — điểm sẽ bị gán lệch chunk."
            )
        bad = [index for index, value in enumerate(scores) if not math.isfinite(value)]
        if bad:
            # NaN so sánh với mọi thứ đều False, nên `sorted` trả về một thứ tự
            # tuỳ ý mà không có gì báo — chunk đúng có thể rơi xuống cuối và
            # metric tụt như thể model kém. Đây là chế độ hỏng thật của fp16:
            # một overflow trung gian trong model là đủ.
            raise RuntimeError(
                f"Reranker {self.reranker.name!r} trả điểm không hữu hạn cho "
                f"{len(bad)}/{len(scores)} ứng viên (vị trí đầu {bad[0]}) — "
                f"thứ hạng sẽ tuỳ ý. Nếu đang dùng fp16, thử `dtype='float32'`."
            )
        limit = min(top_k, self.top_n) if self.top_n is not None else top_k
        order = sorted(range(len(pool)), key=lambda index: (-scores[index], index))
        return [
            RetrievedChunk(
                chunk=pool[index].chunk,
                score=scores[index],
                rank=rank,
                mode=RetrievalMode.RERANKED,
                # Điểm của nhánh nền được giữ để `W2-08` tách được đóng góp:
                # "reranker kéo lên một chunk mà nhánh nào tìm ra?".
                dense_score=pool[index].dense_score,
                sparse_score=pool[index].sparse_score,
                rerank_score=scores[index],
            )
            for rank, index in enumerate(order[:limit], start=1)
        ]

    def fetch_doc_chunks(self, doc_ids: Sequence[str], *, batch: int = 512) -> list[Chunk]:
        """Chuyển tiếp — bắt buộc, xem `sparse.py` về hố im lặng của `getattr`."""
        fetch = getattr(self.base, "fetch_doc_chunks", None)
        if fetch is None:  # pragma: no cover - chỉ xảy ra với base tự viết
            raise AttributeError(
                f"Nhánh nền {self.base.name!r} không có `fetch_doc_chunks`, nên "
                f"nhãn theo span (`TD-12`) không quét được. Xem `sparse.py`."
            )
        return list(fetch(doc_ids, batch=batch))

    def verify_schema(self) -> None:
        verify = getattr(self.base, "verify_schema", None)
        if verify is not None:
            verify()
