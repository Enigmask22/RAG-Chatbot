"""Bọc chuỗi truy hồi để mỗi **lớp** thành một span — `W5-06`.

## ⭐⭐ Cây span chỉ mịn được tới đúng chỗ mã có mối nối

DoD của hạng mục này đòi *"rewrite → retrieve kèm score → rerank → prompt →
completion"*. Nhìn từ `ChatService.prepare` thì truy hồi là **một** lời gọi:

```python
await asyncio.to_thread(snapshot.retriever.retrieve, question, top_k, ...)
```

Một lời gọi, một thời lượng. Muốn có hai span `retrieve` và `rerank` từ đó thì
chỉ có hai lối: bổ đôi con số duy nhất ấy theo một tỉ lệ đoán được, hoặc đi tìm
mối nối thật. Lối thứ nhất **sinh ra dữ liệu** — nó cho ra hai con số trông như
hai phép đo, và không ai đọc trace biết được rằng chúng là một phép đo bị chia.

Mối nối thật có sẵn: `RerankedRetriever` giữ `base` và `reranker` là hai thuộc
tính công khai, và `retrieve()` của nó đúng là `base.retrieve()` rồi
`reranker.score()`. Nên ở đây mỗi lớp được bọc riêng, và mỗi span đo đúng
`perf_counter` quanh đúng lời gọi của lớp ấy. Không có phép chia nào.

Hệ quả phải nói ra: **không có span `embed`** cho đường truy hồi.
`QdrantHybridRetriever.retrieve` gọi `embeddings.embed_query_hybrid` ở *bên
trong* thân hàm, không qua một thuộc tính có thể bọc, nên tách nó ra đòi sửa
`rag_core` — và `rag_core` cố ý không biết gì về quan sát. Thời gian embed nằm
trong span `retrieve.hybrid`; trace nói đúng điều đó chứ không giả vờ có một
con số nó không có. (Đường **cache** thì có span `embed.query` thật, vì ở đó
`prepare()` tự gọi embedder.)

## ⭐ Bọc bằng bản sao nông, không sửa runtime đang phục vụ

`ActiveBundle` là ảnh chụp bất biến và có thể đang được nhiều request cầm
(`W4-02`). Gán `retriever.base = wrapper` là sửa đúng vật thể ấy, và làm hai
lần — reload bundle chẳng hạn — thì được một cái bọc trong một cái bọc, tức mỗi
lượt sinh thêm một tầng span mãi mãi.

`copy.copy` cho một vỏ mới trỏ vào cùng model, cùng client Qdrant, cùng trọng
số trên GPU. Không tốn bộ nhớ, và bản gốc không hề bị chạm.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Sequence
from typing import Any

from rag_core.retrieval.base import Retriever
from rag_core.retrieval.filters import FilterSpec
from rag_core.schemas import RetrievedChunk
from serving.core.tracing import current_trace, hits_summary

__all__ = ["TracedReranker", "TracedRetriever", "instrument_retriever"]

logger = logging.getLogger(__name__)


def _delegate(proxy: Any, item: str) -> Any:
    """`getattr(proxy._inner, item)`, nhưng chịu được lúc `_inner` chưa tồn tại.

    ## ⭐⭐ Một proxy uỷ quyền ngây thơ tự đệ quy tới chết khi bị sao chép

    Bản đầu viết thẳng `return getattr(self._inner, item)`. Nó chạy đúng —
    cho tới khi có ai đó `copy.copy` một lớp bọc. `copy` dựng thực thể mới
    **không qua `__init__`**, rồi tra `__setstate__` trên nó; `_inner` chưa có
    trong `__dict__`, nên `__getattr__("_inner")` gọi lại `__getattr__("_inner")`
    — 961 tầng rồi `RecursionError`.

    Chỗ gọi `copy.copy` là chính `instrument_retriever` bên dưới, nên lỗi này
    xuất hiện đúng khi bọc một chuỗi **đã bọc** — tức đúng kịch bản mà lối
    "sao chép thay vì sửa tại chỗ" được chọn để phục vụ. Cùng họ với mọi lỗi
    `__getattr__`: nó không hỏng ở đường thường, nó hỏng ở đường mà một thư
    viện chuẩn đi qua vật thể của bạn.

    `object.__getattribute__` cắt vòng lặp vì nó **không** rơi vào
    `__getattr__`. Bài test ghim: `test_copying_a_wrapper_does_not_recurse`.
    """
    try:
        inner = object.__getattribute__(proxy, "_inner")
    except AttributeError:
        raise AttributeError(item) from None
    return getattr(inner, item)


class TracedRetriever(Retriever):
    """Uỷ quyền mọi thứ cho `inner`, chỉ chen một span quanh `retrieve()`.

    `__getattr__` uỷ quyền là bắt buộc chứ không phải tiện tay: `embedder_of()`
    của `W4-10` đào `retriever.base.store.embeddings` bằng duck-typing, và một
    lớp bọc không uỷ quyền sẽ làm semantic cache **tắt lặng lẽ** — cache miss
    100%, không log, không lỗi, chỉ có hoá đơn cao hơn.
    """

    def __init__(self, inner: Retriever, span_name: str) -> None:
        self._inner = inner
        self._span_name = span_name
        self.name = inner.name

    def __getattr__(self, item: str) -> Any:
        return _delegate(self, item)

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        *,
        filters: FilterSpec = None,
    ) -> list[RetrievedChunk]:
        trace = current_trace()
        if trace is None:
            return self._inner.retrieve(query, top_k, filters=filters)
        with trace.span(
            self._span_name,
            input={"query": query, "top_k": top_k},
            retriever=self._inner.name,
        ) as span:
            hits = self._inner.retrieve(query, top_k, filters=filters)
            span.end(
                output={"hits": hits_summary(hits)},
                n_hits=len(hits),
                # ⚠️ `top_k` xin và `n_hits` nhận là hai con số khác nhau, và
                # chênh lệch là thứ đáng nhìn nhất trong span này: nó nghĩa là
                # filter hoặc collection không có đủ tài liệu. Ghi cả hai chứ
                # không chỉ ghi cái nhận được.
                truncated=len(hits) < top_k,
            )
            return hits


class TracedReranker:
    """Span quanh `score()` của cross-encoder — bước duy nhất *chỉ* là rerank.

    Không kế thừa `Reranker`: giao thức của nó là `name` + `score`, và
    `__getattr__` lo phần còn lại. Import `rag_core.reranking` ở đây sẽ kéo
    torch vào tiến trình serving lúc import, thứ `pyproject` cố ý tách ra.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.name = getattr(inner, "name", "reranker")

    def __getattr__(self, item: str) -> Any:
        return _delegate(self, item)

    def score(self, query: str, texts: Sequence[str]) -> Sequence[float]:
        trace = current_trace()
        if trace is None:
            result: Sequence[float] = self._inner.score(query, texts)
            return result
        with trace.span(
            "rerank",
            input={"query": query, "n_candidates": len(texts)},
            reranker=self.name,
        ) as span:
            scores: Sequence[float] = self._inner.score(query, texts)
            top = sorted(range(len(scores)), key=lambda i: -scores[i])[:10]
            span.end(
                output={
                    "top": [{"candidate": i, "score": round(float(scores[i]), 6)} for i in top]
                },
                n_candidates=len(texts),
                n_scores=len(scores),
            )
            return scores


def instrument_retriever(retriever: Retriever) -> Retriever:
    """Trả về một chuỗi truy hồi tương đương, mỗi lớp mang một span.

    Nhận diện lớp bằng **thuộc tính**, không bằng `isinstance`: `W2-08` có ít
    nhất ba cách dựng một nhánh, và một `isinstance` sẽ lặng lẽ không bọc gì
    với nhánh thứ tư. Không nhận ra thì bọc một span `retrieve` phẳng — mất độ
    mịn, không mất trace.
    """
    try:
        base = getattr(retriever, "base", None)
        reranker = getattr(retriever, "reranker", None)
        if base is not None and reranker is not None:
            shell = copy.copy(retriever)
            shell.base = instrument_retriever(base)  # type: ignore[attr-defined]
            shell.reranker = TracedReranker(reranker)  # type: ignore[attr-defined]
            return TracedRetriever(shell, "retrieval")
        span_name = "retrieve.hybrid" if getattr(retriever, "k", None) is not None else "retrieve"
        return TracedRetriever(retriever, span_name)
    except Exception:
        # Một retriever không bọc được vẫn là một retriever chạy được. Đây là
        # đúng chỗ mà "quan sát là việc phụ" phải đứng vững nhất: hạng mục này
        # không được phép làm hỏng đường phục vụ ở lúc khởi động.
        logger.warning(
            "tracing: không bọc được chuỗi truy hồi — trace sẽ thiếu span", exc_info=True
        )
        return retriever
