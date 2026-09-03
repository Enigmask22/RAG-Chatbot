"""Chọn nhánh truy hồi từ một chuỗi tên — `W2-03`.

Tồn tại vì từ `W2-03` trở đi, "index nào" và "truy hồi thế nào" là hai câu hỏi
tách nhau: cùng một collection `rag_bgem3` phục vụ được nhánh dense, nhánh
sparse, và (từ `W2-04`) nhánh hybrid. Trước đó chúng dính vào nhau nên
`IndexConfig` một mình là đủ để dựng retriever.

Cố ý **không** đưa nhánh truy hồi vào `IndexConfig`: nó không quyết định vector
nào được ghi, nên nó không thuộc `fingerprint` — mà một trường nằm trong
`IndexConfig` nhưng ngoài `fingerprint` là thứ phải giải thích mỗi lần đọc lại
(đã có hai trường như vậy là `device` và `batch_size`, đủ rồi). Nhánh truy hồi là
tham số của **lần đo**, nên nó là cờ dòng lệnh và sau này là một trường của ma
trận thí nghiệm ở `W2-07`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..schemas import RetrievalMode
from .base import Retriever

if TYPE_CHECKING:
    from .qdrant_store import QdrantDenseRetriever

__all__ = [
    "DEFAULT_RERANK_BASE",
    "HYBRID_OPTIONS",
    "RERANK_OPTIONS",
    "SUPPORTED_MODES",
    "build_branch",
    "check_branch_options",
]

#: Nhánh dựng được ở thời điểm hiện tại — liệt kê tường minh để một tên hợp lệ
#: nhưng chưa cài báo "chưa cài" thay vì báo "tên không hợp lệ", hai chuyện rất
#: khác nhau khi đang gỡ lỗi. `RERANKED` vào đây ở `W2-05`; danh sách này giờ
#: bằng `RetrievalMode`, nên `NotImplementedError` ở cuối `build_branch` là
#: nhánh chết cho tới khi có mode mới — và nó phải ở lại đúng vì lý do đó.
SUPPORTED_MODES: tuple[RetrievalMode, ...] = (
    RetrievalMode.DENSE,
    RetrievalMode.SPARSE,
    RetrievalMode.HYBRID,
    RetrievalMode.RERANKED,
)

#: Tham số chỉ nhánh `reranked` nhận. Mọi thứ còn lại trong `options` được
#: chuyển xuống nhánh nền, nên `--rrf-k 1 --rerank-base dense` vẫn báo lỗi.
RERANK_OPTIONS = frozenset(
    {
        "base",
        "reranker_model",
        "rerank_candidates",
        "rerank_top_n",
        "rerank_max_length",
        "rerank_batch_size",
        "rerank_device",
        "rerank_activation",
        "rerank_dtype",
    }
)

#: Tham số chỉ nhánh `hybrid` nhận — đúng chữ ký `QdrantHybridRetriever.__init__`.
#: Tồn tại vì `check_branch_options` phải trả lời "tham số này hợp lệ không" mà
#: **không** được dựng retriever (xem docstring của nó). Có test đối chiếu tập này
#: với chữ ký thật, nên thêm tham số cho hybrid mà quên ở đây sẽ đỏ.
HYBRID_OPTIONS = frozenset({"k", "candidate_k", "weights"})

#: Nhánh nền mặc định của `reranked`. `W2-04` đo được hybrid là bộ sinh ứng viên
#: tốt nhất (`recall@20` 0,6754 vs dense 0,6324) — nhưng ⚠️ **`k` của nó vẫn lấy
#: mặc định `RRF_K = 60` của bài báo**, mà `W2-04` đo được đó là giá trị tệ nhất.
#: Cố ý không sửa mặc định của thư viện ở đây: một giá trị thắng theo số đo thuộc
#: về **config của thí nghiệm** (`W2-07`), không thuộc về hằng số thư viện, và
#: `retriever.name` mang `rrf60` nên chuyện đó không im lặng. Muốn cấu hình
#: thắng thì truyền `--rrf-k 1 --candidate-k 20 --rrf-weights 1 0.25`.
#:
#: ⭐ `--rrf-weights 1 0.25` là kết quả của `TD-37` (2026-09-03): trọng số **đều**
#: cho hạng 1 của nhánh sparse nửa số phiếu kể cả khi nhánh ấy mù với loại truy
#: vấn đó (`cross_lingual`: sparse `hit_rate@50` = 0,0233). Hạ xuống 0,25 được
#: `cross_lingual` nDCG@10 **+0,0622** và chỉ mất `factoid` −0,0093 — cùng mức
#: lợi như định tuyến sang nền dense nhưng **1/3,6** cái giá.
DEFAULT_RERANK_BASE = RetrievalMode.HYBRID


def check_branch_options(mode: RetrievalMode | str, options: dict[str, Any]) -> RetrievalMode:
    """Kiểm tên nhánh + tham số **mà không dựng gì**. Trả về mode đã phân giải.

    Tồn tại vì `W2-07`. `build_branch` là chỗ duy nhất biết luật "nhánh nào nhận
    tham số nào", nhưng nó chỉ nói được câu trả lời bằng cách **dựng retriever** —
    và với `reranked` việc dựng nạp một cross-encoder 2,2 GB. Một grid 12 ô muốn
    kiểm cả 12 ô *trước khi* chạy ô đầu (xem `pipeline/experiments/runner.py`
    §preflight) thì không thể trả giá đó 12 lần.

    Cách sai là chép luật sang chỗ khác. Luật ở đây gồm cả phần đệ quy của
    `reranked`, nên bản chép sẽ lệch dần và preflight sẽ **cho qua** những ô mà
    `build_branch` từ chối — tức grid chạy 40 phút rồi chết ở ô cuối, đúng cái
    hạng mục này tồn tại để chặn. Nên `build_branch` **gọi chính hàm này**, và có
    test khẳng định hai bên nổ ở cùng những đầu vào.
    """
    try:
        resolved = RetrievalMode(mode)
    except ValueError:
        raise ValueError(
            f"Nhánh truy hồi không hợp lệ: {mode!r}. Hợp lệ: {[m.value for m in RetrievalMode]}"
        ) from None

    supplied = {name: value for name, value in options.items() if value is not None}

    if resolved is RetrievalMode.RERANKED:
        base_mode = supplied.get("base", DEFAULT_RERANK_BASE)
        if str(base_mode) == RetrievalMode.RERANKED.value:
            raise ValueError("`--rerank-base reranked` không có nghĩa: xếp lại hai lần cùng model")
        # Đệ quy đúng như `_build_reranked`: tham số ngoài `RERANK_OPTIONS` đi
        # xuống nhánh nền và bị từ chối ở đó.
        base_options = {k: v for k, v in supplied.items() if k not in RERANK_OPTIONS}
        check_branch_options(base_mode, base_options)
        return resolved

    unknown = RERANK_OPTIONS & supplied.keys()
    if unknown:
        raise ValueError(
            f"Nhánh {resolved.value!r} không nhận tham số {sorted(unknown)} — "
            f"chúng chỉ có nghĩa với nhánh 'reranked'."
        )

    if resolved is RetrievalMode.HYBRID:
        # Trước `W2-07`, `build_branch(store, "hybrid", candidat_k=100)` đi thẳng
        # vào constructor và nhận một `TypeError` trần của Python. Đúng là nổ,
        # nhưng nó không liệt kê tham số hợp lệ — cùng loại thiếu sót với khoá
        # filter gõ sai ở `W2-06`.
        stray = supplied.keys() - HYBRID_OPTIONS
        if stray:
            raise ValueError(
                f"Nhánh 'hybrid' không nhận tham số {sorted(stray)}. "
                f"Hợp lệ: {sorted(HYBRID_OPTIONS)}."
            )
        return resolved

    if supplied:
        raise ValueError(
            f"Nhánh {resolved.value!r} không nhận tham số {sorted(supplied)} — "
            f"chúng chỉ có nghĩa với nhánh 'hybrid'."
        )
    if resolved in SUPPORTED_MODES:
        return resolved

    raise NotImplementedError(  # pragma: no cover - nhánh chết, xem SUPPORTED_MODES
        f"Nhánh {resolved.value!r} là tên hợp lệ nhưng chưa cài. "
        f"Hiện có: {[m.value for m in SUPPORTED_MODES]}."
    )


def build_branch(
    store: QdrantDenseRetriever,
    mode: RetrievalMode | str,
    **options: Any,
) -> Retriever:
    """Bọc `store` thành retriever của nhánh được nêu.

    `RetrievalMode.DENSE` trả về chính `store` — nó *là* retriever dense, bọc
    thêm một lớp chỉ để đối xứng sẽ làm mọi log và mọi `retriever.name` đổi nghĩa
    so với các lần chạy W1/W2-01/W2-02, tức không so được số cũ nữa.

    `options` chỉ dành cho nhánh có tham số (`hybrid`: `k`, `candidate_k`,
    `weights`; `reranked`: xem `RERANK_OPTIONS`). Truyền cho nhánh không nhận là
    **lỗi**, không phải bị bỏ qua: một lần chạy ablation gõ `--rrf-k 10
    --retrieval-mode dense` mà im lặng chạy tiếp sẽ vào bảng `W2-08` như một dòng
    hợp lệ trong khi nó không đo cái nó nói.

    `reranked` **gọi lại chính hàm này** cho nhánh nền, nên phép kiểm trên áp
    dụng nguyên vẹn ở tầng dưới: `--rerank-base dense --rrf-k 1` vẫn nổ.

    Toàn bộ phần *kiểm* nằm ở `check_branch_options`, để `W2-07` kiểm được một
    grid mà không nạp model — xem docstring của nó.
    """
    resolved = check_branch_options(mode, options)
    supplied = {name: value for name, value in options.items() if value is not None}

    if resolved is RetrievalMode.RERANKED:
        return _build_reranked(store, supplied)
    if resolved is RetrievalMode.HYBRID:
        from .hybrid import QdrantHybridRetriever

        return QdrantHybridRetriever(store, **supplied)
    if resolved is RetrievalMode.DENSE:
        return store
    if resolved is RetrievalMode.SPARSE:
        from .sparse import QdrantSparseRetriever

        return QdrantSparseRetriever(store)

    raise NotImplementedError(  # pragma: no cover - nhánh chết, xem SUPPORTED_MODES
        f"Nhánh {resolved.value!r} là tên hợp lệ nhưng chưa cài. "
        f"Hiện có: {[m.value for m in SUPPORTED_MODES]}."
    )


def _build_reranked(store: QdrantDenseRetriever, supplied: dict[str, Any]) -> Retriever:
    """Dựng nhánh `reranked`: một nhánh nền + một cross-encoder bọc ngoài.

    Tách khỏi `build_branch` vì nó là nhánh duy nhất **đệ quy** — mọi tham số
    không thuộc `RERANK_OPTIONS` được chuyển xuống nhánh nền y nguyên, kể cả
    việc bị từ chối ở đó.
    """
    from ..reranking import CrossEncoderReranker
    from .reranked import RerankedRetriever

    base_mode = supplied.get("base", DEFAULT_RERANK_BASE)
    # So bằng **chuỗi**, không qua `RetrievalMode(...)`: một `base` rác phải đi
    # tiếp xuống `build_branch` để nhận đúng câu "nhánh truy hồi không hợp lệ",
    # chứ không nhận một `ValueError` trần của enum ở đây.
    if str(base_mode) == RetrievalMode.RERANKED.value:
        # Rerank hai lần bằng cùng một model là chạy đúng model đó hai lượt trên
        # cùng dữ liệu — không đổi thứ hạng, chỉ đôi chi phí. Chặn ở đây để nó
        # không lọt vào bảng `W2-08` như một dòng "cấu hình mới".
        raise ValueError("`--rerank-base reranked` không có nghĩa: xếp lại hai lần cùng model")

    base_options = {name: value for name, value in supplied.items() if name not in RERANK_OPTIONS}
    base = build_branch(store, base_mode, **base_options)

    reranker_kwargs: dict[str, Any] = {}
    if "rerank_device" in supplied:
        reranker_kwargs["device"] = supplied["rerank_device"]
    if "rerank_batch_size" in supplied:
        reranker_kwargs["batch_size"] = supplied["rerank_batch_size"]
    if "rerank_max_length" in supplied:
        reranker_kwargs["max_length"] = supplied["rerank_max_length"]
    if "rerank_activation" in supplied:
        reranker_kwargs["activation"] = supplied["rerank_activation"]
    if "rerank_dtype" in supplied:
        reranker_kwargs["dtype"] = supplied["rerank_dtype"]
    model_name = supplied.get("reranker_model")
    reranker = (
        CrossEncoderReranker(model_name, **reranker_kwargs)
        if model_name is not None
        else CrossEncoderReranker(**reranker_kwargs)
    )

    return RerankedRetriever(
        base,
        reranker,
        candidates=supplied.get("rerank_candidates"),
        top_n=supplied.get("rerank_top_n"),
    )
