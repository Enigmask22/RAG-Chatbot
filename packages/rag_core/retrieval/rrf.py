"""Reciprocal Rank Fusion — hợp nhất nhiều danh sách xếp hạng. `W2-04`.

Hàm **thuần**, chỉ làm việc với khoá và thứ hạng, không biết gì về chunk hay
Qdrant. Tách như vậy vì đây là chỗ duy nhất trong đường truy hồi mà kết quả
đúng/sai kiểm được bằng số tính tay — và một phép hợp nhất sai thì không hỏng ồn
ào, nó chỉ làm mọi metric tệ đi vài phần trăm.

## Vì sao hợp nhất theo THỨ HẠNG, không theo điểm

Nhánh dense cho cosine ∈ [−1, 1]; nhánh sparse cho dot product của trọng số
không âm nên **không có trần**. Đo thật trên một truy vấn (`W2-02`): dense 0,6682
vs sparse 0,2938 — hai con số này không so được với nhau, và mọi phép chuẩn hoá
(min-max, z-score) đều phải chọn một cửa sổ để chuẩn hoá theo, tức đưa vào một
tham số ẩn phụ thuộc kết quả. Thứ hạng thì không có vấn đề đó.

## `k` làm gì

`1 / (k + rank)`. `k` càng lớn thì chênh lệch giữa các hạng đầu càng **phẳng**:
với `k = 60` thì hạng 1 và hạng 2 chênh 1,6%, còn với `k = 0` thì chênh 50%. Nói
cách khác `k` là "tôi tin thứ hạng trong mỗi danh sách đến mức nào". `k = 60` là
giá trị của bài báo gốc (Cormack et al., 2009) và là mặc định ở đây; nó nằm ngoài
`fingerprint` của index vì nó là tham số của lần **đo**.

⚠️ Qdrant có `Fusion.RRF` server-side nhưng **`k` của nó không cấu hình được** và
nó không trả về thứ hạng của từng nhánh. `W2-08` cần quét `k`, và
`RetrievedChunk.dense_score`/`sparse_score` tồn tại để tách đóng góp — nên tầng
này tự cài. Có test integration đối chiếu với bản của Qdrant để bắt lỗi cài đặt.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["RRF_K", "FusedItem", "reciprocal_rank_fusion"]

#: `k` của bài báo gốc. Xem docstring module về ý nghĩa.
RRF_K = 60


@dataclass(frozen=True, slots=True)
class FusedItem:
    """Một khoá sau khi hợp nhất, kèm thứ hạng của nó trong **từng** danh sách vào.

    `ranks` giữ lại `None` cho danh sách không chứa khoá này, và đó là thông tin
    load-bearing: "chunk này chỉ nhánh sparse tìm ra" khác hẳn "cả hai nhánh đều
    xếp nó hạng 40". Gộp hai thứ đó lại thì `W2-08` không tách được đóng góp của
    mỗi nhánh nữa — cùng lý lẽ với `SparseVector` rỗng ≠ `None` ở `TD-11`.
    """

    key: str
    score: float
    rank: int
    ranks: tuple[int | None, ...]

    @property
    def sources(self) -> int:
        """Số danh sách đã tìm ra khoá này."""
        return sum(1 for r in self.ranks if r is not None)

    @property
    def best_rank(self) -> int | None:
        found = [r for r in self.ranks if r is not None]
        return min(found) if found else None


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = RRF_K,
    weights: Sequence[float] | None = None,
    limit: int | None = None,
) -> list[FusedItem]:
    """Hợp nhất `rankings` (mỗi phần tử là một danh sách khoá đã xếp hạng).

    Thứ tự trong mỗi danh sách vào **là** thứ hạng: phần tử đầu là hạng 1. Trả về
    danh sách đã xếp giảm dần theo điểm RRF, `rank` liên tục từ 1.

    Danh sách **rỗng** là hợp lệ và không đóng góp gì — đó là trạng thái thật của
    nhánh sparse khi truy vấn không trùng token nào (`W2-03`). Cố ý không coi nó
    là lỗi: nếu vậy thì mọi truy vấn kiểu đó sẽ chết thay vì rơi về nhánh còn lại,
    tức mất đúng cái lợi mà hợp nhất mang lại.

    `weights` cho phép nhánh mạnh hơn nặng hơn (weighted RRF). Mặc định `None` =
    đều nhau. Có thật một cơ sở để cân lệch: `W2-03` đo được dense `hit_rate@10`
    0,6268 vs sparse 0,5120 — nhưng cân bao nhiêu là câu hỏi thực nghiệm của
    `W2-08`, không phải câu hỏi thiết kế.

    **Tie-break** (điểm bằng nhau xảy ra *thường xuyên* — một khoá ở hạng 3 của
    danh sách A và một khoá khác ở hạng 3 của danh sách B có cùng điểm), theo thứ
    tự: (1) điểm cao hơn, (2) `best_rank` nhỏ hơn, (3) danh sách nào tìm ra nó
    trước — tức thứ tự `rankings` mang nghĩa, và người gọi đặt nhánh mạnh hơn lên
    đầu, (4) khoá theo thứ tự chữ. Quy tắc (4) hầu như không bao giờ tới nhưng nó
    là thứ biến "gần như xác định" thành "xác định".
    """
    if k < 0:
        raise ValueError(f"k phải không âm, nhận {k}")
    if limit is not None and limit < 0:
        raise ValueError(f"limit phải không âm, nhận {limit}")

    if weights is None:
        resolved = [1.0] * len(rankings)
    else:
        if len(weights) != len(rankings):
            raise ValueError(
                f"weights có {len(weights)} phần tử nhưng có {len(rankings)} danh sách"
            )
        if any(w < 0 for w in weights):
            raise ValueError(f"weights phải không âm, nhận {list(weights)}")
        resolved = [float(w) for w in weights]

    n_lists = len(rankings)
    scores: dict[str, float] = {}
    ranks: dict[str, list[int | None]] = {}

    for list_index, (ranking, weight) in enumerate(zip(rankings, resolved, strict=True)):
        seen: set[str] = set()
        for position, key in enumerate(ranking, start=1):
            if key in seen:
                # Trùng khoá trong **cùng** một danh sách là bug ở tầng trên (hai
                # point mang cùng `chunk_id`). Im lặng bỏ bớt sẽ làm điểm RRF
                # trông hợp lý trong khi index đang có bản trùng.
                raise ValueError(
                    f"khoá {key!r} xuất hiện hai lần trong danh sách #{list_index} "
                    f"(hạng {ranks[key][list_index]} và {position})"
                )
            seen.add(key)
            if key not in scores:
                scores[key] = 0.0
                ranks[key] = [None] * n_lists
            scores[key] += weight / (k + position)
            ranks[key][list_index] = position

    def sort_key(key: str) -> tuple[float, int, int, str]:
        found = [(i, r) for i, r in enumerate(ranks[key]) if r is not None]
        best_rank = min(r for _, r in found)
        first_list = min(i for i, _ in found)
        return (-scores[key], best_rank, first_list, key)

    ordered = sorted(scores, key=sort_key)
    if limit is not None:
        ordered = ordered[:limit]
    return [
        FusedItem(key=key, score=scores[key], rank=i, ranks=tuple(ranks[key]))
        for i, key in enumerate(ordered, start=1)
    ]
