"""Trần chi phí cho một job LLM dài, kiểm **trước** mỗi lời gọi.

Ở đây vì hai hạng mục cần đúng thứ này chứ không phải một: `W3-04` (sinh ngữ
cảnh cho ~15.800 chunk) và `W5-03` (judge chạy trên mọi câu của mọi lần eval).
Cả hai đều là vòng lặp nghìn lời gọi không ai ngồi nhìn.

**Kiểm trước, không phải kiểm sau.** Cộng dồn rồi mới so là để job vượt trần
đúng một lời gọi — vô hại ở đây, nhưng cùng đoạn code sẽ được dùng cho lời gọi
đắt hơn. Vì không biết trước giá của lời gọi kế tiếp, `reserve()` nhận một **ước
lượng** và `charge()` ghi lại số **thật**; chênh lệch tự triệt tiêu qua vòng lặp.
"""

from __future__ import annotations

import logging
import threading

__all__ = ["BudgetExceeded", "CostBudget"]

logger = logging.getLogger(__name__)


class BudgetExceeded(RuntimeError):
    """Job chạm trần chi phí. Không phải lỗi tạm thời — thử lại vẫn chạm."""


class CostBudget:
    """Đếm USD đã tiêu, chặn khi vượt `cap_usd`.

    An toàn luồng vì `NEW-06` đã chốt rằng job LLM dài phải chạy song song (163
    lời gọi tuần tự của `W1-10` mất hơn một giờ; 6 luồng đưa xuống 640 giây), và
    một bộ đếm chi phí không khoá thì trần chỉ đúng khi có một luồng.

    `cap_usd <= 0` nghĩa là **không giới hạn** — phải khai tường minh, vì mặc
    định im lặng không có trần là đúng cái mà lớp này tồn tại để ngăn.
    """

    def __init__(self, cap_usd: float, *, name: str = "job") -> None:
        self.cap_usd = cap_usd
        self.name = name
        self._spent = 0.0
        self._calls = 0
        self._lock = threading.Lock()

    @property
    def unlimited(self) -> bool:
        return self.cap_usd <= 0

    @property
    def spent_usd(self) -> float:
        with self._lock:
            return self._spent

    @property
    def calls(self) -> int:
        with self._lock:
            return self._calls

    @property
    def remaining_usd(self) -> float:
        """Còn lại bao nhiêu; `inf` khi không đặt trần."""
        if self.unlimited:
            return float("inf")
        with self._lock:
            return max(0.0, self.cap_usd - self._spent)

    def reserve(self, estimate_usd: float) -> None:
        """Ném `BudgetExceeded` nếu lời gọi sắp tới có thể vượt trần.

        Raises:
            BudgetExceeded: đã tiêu + ước lượng > `cap_usd`.
        """
        if self.unlimited:
            return
        with self._lock:
            if self._spent + estimate_usd > self.cap_usd:
                raise BudgetExceeded(
                    f"{self.name}: đã tiêu ${self._spent:.4f}, lời gọi kế ước "
                    f"${estimate_usd:.4f}, trần ${self.cap_usd:.4f}"
                )

    def charge(self, actual_usd: float) -> float:
        """Ghi nhận chi phí thật của một lời gọi đã xong. Trả tổng đã tiêu."""
        with self._lock:
            self._spent += actual_usd
            self._calls += 1
            return self._spent

    def cost_per_1000(self) -> float:
        """USD/1000 lời gọi theo mức đã tiêu — con số mà DoD `W3-04` yêu cầu log."""
        with self._lock:
            if self._calls == 0:
                return 0.0
            return self._spent / self._calls * 1000

    def __repr__(self) -> str:
        cap = "không trần" if self.unlimited else f"${self.cap_usd:.4f}"
        return f"CostBudget({self.name!r}, đã tiêu=${self.spent_usd:.4f}, trần={cap})"
