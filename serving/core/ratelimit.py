"""Giới hạn nhịp theo tenant — `W4-04`.

## Vì sao token bucket chứ không cửa sổ cố định

Cửa sổ cố định ("tối đa 60 request mỗi phút, đếm lại lúc :00") cho phép **120
request trong 2 giây** nếu chúng rơi quanh ranh giới phút — 60 cái cuối cửa sổ
này, 60 cái đầu cửa sổ sau. Đó là gấp đôi hạn mức, đúng vào lúc tệ nhất, và nó
không hiện ra trong bất kỳ phép đo trung bình nào.

Token bucket không có ranh giới để mà dồn vào, và nó trả lời được câu mà `429`
bắt buộc phải trả lời: **bao giờ thì thử lại được** — đó là thời gian nạp đủ một
token, tính chính xác chứ không đoán.

⚠️ Cái nó *cho phép* và cần nói rõ: sức chứa bằng hạn mức một phút, nên 60
request trong một giây đầu là **hợp lệ**. Có chủ đích với một API chat (một lần
mở trang bắn vài request), nhưng nó nghĩa là hạn mức này là "60 mỗi phút tính
trung bình", không phải "không bao giờ quá 60 trong bất kỳ giây nào".

## ⚠️ Trong-tiến-trình: N replica = N lần hạn mức

Không có trạng thái chia sẻ, nên 4 replica sau một load balancer cho mỗi tenant
**240** request/phút chứ không phải 60 — và con số đó đổi mỗi lần autoscale, tức
hạn mức thật là một hàm của số pod. `uvicorn --workers N` cũng vậy: mỗi tiến
trình một bộ đếm.

Nói ra chứ không sửa ở đây: cách sửa là Redis (đã có sẵn trong
`docker-compose`, `arq` đang dùng), nhưng nó thêm một phụ thuộc vào **đường
request nóng** và kéo theo một câu hỏi thiết kế riêng — Redis chết thì mở cổng
(hết giới hạn) hay đóng cổng (sập dịch vụ)? Đó là `TD-39`.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

__all__ = ["Decision", "RateLimiter"]

_SWEEP_AT = 1024
"""Số bucket vượt ngưỡng này thì dọn những bucket đã đầy.

Bucket sinh theo tenant và không tự mất đi, nên với tenant sinh động (mỗi khách
một tenant) đây là một chỗ rò bộ nhớ tăng đều. Bucket đã **đầy** không mang
thông tin gì — nó tương đương với một tenant chưa từng gọi — nên xoá nó không
đổi hành vi.
"""


@dataclass(frozen=True)
class Decision:
    allowed: bool
    remaining: int
    retry_after_s: int
    limit: int

    def headers(self) -> dict[str, str]:
        head = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(self.remaining),
        }
        if not self.allowed:
            head["Retry-After"] = str(self.retry_after_s)
        return head


@dataclass
class _Bucket:
    tokens: float
    updated: float


@dataclass
class RateLimiter:
    """Một bucket cho mỗi khoá (ở đây là `tenant_id`)."""

    buckets: dict[str, _Bucket] = field(default_factory=dict)

    def check(self, key: str, limit_per_minute: int) -> Decision:
        """Tiêu một token nếu còn. `limit_per_minute` đi theo từng key vì hạn mức
        là thuộc tính của **hợp đồng với tenant**, không phải hằng số toàn cục."""
        if limit_per_minute < 1:
            raise ValueError(f"limit_per_minute phải ≥ 1, nhận {limit_per_minute}")
        refill_per_second = limit_per_minute / 60.0
        # `monotonic`: đồng hồ tường nhảy lùi (NTP, đổi múa giờ) sẽ làm `updated`
        # nằm ở tương lai và bucket đóng băng đúng bằng độ lệch.
        now = time.monotonic()

        bucket = self.buckets.get(key)
        if bucket is None:
            bucket = _Bucket(tokens=float(limit_per_minute), updated=now)
            self.buckets[key] = bucket
            self._sweep_if_needed()
        else:
            bucket.tokens = min(
                float(limit_per_minute),
                bucket.tokens + (now - bucket.updated) * refill_per_second,
            )
            bucket.updated = now

        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            return Decision(True, int(bucket.tokens), 0, limit_per_minute)

        # ⭐ `ceil`, và sàn là 1. `int()` hay `round()` cho **0** với mọi khoảng
        # chờ dưới một giây, và một client lịch sự đọc `Retry-After: 0` sẽ thử
        # lại ngay — nhận 429 tiếp, thử lại ngay — tức header sinh ra để giảm tải
        # lại biến thành một vòng lặp nóng.
        wait = (1.0 - bucket.tokens) / refill_per_second
        return Decision(False, 0, max(1, math.ceil(wait)), limit_per_minute)

    def _sweep_if_needed(self) -> None:
        if len(self.buckets) <= _SWEEP_AT:
            return
        now = time.monotonic()
        # Giữ lại bucket còn "nợ": cái nào đã im đủ lâu để nạp đầy thì bỏ.
        self.buckets = {
            key: bucket for key, bucket in self.buckets.items() if now - bucket.updated < 60.0
        }
