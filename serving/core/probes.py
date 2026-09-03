"""Phép thử phụ thuộc cho `/ready` — `W4-03`.

`/ready` bị **gọi liên tục**: orchestrator hỏi mỗi vài giây, nhân với số replica.
Ba tính chất dưới đây đều sinh ra từ chỗ đó, và bỏ tính chất nào cũng cho một
kiểu hỏng riêng.

## 1. Có hạn giờ — nếu không, sự cố đổi loại

Qdrant treo (không từ chối, chỉ không trả lời) mà không có hạn giờ thì `/ready`
cũng treo. Orchestrator không nhận được 503; nó nhận **timeout của chính probe**.
Hai thứ đó ánh xạ khác nhau ở mọi hệ thống điều phối, và cái sau còn kéo theo một
kết nối treo mỗi lần hỏi.

⚠️ Hạn giờ ở đây là **lưới chắn cuối**, không phải cách chữa: `asyncio.wait_for`
bỏ *chờ* nhưng không giết được luồng đang chờ socket, nên một phụ thuộc treo vĩnh
viễn vẫn rò một luồng mỗi chu kỳ TTL. Cách chữa đúng nằm ở client
(`QdrantDenseRetriever(timeout=...)`), và `serving.core.runtime` đặt nó ngắn hẳn
so với hạn giờ này để luồng luôn tự quay về trước.

## 2. Có TTL — nếu không, chính phép thử là tải

10 replica × probe mỗi 3 giây = 200 lượt `get_collection` mỗi phút gửi vào một
Qdrant *đang yếu*. Phép thử sức khoẻ khi đó là một phần của nguyên nhân.

Kết quả **hỏng cũng được nhớ**, đúng bằng thời gian như kết quả tốt. Chỉ nhớ kết
quả tốt nghe có vẻ an toàn hơn, nhưng nó nghĩa là phụ thuộc bị dội mạnh nhất
đúng vào lúc nó yếu nhất.

⚠️ Cái giá: hồi phục bị chậm tối đa `ttl_s`. Nên `ttl_s` phải **nhỏ hơn hẳn**
`periodSeconds × failureThreshold` của orchestrator, nếu không thì một pod đã lành
vẫn bị đá ra.

## 3. Hỏng thì đóng — mọi ngoại lệ đều là "chưa sẵn sàng"

Một `except` bắt hẹp theo kiểu lỗi sẽ để lọt loại lỗi chưa lường trước thành
"không sao". Ở phép thử sẵn sàng, chiều mặc định phải là từ chối nhận traffic.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

__all__ = ["Check", "ProbeResult", "ReadinessProbes"]

#: Ném = chưa sẵn sàng; lời của exception thành `detail`. Đồng bộ chứ không
#: async vì mọi client ở tầng dưới (qdrant-client, sau này là driver Postgres
#: đồng bộ) đều đồng bộ — bọc chúng bằng `to_thread` ở một chỗ tốt hơn là bắt
#: mỗi phép thử tự nhớ làm việc đó.
Check = Callable[[], None]


@dataclass(frozen=True)
class ProbeResult:
    name: str
    ok: bool
    detail: str | None
    duration_ms: float

    def as_json(self) -> dict[str, object]:
        return {"ok": self.ok, "detail": self.detail, "duration_ms": self.duration_ms}


@dataclass
class ReadinessProbes:
    """Tập phép thử, chạy song song, có hạn giờ và có nhớ tạm."""

    checks: Mapping[str, Check]
    timeout_s: float = 2.0
    ttl_s: float = 3.0
    _cached: tuple[float, tuple[ProbeResult, ...]] | None = field(
        default=None, init=False, repr=False
    )
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    async def run(self, *, force: bool = False) -> tuple[ProbeResult, ...]:
        # `monotonic` chứ không `time()`: một bước nhảy NTP về quá khứ sẽ đóng
        # băng cache đúng bằng độ lệch, và đó là loại sự cố không ai nghĩ tới khi
        # đọc log.
        now = time.monotonic()
        cached = self._cached
        if not force and cached is not None and now - cached[0] < self.ttl_s:
            return cached[1]

        async with self._lock:
            # Kiểm lại sau khi giành được khoá: nhiều lượt `/ready` cùng đến sẽ
            # xếp hàng ở đây, và nếu không kiểm lại thì tất cả cùng chạy probe —
            # đúng cái TTL sinh ra để tránh.
            cached = self._cached
            now = time.monotonic()
            if not force and cached is not None and now - cached[0] < self.ttl_s:
                return cached[1]

            results = tuple(await asyncio.gather(*(self._run_one(name) for name in self.checks)))
            self._cached = (time.monotonic(), results)
            return results

    async def _run_one(self, name: str) -> ProbeResult:
        check = self.checks[name]
        start = time.perf_counter()
        try:
            await asyncio.wait_for(asyncio.to_thread(check), timeout=self.timeout_s)
        except TimeoutError:
            return ProbeResult(name, False, f"quá hạn {self.timeout_s:g}s", _elapsed_ms(start))
        except Exception as exc:  # hỏng thì đóng, xem docstring §3
            detail = str(exc) or exc.__class__.__name__
            return ProbeResult(name, False, detail, _elapsed_ms(start))
        return ProbeResult(name, True, None, _elapsed_ms(start))


def _elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)
