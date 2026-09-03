"""`/health` và `/ready` — `W4-03`.

## ⭐⭐ Hai endpoint này **không** là hai tên gọi của một thứ

Đây là chỗ dễ làm sai nhất của cả hạng mục, vì làm sai không lộ ra cho tới lần
sự cố đầu tiên — và lúc đó nó *khuếch đại* sự cố.

| | `/health` (liveness) | `/ready` (readiness) |
|---|---|---|
| câu hỏi | "tiến trình này còn cứu được không?" | "gửi traffic vào đây được chưa?" |
| trả lời sai ⇒ | **khởi động lại container** | rút khỏi load balancer |
| được phép chạm phụ thuộc | **không** | có |

⚠️ **Vì sao `/health` tuyệt đối không được hỏi Qdrant.** Giả sử nó có hỏi. Qdrant
chớp 30 giây → mọi replica trả 503 ở `/health` → orchestrator **khởi động lại
toàn bộ chúng cùng lúc**. Việc khởi động lại không chữa được Qdrant, mà mỗi
replica mới lại mất vài phút nạp 3,3 GB trọng số bge-m3. Một trục trặc 30 giây
của phụ thuộc trở thành một sự cố nhiều phút của chính mình, và cái gây ra nó là
phép thử sức khoẻ.

Nên `/health` chỉ trả lời được đúng một điều — vòng lặp sự kiện còn nhận và trả
được một request — và điều đó đã đủ đúng với việc nó điều khiển.

## `/ready` trả **cùng một thân JSON** cho cả 200 và 503

Người vận hành gõ `curl` khi mọi thứ đang hỏng. Một 503 rỗng buộc họ đi tìm log
của đúng pod ấy; một 503 kèm bảng phụ thuộc trả lời ngay tại chỗ.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request, Response, status

from serving.core.probes import ReadinessProbes
from serving.core.registry import BundleRegistry

__all__ = ["router"]

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])


# ⚠️ Phải khai ở **tầng module**, không phải trong một factory: `from __future__
# import annotations` biến annotation thành chuỗi và FastAPI phân giải chúng
# trong không gian tên của module. Một alias `Annotated[...]` định nghĩa bên
# trong hàm sẽ không tìm thấy, và FastAPI im lặng coi tham số là query param —
# đúng cái bẫy đã ghi ở `pipeline/ingest/app.py`.
def get_registry(request: Request) -> BundleRegistry:
    registry: BundleRegistry = request.app.state.registry
    return registry


def get_probes(request: Request) -> ReadinessProbes:
    probes: ReadinessProbes = request.app.state.probes
    return probes


RegistryDep = Annotated[BundleRegistry, Depends(get_registry)]
ProbesDep = Annotated[ReadinessProbes, Depends(get_probes)]


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness. Không chạm gì. Xem bảng ở đầu module trước khi thêm bất cứ gì."""
    return {"status": "alive"}


@router.get("/ready")
async def ready(response: Response, registry: RegistryDep, probes: ProbesDep) -> dict[str, Any]:
    """Readiness: có bundle đang phục vụ **và** mọi phụ thuộc trả lời được.

    Thứ tự có ý nghĩa: hỏi bundle **trước**, và chỉ chạy probe khi đã có bundle.
    Chưa nạp bundle nghĩa là chưa có gì để phục vụ, nên gửi thêm một lượt truy
    vấn vào Qdrant chỉ để cũng trả 503 là tải thừa — nhân với mọi replica đang
    khởi động cùng lúc sau một lần deploy.
    """
    checks: dict[str, Any] = {
        "bundle": {"ok": registry.is_ready, "detail": None if registry.is_ready else "chưa nạp"}
    }
    if registry.is_ready:
        for result in await probes.run():
            checks[result.name] = result.as_json()

    ok = all(entry["ok"] for entry in checks.values())
    if not ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"ready": ok, "checks": checks, **registry.status()}
