"""Ứng dụng FastAPI của Serving Plane — `W4-03`.

    uv run uvicorn serving.api.app:app --port 8000

## ⭐ Khởi động **không** được chết vì bundle nạp lỗi

Phản xạ đầu tiên là fail-fast: nạp bundle trong `lifespan`, nạp không được thì
ném, tiến trình chết. Ở một job offline điều đó đúng (`rag_core.settings` làm
đúng thế). Ở một container serving nó sai, vì ba lý do cộng lại:

1. Container chết → orchestrator khởi động lại → chết → **crashloop**. Log của
   pod đã biến mất khi ta kịp gõ lệnh xem.
2. Không có tiến trình nào sống thì không có `/ready` nào để **hỏi vì sao**.
3. Không có tiến trình nào sống thì `POST /admin/bundle/reload` cũng không có —
   tức cách chữa duy nhất là deploy lại, kể cả khi nguyên nhân chỉ là gõ nhầm
   một số phiên bản.

Khởi động lên nhưng **chưa sẵn sàng** giữ nguyên phần an toàn — load balancer đọc
`/ready` nên không có traffic nào vào — mà đổi lại được cả ba điều trên.

⚠️ Điều này chỉ đúng khi `/ready` thực sự được cấu hình làm readiness probe. Nếu
deploy quên khai nó, mô hình này biến một pod hỏng thành một pod **nhận traffic
và trả 500**. Đó là đánh đổi thật, và nó phải nằm trong checklist deploy của
`W5`.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI

from rag_core.bundle import latest_bundle
from rag_core.llm import (
    DEFAULT_DEEPSEEK_MODEL,
    DEFAULT_GLM_MODEL,
    MIN_REASONING,
    OpenAICompatProvider,
    build_deepseek_provider,
    build_glm_provider,
)
from rag_core.settings import Settings, get_settings
from serving.api import admin, chat, health
from serving.api.middleware import RequestContextMiddleware
from serving.api.security import AuthMiddleware
from serving.core.auth import ApiKeyStore
from serving.core.chat import ChatService
from serving.core.logging import configure_logging
from serving.core.probes import Check, ReadinessProbes
from serving.core.ratelimit import RateLimiter
from serving.core.registry import BundleRegistry, RuntimeBuilder
from serving.core.runtime import QdrantRuntimeBuilder
from serving.core.understanding import QueryUnderstanding

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

__all__ = ["create_app"]  # `app` do `__getattr__` ở cuối file cấp

logger = logging.getLogger(__name__)


def _startup_version(settings: Settings) -> str | None:
    """Bundle nào được kích hoạt lúc khởi động.

    `BUNDLE_VERSION` ghim tường minh thắng. Không ghim thì lấy bản semver cao
    nhất — tiện cho môi trường dev, và ⚠️ **sai cho production**: nó biến một
    lần `save_bundle` vô ý thành một lần deploy, và nó xoá kết quả của mọi lần
    rollback ở lần restart kế tiếp (xem `Settings.bundle_version`).
    """
    if settings.bundle_version is not None:
        return settings.bundle_version
    try:
        newest = latest_bundle(settings.bundle_root)
    except Exception:
        # Một manifest hỏng trong thư mục **không** được ngăn tiến trình lên:
        # `/ready` 503 kèm lý do vẫn gỡ được, một crashloop thì không.
        logger.exception("không quét được %s", settings.bundle_root)
        return None
    return newest.bundle_version if newest is not None else None


def _qdrant_check(registry: BundleRegistry) -> Check:
    """Phép thử Qdrant đi **qua chính retriever đang phục vụ**, không qua một
    client riêng.

    Một client riêng có thể trả lời "Qdrant sống" trong khi collection mà bundle
    này dùng đã bị xoá — tức `/ready` xanh còn mọi `/chat` trả rỗng. Hỏi đúng
    collection đang dùng là phép thử duy nhất đo được thứ mà request thật sự phụ
    thuộc vào.
    """

    def check() -> None:
        _store_of(registry.active.retriever).count()

    return check


def _store_of(retriever: Any) -> Any:
    """Đi xuống tận đáy `reranked[hybrid[dense]]` để tới `QdrantDenseRetriever`.

    ⚠️ Dùng `getattr` chứ không `isinstance` là có chủ đích ngược lại thói quen:
    `isinstance` bắt module này import `rag_core.retrieval.reranked` và
    `.hybrid` ở tầng module, tức kéo `qdrant-client` vào **mọi** lần import
    `serving.api.app` — kể cả trong test dùng builder giả. Đổi lại, `getattr`
    im lặng nếu quy ước tên thuộc tính đổi; đó là lý do có
    `test_the_probe_reaches_the_qdrant_store_through_every_wrapper`.
    """
    seen = retriever
    while True:
        nxt = getattr(seen, "base", None) or getattr(seen, "store", None)
        if nxt is None:
            return seen
        seen = nxt


def _postgres_check() -> Check:
    """⭐ "DB sẵn sàng" nghĩa là **migration đã chạy**, không phải "cắm được".

    `SELECT 1` trả lời câu thứ hai, và đó không phải cách hệ thống này hỏng. Cách
    nó hỏng là: image mới lên, `alembic upgrade head` chưa chạy, pod báo sẵn sàng,
    nhận traffic, rồi mọi request chết bằng `column … does not exist` — với một
    `SELECT 1` xanh suốt. Chi tiết ở `serving/db/engine.py`.

    Engine dựng **một lần** ở đây và giữ trong closure: `create_engine` không mở
    kết nối nào cho tới lượt probe đầu, nhưng dựng lại nó mỗi lượt probe thì mỗi
    lượt là một pool mới và một kết nối mới — tức phép thử sức khoẻ tự trở thành
    tải, đúng thứ mà TTL của `ReadinessProbes` sinh ra để tránh.
    """
    from serving.db.engine import make_engine, postgres_check

    engine = make_engine(pool_size=1, max_overflow=0)

    def check() -> None:
        postgres_check(engine)

    return check


def build_llm(settings: Settings) -> OpenAICompatProvider | None:
    """Nguồn sinh text của `POST /chat` — `W4-06`.

    ⭐ Trả `None` thay vì ném khi thiếu key. Cùng lý lẽ với việc bundle nạp lỗi
    không giết tiến trình (§đầu module): một container không lên được thì không
    có `/ready` nào để hỏi vì sao, và ở đây còn tệ hơn vì `/health`, `/ready`,
    `/admin/bundle` đều còn dùng được bình thường mà không cần LLM. Thiếu key
    làm hỏng đúng **một** endpoint, nên nó phải hỏng đúng một endpoint.

    ⭐ Kiểu trả về là lớp **cụ thể**, không phải `StreamingLLM`, và đó là chỗ
    đúng để hẹp lại: `W4-07` cần cùng client này ở giao diện *không* stream
    (`LLMProvider.complete`) để viết lại câu hỏi. Một client, một pool kết nối,
    một model — và điểm hẹp nằm ở factory, nơi kiểu cụ thể vốn đã biết, chứ
    không lan vào `ChatService` hay `QueryUnderstanding` (cả hai vẫn khai
    Protocol, nên router của `W4-08` vẫn cắm vào được).
    """
    if settings.chat_provider == "none":
        return None
    key = (
        settings.deepseek_api_key if settings.chat_provider == "deepseek" else settings.glm_api_key
    )
    if key is None:
        logger.warning(
            "chat_provider=%s nhưng chưa có API key — POST /chat sẽ trả 503, "
            "phần còn lại của API vẫn chạy",
            settings.chat_provider,
        )
        return None
    if settings.chat_provider == "deepseek":
        return build_deepseek_provider(
            settings.chat_model or DEFAULT_DEEPSEEK_MODEL,
            api_key=key.get_secret_value(),
            base_url=settings.deepseek_base_url,
        )
    return build_glm_provider(
        settings.chat_model or DEFAULT_GLM_MODEL,
        api_key=key.get_secret_value(),
        base_url=settings.glm_base_url,
    )


def build_understanding(settings: Settings, llm: OpenAICompatProvider | None) -> QueryUnderstanding:
    """Bước hiểu câu hỏi của `W4-07`, dùng **chung** client với đường sinh.

    ⚠️ `chat_rewrite=false` tắt **đúng một** trong ba việc: viết lại đa lượt, thứ
    duy nhất tốn tiền và tốn TTFB. Định tuyến và phát hiện ngôn ngữ là luật
    thuần, không có lý do nào để tắt và không có công tắc nào tắt chúng.
    """
    return QueryUnderstanding(
        llm=llm if settings.chat_rewrite else None,
        timeout_s=settings.chat_rewrite_timeout_s,
        extra_body=MIN_REASONING.get(settings.chat_provider),
    )


def build_sessions() -> async_sessionmaker[AsyncSession] | None:
    """Factory phiên async cho đường request (`W4-06`).

    Engine dựng ở đây và sống suốt vòng đời tiến trình — `create_async_engine`
    không mở kết nối nào cho tới lượt dùng đầu, nên nó rẻ lúc khởi động và
    không kéo dài thời gian tới lúc `/health` trả lời.

    Tách khỏi engine **đồng bộ** của `_postgres_check`: probe chạy trong
    threadpool với `pool_size=1`, đường request chạy trên vòng lặp sự kiện. Dùng
    chung một pool thì một `/ready` chậm giữ mất kết nối của một `/chat`.
    """
    try:
        from serving.db.engine import async_session_factory, make_async_engine

        return async_session_factory(make_async_engine())
    except Exception:
        logger.exception("không dựng được engine async — POST /chat sẽ trả 503")
        return None


def build_probes(registry: BundleRegistry) -> ReadinessProbes:
    """Tập phép thử của `/ready` — bundle + Qdrant + Postgres, đủ DoD `W4-03`."""
    return ReadinessProbes(
        checks={"qdrant": _qdrant_check(registry), "postgres": _postgres_check()}
    )


def create_app(
    *,
    settings: Settings | None = None,
    build_runtime: RuntimeBuilder | None = None,
    probe_factory: Callable[[BundleRegistry], ReadinessProbes] = build_probes,
) -> FastAPI:
    """`build_runtime` tiêm được từ ngoài để test chạy không cần Qdrant/GPU —
    cùng lý lẽ đã làm cho `BundleRegistry` nhận Protocol thay vì tự dựng
    (`W4-02` §1)."""
    resolved = settings or get_settings()
    configure_logging(resolved.log_level)

    registry = BundleRegistry(
        root=resolved.bundle_root,
        build_runtime=build_runtime
        or QdrantRuntimeBuilder(
            url=resolved.qdrant_url,
            api_key=(
                resolved.qdrant_api_key.get_secret_value() if resolved.qdrant_api_key else None
            ),
            device=resolved.embedding_device,
            batch_size=resolved.embedding_batch_size,
            allow_runtime_drift=resolved.bundle_allow_runtime_drift,
        ),
    )

    @asynccontextmanager
    async def lifespan(api: FastAPI) -> AsyncIterator[None]:
        version = _startup_version(resolved)
        if version is None:
            logger.error(
                "không tìm thấy bundle nào trong %s — API lên nhưng /ready sẽ trả 503",
                resolved.bundle_root,
            )
        else:
            try:
                # Chặn vòng lặp sự kiện vài chục giây (nạp trọng số). Chấp nhận
                # được **chỉ ở đây**: chưa có traffic, và uvicorn chưa mở cổng
                # cho tới khi lifespan xong. Ở route thì không — xem `admin.py`.
                registry.activate(version)
            except Exception:
                logger.exception(
                    "nạp bundle %s lúc khởi động thất bại — API vẫn lên, /ready trả 503 "
                    "và POST /admin/bundle/reload sửa được mà không cần deploy lại",
                    version,
                )
        yield

    api = FastAPI(
        title="RAG serving",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
    )
    api.state.registry = registry
    api.state.probes = probe_factory(registry)
    llm = build_llm(resolved)
    api.state.chat = ChatService(
        registry=registry,
        sessions=build_sessions(),
        llm=llm,
        top_k=resolved.chat_top_k,
        max_tokens=resolved.chat_max_tokens,
        # Bảng đo được ở `W3-04`, dùng lại nguyên vẹn — xem `ChatService.extra_body`.
        extra_body=MIN_REASONING.get(resolved.chat_provider),
        understanding=build_understanding(resolved, llm),
    )
    # ⚠️ **Thứ tự quan trọng và nó ngược trực giác.** `add_middleware` *chèn lên
    # đầu*, nên cái thêm **sau** nằm **ngoài**. Auth phải thêm trước để
    # `RequestContextMiddleware` bọc ngoài nó — nếu ngược lại thì mọi phản hồi
    # 401/403/429 do auth gửi sẽ không đi qua chỗ gắn `X-Request-ID`, tức đúng
    # những phản hồi mà người vận hành cần truy vết lại là những phản hồi không
    # truy được. Có test ghim (`test_a_401_still_carries_a_request_id`).
    api.state.keys = ApiKeyStore.load(resolved.api_keys_file)
    api.state.limiter = RateLimiter()
    api.add_middleware(AuthMiddleware, keys=api.state.keys, limiter=api.state.limiter)
    # ASGI thuần, không `BaseHTTPMiddleware` — lý do ở docstring của
    # `middleware.py`, nó liên quan trực tiếp tới SSE của `W4-06`.
    api.add_middleware(RequestContextMiddleware)
    api.include_router(health.router)
    api.include_router(admin.router)
    api.include_router(chat.router)
    return api


def __getattr__(name: str) -> FastAPI:
    """`app` dựng **khi được lấy**, không phải khi module được import.

    `uvicorn serving.api.app:app` vẫn chạy đúng (uvicorn import module rồi
    `getattr`). Nhưng `import serving.api.app` trong một test thì không còn kéo
    theo `get_settings()` đọc `.env` thật và `configure_logging()` gỡ sạch handler
    của pytest — hai tác dụng phụ mà một dòng `import` không nên có.
    """
    if name == "app":
        return create_app()
    raise AttributeError(name)
