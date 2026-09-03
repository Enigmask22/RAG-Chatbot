"""Cấu hình đọc từ biến môi trường / `.env`.

Nguyên tắc: **fail-fast và nói rõ thiếu biến nào.** Cấu hình sai mà chương trình
vẫn chạy được rồi hỏng ở giữa job index 2 tiếng là kiểu lỗi đắt nhất.

Secret không có giá trị mặc định và không bao giờ lọt vào `repr` — dùng
`SecretStr` để lỡ log nguyên object settings cũng không rò key.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings", "get_settings"]

_REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(_REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------ LLM API
    deepseek_api_key: SecretStr | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    openrouter_api_key: SecretStr | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    glm_api_key: SecretStr | None = None
    glm_base_url: str = "https://api.z.ai/api/paas/v4"
    """Endpoint quoc te cua Z.ai. Ban dai luc la `https://open.bigmodel.cn/api/paas/v4`."""
    hf_token: SecretStr | None = None

    # ------------------------------------------------------------ Hạ tầng
    qdrant_url: str = "http://127.0.0.1:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "rag_chunks"

    postgres_user: str = "rag"
    postgres_password: SecretStr = SecretStr("rag_local_dev_only")
    postgres_db: str = "rag"
    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5432

    postgres_app_user: str = "rag_app"
    postgres_app_password: SecretStr = SecretStr("rag_app_local_dev_only")
    """⭐⭐ Role mà **ứng dụng** dùng, tách khỏi role chạy migration (`W4-05`).

    Không phải để cho gọn. `POSTGRES_USER` của image postgres là **superuser**, và
    superuser **bỏ qua Row-Level Security hoàn toàn** — kể cả `FORCE ROW LEVEL
    SECURITY`. Nên nếu ứng dụng dùng chung role ấy thì mọi policy trở thành trang
    trí, trong khi `pg_class.relforcerowsecurity` vẫn báo `true` và mọi phép kiểm
    cấu hình vẫn xanh.

    Phát hiện được vì test **hành vi** đỏ trong khi test **cấu hình** xanh —
    xem `tests/integration/test_migrations.py` §2.
    """

    redis_url: str = "redis://127.0.0.1:6379/0"

    ingest_queue: str = "arq:queue"
    """Tên hàng đợi arq cho job ingest.

    Cấu hình được vì hai môi trường dùng chung một Redis sẽ **nhặt job của nhau**:
    worker của staging nhận job của production, thất bại vì thiếu config, và job
    biến mất khỏi hàng đợi. Đo được ở chính test của `W3-08` — các test dùng chung
    một hàng đợi đã làm đúng chuyện đó với nhau.
    """

    ingest_config_dir: Path = Path("configs/indexing")
    """Thư mục mà `POST /ingest` được phép đọc config từ đó (`W3-08`).

    Là **cận trên của quyền đọc**, không phải một tiện ích: `IngestRequest.config`
    nhận một *tên*, và `pipeline.ingest.schemas.resolve_config` ghép nó vào đúng
    thư mục này rồi kiểm lại rằng kết quả không thoát ra ngoài. Đổi được bằng biến
    môi trường vì container có thể mount config ở chỗ khác — và vì test cần một
    thư mục tạm mà không phải nới lỏng hàng rào.
    """

    # ------------------------------------------------------------ Serving
    bundle_root: Path = Path("bundles")
    """Thư mục chứa các `rag-bundle-v*/` mà Serving Plane được nạp từ đó (`W4-03`).

    Là **toàn bộ** đường nối giữa hai plane: pipeline ghi vào đây, serving đọc từ
    đây, và không có kênh nào khác. Cấu hình được vì trong container nó là một
    volume mount, còn trong test nó là `tmp_path`.
    """

    bundle_version: str | None = None
    """Bundle kích hoạt lúc khởi động. `None` = bản semver cao nhất tìm thấy.

    Ghim tường minh là cách duy nhất để một lần rollback **sống sót qua restart**:
    `BundleRegistry.rollback()` chỉ đổi trạng thái trong bộ nhớ, nên nếu deploy
    vẫn để `None` thì container khởi động lại sẽ lặng lẽ quay về đúng bản vừa bị
    lùi khỏi.
    """

    bundle_allow_runtime_drift: bool = False
    """Cho chạy bundle mà runtime không khớp `components.retriever_name` (`TD-38`).

    Chỉ dành cho máy dev không có GPU muốn chạy thử bundle đã eval trên
    `cuda:float16`. Bật ở production nghĩa là mọi số trong manifest có thể đang
    nói về một hệ thống khác hệ thống đang phục vụ.
    """

    api_keys_file: Path | None = Path("secrets/api-keys.json")
    """Kho API key của Serving Plane (`W4-04`) — chứa **digest**, không chứa key.

    Thiếu file = **không có key nào**, tức mọi route cần xác thực trả 401. Cố ý
    không có key mặc định: một key mặc định cho tiện lúc dev là đường ngắn nhất
    để credential của môi trường test đi thẳng vào production, và nó không để
    lại dấu vết nào trong diff.
    """

    # ------------------------------------------------------------ Chat (W4-06)
    chat_provider: Literal["deepseek", "glm", "none"] = "deepseek"
    """Nguồn sinh text của `POST /chat`. `none` = tắt hẳn đường sinh.

    ⚠️ `none` **không** làm `/ready` đỏ. Một phép thử sẵn sàng gọi API trả tiền
    là một phép thử tự sinh hoá đơn, nhân với số replica nhân với tần suất poll.
    Thiếu key biểu hiện ở `POST /chat` bằng `503` kèm lý do, và ở một dòng cảnh
    báo lúc khởi động. `W4-08` sẽ có circuit breaker để nói được nhiều hơn.
    """

    chat_model: str | None = None
    """`None` = mặc định của provider (`DEFAULT_DEEPSEEK_MODEL` / `DEFAULT_GLM_MODEL`)."""

    chat_top_k: int = Field(default=5, ge=1, le=50)
    chat_max_tokens: int = Field(default=1024, ge=1)

    # ------------------------------------------------------------ Model
    embedding_model: str = "bkai-foundation-models/vietnamese-bi-encoder"
    embedding_device: str = "auto"
    embedding_batch_size: int = Field(default=32, ge=1)

    # ------------------------------------------------------------ Khác
    log_level: str = "INFO"
    cache_dir: Path = Path(".cache")

    @field_validator("embedding_device")
    @classmethod
    def _check_device(cls, v: str) -> str:
        allowed = {"auto", "cpu", "cuda", "mps"}
        if v not in allowed:
            raise ValueError(f"EMBEDDING_DEVICE phải thuộc {sorted(allowed)}, nhận được {v!r}")
        return v

    @property
    def postgres_dsn(self) -> str:
        """DSN đồng bộ, dùng cho Alembic và cho phép thử `/ready`.

        ⚠️ Ghi rõ driver `+psycopg` (v3) chứ không để `postgresql://` trần:
        SQLAlchemy 2.0 mặc định `postgresql://` thành **psycopg2**, thứ không có
        trong dự án này — nên DSN trần chết bằng `ModuleNotFoundError` ở lần
        migration đầu, chứ không phải bằng một thông báo về cấu hình.
        """
        pwd = self.postgres_password.get_secret_value()
        return self._dsn(self.postgres_user, pwd)

    @property
    def postgres_app_dsn(self) -> str:
        """DSN của **ứng dụng** — role không superuser, nên RLS áp lên nó.

        Mọi thứ chạm dữ liệu khách hàng phải đi qua DSN này. `postgres_dsn` chỉ
        dành cho migration và cho việc quản trị.
        """
        return self._dsn(self.postgres_app_user, self.postgres_app_password.get_secret_value())

    def _dsn(self, user: str, password: str) -> str:
        return (
            f"postgresql+psycopg://{user}:{password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    def require(self, *names: str) -> None:
        """Khẳng định các secret cần thiết đã có, nếu thiếu thì báo đủ tên một lần.

        Gọi ở đầu mỗi entrypoint cần secret, thay vì để `None` chui xuống tầng
        HTTP client rồi báo 401 khó hiểu.
        """
        missing = [n for n in names if getattr(self, n, None) is None]
        if missing:
            env_names = ", ".join(n.upper() for n in missing)
            raise RuntimeError(
                f"Thiếu biến môi trường bắt buộc: {env_names}. "
                f"Sao chép `.env.example` thành `.env` rồi điền giá trị."
            )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Settings dạng singleton. Gọi `get_settings.cache_clear()` trong test."""
    return Settings()
