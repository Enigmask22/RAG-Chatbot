"""Cấu hình đọc từ biến môi trường / `.env`.

Nguyên tắc: **fail-fast và nói rõ thiếu biến nào.** Cấu hình sai mà chương trình
vẫn chạy được rồi hỏng ở giữa job index 2 tiếng là kiểu lỗi đắt nhất.

Secret không có giá trị mặc định và không bao giờ lọt vào `repr` — dùng
`SecretStr` để lỡ log nguyên object settings cũng không rò key.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

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
        pwd = self.postgres_password.get_secret_value()
        return (
            f"postgresql://{self.postgres_user}:{pwd}"
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
