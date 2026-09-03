"""Alembic env — `W4-05`.

DSN đọc từ `rag_core.settings`, **cùng một chỗ** ứng dụng đọc. Để nó trong
`alembic.ini` thì (a) mật khẩu vào git và (b) tồn tại hai nguồn cấu hình cho
cùng một database — tức migration chạy trên DB này còn ứng dụng nói chuyện với
DB kia, và triệu chứng là "cột không tồn tại" ở production sau một lần deploy
mà migration báo thành công.

⚠️ Không có chế độ *offline*. `alembic upgrade head --sql` sinh ra một script SQL
để ai đó chạy tay, và với schema này script ấy sẽ **thiếu phần RLS** trừ khi
được viết lại lần hai. Một đường sinh SQL đúng-một-nửa nguy hiểm hơn không có
đường nào: nó chạy trót lọt và để lại các bảng không có policy.
"""

from __future__ import annotations

from sqlalchemy import engine_from_config, pool

from alembic import context
from rag_core.settings import get_settings
from serving.db.models import Base

config = context.config
target_metadata = Base.metadata


def run_migrations_online() -> None:
    config.set_main_option("sqlalchemy.url", get_settings().postgres_dsn)
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # So cả **kiểu** cột, không chỉ tên: đổi `String(64)` thành
            # `String(128)` mà không có phép so này thì `--autogenerate` báo "no
            # changes" và cột ở production giữ nguyên độ dài cũ.
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():  # pragma: no cover - xem docstring
    raise SystemExit(
        "chế độ offline (`--sql`) bị tắt có chủ đích: script sinh ra sẽ thiếu "
        "phần RLS, và một đường deploy đúng một nửa tệ hơn không có đường nào."
    )
run_migrations_online()
