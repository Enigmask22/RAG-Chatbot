"""Kết nối Postgres, phiên có tenant, và phép thử sẵn sàng — `W4-05`.

## ⭐⭐ "DB sẵn sàng" nghĩa là **migration đã chạy**, không phải "cắm được"

DoD của `W4-03` viết `/ready` chỉ 200 khi *"bundle + Qdrant + DB sẵn sàng"*, và
hạng mục đó để trống nhánh DB vì lúc ấy chưa có DB. Giờ có, và câu hỏi thật là
**hỏi cái gì**.

`SELECT 1` trả lời "cắm được". Đó không phải cách hệ thống này hỏng. Cách nó
hỏng là: image mới deploy xong, `alembic upgrade head` **chưa** chạy (hoặc chạy
lỗi và bị bỏ qua), pod báo sẵn sàng, nhận traffic, rồi mọi request chết bằng
`column "bundle_version" does not exist`. Một `SELECT 1` xanh suốt.

Nên phép thử so `alembic_version` trong DB với **head của thư mục migration
trong chính image này**. Lệch ⇒ chưa sẵn sàng, kèm cả hai số.

Đọc head từ `ScriptDirectory` chứ không ghim một hằng số trong mã: một hằng số
phải sửa tay mỗi lần thêm migration, và lần quên đầu tiên biến phép thử này
thành phép thử luôn xanh.

⚠️ **Nhiều head là lỗi, không phải là hai lựa chọn.** Hai nhánh migration chưa
merge nghĩa là không có câu trả lời cho "schema đúng là gì" — và nếu chọn bừa một
head thì deploy thành công trên nửa số pod.

## `SET LOCAL`, không `SET SESSION`

Policy RLS đọc `app.tenant_id` từ tham số phiên. Connection pool **tái dùng** kết
nối, nên một `SET SESSION` sót lại làm request kế tiếp mang tenant của request
trước — tức rò dữ liệu chéo tenant do một dòng cấu hình, và nó chỉ xảy ra khi có
tải (lúc pool thật sự tái dùng kết nối), tức không bao giờ xảy ra ở máy dev.
`SET LOCAL` chết cùng transaction.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from rag_core.settings import get_settings
from serving.db.models import TENANT_SETTING

__all__ = [
    "MigrationStateError",
    "expected_revision",
    "make_engine",
    "postgres_check",
    "tenant_session",
]

logger = logging.getLogger(__name__)

_ALEMBIC_DIR = Path(__file__).resolve().parents[2] / "alembic"


class MigrationStateError(RuntimeError):
    """Schema trong DB không phải schema mà mã này giả định."""


@lru_cache(maxsize=1)
def expected_revision() -> str:
    """Head của thư mục migration đi kèm image này."""
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    config = Config()
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    heads = ScriptDirectory.from_config(config).get_heads()
    if len(heads) != 1:
        raise MigrationStateError(
            f"thư mục migration có {len(heads)} head ({sorted(heads)}) — hai nhánh "
            "chưa merge nghĩa là không có câu trả lời cho 'schema đúng là gì'. "
            "Chạy `alembic merge` trước khi deploy."
        )
    return heads[0]


def make_engine(dsn: str | None = None, **kwargs: Any) -> Engine:
    """Engine của **ứng dụng**: role không superuser, nên RLS áp lên nó.

    ⭐ Mặc định là `postgres_app_dsn`, không phải `postgres_dsn`. Đó không phải
    một chi tiết: `POSTGRES_USER` của image postgres là superuser, và superuser
    bỏ qua RLS **hoàn toàn** — kể cả `FORCE ROW LEVEL SECURITY`. Nối bằng DSN
    kia thì năm policy của `0001_initial` trở thành trang trí và mọi phép kiểm
    cấu hình vẫn xanh.
    """
    return create_engine(dsn or get_settings().postgres_app_dsn, pool_pre_ping=True, **kwargs)


def postgres_check(engine: Engine) -> None:
    """Phép thử cho `/ready`. Ném = chưa sẵn sàng, lời của lỗi thành `detail`."""
    with engine.connect() as connection:
        found = connection.execute(
            text("SELECT version_num FROM alembic_version")
        ).scalar_one_or_none()
    want = expected_revision()
    if found is None:
        raise MigrationStateError(
            f"DB chưa có migration nào; mã này cần {want}. Chạy `alembic upgrade head`."
        )
    if found != want:
        raise MigrationStateError(
            f"DB đang ở migration {found} nhưng mã này cần {want} — "
            "deploy đã lên trước khi migration chạy."
        )


@contextmanager
def tenant_session(factory: sessionmaker[Session], tenant_id: str) -> Iterator[Session]:
    """Phiên đã đặt tenant, nên mọi truy vấn trong đó bị RLS thu hẹp sẵn.

    ⭐ Đây là lý do lỗ tenant thứ ba (Postgres) đóng theo cùng một hướng với hai
    lỗ trước: người viết truy vấn **không cần nhớ** thêm `AND tenant_id = …`, và
    nếu họ quên thì kết quả là **rỗng**, không phải là dữ liệu của người khác.
    """
    with factory() as session:
        session.execute(
            # Tham số hoá qua `set_config` chứ không nối chuỗi vào `SET LOCAL`:
            # `SET` không nhận placeholder, nên nối chuỗi là con đường duy nhất
            # còn lại — và `tenant_id` đến từ token, nhưng "đến từ token" không
            # phải một lý do để bỏ phép tham số hoá.
            text(f"SELECT set_config('{TENANT_SETTING}', :tenant, true)"),
            {"tenant": tenant_id},
        )
        yield session
