"""`W4-05` — migration đảo được, và RLS thật sự chặn.

Cần Postgres thật (`make up`). Không mock được: cả hai thứ đáng kiểm ở đây —
`downgrade` có chạy không, và policy có áp cho **owner** không — là hành vi của
chính Postgres, không phải của mã Python.

Nhóm 2 là nhóm quan trọng nhất: nó bắt một lỗ hổng **tự báo cáo là đã đóng**.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from sqlalchemy import Engine, text
from sqlalchemy.exc import OperationalError, ProgrammingError
from sqlalchemy.orm import sessionmaker

from rag_core.settings import get_settings
from serving.db.engine import (
    MigrationStateError,
    expected_revision,
    make_engine,
    postgres_check,
    tenant_session,
)
from serving.db.models import RLS_TABLES

pytestmark = pytest.mark.integration


def _alembic(*argv: str) -> None:
    from alembic.config import Config

    from alembic import command
    from serving.db.engine import _ALEMBIC_DIR

    config = Config(str(_ALEMBIC_DIR.parent / "alembic.ini"))
    config.set_main_option("script_location", str(_ALEMBIC_DIR))
    getattr(command, argv[0])(config, *argv[1:])


@pytest.fixture(scope="module")
def owner_engine() -> Iterator[Engine]:
    """Role chạy migration — superuser trong image postgres."""
    eng = make_engine(get_settings().postgres_dsn)
    try:
        with eng.connect():
            pass
    except OperationalError as exc:  # pragma: no cover - phụ thuộc máy
        pytest.skip(f"không có Postgres: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture(scope="module")
def app_engine(owner_engine: Engine) -> Iterator[Engine]:
    """⭐ Role của **ứng dụng**. Mọi test RLS phải đi qua đây — nối bằng role
    migration thì chúng xanh hết mà không chứng minh gì (superuser bỏ qua RLS)."""
    _alembic("upgrade", "head")
    eng = make_engine()
    yield eng
    eng.dispose()


@pytest.fixture
def migrated(owner_engine: Engine, app_engine: Engine) -> Iterator[Engine]:
    _alembic("upgrade", "head")
    yield app_engine
    with owner_engine.begin() as conn:
        for table in RLS_TABLES:
            conn.execute(text(f"DELETE FROM {table}"))


# ---------------------------------------------------------------------------
# 1. DoD — up → down → up
# ---------------------------------------------------------------------------


def test_upgrade_downgrade_upgrade_from_an_empty_database(owner_engine: Engine) -> None:
    """⭐ `downgrade` mà không ai chạy là một hàm chưa từng được biên dịch.

    ⚠️ Và phải nói rõ nó chứng minh cái gì: chạy trên DB **rỗng** thì nó chứng
    minh **DDL đảo được**, không chứng minh **deploy lùi lại được** — lùi một
    migration có dữ liệu là mất dữ liệu, và không test nào ở đây nói khác.
    """
    _alembic("upgrade", "head")
    _alembic("downgrade", "base")
    with owner_engine.connect() as conn:
        remaining = conn.execute(
            text(
                "SELECT count(*) FROM information_schema.tables "
                "WHERE table_schema='public' AND table_name = ANY(:names)"
            ),
            {"names": list(RLS_TABLES)},
        ).scalar_one()
    assert remaining == 0, "downgrade để sót bảng"

    _alembic("upgrade", "head")
    postgres_check(owner_engine)


def test_every_declared_table_exists_after_upgrade(migrated: Engine) -> None:
    with migrated.connect() as conn:
        found = {
            row[0]
            for row in conn.execute(
                text("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")
            )
        }
    assert set(RLS_TABLES) <= found


# ---------------------------------------------------------------------------
# 2. ⭐⭐ RLS — lỗ hổng tự báo cáo là đã đóng
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("table", RLS_TABLES)
def test_every_tenant_table_has_forced_rls(migrated: Engine, table: str) -> None:
    """⭐⭐ `relforcerowsecurity`, không phải `relrowsecurity`.

    `ENABLE ROW LEVEL SECURITY` bật policy cho mọi role **trừ chủ sở hữu bảng**,
    và ứng dụng kết nối bằng chính owner. Chỉ `ENABLE` thì `pg_tables.rowsecurity`
    trả `true`, mọi dashboard audit xanh, và **không một policy nào** áp lên
    đường mà ứng dụng thật đi.

    Đó là loại lỗ hổng tệ nhất: nó tự báo cáo là đã đóng.
    """
    with migrated.connect() as conn:
        enabled, forced = conn.execute(
            text("SELECT relrowsecurity, relforcerowsecurity FROM pg_class WHERE relname=:t"),
            {"t": table},
        ).one()
    assert enabled, f"{table} chưa bật RLS"
    assert forced, (
        f"{table} mới ENABLE chứ chưa FORCE — owner (`rag`, cũng là user của ứng "
        "dụng) bỏ qua toàn bộ policy, trong khi mọi phép kiểm cấu hình vẫn xanh"
    )


def test_the_owner_does_not_bypass_rls(migrated: Engine) -> None:
    """⭐⭐ Phép kiểm hành vi, không phải phép kiểm cấu hình.

    Test ở trên đọc cờ; test này thử **thật**: ghi hai hàng cho hai tenant bằng
    chính user của ứng dụng, rồi đọc lại bằng một phiên có tenant. Nếu owner bỏ
    qua policy thì nó thấy cả hai.
    """
    factory = sessionmaker(migrated)
    for tenant in ("acme", "globex"):
        with tenant_session(factory, tenant) as session:
            session.execute(
                text("INSERT INTO conversation (id, tenant_id, title) VALUES (:i, :t, :ti)"),
                {"i": f"c-{tenant}", "t": tenant, "ti": f"của {tenant}"},
            )
            session.commit()

    with tenant_session(factory, "acme") as session:
        rows = session.execute(text("SELECT tenant_id FROM conversation")).scalars().all()
    assert rows == ["acme"], f"thấy được dữ liệu của tenant khác: {rows}"


def test_a_query_that_forgets_the_where_clause_returns_nothing(migrated: Engine) -> None:
    """⭐⭐ Toàn bộ lý do RLS tồn tại ở đây, và nó là **hướng hỏng**.

    `SELECT * FROM message WHERE conversation_id = :id` quên `AND tenant_id` là
    câu SQL tự nhiên nhất trên đời, nó chạy đúng trên môi trường dev một-tenant,
    và ở production nó trả về hội thoại của khách hàng khác.

    Với RLS, cùng câu ấy trả **rỗng**. Hướng hỏng đảo chiều — thiếu kết quả thì
    người dùng thấy và báo lại, còn dư kết quả thì không ai thấy, kể cả người bị
    rò. Cùng lý lẽ với `MetadataFilter` (`W2-06`) và `tenant_filter` (`W4-04`).
    """
    factory = sessionmaker(migrated)
    with tenant_session(factory, "globex") as session:
        session.execute(
            text("INSERT INTO conversation (id, tenant_id) VALUES ('c-x', 'globex')"),
        )
        session.commit()

    with tenant_session(factory, "acme") as session:
        forgotten = session.execute(text("SELECT * FROM conversation WHERE id = 'c-x'")).all()
    assert forgotten == []


def test_writing_a_row_for_another_tenant_is_refused(migrated: Engine) -> None:
    """`WITH CHECK` gác chiều ghi. Thiếu nó thì một tenant **ghi được** hàng mang
    `tenant_id` của tenant khác — và hàng đó sau đó vô hình với chính người ghi,
    nên lỗi không bao giờ lộ ra ở phía ghi."""
    factory = sessionmaker(migrated)
    with tenant_session(factory, "acme") as session, pytest.raises(ProgrammingError):
        session.execute(text("INSERT INTO conversation (id, tenant_id) VALUES ('c-y', 'globex')"))
        session.commit()


def test_the_tenant_does_not_leak_into_the_next_transaction(migrated: Engine) -> None:
    """⭐ `SET LOCAL` chứ không `SET SESSION`.

    Connection pool tái dùng kết nối, nên một `SET SESSION` sót lại làm request
    kế tiếp mang tenant của request trước. Đó là rò chéo tenant do một dòng cấu
    hình, và nó chỉ xảy ra **khi có tải** — tức không bao giờ ở máy dev.
    """
    factory = sessionmaker(migrated)
    with tenant_session(factory, "acme") as session:
        session.execute(text("INSERT INTO conversation (id, tenant_id) VALUES ('c-z', 'acme')"))
        session.commit()

    with factory() as bare:
        # Không đặt tenant: `current_setting(..., true)` trả NULL, và
        # `tenant_id = NULL` không đúng với hàng nào.
        assert bare.execute(text("SELECT count(*) FROM conversation")).scalar_one() == 0


# ---------------------------------------------------------------------------
# 3. Phép thử sẵn sàng
# ---------------------------------------------------------------------------


def test_ready_means_migrated_not_merely_reachable(migrated: Engine, owner_engine: Engine) -> None:
    """⭐ `SELECT 1` trả lời "cắm được" — không phải cách hệ thống này hỏng.

    Cách nó hỏng: image mới lên, `alembic upgrade head` chưa chạy, pod báo sẵn
    sàng, nhận traffic, và mọi request chết bằng `column … does not exist`. Một
    `SELECT 1` xanh suốt.
    """
    postgres_check(migrated)  # đang ở head ⇒ không ném

    # Sửa bằng role owner: role ứng dụng chỉ có **SELECT** trên `alembic_version`,
    # và đó là đúng — nó cần *đọc* phiên bản schema, không cần đổi.
    with owner_engine.begin() as conn:
        conn.execute(text("UPDATE alembic_version SET version_num = 'cũ_hơn'"))
    try:
        with pytest.raises(MigrationStateError, match="cũ_hơn"):
            postgres_check(migrated)
    finally:
        with owner_engine.begin() as conn:
            conn.execute(
                text("UPDATE alembic_version SET version_num = :v"),
                {"v": expected_revision()},
            )


def test_the_expected_revision_comes_from_the_migration_folder() -> None:
    """Ghim một hằng số trong mã thì phải sửa tay mỗi lần thêm migration, và lần
    quên đầu tiên biến phép thử sẵn sàng thành một phép thử luôn xanh."""
    assert expected_revision() == "0001_initial"
