"""schema ban đầu + RLS cưỡng chế

Revision ID: 0001_initial
Revises:
Create Date: 2026-09-03

## ⭐⭐ Vì sao `FORCE ROW LEVEL SECURITY` chứ không chỉ `ENABLE`

`ENABLE ROW LEVEL SECURITY` bật policy cho **mọi role trừ chủ sở hữu bảng**.
Ứng dụng ở đây kết nối bằng `rag`, chính là owner. Nên chỉ `ENABLE` thì:

* `pg_tables.rowsecurity` trả `true`;
* mọi phép kiểm cấu hình, mọi dashboard, mọi checklist audit đều xanh;
* và **không một policy nào được áp dụng** cho đường mà ứng dụng thật đi.

Đó là loại lỗ hổng tệ nhất của cả dự án này: nó tự báo cáo là đã đóng. `FORCE`
bắt owner cũng phải qua policy. Có test ghim (`test_the_owner_does_not_bypass_rls`).

## Vì sao `downgrade` có thật và bị chạy trong CI

`downgrade` mà không ai chạy là một hàm chưa từng được biên dịch. `W4-05` bắt
up → down → up.

⚠️ Nhưng phải nói rõ nó chứng minh cái gì: chạy trên **DB rỗng** thì nó chứng
minh **DDL đảo được**, chứ không chứng minh **deploy lùi lại được** — lùi một
migration có dữ liệu là mất dữ liệu, và không phép kiểm nào ở đây nói khác.
"""

from __future__ import annotations

import re

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op
from rag_core.settings import get_settings
from serving.db.models import RLS_TABLES, TENANT_SETTING


def _app_role() -> str:
    """Tên role, đã kiểm là một identifier SQL an toàn.

    Nó bị **nội suy vào SQL** — `CREATE ROLE` và `GRANT` không nhận placeholder —
    và nó đến từ biến môi trường. "Cấu hình của chính mình" không phải một lý do
    để bỏ phép kiểm: một `POSTGRES_APP_USER` gõ nhầm dấu nháy sẽ cho một lỗi cú
    pháp ở giữa migration, tức schema dừng lại ở nửa đường.
    """
    name = get_settings().postgres_app_user
    if not re.fullmatch(r"[a-z_][a-z0-9_]{0,62}", name):
        raise ValueError(f"`POSTGRES_APP_USER` phải là identifier SQL viết thường: {name!r}")
    return name


_APP_ROLE = _app_role()

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None

_TS = sa.DateTime(timezone=True)


def upgrade() -> None:
    op.create_table(
        "conversation",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("created_at", _TS, server_default=sa.func.now(), nullable=False),
        sa.Column("title", sa.String(512)),
        sa.Column("bundle_version", sa.String(64)),
    )
    op.create_index("ix_conversation_tenant_id", "conversation", ["tenant_id"])

    op.create_table(
        "message",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("created_at", _TS, server_default=sa.func.now(), nullable=False),
        sa.Column(
            "conversation_id",
            sa.String(32),
            sa.ForeignKey("conversation.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("citations", postgresql.JSONB),
        sa.Column("latency_ms", sa.Integer),
        sa.CheckConstraint("role IN ('user', 'assistant', 'system')", name="ck_message_role"),
    )
    op.create_index("ix_message_tenant_id", "message", ["tenant_id"])
    op.create_index("ix_message_history", "message", ["tenant_id", "conversation_id", "created_at"])

    op.create_table(
        "document",
        sa.Column("id", sa.String(64), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("created_at", _TS, server_default=sa.func.now(), nullable=False),
        sa.Column("title", sa.String(1024)),
        sa.Column("source_uri", sa.Text),
        sa.Column("lang", sa.String(8)),
        sa.Column("n_chunks", sa.Integer),
        sa.Column("meta", postgresql.JSONB),
    )
    op.create_index("ix_document_tenant_id", "document", ["tenant_id"])

    op.create_table(
        "ingest_job",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("created_at", _TS, server_default=sa.func.now(), nullable=False),
        sa.Column("state", sa.String(16), nullable=False),
        sa.Column("config", sa.String(128)),
        sa.Column("n_documents", sa.Integer),
        sa.Column("error", sa.Text),
        sa.Column("finished_at", _TS),
    )
    op.create_index("ix_ingest_job_tenant_id", "ingest_job", ["tenant_id"])

    op.create_table(
        "feedback",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("tenant_id", sa.String(64), nullable=False),
        sa.Column("created_at", _TS, server_default=sa.func.now(), nullable=False),
        sa.Column(
            "message_id",
            sa.String(32),
            sa.ForeignKey("message.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("rating", sa.Integer, nullable=False),
        sa.Column("comment", sa.Text),
        sa.CheckConstraint("rating IN (-1, 1)", name="ck_feedback_rating"),
    )
    op.create_index("ix_feedback_tenant_id", "feedback", ["tenant_id"])
    op.create_index("ix_feedback_message", "feedback", ["tenant_id", "message_id"])

    _create_app_role()

    for table in RLS_TABLES:
        op.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        # ⚠️ Dòng dưới là dòng duy nhất làm cho RLS có tác dụng thật ở đây —
        # xem docstring module.
        op.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY {table}_tenant ON {table} "
            f"USING (tenant_id = current_setting('{TENANT_SETTING}', true)) "
            f"WITH CHECK (tenant_id = current_setting('{TENANT_SETTING}', true))"
        )
        # `USING` gác đọc/xoá, `WITH CHECK` gác ghi.
        #
        # ⚠️ Viết tường minh dù nó **không đổi hành vi hôm nay**: bỏ `WITH CHECK`
        # thì Postgres dùng chính biểu thức `USING` cho chiều ghi. Tôi đã tưởng
        # ngược lại và viết một chú thích sai ở đây; phép tiêm lỗi (`xoá WITH
        # CHECK` → không test nào đỏ) mới là thứ chỉ ra điều đó.
        #
        # Giữ lại vì nó là **hàng rào cho tương lai**: ngày nào `USING` được nới
        # ra (ví dụ cho một role đọc chéo tenant để báo cáo), thiếu `WITH CHECK`
        # sẽ nới luôn cả chiều ghi — im lặng.
        op.execute(f"GRANT SELECT, INSERT, UPDATE, DELETE ON {table} TO {_APP_ROLE}")


def _create_app_role() -> None:
    """⭐⭐ Role riêng cho ứng dụng, và đây là dòng làm RLS có thật.

    `POSTGRES_USER` của image postgres là **superuser**, và superuser bỏ qua RLS
    **hoàn toàn** — kể cả `FORCE`. Nên chừng nào ứng dụng còn dùng role ấy thì
    năm policy ở trên là trang trí, trong khi `relforcerowsecurity` vẫn báo
    `true`.

    Đo được, không suy ra: bốn test **hành vi** đỏ trong khi năm test **cấu
    hình** xanh. Đó là toàn bộ lý do có cả hai loại test.

    `NOSUPERUSER NOBYPASSRLS` viết tường minh dù đó là mặc định — hai thuộc tính
    này là *lý do tồn tại* của role, không phải chi tiết cài đặt.
    """
    settings = get_settings()
    password = settings.postgres_app_password.get_secret_value().replace("'", "''")
    op.execute(
        f"""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{_APP_ROLE}') THEN
                CREATE ROLE {_APP_ROLE} LOGIN NOSUPERUSER NOBYPASSRLS
                    PASSWORD '{password}';
            ELSE
                ALTER ROLE {_APP_ROLE} LOGIN NOSUPERUSER NOBYPASSRLS
                    PASSWORD '{password}';
            END IF;
        END
        $$;
        """
    )
    op.execute(f"GRANT USAGE ON SCHEMA public TO {_APP_ROLE}")
    # ⭐ Chỉ **SELECT**, và chỉ trên bảng này. `/ready` phải trả lời được "schema
    # trong DB có đúng schema mã này giả định không" (xem `serving/db/engine.py`),
    # và câu đó không trả lời được nếu tiến trình serving không đọc nổi
    # `alembic_version`. Lối sai là cho serving dùng DSN của owner — tức mang
    # credential superuser vào tiến trình nhận request từ Internet.
    op.execute(f"GRANT SELECT ON alembic_version TO {_APP_ROLE}")


def downgrade() -> None:
    # Ngược đúng thứ tự tạo: `feedback` → `message` → `conversation`, vì khoá
    # ngoại. Thả bảng cũng thả policy và grant của nó, nên không cần dọn riêng.
    for table in ("feedback", "ingest_job", "document", "message", "conversation"):
        op.drop_table(table)
    # Có điều kiện: `downgrade` phải chạy được cả trên một DB được migrate bởi
    # **phiên bản cũ hơn** của chính revision này (chưa tạo role). Một
    # `downgrade` chỉ chạy được trên đúng một lịch sử là một `downgrade` không
    # dùng được lúc cần.
    op.execute(
        f"""
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{_APP_ROLE}') THEN
                REVOKE USAGE ON SCHEMA public FROM {_APP_ROLE};
            END IF;
        END
        $$;
        """
    )
    # ⚠️ **Không** `DROP ROLE`. Role là đối tượng của cả cluster, không của một
    # database: thả nó ở đây làm hỏng mọi database khác đang dùng chung nó, và
    # `downgrade` của một schema không có quyền quyết định chuyện ngoài schema.
