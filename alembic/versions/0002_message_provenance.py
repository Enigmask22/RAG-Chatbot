"""Nguồn gốc của một câu trả lời: model nào sinh, và vì sao nó dừng.

`W4-06` — hai cột trên `message`.

## Vì sao là một migration riêng chứ không sửa `0001`

`0001` đã chạy trên máy này và trên máy CI. Sửa nó tại chỗ nghĩa là hai database
mang cùng một `version_num` mà schema khác nhau — đúng chế độ hỏng mà
`postgres_check` của `W4-05` sinh ra để bắt, chỉ khác là lần này chính tôi tạo
ra nó và phép kiểm sẽ báo **xanh** vì số revision vẫn khớp.

## `nullable=True` cho cả hai, và **không** backfill

Message của người dùng không có model. Message trợ lý đã ghi trước hạng mục này
thì không có cách nào biết model nào sinh ra chúng — và điền một giá trị đoán
vào đó là biến "không biết" thành "biết sai", thứ không phục hồi được. `NULL`
nói đúng cái nó biết.

Revision ID: 0002_message_provenance
Revises: 0001_initial
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0002_message_provenance"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("message", sa.Column("model", sa.String(length=128), nullable=True))
    op.add_column("message", sa.Column("finish_reason", sa.String(length=32), nullable=True))
    # ⚠️ Không cần `GRANT` lại: quyền trong Postgres là quyền trên **bảng**, không
    # trên từng cột, khi `GRANT` được cấp ở mức bảng. Policy RLS của `0001` cũng
    # áp nguyên vẹn — nó nói về `tenant_id`, thứ không đổi.


def downgrade() -> None:
    op.drop_column("message", "finish_reason")
    op.drop_column("message", "model")
