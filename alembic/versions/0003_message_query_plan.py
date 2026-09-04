"""Vì sao lượt này truy hồi ra những chunk đó: nhánh nào, và bằng chuỗi nào.

`W4-07` — hai cột trên `message`, đều trên hàng của **người dùng**.

## Vì sao `rewritten_query` phải là một cột chứ không phải một dòng log

Sau `W4-07`, chuỗi đưa vào truy hồi có thể **khác** chuỗi người dùng gõ. Không
ghi lại thì một lượt đã viết lại là không giải thích nổi về sau: `content` ghi
"cái đó thì sao?" còn `citations` nói về di cư lao động, và không có gì trong
database nối hai thứ ấy lại.

Đây đúng lý lẽ đã đưa `model` và `finish_reason` vào `0002`: một thuộc tính chỉ
sống trong log là một thuộc tính biến mất theo chính sách giữ log, và câu hỏi
cần nó luôn được đặt ra muộn hơn thế.

## `route` có `CHECK`, `rewritten_query` thì không

`route` là một tập **đóng** ba giá trị do mã sinh ra; một giá trị thứ tư trong
cột này nghĩa là có đường ghi nào đó không đi qua `QueryUnderstanding`, và đó là
thứ đáng nổ ngay lúc `INSERT` chứ không phải lúc ai đó đọc báo cáo. Nội dung câu
hỏi thì ngược lại — nó là text tự do của người dùng.

## `nullable=True`, **không** backfill

Message ghi trước hạng mục này không đi qua bộ phân loại nào cả. Điền
`route = 'retrieve'` cho chúng thì đúng về mặt hành vi (`W4-06` luôn truy hồi)
nhưng nói dối về nguồn gốc: nó khiến một báo cáo "tỉ lệ lượt bỏ truy hồi" tính
cả những lượt chưa từng có lựa chọn nào. `NULL` = "hàng này có trước `W4-07`".

Revision ID: 0003_message_query_plan
Revises: 0002_message_provenance
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0003_message_query_plan"
down_revision = "0002_message_provenance"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("message", sa.Column("route", sa.String(length=16), nullable=True))
    op.add_column("message", sa.Column("rewritten_query", sa.Text(), nullable=True))
    op.create_check_constraint(
        "ck_message_route",
        "message",
        "route IS NULL OR route IN ('retrieve', 'no_retrieval', 'clarify')",
    )


def downgrade() -> None:
    op.drop_constraint("ck_message_route", "message", type_="check")
    op.drop_column("message", "rewritten_query")
    op.drop_column("message", "route")
