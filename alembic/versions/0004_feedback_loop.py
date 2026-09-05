"""Vòng phản hồi: khoá nối tới trace, và hai cái tên bị gộp làm một.

`W5-08` — bốn thay đổi trên `message` + `feedback`.

## ⭐⭐ Cột `citations` của `0001` không giữ citations

`serving/core/chat.py` mở đầu bằng một cảnh báo tôi tự viết ở `W4-06`:

    Đó là lý do khung ở đây tên là `sources` (cái đã đưa cho model) chứ không
    phải `citations` (cái đã kiểm) — hai thứ khác nhau, và gộp tên chúng lại là
    cách chắc chắn để `W4-09` trở nên vô hình với client.

Rồi `_save()` ghi `citations=turn.sources()`. Đúng cái bẫy vừa mô tả, cách chỗ
mô tả nó một nghìn dòng — và không có gì bắt được, vì một cột JSONB nhận mọi
hình dạng.

Hệ quả không phải là thẩm mỹ. `W4-09` xác minh từng quote rồi phát kết quả ra
khung SSE, và **kết quả ấy chưa từng được ghi xuống**: một câu trả lời đọc lại
từ lịch sử trông y hệt nhau dù mọi citation của nó là thật hay bịa. Đó là nội
dung thật của `TD-50`, và hạng mục này là chỗ nó phải đóng — bởi vì phân biệt
"model bịa nguồn" với "truy hồi lấy nhầm tài liệu" là câu hỏi đầu tiên người
review một câu 👎 phải trả lời.

Nên: `citations` → **`retrieved_sources`** (đúng thứ nó vẫn chứa), và cột
`citations_verified` mới cho khung của `W4-09`.

⚠️ **Không backfill, và lần này vì lý do mạnh hơn `0002`/`0003`.** Ở đó backfill
là *đoán*; ở đây nó **bất khả**: xác minh cần nguyên văn chunk tại thời điểm trả
lời, mà chunk ấy có thể đã bị reindex. `NULL` = "hàng này có trước `0004`".

## `trace_id` trên `message`, và vì sao nó không phải một tham số của API

Điểm số Langfuse gắn theo `traceId`. Nếu client gửi `trace_id` kèm feedback thì
một tenant gắn được điểm vào trace của tenant khác — `TD-73` đã ghi rằng tenant
trong Langfuse là một *nhãn*, không phải một *hàng rào*, nên bên kia không có
gì chặn lại. Cột này làm cho khoá nối được **suy ra từ một hàng mà người gọi
chứng minh được là của mình** (RLS đã lọc), thay vì được người gọi khai.

## `feedback`: một hàng cho mỗi (tenant, message)

Không có ràng buộc duy nhất thì một cú double-click là hai hàng, và hàng đợi
review là một phép đếm. Upsert ở tầng ứng dụng cần đúng chỉ mục này để bám vào.

⚠️ Khoá là `(tenant_id, message_id)` chứ không phải người dùng, vì mô hình xác
thực của `W4-04` chỉ biết tới tenant. Một tenant nhiều người dùng ⇒ ghi đè lẫn
nhau. Ghi vào `TD-79` chứ không giả vờ rằng cột `user_id` không tồn tại là một
lựa chọn thiết kế.

Revision ID: 0004_feedback_loop
Revises: 0003_message_query_plan
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision = "0004_feedback_loop"
down_revision = "0003_message_query_plan"
branch_labels = None
depends_on = None

REASONS = ("wrong", "incomplete", "not_found", "citation", "language", "slow", "other")
"""Tập **đóng**, và mỗi giá trị chỉ vào một bộ phận cụ thể của pipeline.

Đó là tiêu chí để một mã lý do đáng tồn tại: `"wrong"` → bộ sinh, `"not_found"`
→ truy hồi, `"citation"` → `W4-09`, `"language"` → prompt, `"slow"` → serving.
Một mã không chỉ được vào đâu thì nó chỉ là `"other"` viết dài hơn.
"""

_CHECK = "reason IS NULL OR reason IN (" + ", ".join(f"'{r}'" for r in REASONS) + ")"


def upgrade() -> None:
    op.alter_column("message", "citations", new_column_name="retrieved_sources")
    op.add_column("message", sa.Column("citations_verified", postgresql.JSONB, nullable=True))
    op.add_column("message", sa.Column("trace_id", sa.String(length=32), nullable=True))

    op.add_column("feedback", sa.Column("reason", sa.String(length=32), nullable=True))
    op.create_check_constraint("ck_feedback_reason", "feedback", _CHECK)
    # ⚠️ `UNIQUE` chứ không phải một `INSERT ... WHERE NOT EXISTS` ở tầng ứng
    # dụng: hai request đồng thời của cùng một người (một cú double-click **là**
    # hai request) đi qua phép kiểm ấy cùng lúc và cả hai đều ghi.
    op.create_index(
        "uq_feedback_tenant_message", "feedback", ["tenant_id", "message_id"], unique=True
    )


def downgrade() -> None:
    op.drop_index("uq_feedback_tenant_message", table_name="feedback")
    op.drop_constraint("ck_feedback_reason", "feedback", type_="check")
    op.drop_column("feedback", "reason")

    op.drop_column("message", "trace_id")
    op.drop_column("message", "citations_verified")
    op.alter_column("message", "retrieved_sources", new_column_name="citations")
