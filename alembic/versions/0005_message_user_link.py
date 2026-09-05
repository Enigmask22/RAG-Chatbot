"""Khoá nối câu hỏi → câu trả lời, thay cho suy luận theo thời gian.

`NEW-08`/`AU-07` — một cột trên `message`.

## Vì sao `created_at` không phải một khoá nối

`_questions_for` của `W5-08` ghép "câu hỏi đứng ngay trước mỗi câu trả lời"
bằng cách chọn user message **muộn nhất** có `created_at <=` answer. Suy luận
ấy đúng khi các lượt tuần tự tuyệt đối — và thứ tự ghi của hệ thống này không
tuần tự: hàng user ghi **ngay** trong `_open_turn()`, còn hàng assistant ghi
trong một task nền **sau khi toàn bộ stream kết thúc**, tức vài giây sau. Hai
request cùng `conversation_id` chạy chồng nhau (đa tab, client retry) là đủ để
user_B chen vào giữa user_A và assistant_A — và ứng viên golden xuất ra mang
**câu hỏi B dán lên câu trả lời của A**, sai không dấu vết vì JSONB không kiểm
được logic này.

Khoá thật đã tồn tại từ `W4-06`: `ChatTurn.user_message_id`. Cột này chỉ là
việc ghi nó xuống hàng assistant lúc `_save()`, để `_questions_for` join theo
id thay vì đoán theo đồng hồ.

⚠️ **Không backfill.** Với các hàng cũ, thông tin "câu trả lời này trả lời câu
hỏi nào" chỉ tồn tại dưới dạng đúng cái suy luận thời gian mà migration này
thay thế — backfill bằng suy luận ấy là đóng dấu "khoá thật" lên một phép đoán.
`NULL` = "hàng ghi trước `0005`, ghép theo đường cũ và mang theo rủi ro cũ".
"""

from __future__ import annotations

import sqlalchemy as sa

from alembic import op

revision = "0005_message_user_link"
down_revision = "0004_feedback_loop"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("message", sa.Column("user_message_id", sa.String(length=32), nullable=True))


def downgrade() -> None:
    op.drop_column("message", "user_message_id")
