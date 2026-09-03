"""Schema Postgres của Serving Plane — `W4-05`.

## ⭐⭐ Chỗ ép tenant, lần thứ ba, và lần này ở một tầng khác

`W2-06` ghi rằng `rag_core` không ép được `tenant_id`; `W4-04` đóng chỗ đó ở tầng
HTTP bằng `tenant_filter()`. Postgres là **lỗ thứ ba của cùng một họ**, và nó
hỏng theo cách y hệt: một câu `SELECT * FROM message WHERE conversation_id = :id`
quên `AND tenant_id = :tenant` trả về hội thoại của khách hàng khác, và nó *chạy
đúng* trên môi trường dev một-tenant.

Cách chặn ở SQL là **Row-Level Security**, và lý lẽ giống hệt lý lẽ
middleware-vs-`Depends` của `W4-04`: RLS chặn theo mặc định ở tầng engine, nên
quên `WHERE` trả về **rỗng** thay vì trả về tất cả. Hướng hỏng đảo chiều, và đó
là toàn bộ giá trị.

⚠️ **Cái bẫy làm cho RLS thành trang trí:** chủ sở hữu bảng **được miễn** mọi
policy. Ứng dụng kết nối bằng `rag` — chính là owner — nên `ENABLE ROW LEVEL
SECURITY` một mình không chặn gì cả, trong khi `pg_tables.rowsecurity` vẫn báo
`true` và mọi phép kiểm cấu hình vẫn xanh. Phải là **`FORCE ROW LEVEL
SECURITY`**. Có test ghim đúng chỗ này (`test_the_owner_does_not_bypass_rls`).

Tenant hiện tại đọc từ tham số phiên `app.tenant_id`, đặt bằng `SET LOCAL` trong
mỗi transaction — `LOCAL` chứ không `SESSION`, vì connection pool tái dùng kết
nối và một `SET SESSION` sót lại sẽ cho request kế tiếp mang tenant của request
trước.

## Hai quyết định nhỏ dễ làm sai

* **`timestamptz`, không `timestamp`.** `timestamp` bỏ lặng lẽ phần offset khi
  ghi, nên hai request từ hai múi giờ ghi cùng một thời điểm ra hai giá trị khác
  nhau, và không có lỗi nào.
* **`ingest_job` ở đây KHÔNG thay Redis của `W3-08`.** Redis giữ *hàng đợi* và
  trạng thái đang chạy (ephemeral, và `arq` mới là chủ của nó); bảng này giữ
  *bản ghi bền* để trả lời "tài liệu này vào index lúc nào, bằng config nào" sau
  khi Redis đã hết TTL. Nguồn sự thật khi hai bên lệch nhau là **Redis** trong
  lúc job còn chạy, và **Postgres** sau khi nó kết thúc.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

__all__ = [
    "RLS_TABLES",
    "TENANT_SETTING",
    "Base",
    "Conversation",
    "Document",
    "Feedback",
    "IngestJob",
    "Message",
]

TENANT_SETTING = "app.tenant_id"
"""Tham số phiên mà policy RLS đọc. `SET LOCAL`, không `SET SESSION` — xem docstring."""

RLS_TABLES = ("conversation", "message", "document", "ingest_job", "feedback")
"""Mọi bảng mang dữ liệu của khách hàng.

Là một hằng số chứ không phải một danh sách trong migration, vì
`test_every_tenant_table_has_forced_rls` duyệt nó: một bảng mới thêm vào model mà
quên bật RLS sẽ đỏ, thay vì lặng lẽ thành bảng công khai.
"""


class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return uuid.uuid4().hex


class _Tenanted:
    """Phần chung của mọi bảng có chủ.

    `tenant_id` **không** nullable và **không** có mặc định: một mặc định ở đây
    (kể cả `'default'`) nghĩa là một `INSERT` quên tenant vẫn thành công, và
    hàng đó sau đó vô hình với mọi policy — dữ liệu tồn tại mà không ai đọc được.
    """

    tenant_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class Conversation(Base, _Tenanted):
    __tablename__ = "conversation"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    title: Mapped[str | None] = mapped_column(String(512))
    bundle_version: Mapped[str | None] = mapped_column(String(64))
    """Bundle đang phục vụ lúc hội thoại bắt đầu.

    ⭐ Ghi lại để một câu trả lời cũ còn giải thích được: sau một lần hot-swap
    (`W4-02`) hoặc rollback, cùng một câu hỏi cho câu trả lời khác, và nếu không
    có trường này thì không cách nào biết bản nào đã trả lời.
    """

    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", passive_deletes=True
    )


class Message(Base, _Tenanted):
    __tablename__ = "message"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    conversation_id: Mapped[str] = mapped_column(
        # `ondelete="CASCADE"` ở tầng DB, không chỉ ở tầng ORM: xoá một hội thoại
        # bằng SQL trực tiếp (script vận hành, yêu cầu GDPR) phải cũng dọn message,
        # nếu không thì "đã xoá" chỉ đúng khi việc xoá đi qua Python.
        ForeignKey("conversation.id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[list[dict[str, object]] | None] = mapped_column(JSONB)
    """Citation dạng JSONB, không phải bảng riêng — chúng luôn được đọc **cùng**
    message và không bao giờ được truy vấn độc lập. `W4-09` xác minh chúng trước
    khi ghi, nên cái nằm ở đây đã là dữ liệu đã kiểm."""

    latency_ms: Mapped[int | None] = mapped_column(Integer)
    conversation: Mapped[Conversation] = relationship(back_populates="messages")

    __table_args__ = (
        CheckConstraint("role IN ('user', 'assistant', 'system')", name="ck_message_role"),
        # Đúng thứ tự mà đường đọc lịch sử cần: một hội thoại của một tenant, theo
        # thời gian. Thiếu index này thì mỗi lần mở hội thoại là một seq scan trên
        # toàn bộ bảng message của mọi khách hàng.
        Index("ix_message_history", "tenant_id", "conversation_id", "created_at"),
    )


class Document(Base, _Tenanted):
    __tablename__ = "document"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    """`doc_id` — **cùng** giá trị với `doc_id` trong payload Qdrant.

    ⚠️ Không sinh id riêng ở đây: hai không gian id cho cùng một tài liệu là chỗ
    mà citation trỏ vào một hàng không tồn tại, và phép nối sẽ trông đúng cho tới
    khi có tài liệu thứ hai cùng tên.
    """

    title: Mapped[str | None] = mapped_column(String(1024))
    source_uri: Mapped[str | None] = mapped_column(Text)
    lang: Mapped[str | None] = mapped_column(String(8))
    n_chunks: Mapped[int | None] = mapped_column(Integer)
    meta: Mapped[dict[str, object] | None] = mapped_column(JSONB)


class IngestJob(Base, _Tenanted):
    __tablename__ = "ingest_job"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    """Cùng `job_id` với bản ghi trong Redis của `W3-08` — xem docstring module."""

    state: Mapped[str] = mapped_column(String(16), nullable=False)
    config: Mapped[str | None] = mapped_column(String(128))
    n_documents: Mapped[int | None] = mapped_column(Integer)
    error: Mapped[str | None] = mapped_column(Text)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Feedback(Base, _Tenanted):
    __tablename__ = "feedback"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_uuid)
    message_id: Mapped[str] = mapped_column(
        ForeignKey("message.id", ondelete="CASCADE"), nullable=False
    )
    rating: Mapped[int] = mapped_column(Integer, nullable=False)
    comment: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (
        # Chỉ `-1` và `1`. Thang 1–5 nghe giàu thông tin hơn nhưng nó cho ra một
        # phân bố dồn ở hai đầu mà không ai gán nhãn được ngưỡng, còn thumbs
        # up/down ghép thẳng được vào bộ eval của `W5` như nhãn nhị phân.
        CheckConstraint("rating IN (-1, 1)", name="ck_feedback_rating"),
        Index("ix_feedback_message", "tenant_id", "message_id"),
    )
