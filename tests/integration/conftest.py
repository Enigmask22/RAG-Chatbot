"""Fixture chung cho integration: `database`/`workspace` của `test_chat_stream`.

Nạp ở conftest thay vì import trong từng file test: một fixture import rồi lại
xuất hiện làm tham số test là F811 (redefinition) với ruff — còn qua conftest
thì pytest tự phân giải theo tên, không cần import nào ở file dùng.
"""

from tests.integration.test_chat_stream import database, workspace

__all__ = ["database", "workspace"]
