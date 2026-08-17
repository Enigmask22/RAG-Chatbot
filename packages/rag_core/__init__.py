"""`rag_core` — thư viện dùng chung giữa Pipeline Plane và Serving Plane.

Quy tắc kiến trúc bắt buộc:

* `pipeline/` và `serving/` đều **được phép** import `rag_core`.
* `rag_core` **không được** import `pipeline` hay `serving`.
* `serving` **không được** import `pipeline` (và ngược lại). Hai plane chỉ nối
  với nhau qua artifact `RagBundle` có version.

Vi phạm chiều phụ thuộc này làm mất ranh giới hai plane — thứ là lý do tồn tại
của cả kiến trúc. Có test canh ở `tests/unit/test_architecture_boundaries.py`.
"""

__version__ = "0.1.0"
