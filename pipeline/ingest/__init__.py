"""API điều khiển ingestion + worker chạy nền. `W3-08`.

## Vì sao nằm ở `pipeline/`, không phải `serving/`

`serving/` là **Serving Plane**: đường phục vụ truy vấn của người dùng cuối, và
theo mệnh lệnh kiến trúc của dự án nó chỉ được nối với Pipeline Plane qua một
`RagBundle` bất biến có version. Một endpoint **ghi vào index** không thuộc về
đó — nó là điều khiển từ xa của chính pipeline. Đặt nhầm chỗ thì ranh giới mà cả
kiến trúc dựa vào bị nhoè ngay ở hạng mục đầu tiên chạm tới HTTP.

Nên: đây là *ingestion control plane*. API truy vấn của `W4` sẽ nằm ở `serving/`
và **không** import gì từ đây.

## Vì sao `arq` chứ không Celery

Ba lý do cụ thể ở chỗ này: nó là asyncio thuần nên dùng chung vòng lặp với
FastAPI thay vì thêm một mô hình đồng thời thứ hai; nó chỉ cần Redis, thứ
`docker-compose` đã có, thay vì thêm một broker; và nó đủ nhỏ để đọc hết khi cần
biết **chính xác** hành vi retry — thứ mà DoD của `W3-08` yêu cầu chứng minh.

## Retry rẻ là nhờ `W3-07`

Job bị chạy lại từ đầu khi worker chết. Điều đó chỉ chấp nhận được vì
`build_index` idempotent (ba tầng, xem docstring của nó) **và** vì `W3-07` khiến
lượt chạy lại chỉ embed phần thật sự đổi: một job chết ở 90% khi chạy lại không
trả lại tiền embed của 90% đã xong. Không có `W3-07` thì "retry" nghĩa là "làm
lại 376 giây".
"""

from __future__ import annotations

__all__ = ["IngestRequest", "JobState", "JobStatus"]

from .schemas import IngestRequest, JobState, JobStatus
