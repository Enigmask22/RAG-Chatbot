"""Prompt sinh câu hỏi cho từng nhóm.

Nguyên tắc chung của cả bộ prompt:

* **Model không được tự bịa `chunk_id`.** Nó chỉ trả về **chỉ số** của chunk
  trong danh sách được đưa; việc ánh xạ sang `chunk_id` thật do code làm. Cho
  model tự viết id là mở đường cho cả một golden set trỏ vào hư không.
* **Bắt buộc có `quote` trích nguyên văn.** Code đối chiếu lại với chunk; không
  khớp nghĩa là model bịa, và câu đó bị đánh dấu để người review đọc trước.
* Câu hỏi phải **đứng một mình được** — người đọc câu hỏi mà không thấy chunk
  vẫn hiểu đang hỏi gì. "Con số này là bao nhiêu?" là câu hỏi vô dụng khi đem đi
  truy hồi, vì nó không mang tín hiệu nào để tìm kiếm.

Prompt viết bằng tiếng Việt vì hệ thống phục vụ người dùng Việt và phần lớn câu
hỏi sẽ là tiếng Việt; phần chunk tiếng Anh vẫn để nguyên trong prompt.
"""

from __future__ import annotations

from collections.abc import Sequence

from pipeline.eval.golden import QueryCategory
from rag_core.schemas import Chunk

__all__ = ["CATEGORY_BRIEF", "build_messages", "render_chunks"]

_SYSTEM = """\
Bạn là kỹ sư đánh giá đang xây tập kiểm thử (golden set) cho một hệ thống RAG \
tiếng Việt. Corpus là báo cáo phát triển của World Bank về Việt Nam, gồm cả bản \
tiếng Việt và tiếng Anh.

Nhiệm vụ: đọc các đoạn văn được đánh số rồi viết câu hỏi dùng để ĐO chất lượng \
truy hồi. Đây không phải bài tập viết câu hỏi hay — nó là dữ liệu đo, nên tính \
chính xác quan trọng hơn sự tự nhiên.

Ràng buộc bắt buộc:
1. Câu hỏi phải ĐỨNG MỘT MÌNH ĐƯỢC: người đọc không nhìn thấy đoạn văn vẫn hiểu \
đang hỏi về cái gì. Cấm dùng "đoạn này", "bảng trên", "con số đó".
2. Câu hỏi phải nêu đủ ngữ cảnh định danh (chủ thể, năm, địa bàn, chỉ tiêu) để \
tìm kiếm được. "Tỷ lệ nghèo là bao nhiêu?" là câu hỏi hỏng; \
"Tỷ lệ nghèo đa chiều của Việt Nam năm 2020 theo World Bank là bao nhiêu?" là được.
3. `quote` phải là đoạn TRÍCH NGUYÊN VĂN, sao chép chính xác từ đoạn văn tương \
ứng, dài 10–40 từ. Không diễn đạt lại, không cắt ghép hai chỗ.
4. `used_chunks` chỉ được chứa số thứ tự của đoạn văn đã cho. Không bịa số khác.
5. Trả về DUY NHẤT một object JSON hợp lệ, không kèm giải thích, không kèm ```.

Định dạng trả về:
{"questions": [{"query": "...", "lang": "vi"|"en", "used_chunks": [1], \
"quote": "...", "reference_answer": "..."}]}
"""

CATEGORY_BRIEF: dict[QueryCategory, str] = {
    QueryCategory.FACTOID: (
        "Nhóm `factoid`: câu hỏi có MỘT đáp án dứt khoát nằm gọn trong một đoạn văn — "
        "một con số, một mốc thời gian, một tên gọi, một định nghĩa. "
        "`used_chunks` có đúng một phần tử."
    ),
    QueryCategory.MULTI_HOP: (
        "Nhóm `multi_hop`: câu hỏi CHỈ trả lời được khi ghép thông tin từ CẢ HAI đoạn văn. "
        "Kiểm tra lại: nếu chỉ đọc một trong hai đoạn mà đã trả lời được thì câu đó SAI nhóm, "
        "phải viết lại. `used_chunks` có đúng hai phần tử."
    ),
    QueryCategory.AGGREGATION: (
        "Nhóm `aggregation`: câu hỏi đòi tổng hợp nhiều mẩu rời rạc thành một câu trả lời — "
        "liệt kê đầy đủ, so sánh nhiều đối tượng, hoặc tóm tắt xu hướng qua nhiều giai đoạn. "
        "Không phải phép cộng số học đơn thuần. Dùng từ hai đoạn văn trở lên."
    ),
    QueryCategory.TABLE_LOOKUP: (
        "Nhóm `table_lookup`: câu hỏi tra một ô dữ liệu cụ thể trong bảng biểu — "
        "giao của một hàng (đối tượng) và một cột (năm/chỉ tiêu). "
        "NẾU đoạn văn không chứa bảng hay chuỗi số liệu có cấu trúc thì TRẢ VỀ DANH SÁCH RỖNG, "
        "đừng cố nặn ra câu hỏi."
    ),
    QueryCategory.CROSS_LINGUAL: (
        "Nhóm `cross_lingual`: đoạn văn ở một ngôn ngữ, câu hỏi viết ở ngôn ngữ CÒN LẠI. "
        "Đoạn tiếng Anh thì hỏi bằng tiếng Việt, và ngược lại. "
        "Giữ nguyên tên riêng và số liệu. `lang` là ngôn ngữ của CÂU HỎI."
    ),
    QueryCategory.UNANSWERABLE: (
        "Nhóm `unanswerable`: câu hỏi NGHE NHƯ thuộc cùng chủ đề với đoạn văn nhưng corpus "
        "chắc chắn KHÔNG trả lời được — hỏi về năm nằm ngoài phạm vi báo cáo, về một quốc gia "
        "khác, về dự báo tương lai, hoặc về một chỉ tiêu không hề được nhắc tới. "
        "Câu hỏi vẫn phải hợp lý, không được vô lý lộ liễu. "
        "`used_chunks` phải là DANH SÁCH RỖNG và `quote` là chuỗi rỗng."
    ),
    QueryCategory.ADVERSARIAL: (
        "Nhóm `adversarial`: câu hỏi cài sẵn một TIỀN ĐỀ SAI mà đoạn văn bác bỏ được — "
        "sai con số, sai năm, sai chủ thể, hoặc khẳng định điều ngược lại với văn bản. "
        "Hệ thống tốt phải tìm ra đúng đoạn văn này rồi đính chính. "
        "`reference_answer` phải nêu rõ tiền đề sai ở chỗ nào."
    ),
}


def render_chunks(chunks: Sequence[Chunk], *, max_chars: int = 3000) -> str:
    """Đánh số đoạn văn từ 1 — chính là chỉ số model sẽ trả về ở `used_chunks`."""
    parts: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        lang = chunk.metadata.lang.value if chunk.metadata is not None else "unknown"
        title = (chunk.metadata.title if chunk.metadata is not None else None) or "(không rõ)"
        body = chunk.content.strip()[:max_chars]
        parts.append(f"[{index}] (ngôn ngữ: {lang} · tài liệu: {title})\n{body}")
    return "\n\n".join(parts)


def build_messages(
    category: QueryCategory,
    chunks: Sequence[Chunk],
    *,
    n_questions: int = 2,
) -> tuple[str, str]:
    """Trả về `(system, user)`. Tách ra để test kiểm được nội dung prompt."""
    brief = CATEGORY_BRIEF[category]
    user = (
        f"{brief}\n\n"
        f"Hãy viết {n_questions} câu hỏi thuộc nhóm `{category.value}` "
        f"dựa trên các đoạn văn dưới đây.\n\n"
        f"{render_chunks(chunks)}\n\n"
        f"Nhắc lại: trả về DUY NHẤT một object JSON dạng "
        f'{{"questions": [...]}}, mỗi phần tử có đủ các khoá '
        f"`query`, `lang`, `used_chunks`, `quote`, `reference_answer`."
    )
    return _SYSTEM, user
