"""Cross-encoder reranker — `bge-reranker-v2-m3`. `W2-05`.

Bốn quyết định của tầng này:

**Cùng họ với embedding model.** `bge-reranker-v2-m3` dùng chính backbone
XLM-RoBERTa của `bge-m3` (`W2-01`), nên nó đa ngữ theo cùng một cách và không
mang thêm một tokenizer thứ hai vào hệ thống.

⚠️ Tôi viết ở đây rằng "đây cũng là lý do nó **không** sửa được `TD-18`: cùng vocab
subword thì cùng xé `P171645`". **`W2-05` §7 đã phản chứng.** Nó sửa được phần
lớn: known-item hit@1 0,0980 → **0,5490**, và nó thắng cả nhánh sparse. Chỗ tôi
lẫn là giữa **truy hồi** một mã và **nhận ra** một mã: điểm sparse là tích vô
hướng trên *túi* subword nên `['▁P','171','645']` mất hết thông tin thứ tự và liền
kề, còn cross-encoder có attention trên **cả cặp** nên nó thấy ba mảnh ấy xuất
hiện liền nhau, đúng thứ tự. Cùng vocab, hai bài toán khác nhau. Phần `TD-18` còn
lại là 35% mã **không vào được pool**, và đó là giới hạn của `candidates`, không
phải của tokenizer.

**Logit thô, không sigmoid.** Sigmoid đơn điệu nên nó không thêm gì cho việc xếp
hạng, và bỏ một phép biến đổi là bỏ một chỗ có thể sai. Đó là toàn bộ lý do, và
nó cố tình yếu — xem cảnh báo dưới. Có test canh `activation="sigmoid"` cho cùng
thứ tự.

⚠️ Lý do **ban đầu** tôi viết ở đây mạnh hơn: "sigmoid bão hoà, `float32` cho
`sigmoid(x) == 1.0` từ khoảng `x > 17`, nên nó sinh ties nhân tạo ở đúng chỗ quan
trọng nhất". `W2-05` đã **đo** và phản chứng: 2000 logit thật trên corpus này nằm
trong khoảng `[−10,87 ; +8,67]`, **0,0%** vượt ngưỡng bão hoà. Câu đó không sai về
nguyên lý — sigmoid *có* bão hoà — nhưng tôi đã dùng một sự thật về kiểu số để kết
luận về một phân bố chưa đo. Ghi lại nguyên văn thay vì xoá, vì cái sai ở đây là
về **cách lập luận**, không phải về mặc định. Xem `reports/w2-05-reranker.md` §4.

**`max_length` nằm trong `name`, `batch_size` thì không.** Cùng lý lẽ với
`IndexConfig.fingerprint` (`W1-06`): `max_length` **cắt nội dung** nên nó đổi
điểm, còn `batch_size` chỉ đổi tốc độ. Một cần điều khiển đổi kết quả mà không
xuất hiện trong nhãn là cách chắc chắn để bảng ablation `W2-08` có hai dòng
trùng tên mà khác số.

**Mặc định `max_length=512`, không phải 8192.** Cửa sổ của model là 8192 nhưng
cặp (truy vấn, chunk 1000 ký tự) chỉ khoảng 300–400 token, và chi phí
attention là bậc hai theo độ dài. `count_pair_tokens` tồn tại để **đo** tỉ lệ bị
cắt thay vì tin — đúng bài học `TD-11`, nơi một giả định về truncation không được
đo đã dẫn cả tuần đi sai hướng.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast

from ..embedding.huggingface import resolve_device
from .base import Reranker

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

__all__ = [
    "BGE_RERANKER_V2_M3",
    "DEFAULT_RERANK_BATCH_SIZE",
    "DEFAULT_RERANK_MAX_LENGTH",
    "CrossEncoderReranker",
]

BGE_RERANKER_V2_M3 = "BAAI/bge-reranker-v2-m3"

#: Độ chính xác cho phép. `bfloat16` có mặt vì Ampere trở lên chạy nó tốt hơn
#: fp16 về ổn định số, nhưng 4060 (Ada) chạy cả hai như nhau.
SUPPORTED_DTYPES = frozenset({"float16", "bfloat16", "float32"})

#: Số cặp mỗi forward pass. Chỉ là cần điều khiển tốc độ — có test canh nó không
#: làm đổi điểm, nên nó **không** vào `name`.
DEFAULT_RERANK_BATCH_SIZE = 16

#: Trần token cho **cả cặp** (truy vấn + chunk + special token). Xem docstring
#: module: đây là cần điều khiển làm đổi kết quả, nên nó vào `name`.
DEFAULT_RERANK_MAX_LENGTH = 512


def _resolve_dtype(dtype: str | None, device: str) -> str | None:
    """`auto` = fp16 trên CUDA, fp32 ở nơi khác.

    Vì sao mặc định là `auto` chứ không phải fp32 cho chắc: đo được fp32 tốn
    **1794,7 ms** để chấm 50 cặp trên 4060 (corpus thật, p50), tức gấp 4,5 lần
    ngân sách 400 ms của DoD `W2-05`; fp16 đưa xuống **510,1 ms**, tức **3,52×**.
    Đó là cần điều khiển duy nhất thu hẹp được khoảng đó mà không phải bỏ ứng
    viên — `max_length` chỉ cắt 1/12.100 cặp và `batch_size` không mua được gì
    (`reports/w2-05-reranker.md` §3, §5.3). Cái giá là điểm khác đi ở chữ số thấp
    (trùng top-1 98,3%, lệch max 0,08% khoảng logit), nên nó nằm trong `name`.

    CPU cố ý **không** dùng fp16: hầu hết CPU không có kernel fp16 nên PyTorch
    lùi về emulate và chạy *chậm hơn* fp32 — nhanh trên GPU không suy ra nhanh
    ở mọi nơi.
    """
    if dtype != "auto":
        return dtype
    return "float16" if device.startswith("cuda") else None


@lru_cache(maxsize=2)
def _load_cross_encoder(
    model_name: str, device: str, max_length: int, dtype: str | None
) -> CrossEncoder:
    """Nạp một lần cho mỗi bộ bốn `(model, device, max_length, dtype)`.

    Giới hạn 2 chứ không phải 4 như `_load_model` của embedding: trên 4060 8GB thì
    `bge-m3` đã chiếm ~3,3GB và reranker fp32 thêm ~2,3GB nữa.

    ⚠️ **Ngay cả 2 cũng là một lời hứa phần cứng này không giữ được.** Phiên pytest
    đầy đủ của `W2-05` đã **OOM thật**: bge-m3 3,3GB + fp16 1,15GB + fp32 2,3GB +
    một bản `max_length=32` 2,3GB = 9,05GB trên GPU 8,0GB — và `max_length` nằm
    trong khoá cache nên bản thứ ba ấy là *một model nữa*, không phải cùng model
    chạy khác cấu hình. Cách sửa ở tầng test là đưa bản đó sang CPU; cách sửa đúng
    là `W0-06` đo ngân sách theo **tổ hợp model đồng thời**, không theo từng model.

    `dtype` nằm trong khoá cache vì bản fp16 và bản fp32 là **hai** model khác
    nhau về số, không phải một model chạy hai chế độ.
    """
    import torch
    from sentence_transformers import CrossEncoder

    kwargs: dict[str, Any] = {}
    if dtype is not None:
        kwargs["model_kwargs"] = {"torch_dtype": getattr(torch, dtype)}
    return cast(
        "CrossEncoder", CrossEncoder(model_name, device=device, max_length=max_length, **kwargs)
    )


class CrossEncoderReranker(Reranker):
    """Chấm cặp (truy vấn, chunk) bằng một cross-encoder HuggingFace."""

    def __init__(
        self,
        model_name: str = BGE_RERANKER_V2_M3,
        *,
        device: str = "auto",
        batch_size: int = DEFAULT_RERANK_BATCH_SIZE,
        max_length: int = DEFAULT_RERANK_MAX_LENGTH,
        activation: str = "none",
        dtype: str | None = "auto",
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size phải ≥ 1, nhận {batch_size}")
        if max_length < 1:
            raise ValueError(f"max_length phải ≥ 1, nhận {max_length}")
        if activation not in ("none", "sigmoid"):
            raise ValueError(f"activation phải là 'none' hoặc 'sigmoid', nhận {activation!r}")
        if dtype is not None and dtype != "auto" and dtype not in SUPPORTED_DTYPES:
            raise ValueError(
                f"dtype phải là None, 'auto' hoặc thuộc {sorted(SUPPORTED_DTYPES)}, nhận {dtype!r}"
            )
        self.model_name = model_name
        # `resolve_device` là chỗ CPU fallback xảy ra: không có CUDA thì trả
        # "cpu" thay vì nổ. Dùng lại của tầng embedding chứ không viết bản thứ
        # hai — hai hàm chọn device là hai câu trả lời khác nhau chờ xảy ra.
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.activation = activation
        # `auto` được phân giải **ngay** thành tên dtype thật, không để lại
        # "auto" trong `name`: một nhãn ghi "auto" nghĩa là đọc log xong vẫn
        # không biết model đã chạy ở độ chính xác nào.
        self.dtype = _resolve_dtype(dtype, self.device)
        suffix = f":L{max_length}"
        if self.dtype is not None:
            suffix += f":{self.dtype}"
        if activation != "none":
            suffix += f":{activation}"
        self.name = f"{model_name}@{self.device}{suffix}"
        self._model: CrossEncoder | None = None

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            self._model = _load_cross_encoder(
                self.model_name, self.device, self.max_length, self.dtype
            )
        return self._model

    def _activation_fn(self) -> Any:
        """Hàm kích hoạt truyền tường minh cho `predict`.

        Truyền tường minh chứ không để `None` (mặc định của
        `sentence-transformers`): mặc định của ST **đổi theo phiên bản** và theo
        `config.json` của model, nên để nó tự chọn là chấp nhận việc điểm log ra
        khác nhau giữa laptop và pod RunPod mà không có gì báo.
        """
        import torch

        return torch.nn.Sigmoid() if self.activation == "sigmoid" else torch.nn.Identity()

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        # `list[Any]` có chủ ý: `PairInput` của sentence-transformers là một hợp
        # rất rộng (text/image/audio/video) và **đổi theo phiên bản**. Ghim đúng
        # kiểu đó vào đây là tự tạo việc phải sửa mỗi lần nâng ST, mà `Reranker`
        # chỉ hứa nhận `str`.
        pairs: list[Any] = [(query, text) for text in texts]
        raw = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            activation_fn=self._activation_fn(),
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [float(value) for value in cast("Sequence[float]", raw)]

    def count_pair_tokens(self, query: str, texts: Sequence[str]) -> list[int]:
        """Số token thật của từng cặp, **không cắt** — để đo truncation.

        `truncation=False` là điểm quan trọng, y như `count_tokens` ở
        `HuggingFaceEmbeddingProvider`: mặc định của tokenizer là cắt ở
        `model_max_length`, và khi đó phép đo "bao nhiêu phần trăm bị cắt" trả về
        hằng số 0 cho mọi corpus.
        """
        if not texts:
            return []
        encoded = self.model.tokenizer(
            [query] * len(texts),
            list(texts),
            add_special_tokens=True,
            truncation=False,
            padding=False,
            verbose=False,
        )
        return [len(ids) for ids in encoded["input_ids"]]
