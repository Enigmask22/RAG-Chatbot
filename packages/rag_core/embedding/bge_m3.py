"""BGE-M3: dense 1024-d và sparse lexical weights từ **một** forward pass.

Vì sao có module riêng thay vì dùng `HuggingFaceEmbeddingProvider` trực tiếp:
`sentence-transformers` chỉ trả dense. Sparse của BGE-M3 nằm ở một `Linear(1024
→ 1)` đặt lên `last_hidden_state`, và trọng số của nó ở `sparse_linear.pt` trong
repo model — không thuộc `modules.json` nên ST không nạp.

Hai lý do chọn BGE-M3 cho `W2-01`, cả hai đến từ số đo của `TD-11`:

* **Cửa sổ 8192 token** xoá truncation mà *không* phải hạ `chunk_size`. `TD-11`
  đã đo: hạ `chunk_size` là đánh đổi (mỗi vector đọc 950 → 678 ký tự), không
  phải thu hồi nội dung, và không cải thiện gì đo được (`p = 0,711`).
* **Đa ngữ.** Baseline có `cross_lingual` recall@5 = 0, và tiếng Anh mất 19,4%
  token vs tiếng Việt 7,5% vì PhoBERT xé chữ Anh vụn hơn.

⚠️ **`_forward` là đường code DUY NHẤT.** `_encode` (dense) cũng đi qua nó rồi
bỏ phần sparse, chứ không gọi `model.encode()`. Hai đường sinh dense song song
là cách chắc chắn để hai nhánh ablation vô tình đo hai thứ khác nhau — có test
canh dense của module này khớp `SentenceTransformer.encode()`.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .base import FloatArray, HybridVectors, l2_normalize
from .huggingface import HuggingFaceEmbeddingProvider
from .sparse import SparseVector

if TYPE_CHECKING:
    import torch

__all__ = ["BGE_M3_MODEL", "BgeM3EmbeddingProvider"]

BGE_M3_MODEL = "BAAI/bge-m3"

#: Trần token mỗi forward pass (`số câu × độ dài câu dài nhất trong batch`).
#: Cửa sổ của BGE-M3 là 8192 nên `batch_size` một mình không chặn được gì: một
#: batch 16 câu dài 8192 token là 131k token và OOM ngay trên 4060 8GB. Trần này
#: chia nhỏ batch theo *độ dài thật*, nên chunk ngắn vẫn chạy full batch.
DEFAULT_MAX_BATCH_TOKENS = 8192


@lru_cache(maxsize=4)
def _load_sparse_head(model_name: str, device: str) -> torch.nn.Linear:
    """Nạp `sparse_linear.pt` — trọng số sparse không nằm trong `modules.json`.

    Cache theo cặp `(model, device)` giống `_load_model`, vì cùng lý do: một lần
    ablation quét nhiều cấu hình không được giữ nhiều bản trong VRAM.
    """
    import torch
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(model_name, "sparse_linear.pt")
    state = torch.load(path, map_location="cpu", weights_only=True)
    weight = state["weight"]
    head = torch.nn.Linear(weight.shape[1], weight.shape[0], bias="bias" in state)
    head.load_state_dict(state)
    head.to(device).eval()
    return head


class BgeM3EmbeddingProvider(HuggingFaceEmbeddingProvider):
    """Provider BGE-M3 — dense + sparse cùng lúc.

    `query_prefix`/`document_prefix` để rỗng có chủ ý: BGE-M3 **không** dùng
    instruction prefix, khác BGE v1.5 và E5. Thêm prefix vào là làm lệch phân bố
    đầu vào so với lúc train mà không có lỗi nào báo ra.
    """

    def __init__(
        self,
        model_name: str = BGE_M3_MODEL,
        *,
        device: str = "auto",
        batch_size: int = 16,
        normalize: bool = True,
        max_batch_tokens: int = DEFAULT_MAX_BATCH_TOKENS,
        sparse_head_name: str | None = None,
    ) -> None:
        super().__init__(
            model_name,
            device=device,
            batch_size=batch_size,
            normalize=normalize,
        )
        if max_batch_tokens < 1:
            raise ValueError(f"max_batch_tokens phải dương, nhận {max_batch_tokens}")
        self.max_batch_tokens = max_batch_tokens
        # Trọng số sparse mặc định lấy từ chính repo model. Tách ra được để
        # `W2-08` thử checkpoint fine-tune mà không phải fork provider.
        self.sparse_head_name = sparse_head_name or model_name
        # Nhớ kết quả `len(tokenizer)` — xem `sparse_vocab_size`. Không tính ở đây
        # để việc dựng provider không kéo theo 64 ms cho một thứ có thể không dùng.
        self._sparse_vocab_size: int | None = None

    @property
    def sparse_head(self) -> torch.nn.Linear:
        return _load_sparse_head(self.sparse_head_name, self.device)

    @property
    def backbone(self) -> torch.nn.Module:
        """`XLMRobertaModel` nằm dưới module `Transformer` của ST.

        Phải đi thẳng vào đây vì `encode()` không trả `last_hidden_state`, mà
        sparse cần nó — và cần **đúng cái** đã dùng để lấy dense. Không nạp thêm
        bản thứ hai: 2,2GB trọng số nhân đôi thì 4060 8GB hết chỗ.
        """
        return cast("torch.nn.Module", self.model[0].auto_model)

    @property
    def sparse_vocab_size(self) -> int | None:
        """Số chiều của không gian sparse = vocab của tokenizer (250.002).

        ⚠️ **Phải nhớ kết quả.** `len(tokenizer)` gọi `get_vocab()`, và với XLM-R
        thì đó là dựng lại một dict 250.002 phần tử — đo được **64 ms mỗi lần**.
        Thuộc tính này bị đọc ở đường nóng: `QdrantDenseRetriever.writes_sparse`
        đọc nó ở **mỗi** truy vấn sparse và ở **mỗi lô** upsert.

        Không dùng `tokenizer.vocab_size` (nhanh nhưng khác nghĩa): nó là kích cỡ
        vocab gốc, không tính token thêm vào. Token id do tokenizer sinh ra có thể
        vượt quá nó, và một chặn trên sai ở đây thì hỏng im lặng — `SparseVector`
        vẫn nhận index lớn hơn.
        """
        if self._sparse_vocab_size is None:
            self._sparse_vocab_size = len(self.model.tokenizer)
        return self._sparse_vocab_size

    # ---------------------------------------------------------------- forward

    def _unused_token_ids(self) -> frozenset[int]:
        """Token id bị loại khỏi sparse: `[CLS]`, `[SEP]`, `[PAD]`, `[UNK]`.

        Chúng có trọng số dương nhưng không mang nghĩa từ vựng, và `[CLS]`
        thường là chiều nặng nhất — để lại thì mọi cặp text đều khớp nhau ở
        đúng chiều đó và điểm sparse gần như thành hằng số.
        """
        tokenizer = self.model.tokenizer
        ids = (
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.pad_token_id,
            tokenizer.unk_token_id,
        )
        return frozenset(int(i) for i in ids if i is not None)

    def _plan_batches(self, lengths: Sequence[int], limit: int) -> list[list[int]]:
        """Chia chỉ số thành batch, sắp giảm dần theo độ dài token.

        Hai lý do phải sắp: padding tới câu dài nhất trong batch, nên trộn câu
        40 token với câu 500 token làm hầu hết phép tính đổ vào padding; và câu
        dài nhất rơi vào batch đầu, nên cấu hình sẽ OOM thì OOM ở giây thứ nhất
        chứ không phải ở phút thứ ba của một job 31.000 chunk.

        `SentenceTransformer.encode()` cũng sắp theo độ dài; đây là chỗ phải làm
        lại vì `_forward` không đi qua nó nữa.
        """
        order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
        batches: list[list[int]] = []
        current: list[int] = []
        widest = 0
        for idx in order:
            width = max(min(lengths[idx], limit), 1)
            next_widest = max(widest, width)
            full = len(current) >= self.batch_size
            over_budget = next_widest * (len(current) + 1) > self.max_batch_tokens
            if current and (full or over_budget):
                batches.append(current)
                current = []
                next_widest = width
            current.append(idx)
            widest = next_widest
        if current:
            batches.append(current)
        return batches

    def _forward(self, texts: Sequence[str]) -> HybridVectors:
        """Chạy model, trả cả dense và sparse. Thứ tự đầu ra khớp đầu vào.

        ⭐ Khoá ở **đây** chứ không chỉ ở `_encode` của lớp cha: lớp này ghi đè
        `_encode`, nên khoá đặt ở lớp cha bị đi vòng hoàn toàn. Đã trả tiền để
        biết — bản vá đầu chỉ khoá `HuggingFaceEmbeddingProvider._encode` và
        container **vẫn** trả `RuntimeError: Already borrowed` ở đúng dòng
        `tokenizer(...)` dưới đây, 6/9 request.
        """
        import torch

        if not texts:
            return HybridVectors(np.zeros((0, self.dimension), dtype=np.float32), [])

        with self.lock:
            return self._forward_locked(texts, torch)

    def _forward_locked(self, texts: Sequence[str], torch: Any) -> HybridVectors:
        limit = self.max_sequence_tokens or 8192
        tokenizer = self.model.tokenizer
        lengths = [len(ids) for ids in tokenizer(list(texts), truncation=False)["input_ids"]]
        unused = self._unused_token_ids()

        dense_rows: list[FloatArray | None] = [None] * len(texts)
        sparse_rows: list[SparseVector | None] = [None] * len(texts)

        for batch in self._plan_batches(lengths, limit):
            encoded = tokenizer(
                [texts[i] for i in batch],
                padding=True,
                truncation=True,
                max_length=limit,
                return_tensors="pt",
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.no_grad():
                hidden = self.backbone(**encoded, return_dict=True).last_hidden_state
                # CLS pooling — khớp `1_Pooling/config.json` của BGE-M3.
                dense = hidden[:, 0].float()
                weights = torch.relu(self.sparse_head(hidden)).squeeze(-1).float()

            dense_np = cast(FloatArray, dense.cpu().numpy().astype(np.float32))
            if self.normalize:
                dense_np = l2_normalize(dense_np)
            ids_np = encoded["input_ids"].cpu().numpy()
            mask_np = encoded["attention_mask"].cpu().numpy()
            weights_np = weights.cpu().numpy()

            for row, target in enumerate(batch):
                dense_rows[target] = dense_np[row]
                sparse_rows[target] = _collapse(ids_np[row], mask_np[row], weights_np[row], unused)

        # `None` còn lại nghĩa là `_plan_batches` bỏ sót chỉ số — bug, không phải
        # trạng thái hợp lệ. Nói ra thay vì lặng lẽ trả thiếu hàng.
        missing = [i for i, row in enumerate(dense_rows) if row is None]
        if missing:  # pragma: no cover - chỉ xảy ra nếu _plan_batches sai
            raise RuntimeError(f"{len(missing)} text không được embed, ví dụ chỉ số {missing[:5]}")

        return HybridVectors(
            np.vstack([cast(FloatArray, row) for row in dense_rows]).astype(np.float32),
            [cast(SparseVector, row) for row in sparse_rows],
        )

    # ------------------------------------------------------------ public API

    def _encode(self, texts: Sequence[str]) -> FloatArray:
        """Dense-only, nhưng đi qua đúng `_forward` như đường hybrid."""
        return self._forward(texts).dense

    def embed_documents_hybrid(self, texts: Sequence[str]) -> HybridVectors:
        return self._forward(texts)

    def embed_query_hybrid(self, text: str) -> tuple[FloatArray, SparseVector]:
        dense, sparse = self._forward([text])
        return cast(FloatArray, dense[0]), sparse[0]


def _collapse(
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    weights: np.ndarray,
    unused: frozenset[int],
) -> SparseVector:
    """Gộp trọng số theo token: mỗi token id lấy **max** qua các vị trí.

    Max chứ không phải sum — đây là định nghĩa của BGE-M3, và nó có nghĩa: trọng
    số là "token này quan trọng thế nào cho đoạn text", không phải "nó xuất hiện
    bao nhiêu lần". Đổi sang sum thì token lặp nhiều lấn át và sparse retrieval
    biến thành đếm tần suất thô.
    """
    collapsed: dict[int, float] = {}
    for token_id, keep, weight in zip(input_ids, attention_mask, weights, strict=True):
        if not keep:
            continue
        key = int(token_id)
        if key in unused:
            continue
        value = float(weight)
        if value > collapsed.get(key, 0.0):
            collapsed[key] = value
    return SparseVector.from_weights(collapsed)
