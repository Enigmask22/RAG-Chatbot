"""`W3-04` — job sinh ngữ cảnh cho chunk, chạy trên API hoặc trên GPU thuê.

Hai lệnh con, và ranh giới giữa chúng chính là ranh giới bảo mật của `W0-08`:

```
prepare  corpus + chunker  ->  requests.jsonl   (chạy ở laptop)
run      requests.jsonl    ->  contexts.jsonl   (chạy ở laptop HOẶC trên pod)
```

**Pod chỉ nhận `requests.jsonl`.** Không manifest, không corpus, không config,
không `.env`, không model embedding. Prompt vào, ngữ cảnh ra. Quy tắc cứng #2
nói pod chỉ chạy job GPU-bound tự chứa; cách chắc chắn nhất để giữ lời hứa đó là
làm cho pod **không có gì khác để chạy**.

## Một client cho cả hai backend

vLLM phục vụ đúng giao thức `chat/completions` của OpenAI, nên
`OpenAICompatProvider` dùng được nguyên xi: đổi `base_url` sang
`http://127.0.0.1:8000/v1` và bảng giá về 0. Không có đường code thứ hai cho
GPU, tức không có đường code nào chỉ được chạy thử đúng một lần trên máy thuê.

## Chi phí trên GPU thuê không tính theo token

Pod tính theo giờ. Nên `--gpu-hourly-usd` quy thời gian chạy thành USD để
`cost/1000 chunk` — con số DoD yêu cầu — có nghĩa ở cả hai backend và **so được
với nhau**. Không có nó thì đường vLLM báo chi phí bằng 0, đúng về mặt token và
vô dụng về mặt quyết định.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
import threading
import time
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import IO, Any, cast

from rag_core.chunking.contextual import (
    BatchParseError,
    ContextRequest,
    ContextualConfig,
    parse_response,
)
from rag_core.chunking.tokens import TokenCounter
from rag_core.llm.base import ChatMessage, LLMError, LLMProvider
from rag_core.llm.budget import BudgetExceeded, CostBudget

__all__ = [
    "RunReport",
    "build_provider",
    "load_contexts",
    "load_done_keys",
    "read_requests",
    "run_requests",
    "write_requests",
]

logger = logging.getLogger(__name__)

VLLM_BASE_URL = "http://127.0.0.1:8000/v1"

DEFAULT_MODEL: dict[str, str] = {
    "glm": "glm-5.3-flash",
    "deepseek": "deepseek-v4-flash",
    "vllm": "Qwen/Qwen3-8B",
}
"""Slug mặc định **theo backend**.

⚠️ Bản đầu để một mặc định duy nhất (`Qwen/Qwen3-8B`) cho mọi backend, và lần
đầu chạy `--backend glm` mà quên `--model` thì cả 20 request trả `HTTP 400
modelCode: does not exist`. Ở laptop thì vô hại — xử lý lỗi đúng như thiết kế,
20 lỗi ghi ra file riêng, artifact chính không bẩn, chạy lại là thử lại. Trên
pod thì đó là mấy phút tiền thuê đổi lấy một file lỗi."""

MIN_REASONING: dict[str, dict[str, Any]] = {
    # `deepseek-v4-flash`, đo 2026-09-03, cùng prompt, `max_tokens=512`:
    #   khong dat                    -> reasoning 275, completion 328
    #   thinking={"type":"disabled"} -> reasoning   0, completion 138
    #   reasoning_effort="none"      -> reasoning   0, completion  87
    #   chat_template_kwargs=...     -> reasoning 159, completion 219  (NHAN roi BO QUA)
    "deepseek": {"thinking": {"type": "disabled"}},
    # `glm-5.3-flash`, cung prompt, cung ngay. Model nay **khong tat duoc** suy
    # luan -- API tra HTTP 400 ma`1210`: "This model always engages in thinking
    # and cannot be disabled; please use low, high, or max". Ba muc hop le, do
    # tren cung mot prompt:
    #   khong dat            -> reasoning 165, completion 243, content 401 ky tu
    #   reasoning_effort=low -> reasoning   0, completion  70, content 383
    #   reasoning_effort=high-> reasoning  40, completion 118, content 393
    #   reasoning_effort=max -> reasoning 180, completion 255, content 411
    # `low` cho `reasoning_content` **rong that** (0 ky tu), khong phai field bi
    # giau di -- da doc response tho de kiem. Do dai content khong doi dang ke,
    # nen 3,5x output do la tiet kiem sach.
    "glm": {"reasoning_effort": "low"},
    # vLLM/Qwen3: chat template cua Qwen3 doc `enable_thinking`. Day moi la cho
    # tham so ay CO tac dung -- DeepSeek nhan no roi bo qua.
    "vllm": {"chat_template_kwargs": {"enable_thinking": False}},
}
"""Tham số **giảm suy luận tới mức thấp nhất provider cho phép**, đã đo từng cái.

Tên là `MIN_REASONING` chứ không phải `NO_THINKING` vì GLM-5.3-Flash **không tắt
được**: nó chỉ nhận `low`/`high`/`max`. Gọi tên sai ở đây sẽ dẫn tới đọc sai một
báo cáo chi phí về sau.

⚠️ Hai trong bốn dòng trên là **tham số được nhận, không lỗi, và không có tác
dụng** nếu đặt nhầm nhà: `chat_template_kwargs` với DeepSeek (vẫn 159 token suy
luận), và `thinking={"type":...}` với GLM (HTTP 400). Không đo thì tưởng đã tắt."""


# ---------------------------------------------------------------- artifact


def _open_text(path: Path, mode: str) -> IO[str]:
    """Mở file, tự nén khi đuôi là `.gz`.

    Cần thật chứ không phải cho gọn: `document_head` lặp lại ở **mọi** dòng của
    cùng một tài liệu, nên gói request thô nặng ~285 MB cho 15.814 chunk. Chính
    phần lặp ấy làm nó nén rất tốt. Cách khác — tách head ra một bảng rồi ghép
    lại trên pod — rẻ hơn nữa nhưng thêm một chỗ để `_render_user` trôi khỏi bản
    dựng lại, và cái đó thì hỏng âm thầm.
    """
    if path.suffix == ".gz":
        return cast("IO[str]", gzip.open(path, mode + "t", encoding="utf-8"))
    return cast("IO[str]", path.open(mode, encoding="utf-8"))


def _read_lines(path: Path) -> list[str]:
    with _open_text(path, "r") as handle:
        return handle.read().splitlines()


def write_requests(path: Path, requests: Iterable[ContextRequest]) -> int:
    """Ghi gói job. Đây là **toàn bộ** thứ được mang lên pod."""
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with _open_text(path, "w") as handle:
        for request in requests:
            handle.write(
                json.dumps(
                    {
                        "key": request.key,
                        "chunk_ids": list(request.chunk_ids),
                        "cfg": request.chunk_fingerprint,
                        "echoes": list(request.echoes),
                        "doc_id": request.doc_id,
                        "system": request.messages[0].content,
                        "user": request.messages[1].content,
                        "est_prompt_tokens": request.est_prompt_tokens,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1
    return written


def read_requests(path: Path) -> list[ContextRequest]:
    out: list[ContextRequest] = []
    for line_no, line in enumerate(_read_lines(path), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            out.append(
                ContextRequest(
                    key=row["key"],
                    chunk_ids=tuple(row.get("chunk_ids") or [row["chunk_id"]]),
                    chunk_fingerprint=str(row.get("cfg", "")),
                    echoes=tuple(row.get("echoes") or ()),
                    doc_id=row["doc_id"],
                    messages=(
                        ChatMessage(role="system", content=row["system"]),
                        ChatMessage(role="user", content=row["user"]),
                    ),
                    est_prompt_tokens=int(row.get("est_prompt_tokens", 0)),
                )
            )
        except (json.JSONDecodeError, KeyError, ValueError):
            logger.warning("Bỏ dòng request hỏng %s:%d", path, line_no)
    return out


def load_done_keys(path: Path) -> set[str]:
    """Khoá đã có ngữ cảnh. Artifact **là** checkpoint — không có file thứ hai.

    Dòng hỏng bị bỏ qua thay vì làm hỏng cả lần chạy tiếp: mất điện giữa lúc ghi
    để lại một dòng JSON dở, và checkpoint tồn tại để cứu công đã làm chứ không
    phải để thêm một chỗ hỏng nữa (`W1-10` đã học bài này một lần).
    """
    if not path.exists():
        return set()
    keys: set[str] = set()
    for line_no, line in enumerate(_read_lines(path), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("Bỏ dòng checkpoint hỏng %s:%d", path, line_no)
            continue
        key = row.get("key")
        if isinstance(key, str) and (row.get("context") or "").strip():
            keys.add(key)
    return keys


def load_contexts(
    path: Path,
    keys: Collection[str] | None = None,
    *,
    fingerprint: str | None = None,
) -> dict[str, str]:
    """`chunk_id -> context`, dạng mà `apply_contexts` nhận.

    `keys` là tập khoá request của **lần chạy hiện tại**. Truyền vào thì các dòng
    sinh bởi cấu hình khác bị bỏ qua — đó là chỗ giữ lại tính vô hiệu hoá theo
    cấu hình sau khi `apply_contexts` chuyển sang khoá theo `chunk_id`. Không
    truyền thì lấy tất, và khi ấy artifact lẫn hai cấu hình sẽ cho kết quả tuỳ
    theo dòng nào ghi sau.

    `fingerprint` lọc theo **cấu hình chunk** đã sinh ra ngữ cảnh. Đây là thứ
    chặn lỗi im lặng nguy hiểm hơn: `chunk_id` là `doc::index`, nên ngữ cảnh
    sinh cho `chunk_size=1000` vẫn khớp id với chunk của `chunk_size=550` trong
    khi nội dung khác hẳn — coverage báo 100% và index nhận toàn câu mô tả sai.
    """
    out: dict[str, str] = {}
    for row in _iter_rows(path):
        if keys is not None and str(row.get("key")) not in keys:
            continue
        if fingerprint is not None and str(row.get("cfg", "")) != fingerprint:
            continue
        context = (row.get("context") or "").strip()
        chunk_id = row.get("chunk_id")
        if context and isinstance(chunk_id, str):
            out[chunk_id] = context
    return out


def _iter_rows(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    for line in _read_lines(path):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and "key" in row:
            yield row


# ---------------------------------------------------------------- backend


def build_provider(
    backend: str,
    *,
    model: str,
    api_key: str = "",
    base_url: str = "",
    timeout: float = 300.0,
) -> LLMProvider:
    """`deepseek` hoặc `vllm`. Cùng một lớp client, khác `base_url` và bảng giá."""
    from rag_core.llm import (
        DEEPSEEK_PRICING,
        GLM_BASE_URL,
        GLM_PRICING,
        ModelPricing,
        OpenAICompatProvider,
    )

    if backend == "glm":
        if not api_key:
            raise SystemExit("Backend `glm` cần GLM_API_KEY.")
        return OpenAICompatProvider(
            model,
            api_key=api_key,
            base_url=base_url or GLM_BASE_URL,
            pricing=GLM_PRICING.get(model, ModelPricing()),
            timeout=timeout,
        )
    if backend == "deepseek":
        if not api_key:
            raise SystemExit("Backend `deepseek` cần DEEPSEEK_API_KEY.")
        return OpenAICompatProvider(
            model,
            api_key=api_key,
            base_url=base_url or "https://api.deepseek.com",
            pricing=DEEPSEEK_PRICING.get(model, ModelPricing()),
            timeout=timeout,
        )
    if backend == "vllm":
        # Không API key, cố ý: đây là đường chạy trên máy thuê. Chuỗi rỗng ở
        # `Authorization` là thứ vLLM bỏ qua khi server không bật `--api-key`.
        return OpenAICompatProvider(
            model,
            api_key="EMPTY",
            base_url=base_url or VLLM_BASE_URL,
            pricing=ModelPricing(),
            timeout=timeout,
        )
    raise SystemExit(f"Backend không biết: {backend!r} (chọn `glm`, `deepseek` hoặc `vllm`)")


# ---------------------------------------------------------------- vòng chạy


@dataclass
class RunReport:
    n_requests: int = 0
    n_done: int = 0
    n_skipped: int = 0
    n_failed: int = 0
    n_rejected: int = 0
    """Số **lô** bị chốt chặn từ chối. Khác `n_failed`, thứ đếm theo chunk.

    Tách riêng vì hai con số trả lời hai câu khác nhau: `n_failed` nói mất bao
    nhiêu công, `n_rejected` nói định dạng lô có đáng tin không. Một tỉ lệ từ
    chối vài phần trăm là chuyện thường và tự chữa ở lần chạy sau; vài chục phần
    trăm nghĩa là prompt hoặc trần output sai, và chạy tiếp chỉ đốt tiền.
    """
    n_empty: int = 0
    n_truncated: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    api_cost_usd: float = 0.0
    gpu_cost_usd: float = 0.0
    elapsed_s: float = 0.0
    models_served: list[str] = field(default_factory=list)
    stopped_early: str = ""

    @property
    def total_cost_usd(self) -> float:
        return self.api_cost_usd + self.gpu_cost_usd

    @property
    def cost_per_1000(self) -> float:
        """Con số DoD yêu cầu, tính trên số chunk **thực sự sinh được**."""
        return self.total_cost_usd / self.n_done * 1000 if self.n_done else 0.0

    @property
    def cache_hit_rate(self) -> float:
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["total_cost_usd"] = round(self.total_cost_usd, 6)
        data["cost_per_1000_usd"] = round(self.cost_per_1000, 6)
        data["cache_hit_rate"] = round(self.cache_hit_rate, 4)
        return data


def run_requests(
    requests: Sequence[ContextRequest],
    provider: LLMProvider,
    out_path: Path,
    *,
    budget: CostBudget,
    max_tokens: int,
    concurrency: int = 6,
    seed: int = 0,
    gpu_hourly_usd: float = 0.0,
    progress_every: int = 50,
    extra_body: dict[str, Any] | None = None,
    skip_done_chunks: bool = False,
) -> RunReport:
    """Gọi LLM cho từng request còn thiếu, ghi nối vào `out_path`.

    Ba kiểu hỏng được đối xử khác nhau, và đó là phần đáng đọc của hàm này:

    * **Một request lỗi** → đếm vào `n_failed`, ghi sang `.failures.jsonl`, job
      chạy tiếp. Đây là nửa "fail 1 chunk không làm sập cả job" của DoD. Cố ý
      **không** ghi vào artifact chính, để lần chạy sau tự thử lại — ghi vào đó
      là biến một lỗi tạm thời thành một lỗ vĩnh viễn.
    * **Trả về rỗng** → cũng không ghi. Với model suy luận thì đây là triệu
      chứng của `max_tokens` bị chuỗi suy luận ăn hết, nên `n_truncated` được
      đếm riêng để phân biệt "model không có gì để nói" với "model bị cắt lời".
    * **Chạm trần chi phí** → dừng **cả job**. Thử lại vẫn chạm; chạy tiếp chỉ
      để đốt thêm tiền cho tới khi hết request.
    """
    _warn_if_not_grouped_by_document(requests)
    done = load_done_keys(out_path)
    todo = [r for r in requests if r.key not in done]
    if skip_done_chunks:
        # Lượt lùi: chunk nào đã có ngữ cảnh rồi thì thôi, bất kể nó được
        # sinh bởi lô nào. Đây là chỗ nối lượt gộp với lượt một-chunk.
        covered = set(load_contexts(out_path))
        todo = [r for r in todo if not set(r.chunk_ids) <= covered]
    report = RunReport(n_requests=len(requests), n_skipped=len(requests) - len(todo))
    if not todo:
        logger.info("Không còn request nào — artifact đã đủ %d khoá", report.n_skipped)
        return report

    out_path.parent.mkdir(parents=True, exist_ok=True)
    failures_path = out_path.with_suffix(".failures.jsonl")
    write_lock = threading.Lock()
    stop = threading.Event()
    served: set[str] = set()
    started = time.perf_counter()

    def call(request: ContextRequest) -> None:
        if stop.is_set():
            return
        # Ước lượng để `reserve` có cái mà so. Chỉ dùng phần input vì output bị
        # `max_tokens` chặn cứng và nhỏ hơn input hai bậc.
        budget.reserve(request.est_prompt_tokens / 1_000_000 * 1.0)
        response = provider.complete(
            request.messages,
            temperature=0.0,
            # Trần output phải theo SỐ CHUNK của lời gọi. Một lô 8 chunk chạy với
            # trần của một chunk sẽ bị cắt lời ở chunk thứ hai, và triệu chứng là
            # `BatchParseError` hàng loạt — một thông báo không hề nói ra nguyên nhân.
            max_tokens=max_tokens * request.n_chunks,
            seed=seed,
            extra_body=extra_body,
        )
        cached = int(response.raw.get("cached_tokens", 0) or 0)
        reasoning = int(response.raw.get("reasoning_tokens", 0) or 0)

        with write_lock:
            report.prompt_tokens += response.usage.prompt_tokens
            report.completion_tokens += response.usage.completion_tokens
            report.cached_tokens += cached
            report.reasoning_tokens += reasoning
            report.api_cost_usd += response.usage.cost_usd
            served.add(response.model)
            budget.charge(response.usage.cost_usd)

            if response.finish_reason == "length":
                report.n_truncated += 1

            try:
                contexts = parse_response(request, response.text)
            except BatchParseError as exc:
                # Cả lô bị từ chối, kể cả phần bóc được. Ghi sang `.failures.jsonl`
                # để lần chạy sau thử lại — xem docstring của `BatchParseError`.
                report.n_rejected += 1
                report.n_failed += request.n_chunks
                _append(
                    failures_path,
                    request,
                    reason=f"lô hỏng: {exc}",
                    finish=response.finish_reason,
                    text=response.text,
                )
                return

            if not contexts:
                report.n_empty += request.n_chunks
                _append(failures_path, request, reason="rỗng", finish=response.finish_reason)
                return

            _append_contexts(out_path, request, contexts, response, cached)
            report.n_done += len(contexts)
            if progress_every and report.n_done % progress_every == 0:
                logger.info(
                    "%d/%d · $%.4f · %.0f%% cache",
                    report.n_done,
                    len(todo),
                    budget.spent_usd,
                    report.cache_hit_rate * 100,
                )

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {pool.submit(call, r): r for r in todo}
        for future in as_completed(futures):
            request = futures[future]
            try:
                future.result()
            except BudgetExceeded as exc:
                if not stop.is_set():
                    stop.set()
                    report.stopped_early = str(exc)
                    logger.error("Dừng job: %s", exc)
            except (LLMError, OSError) as exc:
                with write_lock:
                    report.n_failed += 1
                _append(failures_path, request, reason=str(exc)[:400])

    report.elapsed_s = time.perf_counter() - started
    report.gpu_cost_usd = gpu_hourly_usd * report.elapsed_s / 3600
    report.models_served = sorted(served)
    return report


def _warn_if_not_grouped_by_document(requests: Sequence[ContextRequest]) -> int:
    """Đếm số lần `doc_id` đổi. Trả về số ấy để test bám vào, không chỉ để log.

    ⭐ Đo được, và đo được **tình cờ**: một mẫu 20 request lấy ngẫu nhiên khắp
    corpus cho cache trúng **10,5%**, trong khi 40 request liên tiếp cùng một tài
    liệu cho **49,1%**. Prefix cache — cả của vLLM lẫn của DeepSeek — chỉ trúng
    khi tiền tố **vừa mới** đi qua; xáo trộn để "chia tải" là vứt đi một nửa
    trong 31,3 triệu token tiền tố dùng chung.

    `prepare` sinh ra thứ tự đúng sẵn (duyệt theo tài liệu). Cảnh báo này để
    ai đó lọc/xáo file request về sau không âm thầm trả giá gấp đôi.
    """
    switches = sum(1 for a, b in pairwise(requests) if a.doc_id != b.doc_id)
    distinct = len({r.doc_id for r in requests})
    if switches > distinct:
        logger.warning(
            "Request KHÔNG gom theo tài liệu: %d lần đổi doc_id cho %d tài liệu. "
            "Prefix cache sẽ trượt phần lớn — sắp xếp theo doc_id trước khi chạy.",
            switches,
            distinct,
        )
    return switches


def _append_contexts(
    path: Path,
    request: ContextRequest,
    contexts: Mapping[str, str],
    response: Any,
    cached: int,
) -> None:
    """Một dòng cho mỗi chunk, kèm **phần chia** của số liệu lời gọi.

    Số token và chi phí là của cả lô, nên chia đều cho các chunk trong lô và ghi
    `batch_size` để người đọc biết đó là phần chia chứ không phải số đo riêng của
    chunk ấy. Tổng cộng lại vẫn đúng bằng chi phí thật; con số duy nhất mà DoD
    hỏi — `cost/1000 chunk` — được tính từ `RunReport`, không từ những dòng này.
    """
    n = max(1, len(contexts))
    with path.open("a", encoding="utf-8") as handle:
        for chunk_id, context in contexts.items():
            handle.write(
                json.dumps(
                    {
                        "key": request.key,
                        "chunk_id": chunk_id,
                        "doc_id": request.doc_id,
                        "cfg": request.chunk_fingerprint,
                        "context": context,
                        "model": response.model,
                        "batch_size": len(request.chunk_ids),
                        "prompt_tokens": round(response.usage.prompt_tokens / n, 1),
                        "completion_tokens": round(response.usage.completion_tokens / n, 1),
                        "cached_tokens": round(cached / n, 1),
                        "cost_usd": round(response.usage.cost_usd / n, 8),
                        "latency_ms": round(response.latency_ms, 1),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def _append(
    path: Path,
    request: ContextRequest,
    *,
    reason: str,
    finish: str | None = None,
    text: str = "",
) -> None:
    """Ghi một lô hỏng, **kèm câu trả lời thô**.

    ⚠️ Bản đầu chỉ ghi lý do. Lượt canary đầu tiên của `TD-32` từ chối 109/110 lô
    vì chốt chặn quá chặt; sau khi sửa chốt chặn thì 92 lô trong số đó lẽ ra bóc
    được — nhưng câu trả lời đã không còn, nên phải trả tiền gọi lại lần nữa.
    Lưu lại văn bản biến "sửa parser" từ việc **mua lại dữ liệu** thành việc
    **bóc lại dữ liệu đã mua**.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "key": request.key,
                    "chunk_ids": list(request.chunk_ids),
                    "reason": reason,
                    "finish_reason": finish,
                    "response_text": text,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


# ---------------------------------------------------------------- CLI


def _prepare(args: argparse.Namespace) -> int:
    """Corpus -> `requests.jsonl`. Chunk **đúng như `build_index`**, không xấp xỉ.

    Dùng lại `IndexConfig.build_chunker` và gọi `chunker.prepare(len(documents))`
    vì `NEW-03` đã chỉ ra: `HybridChunker` chọn chiến lược theo kích thước lô
    thật, nên chunk từng tài liệu mà không khai lô trước là chunk bằng một chiến
    lược khác với chiến lược sẽ được index. Ngữ cảnh sinh cho chunk không tồn tại
    thì `key` không bao giờ khớp, và triệu chứng là "coverage 0%" chứ không phải
    một lỗi.
    """
    from rag_core.chunking.contextual import build_requests
    from rag_core.llm.tokenizer import HFTokenCounter

    from .config import load_index_config
    from .corpus_loader import load_documents

    config = load_index_config(args.config)
    documents = load_documents(
        config.manifest_path,
        config.corpus_dir,
        max_documents=args.limit_docs,
    )
    if not documents:
        raise SystemExit("Corpus rỗng — kiểm tra manifest_path/corpus_dir trong config.")

    embeddings = config.build_embeddings()
    chunker = config.build_chunker(embeddings)
    chunker.prepare(len(documents))

    contextual = ContextualConfig(
        chunk_fingerprint=config.chunking_fingerprint,
        model=args.model,
        head_tokens=args.head_tokens,
        window_tokens=args.window_tokens,
        max_context_tokens=args.max_context_tokens,
        prompt_version=args.prompt_version,
        batch_size=args.batch_size,
    )
    counter = HFTokenCounter(args.tokenizer, max_tokens=args.context_window)

    requests: list[ContextRequest] = []
    for document in documents:
        chunks = chunker.chunk([document])
        requests.extend(build_requests(document, chunks, config=contextual, counter=counter))

    est = sum(r.est_prompt_tokens for r in requests)
    shared = _shared_prefix_tokens(requests, counter)
    n_chunks = sum(r.n_chunks for r in requests)
    print(f"tài liệu           {len(documents)}")
    print(f"chunk              {n_chunks:,}")
    print(f"request            {len(requests):,} (gộp {args.batch_size} chunk/lời gọi)")
    print(f"prefill ước tính   {est:,} token (không cache tiền tố)")
    print(f"  mỗi chunk        {est / max(n_chunks, 1):,.0f} token")
    print(f"  tiền tố dùng chung {shared:,} token → còn ~{est - shared:,} nếu cache trúng")
    print(f"trần cửa sổ        {args.context_window:,} token · dài nhất {_max_prompt(requests):,}")

    if args.dry_run:
        print("\n--- prompt mẫu (request đầu) ---")
        print(requests[0].messages[1].content[:1200] + "\n[...]")
        return 0

    out = Path(args.out)
    print(f"\nđã ghi {write_requests(out, requests):,} request → {out}")
    return 0


def _shared_prefix_tokens(requests: Sequence[ContextRequest], counter: TokenCounter) -> int:
    """Phần prefill mà prefix caching **bỏ được**, đo bằng tokenizer thật.

    Không suy từ `head_tokens`: `_snap_right` nới biên tới 200 ký tự và
    `document_head` bị `strip()`, nên con số khai báo và con số thật lệch nhau.
    """
    if not requests:
        return 0
    system = requests[0].messages[0].content
    heads = {r.doc_id: r.messages[1].content.split("</document_head>")[0] for r in requests}
    counts = counter.count_tokens([system, *heads.values()])
    if counts is None:  # pragma: no cover - HFTokenCounter luon dem duoc
        return 0

    per_doc_chunks: dict[str, int] = dict.fromkeys(heads, 0)
    for request in requests:
        per_doc_chunks[request.doc_id] += 1

    # `system` dùng lại trên CẢ job; `head` chỉ dùng lại trong một tài liệu.
    saved = counts[0] * (len(requests) - 1)
    for doc, head_tokens in zip(heads, counts[1:], strict=True):
        saved += head_tokens * (per_doc_chunks[doc] - 1)
    return saved


def _max_prompt(requests: Sequence[ContextRequest]) -> int:
    return max((r.est_prompt_tokens for r in requests), default=0)


def _run(args: argparse.Namespace) -> int:
    from rag_core.settings import get_settings

    requests = read_requests(Path(args.requests))
    if args.limit:
        requests = requests[: args.limit]
    if not requests:
        raise SystemExit(f"Không đọc được request nào từ {args.requests}")

    api_key = ""
    base_url = args.base_url
    if args.backend in ("deepseek", "glm"):
        field = f"{args.backend}_api_key"
        settings = get_settings()
        settings.require(field)
        secret = getattr(settings, field)
        if secret is None:  # pragma: no cover - `require` đã chặn
            raise SystemExit(f"Thiếu {field.upper()}")
        api_key = secret.get_secret_value()
        base_url = base_url or getattr(settings, f"{args.backend}_base_url", "")

    model = args.model or DEFAULT_MODEL[args.backend]
    provider = build_provider(args.backend, model=model, api_key=api_key, base_url=base_url)
    budget = CostBudget(args.cost_cap, name=f"W3-04/{args.backend}")

    report = run_requests(
        requests,
        provider,
        Path(args.out),
        budget=budget,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        seed=args.seed,
        gpu_hourly_usd=args.gpu_hourly_usd,
        extra_body=None if args.thinking else MIN_REASONING.get(args.backend),
        skip_done_chunks=args.skip_done_chunks,
    )
    _print_report(report)

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            json.dumps(report.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 1 if report.stopped_early else 0


def _coverage(args: argparse.Namespace) -> int:
    """Bao nhiêu chunk đã có ngữ cảnh, và thiếu tập trung ở tài liệu nào.

    Câu này phải trả lời được **trước** khi tuyên bố `W3-04` xong. `apply_contexts`
    cố ý giữ nguyên chunk khi thiếu ngữ cảnh — nửa "fail 1 chunk không làm sập cả
    job" của DoD — nên thiếu 2.000 chunk và thiếu 0 chunk **trông giống hệt nhau**
    ở phía build index: cả hai đều chạy xong và không báo gì.
    """
    requests = read_requests(Path(args.requests))
    if not requests:
        raise SystemExit(f"Không đọc được request nào từ {args.requests}")
    wanted = [cid for r in requests for cid in r.chunk_ids]
    have = load_contexts(Path(args.out))
    missing = [cid for cid in wanted if cid not in have]

    by_doc: dict[str, int] = {}
    for cid in missing:
        by_doc[cid.split("::")[0]] = by_doc.get(cid.split("::")[0], 0) + 1

    print(f"chunk cần      {len(wanted):,}")
    print(f"đã có ngữ cảnh {len(wanted) - len(missing):,} ({1 - len(missing) / len(wanted):.1%})")
    print(f"còn thiếu      {len(missing):,}")
    if by_doc:
        print("\nthiếu nhiều nhất:")
        for doc, n in sorted(by_doc.items(), key=lambda kv: -kv[1])[:10]:
            print(f"  {n:6,}  {doc}")
    return 1 if missing else 0


def _print_report(report: RunReport) -> None:
    empty_line = f"lỗi {report.n_failed:,} · rỗng {report.n_empty:,}"
    print()
    print(f"  request        {report.n_requests:,} (bỏ qua vì đã có {report.n_skipped:,})")
    print(f"  sinh được      {report.n_done:,}")
    print(f"  {empty_line} · cắt lời {report.n_truncated:,}")
    if report.n_rejected:
        print(f"  ⚠ lô bị từ chối {report.n_rejected:,} (chốt chặn echo/định dạng)")
    print(f"  token          prompt {report.prompt_tokens:,} · out {report.completion_tokens:,}")
    print(f"                 cache trúng {report.cached_tokens:,} ({report.cache_hit_rate:.1%})")
    if report.reasoning_tokens:
        print(f"                 suy luận {report.reasoning_tokens:,} (KHÔNG có trong content)")
    print(f"  thời gian      {report.elapsed_s:.1f}s")
    print(f"  chi phí        API ${report.api_cost_usd:.4f} + GPU ${report.gpu_cost_usd:.4f}")
    print(f"  cost/1000      ${report.cost_per_1000:.4f}")
    print(f"  model phục vụ  {', '.join(report.models_served) or 'khong ro'}")
    if report.stopped_early:
        print(f"  DUNG SOM       {report.stopped_early}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="W3-04 — sinh ngữ cảnh cho chunk")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="corpus → requests.jsonl (chạy ở laptop)")
    prepare.add_argument("--config", default="configs/indexing/bgem3.yaml")
    prepare.add_argument("--out", default="data/contexts/requests.jsonl.gz")
    prepare.add_argument("--model", default="Qwen/Qwen3-8B", help="ghi vào khoá cache")
    prepare.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    prepare.add_argument("--context-window", type=int, default=40960)
    prepare.add_argument("--head-tokens", type=int, default=2000)
    prepare.add_argument("--window-tokens", type=int, default=1500)
    prepare.add_argument("--max-context-tokens", type=int, default=120)
    prepare.add_argument("--prompt-version", default="ctx-v1")
    prepare.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="số chunk mỗi lời gọi. >1 bật prompt gộp + chốt chặn echo (TD-32)",
    )
    prepare.add_argument("--limit-docs", type=int, default=None)
    prepare.add_argument("--dry-run", action="store_true", help="in thống kê + prompt mẫu")
    prepare.set_defaults(func=_prepare)

    run = subparsers.add_parser("run", help="requests.jsonl → contexts.jsonl")
    run.add_argument("--requests", default="data/contexts/requests.jsonl.gz")
    run.add_argument("--out", default="data/contexts/contexts.jsonl")
    run.add_argument("--backend", choices=("glm", "deepseek", "vllm"), default="vllm")
    run.add_argument("--model", default="", help="bo trong = mac dinh cua backend")
    run.add_argument("--base-url", default="")
    run.add_argument("--concurrency", type=int, default=6)
    run.add_argument("--max-tokens", type=int, default=256)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--limit", type=int, default=None, help="chỉ chạy N request đầu")
    run.add_argument("--cost-cap", type=float, default=5.0, help="USD; 0 = không trần")
    run.add_argument("--gpu-hourly-usd", type=float, default=0.0)
    run.add_argument(
        "--thinking",
        action="store_true",
        help="BẬT lại suy luận (mặc định giảm tối đa — xem MIN_REASONING)",
    )
    run.add_argument(
        "--skip-done-chunks",
        action="store_true",
        help=(
            "bỏ request mà MỌI chunk của nó đã có ngữ cảnh trong --out. "
            "Dùng cho lượt lùi: sinh lại các chunk mà lượt gộp đã bị chốt chặn từ chối"
        ),
    )
    run.add_argument("--report", default="")
    run.set_defaults(func=_run)

    coverage = subparsers.add_parser("coverage", help="đếm chunk còn thiếu ngữ cảnh")
    coverage.add_argument("--requests", default="data/contexts/requests-b1.jsonl.gz")
    coverage.add_argument("--out", default="data/contexts/contexts.jsonl")
    coverage.set_defaults(func=_coverage)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
