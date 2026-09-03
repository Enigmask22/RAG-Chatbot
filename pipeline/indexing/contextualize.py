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
from collections.abc import Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import IO, Any, cast

from rag_core.chunking.contextual import ContextRequest, ContextualConfig
from rag_core.chunking.tokens import TokenCounter
from rag_core.llm.base import ChatMessage, LLMError, LLMProvider
from rag_core.llm.budget import BudgetExceeded, CostBudget

__all__ = [
    "RunReport",
    "build_provider",
    "load_done_keys",
    "read_requests",
    "run_requests",
    "write_requests",
]

logger = logging.getLogger(__name__)

VLLM_BASE_URL = "http://127.0.0.1:8000/v1"

NO_THINKING: dict[str, dict[str, Any]] = {
    # Đo ngày 2026-09-03 trên `deepseek-v4-flash`, cùng một prompt, `max_tokens=512`:
    #   khong tat                -> reasoning 275, completion 328, content 255 ky tu
    #   thinking={"type":"disabled"} -> reasoning   0, completion 138, content 361
    #   reasoning_effort="none"      -> reasoning   0, completion  87, content 374
    #   chat_template_kwargs=...     -> reasoning 159, completion 219, content 288
    # Hai cai dau tat that. Cai cuoi duoc NHAN roi BO QUA -- day la vi du cua tham
    # so khong loi, khong tac dung: khong do thi tuong da tat.
    "deepseek": {"thinking": {"type": "disabled"}},
    # Voi vLLM day moi la co che dung: Qwen3 la model lai, chat template cua no
    # doc `enable_thinking`. Khong tat thi phan lon `max_tokens` di vao chuoi suy
    # luan khong nam trong `content` -- dry-run do duoc 83%, va 6/30 request tra rong.
    "vllm": {"chat_template_kwargs": {"enable_thinking": False}},
}
"""Tham số tắt suy luận, **khác nhau theo nhà cung cấp** và đã đo chứ không đoán."""


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
                        "chunk_id": request.chunk_id,
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
                    chunk_id=row["chunk_id"],
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


def load_contexts(path: Path) -> dict[str, str]:
    """`key -> context`, dạng mà `apply_contexts` nhận."""
    out: dict[str, str] = {}
    for key_row in _iter_rows(path):
        context = (key_row.get("context") or "").strip()
        if context:
            out[str(key_row["key"])] = context
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
    from rag_core.llm import DEEPSEEK_PRICING, ModelPricing, OpenAICompatProvider

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
    raise SystemExit(f"Backend không biết: {backend!r} (chọn `deepseek` hoặc `vllm`)")


# ---------------------------------------------------------------- vòng chạy


@dataclass
class RunReport:
    n_requests: int = 0
    n_done: int = 0
    n_skipped: int = 0
    n_failed: int = 0
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
    todo = [r for r in requests if r.key not in load_done_keys(out_path)]
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
            max_tokens=max_tokens,
            seed=seed,
            extra_body=extra_body,
        )
        context = response.text.strip()
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
            if not context:
                report.n_empty += 1
                _append(failures_path, request, reason="rỗng", finish=response.finish_reason)
                return

            _append_context(out_path, request, context, response, cached)
            report.n_done += 1
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


def _append_context(
    path: Path, request: ContextRequest, context: str, response: Any, cached: int
) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "key": request.key,
                    "chunk_id": request.chunk_id,
                    "doc_id": request.doc_id,
                    "context": context,
                    "model": response.model,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "cached_tokens": cached,
                    "cost_usd": round(response.usage.cost_usd, 8),
                    "latency_ms": round(response.latency_ms, 1),
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def _append(path: Path, request: ContextRequest, *, reason: str, finish: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "key": request.key,
                    "chunk_id": request.chunk_id,
                    "reason": reason,
                    "finish_reason": finish,
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
        model=args.model,
        head_tokens=args.head_tokens,
        window_tokens=args.window_tokens,
        max_context_tokens=args.max_context_tokens,
        prompt_version=args.prompt_version,
    )
    counter = HFTokenCounter(args.tokenizer, max_tokens=args.context_window)

    requests: list[ContextRequest] = []
    for document in documents:
        chunks = chunker.chunk([document])
        requests.extend(build_requests(document, chunks, config=contextual, counter=counter))

    est = sum(r.est_prompt_tokens for r in requests)
    shared = _shared_prefix_tokens(requests, counter)
    print(f"tài liệu           {len(documents)}")
    print(f"request            {len(requests):,}")
    print(f"prefill ước tính   {est:,} token (không cache tiền tố)")
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
    if args.backend == "deepseek":
        settings = get_settings()
        settings.require("deepseek_api_key")
        if settings.deepseek_api_key is None:  # pragma: no cover - `require` đã chặn
            raise SystemExit("Thiếu DEEPSEEK_API_KEY")
        api_key = settings.deepseek_api_key.get_secret_value()

    provider = build_provider(
        args.backend, model=args.model, api_key=api_key, base_url=args.base_url
    )
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
        extra_body=None if args.thinking else NO_THINKING.get(args.backend),
    )
    _print_report(report)

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            json.dumps(report.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return 1 if report.stopped_early else 0


def _print_report(report: RunReport) -> None:
    empty_line = f"lỗi {report.n_failed:,} · rỗng {report.n_empty:,}"
    print()
    print(f"  request        {report.n_requests:,} (bỏ qua vì đã có {report.n_skipped:,})")
    print(f"  sinh được      {report.n_done:,}")
    print(f"  {empty_line} · cắt lời {report.n_truncated:,}")
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
    prepare.add_argument("--limit-docs", type=int, default=None)
    prepare.add_argument("--dry-run", action="store_true", help="in thống kê + prompt mẫu")
    prepare.set_defaults(func=_prepare)

    run = subparsers.add_parser("run", help="requests.jsonl → contexts.jsonl")
    run.add_argument("--requests", default="data/contexts/requests.jsonl.gz")
    run.add_argument("--out", default="data/contexts/contexts.jsonl")
    run.add_argument("--backend", choices=("deepseek", "vllm"), default="vllm")
    run.add_argument("--model", default="Qwen/Qwen3-8B")
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
        help="BẬT lại suy luận (mặc định tắt — xem NO_THINKING)",
    )
    run.add_argument("--report", default="")
    run.set_defaults(func=_run)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
