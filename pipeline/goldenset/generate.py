"""Sinh bản nháp golden set bằng LLM, từ chunk thật trong index.

Đây là bước **rẻ** của quy trình hai bước: LLM sinh ra nhiều nháp, người đọc và
đóng băng ở `W1-11`. Không được nhầm đầu ra ở đây với golden set — nhãn do model
tự chấm cho chính đầu ra của model là thứ vô giá trị để đo hệ thống.

Ba thứ code kiểm ngay tại đây thay vì đẩy hết cho người review:

1. **`used_chunks` phải nằm trong khoảng đã đưa.** Model bịa chỉ số thì bỏ câu
   đó — không có cách nào cứu, và giữ lại là mở đường cho `chunk_id` trỏ sai.
2. **`quote` phải tìm thấy trong chunk được viện dẫn.** Không khớp thì vẫn giữ
   câu, nhưng đánh dấu `quotes_verified=False` để xếp lên đầu hàng đợi review.
3. **Ràng buộc của từng nhóm.** `unanswerable` mà có chunk liên quan, hay
   `multi_hop` chỉ dùng một chunk, đều là model làm sai việc được giao —
   `GoldenQuery` từ chối cái thứ nhất, còn cái thứ hai bị hạ nhóm về `factoid`
   và ghi lại là đã trôi nhóm.

Chi phí được cộng dồn theo từng lời gọi và ghi vào báo cáo. Không đo được tiền
thì không trả lời được "cải thiện 3% recall này giá bao nhiêu".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import threading
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from pipeline.eval.golden import GoldenQuery, QueryCategory
from pipeline.indexing.config import load_index_config
from rag_core.llm import (
    DEFAULT_DEEPSEEK_MODEL,
    ChatMessage,
    LLMError,
    LLMProvider,
    build_deepseek_provider,
)
from rag_core.schemas import Chunk, Language
from rag_core.settings import get_settings

from .dedupe import deduplicate_drafts
from .prompts import build_messages
from .sampling import ChunkGroup, sample_groups
from .schema import DraftProvenance, GoldenDraft, summary_json, write_drafts

__all__ = ["GenerationStats", "generate_drafts", "main", "parse_response"]

logger = logging.getLogger("pipeline.goldenset")

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)
_WS_RE = re.compile(r"\s+", re.UNICODE)

#: Hạn mức mặc định. `table_lookup` để thấp vì corpus hiện chỉ có bản `.txt` của
#: World Bank — bảng biểu đã bị làm phẳng, nhóm này chỉ thật sự đo được khi có
#: nguồn (c) báo cáo thường niên HOSE và Docling ở `W3-01`.
DEFAULT_QUOTAS: dict[QueryCategory, int] = {
    QueryCategory.FACTOID: 45,
    QueryCategory.MULTI_HOP: 25,
    QueryCategory.AGGREGATION: 20,
    QueryCategory.CROSS_LINGUAL: 25,
    QueryCategory.ADVERSARIAL: 20,
    QueryCategory.UNANSWERABLE: 20,
    QueryCategory.TABLE_LOOKUP: 10,
}

QUESTIONS_PER_CALL = 2

#: `deepseek-v4-flash` là model suy luận: nó tiêu 1.500–2.000 token cho chuỗi
#: suy luận **không** nằm trong `content`, rồi mới viết JSON. Với ngân sách 2.000
#: thì content thường rỗng hoặc bị cắt giữa chừng, và triệu chứng hiện ra là
#: "response không phải JSON hợp lệ" — chẩn đoán sai hoàn toàn nguyên nhân.
MAX_TOKENS = 6000

#: Số lời gọi chạy song song. Mỗi lời gọi ~25 giây và gần như toàn bộ là chờ
#: mạng, nên chạy tuần tự 163 lô mất hơn một tiếng. Để 6 cho vừa hạn mức của
#: DeepSeek — đẩy cao hơn thì gặp 429 và phần thắng bị backoff ăn hết.
DEFAULT_CONCURRENCY = 6


@dataclass
class GenerationStats:
    n_groups: int = 0
    n_calls: int = 0
    n_failed_calls: int = 0
    n_skipped_groups: int = 0
    n_raw_questions: int = 0
    n_rejected_bad_index: int = 0
    n_rejected_schema: int = 0
    n_truncated: int = 0
    n_category_drifted: int = 0
    n_quotes_unverified: int = 0
    n_duplicates_removed: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    seconds: float = 0.0
    models_served: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_groups": self.n_groups,
            "n_calls": self.n_calls,
            "n_failed_calls": self.n_failed_calls,
            "n_skipped_groups": self.n_skipped_groups,
            "n_raw_questions": self.n_raw_questions,
            "n_rejected_bad_index": self.n_rejected_bad_index,
            "n_rejected_schema": self.n_rejected_schema,
            "n_truncated": self.n_truncated,
            "n_category_drifted": self.n_category_drifted,
            "n_quotes_unverified": self.n_quotes_unverified,
            "n_duplicates_removed": self.n_duplicates_removed,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "seconds": round(self.seconds, 1),
            "models_served": self.models_served,
        }


def query_id_for(category: QueryCategory, query: str) -> str:
    """ID xác định theo nội dung câu hỏi.

    Chạy lại sinh ra cùng câu thì cùng ID, nên hai lần chạy gộp được với nhau
    mà không nhân đôi. ID chạy theo số thứ tự sẽ đổi hết mỗi lần thêm một câu ở
    giữa — và mọi tham chiếu trong báo cáo cũ thành sai.
    """
    digest = hashlib.sha1(_WS_RE.sub(" ", query.strip().lower()).encode("utf-8")).hexdigest()
    return f"{category.value}-{digest[:10]}"


def _strip_fences(text: str) -> str:
    return _FENCE_RE.sub("", text).strip()


def quote_found(quote: str, chunks: Sequence[Chunk]) -> bool:
    """Trích dẫn có thật sự nằm trong một trong các chunk không.

    So sau khi gom khoảng trắng: chunk giữ nguyên xuống dòng của PDF gốc, còn
    model chép lại thành một dòng — đó là khác biệt về hình thức, không phải
    dấu hiệu bịa đặt.
    """
    needle = _WS_RE.sub(" ", quote.strip().lower())
    if len(needle) < 12:
        return False
    return any(needle in _WS_RE.sub(" ", c.content.lower()) for c in chunks)


@dataclass(frozen=True)
class ParsedQuestion:
    query: str
    lang: Language
    used_chunks: list[int]
    quote: str
    reference_answer: str


def parse_response(text: str, n_chunks: int) -> tuple[list[ParsedQuestion], int, int]:
    """Đọc JSON model trả về. Trả `(câu hợp lệ, số bị loại vì chỉ số, số sai schema)`.

    Cố ý khoan dung với hình thức (rào ```json, khoá thiếu, `used_chunks` là số
    đơn) nhưng nghiêm với nội dung: chỉ số ngoài khoảng là loại thẳng, vì đó là
    thứ duy nhất không sửa được ở bước review.
    """
    try:
        payload = json.loads(_strip_fences(text))
    except json.JSONDecodeError:
        logger.warning("Response không phải JSON hợp lệ: %s", text[:200])
        return [], 0, 1

    raw_items = payload.get("questions") if isinstance(payload, dict) else payload
    if not isinstance(raw_items, list):
        logger.warning("Response thiếu mảng `questions`: %s", text[:200])
        return [], 0, 1

    questions: list[ParsedQuestion] = []
    bad_index = 0
    bad_schema = 0
    for item in raw_items:
        if not isinstance(item, dict):
            bad_schema += 1
            continue
        query = str(item.get("query", "")).strip()
        if not query:
            bad_schema += 1
            continue

        raw_used = item.get("used_chunks", [])
        if isinstance(raw_used, int):
            raw_used = [raw_used]
        if not isinstance(raw_used, list):
            bad_schema += 1
            continue

        indices: list[int] = []
        out_of_range = False
        for value in raw_used:
            try:
                index = int(value)
            except (TypeError, ValueError):
                out_of_range = True
                break
            if not 1 <= index <= n_chunks:
                out_of_range = True
                break
            if index not in indices:
                indices.append(index)
        if out_of_range:
            bad_index += 1
            continue

        try:
            lang = Language(str(item.get("lang", "unknown")).lower())
        except ValueError:
            lang = Language.UNKNOWN

        questions.append(
            ParsedQuestion(
                query=query,
                lang=lang,
                used_chunks=indices,
                quote=str(item.get("quote", "")).strip(),
                reference_answer=str(item.get("reference_answer", "")).strip(),
            )
        )
    return questions, bad_index, bad_schema


def _resolve_category(requested: QueryCategory, n_used: int) -> QueryCategory:
    """Hạ nhóm khi model không dùng đủ số chunk mà nhóm đó đòi hỏi.

    Một câu `multi_hop` chỉ dựa vào một chunk thì nó là `factoid`. Giữ nguyên
    nhãn sai sẽ làm cột `multi_hop` trong bảng breakdown báo một năng lực mà hệ
    thống chưa từng được đo.
    """
    if requested is QueryCategory.UNANSWERABLE:
        return requested
    if requested in {QueryCategory.MULTI_HOP, QueryCategory.AGGREGATION} and n_used < 2:
        return QueryCategory.FACTOID
    return requested


def _build_draft(
    parsed: ParsedQuestion,
    group: ChunkGroup,
    *,
    response_model: str,
    requested_model: str,
    cost_usd: float,
    prompt_tokens: int,
    completion_tokens: int,
    batch_id: str,
) -> GoldenDraft | None:
    is_unanswerable = group.category is QueryCategory.UNANSWERABLE
    cited = [group.chunks[i - 1] for i in parsed.used_chunks]
    relevant_ids = [] if is_unanswerable else [c.chunk_id for c in cited]

    if not is_unanswerable and not relevant_ids:
        return None

    category = _resolve_category(group.category, len(relevant_ids))
    verified = True if is_unanswerable else bool(parsed.quote) and quote_found(parsed.quote, cited)

    try:
        query = GoldenQuery(
            query_id=query_id_for(category, parsed.query),
            query=parsed.query,
            category=category,
            lang=parsed.lang,
            relevant_chunk_ids=relevant_ids,
            reference_answer=parsed.reference_answer or None,
        )
    except Exception as exc:
        logger.warning("Câu không qua được schema GoldenQuery: %s", exc)
        return None

    return GoldenDraft(
        query=query,
        provenance=DraftProvenance(
            generator_model=response_model,
            generator_model_requested=requested_model,
            category_requested=group.category,
            source_chunk_ids=group.chunk_ids,
            supporting_quotes=[parsed.quote] if parsed.quote else [],
            quotes_verified=verified,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
            generated_at=datetime.now(UTC).isoformat(timespec="seconds"),
            batch_id=batch_id,
        ),
    )


def generate_drafts(
    provider: LLMProvider,
    groups: Sequence[ChunkGroup],
    *,
    questions_per_call: int = QUESTIONS_PER_CALL,
    seed: int = 20260817,
    max_tokens: int = MAX_TOKENS,
    progress: bool = False,
    checkpoint_path: Path | None = None,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> tuple[list[GoldenDraft], GenerationStats]:
    """Gọi model cho từng lô chunk và dựng bản nháp.

    `checkpoint_path` ghi từng bản nháp ngay khi có, và bỏ qua lô nào đã xong ở
    lần chạy trước. Một lượt đầy đủ tốn hơn một tiếng và tiền thật; đứt mạng ở
    lời gọi thứ 160 mà mất sạch là cái giá không đáng trả cho một file JSONL.

    `concurrency` chạy nhiều lời gọi song song. Mỗi lời gọi mất ~25 giây vì
    `deepseek-v4-flash` sinh 1.500–3.000 token suy luận trước khi viết JSON, mà
    thời gian đó là chờ mạng chứ không phải chờ CPU — chạy tuần tự là để máy
    ngồi không. Kết quả vẫn được ráp lại **theo đúng thứ tự lô** để hai lần chạy
    cùng seed cho ra cùng một file.
    """
    stats = GenerationStats(n_groups=len(groups))
    started = time.perf_counter()
    resumed: list[GoldenDraft] = []
    batch_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")

    done_groups: set[str] = set()
    if checkpoint_path is not None and checkpoint_path.exists():
        resumed, done_groups = _load_checkpoint(checkpoint_path)
        logger.info(
            "Tiếp tục từ checkpoint %s: đã có %d câu của %d lô",
            checkpoint_path,
            len(resumed),
            len(done_groups),
        )

    pending = [g for g in groups if _group_key(g.chunk_ids) not in done_groups]
    stats.n_skipped_groups = len(groups) - len(pending)

    lock = threading.Lock()
    results: dict[int, list[GoldenDraft]] = {}

    def work(index: int, group: ChunkGroup) -> None:
        produced = _run_one_group(
            provider,
            group,
            stats=stats,
            lock=lock,
            questions_per_call=questions_per_call,
            seed=seed,
            max_tokens=max_tokens,
            batch_id=batch_id,
        )
        with lock:
            results[index] = produced
            if checkpoint_path is not None:
                _append_checkpoint(checkpoint_path, produced, group)

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = [pool.submit(work, i, g) for i, g in enumerate(pending)]
        for future in _completed_with_progress(futures, enabled=progress):
            future.result()

    drafts = resumed + [d for index in sorted(results) for d in results[index]]
    stats.seconds = time.perf_counter() - started
    return drafts, stats


def _run_one_group(
    provider: LLMProvider,
    group: ChunkGroup,
    *,
    stats: GenerationStats,
    lock: threading.Lock,
    questions_per_call: int,
    seed: int,
    max_tokens: int,
    batch_id: str,
) -> list[GoldenDraft]:
    """Một lời gọi LLM cho một lô chunk. Chạy trong thread, nên mọi cập nhật
    `stats` đều phải nằm trong `lock`."""
    system, user = build_messages(group.category, group.chunks, n_questions=questions_per_call)
    try:
        response = provider.complete(
            [
                ChatMessage(role="system", content=system),
                ChatMessage(role="user", content=user),
            ],
            temperature=0.0,
            max_tokens=max_tokens,
            json_mode=True,
            seed=seed,
        )
    except LLMError as exc:
        with lock:
            stats.n_failed_calls += 1
        logger.warning("Bỏ lô %s: %s", group.chunk_ids[:1], exc)
        return []

    questions, bad_index, bad_schema = parse_response(response.text, len(group.chunks))

    with lock:
        stats.n_calls += 1
        stats.prompt_tokens += response.usage.prompt_tokens
        stats.completion_tokens += response.usage.completion_tokens
        stats.cost_usd += response.usage.cost_usd
        stats.models_served[response.model] = stats.models_served.get(response.model, 0) + 1
        stats.n_raw_questions += len(questions) + bad_index + bad_schema
        stats.n_rejected_bad_index += bad_index
        stats.n_rejected_schema += bad_schema
        if response.finish_reason == "length":
            # Phân biệt rõ với JSON hỏng: đây là hết ngân sách token, sửa bằng
            # `--max-tokens`, chứ không phải model trả về rác.
            stats.n_truncated += 1

    if response.finish_reason == "length":
        logger.warning(
            "Response bị cắt vì hết token (%d completion, %d dành cho suy luận). "
            "Tăng --max-tokens.",
            response.usage.completion_tokens,
            int(response.raw.get("reasoning_tokens", 0)),
        )

    # Chi phí của lời gọi chia đều cho các câu nó sinh ra — để cộng lại vẫn bằng
    # tổng thật, và để biết một câu dùng được giá bao nhiêu.
    divisor = max(1, len(questions))
    produced: list[GoldenDraft] = []
    for parsed in questions:
        draft = _build_draft(
            parsed,
            group,
            response_model=response.model,
            requested_model=response.model_requested,
            cost_usd=response.usage.cost_usd / divisor,
            prompt_tokens=response.usage.prompt_tokens // divisor,
            completion_tokens=response.usage.completion_tokens // divisor,
            batch_id=batch_id,
        )
        if draft is None:
            with lock:
                stats.n_rejected_schema += 1
            continue
        with lock:
            if draft.category_drifted:
                stats.n_category_drifted += 1
            if not draft.provenance.quotes_verified:
                stats.n_quotes_unverified += 1
        produced.append(draft)
    return produced


def _load_checkpoint(path: Path) -> tuple[list[GoldenDraft], set[str]]:
    """Đọc checkpoint, bỏ qua dòng hỏng thay vì làm hỏng cả lần chạy tiếp.

    Đứt điện giữa lúc ghi để lại một dòng JSON dở. Checkpoint là thứ để cứu công
    đã bỏ ra — nó mà tự làm sập lần chạy sau thì vô nghĩa.
    """
    drafts: list[GoldenDraft] = []
    done: set[str] = set()
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("Bỏ dòng checkpoint hỏng %s:%d", path, line_no)
            continue
        marker = payload.get("__empty_group__") if isinstance(payload, dict) else None
        if marker:
            done.add(str(marker))
            continue
        try:
            draft = GoldenDraft.model_validate(payload)
        except Exception:
            logger.warning("Bỏ dòng checkpoint không hợp schema %s:%d", path, line_no)
            continue
        drafts.append(draft)
        done.add(_group_key(draft.provenance.source_chunk_ids))
    return drafts, done


def _group_key(chunk_ids: Sequence[str]) -> str:
    """Khoá nhận diện một lô, dùng để biết lô nào đã xong ở lần chạy trước."""
    return "|".join(sorted(chunk_ids))


def _append_checkpoint(path: Path, drafts: Sequence[GoldenDraft], group: ChunkGroup) -> None:
    """Ghi nối vào checkpoint. Lô không sinh ra câu nào cũng phải ghi dấu.

    Nếu chỉ ghi khi có câu thì lô bị model trả về rỗng (thường gặp với
    `table_lookup` khi đoạn văn không có bảng) sẽ được gọi lại ở mỗi lần chạy —
    trả tiền nhiều lần cho cùng một câu trả lời rỗng.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for draft in drafts:
            handle.write(draft.model_dump_json() + "\n")
        if not drafts:
            handle.write(
                json.dumps({"__empty_group__": _group_key(group.chunk_ids)}, ensure_ascii=False)
                + "\n"
            )


def _completed_with_progress(
    futures: Sequence[Future[None]], *, enabled: bool
) -> Iterable[Future[None]]:
    """Duyệt future theo thứ tự hoàn thành, có thanh tiến trình nếu bật."""
    stream = as_completed(futures)
    if not enabled:
        return stream
    try:
        from tqdm import tqdm  # type: ignore[import-untyped]
    except ImportError:  # pragma: no cover - tqdm nằm ở extra `pipeline`
        return stream
    return cast(
        "Iterable[Future[None]]", tqdm(stream, total=len(futures), desc="golden", unit="lô")
    )


def _parse_quotas(raw: str | None) -> dict[QueryCategory, int]:
    if not raw:
        return dict(DEFAULT_QUOTAS)
    quotas: dict[QueryCategory, int] = {}
    for piece in raw.split(","):
        name, _, count = piece.partition("=")
        quotas[QueryCategory(name.strip())] = int(count)
    return quotas


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sinh bản nháp golden set từ chunk thật trong index"
    )
    parser.add_argument("--index-config", type=Path, default=Path("configs/indexing/baseline.yaml"))
    parser.add_argument("--out", type=Path, default=Path("data/golden/draft_v1.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("plans/reports/goldenset-draft.json"))
    parser.add_argument(
        "--model",
        default=DEFAULT_DEEPSEEK_MODEL,
        help="Slug model tường minh. KHÔNG dùng preset, và tránh cả bí danh "
        "`deepseek-chat` — nó là con trỏ do server nắm.",
    )
    parser.add_argument(
        "--quotas",
        help="Ghi đè hạn mức, dạng `factoid=45,multi_hop=25`. Mặc định: xem DEFAULT_QUOTAS",
    )
    parser.add_argument("--seed", type=int, default=20260817)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/golden/.draft_checkpoint.jsonl"),
        help="Ghi từng câu ngay khi có; chạy lại sẽ bỏ qua lô đã xong",
    )
    parser.add_argument(
        "--no-checkpoint", action="store_true", help="Tắt checkpoint (chạy lại từ đầu)"
    )
    parser.add_argument("--questions-per-call", type=int, default=QUESTIONS_PER_CALL)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Số lời gọi song song. Cao hơn thì gặp 429 và backoff ăn hết phần thắng",
    )
    parser.add_argument("--jaccard-threshold", type=float, default=0.8)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ chọn mẫu và in prompt của lô đầu tiên, không gọi API",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    settings = get_settings()
    index_config = load_index_config(args.index_config)
    embeddings = index_config.build_embeddings()
    retriever = index_config.build_retriever(
        embeddings,
        url=settings.qdrant_url,
        api_key=(settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None),
    )

    quotas = _parse_quotas(args.quotas)
    groups = sample_groups(retriever, quotas, seed=args.seed)
    logger.info("Chọn được %d lô chunk cho %d nhóm", len(groups), len(quotas))

    if args.dry_run:
        return _run_dry(groups, args.questions_per_call)

    settings.require("deepseek_api_key")
    api_key = settings.deepseek_api_key
    if api_key is None:  # pragma: no cover - `require` ở trên đã chặn
        raise RuntimeError("Thiếu DEEPSEEK_API_KEY")
    provider = build_deepseek_provider(
        args.model,
        api_key=api_key.get_secret_value(),
        base_url=settings.deepseek_base_url,
    )

    checkpoint = None if args.no_checkpoint else args.checkpoint
    drafts, stats = generate_drafts(
        provider,
        groups,
        questions_per_call=args.questions_per_call,
        seed=args.seed,
        max_tokens=args.max_tokens,
        concurrency=args.concurrency,
        progress=True,
        checkpoint_path=checkpoint,
    )
    result = deduplicate_drafts(drafts, jaccard_threshold=args.jaccard_threshold)
    stats.n_duplicates_removed = result.n_removed

    write_drafts(args.out, result.kept)
    _write_report(args.report, result.kept, stats, args)
    if checkpoint is not None and checkpoint.exists():
        # Kết quả đã nằm an toàn ở `--out`; giữ checkpoint lại sẽ khiến lần chạy
        # sau tưởng mọi lô đều xong và không sinh gì mới.
        checkpoint.unlink()
    _log_summary(result.kept, stats)
    return 0


def _run_dry(groups: Sequence[ChunkGroup], questions_per_call: int) -> int:
    logger.info("DRY RUN — không gọi API, không tốn tiền")
    by_category: dict[str, int] = {}
    for group in groups:
        by_category[group.category.value] = by_category.get(group.category.value, 0) + 1
    for name, count in sorted(by_category.items()):
        logger.info("  %-14s %3d lô", name, count)
    logger.info(
        "  Ước tính %d lời gọi → tối đa %d câu",
        len(groups),
        len(groups) * questions_per_call,
    )
    if groups:
        system, user = build_messages(
            groups[0].category, groups[0].chunks, n_questions=questions_per_call
        )
        logger.info(
            "Prompt của lô đầu tiên (%s):\n%s\n---\n%s",
            groups[0].category.value,
            system,
            user[:1500],
        )
    return 0


def _write_report(
    path: Path,
    drafts: Sequence[GoldenDraft],
    stats: GenerationStats,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "index_config": str(args.index_config),
        "model_requested": args.model,
        "seed": args.seed,
        "generation": stats.as_dict(),
        "drafts": json.loads(summary_json(drafts)),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Đã ghi báo cáo %s", path)


def _log_summary(drafts: Sequence[GoldenDraft], stats: GenerationStats) -> None:
    summary = json.loads(summary_json(drafts))
    logger.info("─" * 62)
    logger.info("Bản nháp golden set")
    logger.info(
        "  lời gọi          %d (hỏng %d · bỏ qua vì checkpoint %d)",
        stats.n_calls,
        stats.n_failed_calls,
        stats.n_skipped_groups,
    )
    logger.info(
        "  câu              %d thô → %d giữ lại (bỏ trùng %d · sai chỉ số %d · sai schema %d)",
        stats.n_raw_questions,
        len(drafts),
        stats.n_duplicates_removed,
        stats.n_rejected_bad_index,
        stats.n_rejected_schema,
    )
    if stats.n_truncated:
        logger.warning("  BỊ CẮT           %d response hết ngân sách token", stats.n_truncated)
    for name, count in summary["by_category"].items():
        logger.info("    %-14s %3d", name, count)
    logger.info(
        "  cần đọc kỹ       %d (trích dẫn không kiểm chứng được %d · trôi nhóm %d)",
        summary["needs_close_review"],
        summary["quotes_unverified"],
        summary["category_drifted"],
    )
    logger.info(
        "  token            %s vào · %s ra",
        f"{stats.prompt_tokens:,}",
        f"{stats.completion_tokens:,}",
    )
    logger.info(
        "  chi phí          $%.4f (%.5f $/câu giữ lại)",
        stats.cost_usd,
        stats.cost_usd / max(1, len(drafts)),
    )
    logger.info("  model phục vụ    %s", stats.models_served)
    logger.info("  thời gian        %.1fs", stats.seconds)
    logger.info("─" * 62)


if __name__ == "__main__":
    sys.exit(main())
