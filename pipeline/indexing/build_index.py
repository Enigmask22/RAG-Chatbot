"""Corpus → chunk → embed → Qdrant. Chạy lại bao nhiêu lần cũng cho cùng index.

Tính idempotent ở đây gồm ba tầng, và cả ba đều cần thiết:

1. **Point ID xác định** (`chunk_point_id` = UUIDv5 của `chunk_id`, làm ở `W1-07`).
   Upsert lại cùng một chunk ghi đè đúng point cũ. Tầng này lo phần "không sinh
   bản trùng".
2. **Dọn chunk thừa.** Sửa một tài liệu cho ra ít chunk hơn trước thì các point
   `doc::00042`…`doc::00050` vẫn nằm lại trong collection. Chúng không trùng
   lặp, chúng là **rác trỏ tới văn bản không còn tồn tại** — và retriever vẫn
   trả chúng về. Tầng 1 không bắt được; phải nhớ số chunk cũ mới xoá đúng.
3. **Chặn trộn hai cấu hình.** Chạy config A rồi chạy config B vào cùng
   collection cho ra một index nửa nọ nửa kia, mà `count()` vẫn trông hợp lý.
   Mọi số eval sau đó đều vô nghĩa. State file giữ `fingerprint`; khác là dừng.

State file (`.cache/index_state/{name}.json`) là bộ nhớ cho tầng 2 và 3. Nó
**không** phải nguồn sự thật — Qdrant mới là. Nên trước khi tin state, script
đối chiếu tổng số chunk với `collection.count()`; lệch thì bỏ state và làm lại
từ đầu (rẻ hơn nhiều so với một index sai âm thầm).
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from rag_core.chunking import Chunker
from rag_core.chunking.cache import CachedChunker
from rag_core.embedding import EmbeddingProvider
from rag_core.embedding.truncation import token_stats
from rag_core.retrieval import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME
from rag_core.schemas import Chunk, Document
from rag_core.settings import get_settings

from .config import IndexConfig, load_index_config
from .corpus_loader import load_documents

if TYPE_CHECKING:
    from rag_core.retrieval.qdrant_store import QdrantDenseRetriever

__all__ = ["BuildReport", "IndexState", "build_index", "main"]

logger = logging.getLogger("pipeline.indexing")


class DocState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content_hash: str
    n_chunks: int = Field(ge=0)


class IndexState(BaseModel):
    """Những gì đã được ghi vào collection, ở lần build gần nhất."""

    model_config = ConfigDict(extra="forbid")

    config_name: str
    fingerprint: str
    collection: str
    embedding_model: str
    embedding_dim: int
    chunker_name: str
    documents: dict[str, DocState] = Field(default_factory=dict)
    updated_at: str = ""

    @property
    def total_chunks(self) -> int:
        return sum(doc.n_chunks for doc in self.documents.values())

    @classmethod
    def load(cls, path: str | Path) -> IndexState | None:
        source = Path(path)
        if not source.exists():
            return None
        try:
            return cls.model_validate_json(source.read_text(encoding="utf-8"))
        except Exception:
            # State hỏng hoặc schema đã đổi. Nó là cache tái sinh được, không
            # phải dữ liệu gốc — bỏ đi và build lại còn hơn đoán mò.
            logger.warning("State %s không đọc được, coi như chưa có", source)
            return None

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = datetime.now(UTC).isoformat(timespec="seconds")
        target.write_text(self.model_dump_json(indent=2), encoding="utf-8")


@dataclass
class BuildReport:
    config_name: str
    collection: str
    fingerprint: str
    embedding_model: str
    embedding_device: str
    embedding_dim: int
    chunker_name: str
    n_documents: int
    n_documents_indexed: int
    n_documents_skipped: int
    n_documents_removed: int
    n_chunks_written: int
    n_stale_points_deleted: int
    collection_count: int
    chars_in: int
    chars_out: int
    vector_names: tuple[str, ...] = (DENSE_VECTOR_NAME,)
    """Named vector thật sự đã được ghi. Ghi lại vì `W2-02` làm collection có
    thể mang một hoặc hai loại vector, và một report chỉ nói "15.814 chunk" thì
    không phân biệt được index dense-only với index hybrid — hai thứ cho ra hai
    bảng eval khác nhau."""
    chunk_len: dict[str, float] = field(default_factory=dict)
    truncation: dict[str, float] = field(default_factory=dict)
    seconds: dict[str, float] = field(default_factory=dict)
    cache: dict[str, float] = field(default_factory=dict)
    created_at: str = ""

    @property
    def context_inflation(self) -> float:
        """`chars_out / chars_in` — chi phí thật của `neighbor_context_chars`.

        Bằng ~1.0 nếu tắt; bản POC bật 100 ký tự nên số này sẽ > 1 và đó chính
        là phần token phải trả thêm cho mỗi lần embed.
        """
        return self.chars_out / self.chars_in if self.chars_in else 0.0

    def to_json(self) -> str:
        payload = asdict(self)
        payload["context_inflation"] = round(self.context_inflation, 4)
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def log_summary(self) -> None:
        logger.info("─" * 62)
        logger.info("Index `%s` → collection `%s`", self.config_name, self.collection)
        logger.info("  fingerprint      %s", self.fingerprint[:16])
        logger.info(
            "  embedding        %s (%s, %d chiều)",
            self.embedding_model,
            self.embedding_device,
            self.embedding_dim,
        )
        logger.info("  chunker          %s", self.chunker_name)
        logger.info(
            "  tài liệu         %d (index %d · bỏ qua %d · gỡ %d)",
            self.n_documents,
            self.n_documents_indexed,
            self.n_documents_skipped,
            self.n_documents_removed,
        )
        logger.info(
            "  chunk            ghi %d · xoá thừa %d · tổng trong collection %d",
            self.n_chunks_written,
            self.n_stale_points_deleted,
            self.collection_count,
        )
        logger.info(
            "  named vector     %s%s",
            ", ".join(self.vector_names),
            "" if SPARSE_VECTOR_NAME in self.vector_names else "  (chưa có sparse — W2-02)",
        )
        if self.chunk_len:
            logger.info(
                "  độ dài chunk     trung bình %.0f · p50 %.0f · p95 %.0f · max %.0f ký tự",
                self.chunk_len.get("mean", 0),
                self.chunk_len.get("p50", 0),
                self.chunk_len.get("p95", 0),
                self.chunk_len.get("max", 0),
            )
        self._log_truncation()
        logger.info(
            "  ký tự            vào %s → embed %s (hệ số %.2fx)",
            f"{self.chars_in:,}",
            f"{self.chars_out:,}",
            self.context_inflation,
        )
        if self.cache:
            logger.info(
                "  cache chunk      hit %d · miss %d (%.0f%%)",
                int(self.cache.get("hits", 0)),
                int(self.cache.get("misses", 0)),
                100 * self.cache.get("hit_rate", 0.0),
            )
        total = self.seconds.get("total", 0.0)
        logger.info(
            "  thời gian        nạp %.1fs · chunk %.1fs · embed+ghi %.1fs · tổng %.1fs",
            self.seconds.get("load", 0.0),
            self.seconds.get("chunk", 0.0),
            self.seconds.get("upsert", 0.0),
            total,
        )
        if total > 0 and self.n_chunks_written:
            logger.info("  thông lượng      %.1f chunk/giây", self.n_chunks_written / total)
        logger.info("─" * 62)

    def _log_truncation(self) -> None:
        """In phần `TD-11`. Cố ý dùng WARNING, không phải INFO.

        Đây là lỗi đã trốn được suốt W1 vì nó **không có triệu chứng**: index
        đủ chunk, đủ chiều, mọi số trong report đều đẹp, chỉ có phần đuôi văn
        bản là không tồn tại đối với vector. Một dòng INFO giữa mười dòng INFO
        khác thì cũng trốn được lần nữa.
        """
        if not self.truncation and not self.n_chunks_written:
            logger.info("  cắt token        không có chunk mới nên không đo lại")
            return
        if not self.truncation:
            logger.warning(
                "  ⚠️ TD-11          KHÔNG đo được (provider không cho biết giới hạn token). "
                "Đây là 'không biết', không phải 'không bị cắt'"
            )
            return
        lost = self.truncation.get("tokens_lost_ratio", 0.0)
        n_trunc = int(self.truncation.get("n_truncated", 0))
        line = (
            "  cắt token        %d/%d chunk vượt %d token (%.1f%%) · "
            "%.1f%% token không tới được vector"
        )
        args = (
            n_trunc,
            int(self.truncation.get("n_texts", 0)),
            int(self.truncation.get("limit", 0)),
            100 * self.truncation.get("truncated_ratio", 0.0),
            100 * lost,
        )
        if n_trunc:
            logger.warning(line.replace("  cắt token       ", "  ⚠️ cắt token    "), *args)
        else:
            logger.info(line, *args)


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(pct / 100.0 * (len(ordered) - 1)))
    return float(ordered[index])


def _chunk_length_stats(lengths: Sequence[int]) -> dict[str, float]:
    if not lengths:
        return {}
    return {
        "mean": float(statistics.fmean(lengths)),
        "p50": _percentile(lengths, 50),
        "p95": _percentile(lengths, 95),
        "min": float(min(lengths)),
        "max": float(max(lengths)),
    }


def _stale_point_ids(doc_id: str, old_count: int, new_count: int) -> list[str]:
    """ID của các point thuộc chỉ số ≥ `new_count` — chunk của bản cũ, giờ thừa."""
    from rag_core.retrieval.qdrant_store import chunk_point_id

    return [chunk_point_id(f"{doc_id}::{i:05d}") for i in range(new_count, old_count)]


def _reconcile_state(
    state: IndexState | None,
    config: IndexConfig,
    retriever: QdrantDenseRetriever,
    embedding_dim: int,
    chunker_name: str,
    *,
    allow_mixed: bool,
) -> IndexState:
    """Quyết định có tin được state cũ không; trả về state để làm việc tiếp."""
    fresh = IndexState(
        config_name=config.name,
        fingerprint=config.fingerprint,
        collection=config.collection_name,
        embedding_model=config.embedding_model,
        embedding_dim=embedding_dim,
        chunker_name=chunker_name,
    )
    if state is None:
        return fresh

    if state.fingerprint != config.fingerprint:
        message = (
            f"Collection `{config.collection_name}` đang chứa index của fingerprint "
            f"{state.fingerprint[:16]} nhưng config hiện tại là {config.fingerprint[:16]}. "
            "Ghi đè lên sẽ tạo ra index trộn hai cấu hình — mọi số eval sau đó vô nghĩa. "
            "Chạy lại với `--recreate` để build sạch, hoặc đổi `name`/`collection` "
            "trong config để hai lần chạy không giẫm lên nhau."
        )
        if not allow_mixed:
            raise RuntimeError(message)
        logger.warning("%s (bỏ qua vì --allow-mixed)", message)
        return fresh

    actual = retriever.count()
    if actual != state.total_chunks:
        # Collection bị xoá/sửa ngoài script, hoặc lần chạy trước đứt giữa chừng.
        logger.warning(
            "State ghi %d chunk nhưng collection có %d — bỏ state, index lại toàn bộ",
            state.total_chunks,
            actual,
        )
        return fresh
    return state


def build_index(
    config: IndexConfig,
    *,
    qdrant_url: str,
    qdrant_api_key: str | None = None,
    recreate: bool = False,
    allow_mixed: bool = False,
    verify_hash: bool = True,
    progress: bool = False,
) -> BuildReport:
    started = time.perf_counter()

    t_load = time.perf_counter()
    documents = load_documents(
        config.manifest_path,
        config.corpus_dir,
        languages=config.languages,
        doc_types=config.doc_types,
        max_documents=config.max_documents,
        verify_hash=verify_hash,
    )
    load_seconds = time.perf_counter() - t_load
    logger.info("Nạp %d tài liệu trong %.1fs", len(documents), load_seconds)

    embeddings = config.build_embeddings()
    chunker = config.build_chunker(embeddings)
    # Khai báo tổng số tài liệu **trước** vòng lặp: chunker hybrid chọn nhánh
    # theo con số này, và nếu để nó tự suy từ lô 1 tài liệu thì luôn ra semantic.
    chunker.prepare(len(documents))
    dimension = embeddings.dimension
    chunker_name = chunker.name
    logger.info("Embedding %s · %d chiều · chunker %s", embeddings.name, dimension, chunker_name)

    retriever = config.build_retriever(embeddings, url=qdrant_url, api_key=qdrant_api_key)
    retriever.ensure_collection(recreate=recreate)

    state = None if recreate else IndexState.load(config.state_path)
    state = _reconcile_state(
        state, config, retriever, dimension, chunker_name, allow_mixed=allow_mixed
    )

    chunk_seconds = 0.0
    upsert_seconds = 0.0
    chars_in = 0
    chars_out = 0
    lengths: list[int] = []
    # `None` = provider không đếm được token. Phân biệt với list rỗng (đếm được
    # nhưng chưa có chunk nào) — hai thứ này in ra hai dòng log khác nhau.
    token_counts: list[int] | None = [] if embeddings.max_sequence_tokens else None
    n_written = 0
    n_indexed = 0
    n_skipped = 0
    n_stale = 0

    seen_doc_ids: set[str] = set()
    for doc in _with_progress(documents, enabled=progress):
        seen_doc_ids.add(doc.doc_id)
        chars_in += len(doc.content)
        previous = state.documents.get(doc.doc_id)
        if previous is not None and previous.content_hash == doc.content_hash:
            n_skipped += 1
            continue

        t0 = time.perf_counter()
        chunks = chunker.chunk([doc])
        chunk_seconds += time.perf_counter() - t0

        chars_out += sum(len(c.content) for c in chunks)
        lengths.extend(len(c.content) for c in chunks)
        token_counts = _accumulate_tokens(token_counts, embeddings, chunks)

        t0 = time.perf_counter()
        n_written += retriever.upsert(chunks, batch_size=config.upsert_batch_size)
        n_stale += _delete_stale(retriever, doc.doc_id, previous, len(chunks))
        upsert_seconds += time.perf_counter() - t0

        state.documents[doc.doc_id] = DocState(content_hash=doc.content_hash, n_chunks=len(chunks))
        n_indexed += 1

    # Tài liệu đã rời manifest phải rời cả index — nếu không, index là hợp của
    # mọi lần chạy trong quá khứ chứ không phải ảnh của manifest hiện tại.
    removed = [doc_id for doc_id in state.documents if doc_id not in seen_doc_ids]
    for doc_id in removed:
        logger.info("Tài liệu %s không còn trong manifest — xoá khỏi collection", doc_id)
        retriever.delete_by_doc(doc_id)
        del state.documents[doc_id]

    state.save(config.state_path)

    cache_stats: dict[str, float] = {}
    if isinstance(chunker, CachedChunker):
        stats = chunker.cache.stats()
        cache_stats = {
            "hits": float(stats.hits),
            "misses": float(stats.misses),
            "hit_rate": stats.hit_rate,
            "entries": float(stats.entries),
        }

    limit = embeddings.max_sequence_tokens
    truncation: dict[str, float] = {}
    if token_counts and limit is not None:
        truncation = token_stats(token_counts, limit=limit, chars=lengths).as_dict()

    report = BuildReport(
        config_name=config.name,
        collection=config.collection_name,
        fingerprint=config.fingerprint,
        embedding_model=config.embedding_model,
        embedding_device=getattr(embeddings, "device", config.embedding_device),
        embedding_dim=dimension,
        chunker_name=chunker_name,
        n_documents=len(documents),
        n_documents_indexed=n_indexed,
        n_documents_skipped=n_skipped,
        n_documents_removed=len(removed),
        n_chunks_written=n_written,
        n_stale_points_deleted=n_stale,
        collection_count=retriever.count(),
        chars_in=chars_in,
        vector_names=(
            (DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME)
            if retriever.writes_sparse
            else (DENSE_VECTOR_NAME,)
        ),
        chars_out=chars_out,
        chunk_len=_chunk_length_stats(lengths),
        truncation=truncation,
        seconds={
            "load": load_seconds,
            "chunk": chunk_seconds,
            "upsert": upsert_seconds,
            "total": time.perf_counter() - started,
        },
        cache=cache_stats,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
    )
    report.log_summary()
    return report


def _accumulate_tokens(
    counts: list[int] | None,
    embeddings: EmbeddingProvider,
    chunks: Sequence[Chunk],
) -> list[int] | None:
    """Cộng dồn số token của lô chunk vừa sinh; `None` nghĩa là không đếm được.

    Một lần trả `None` là bỏ luôn cả phép đo cho lần build này. Cố ý không
    "đếm được bao nhiêu thì đếm": một thống kê trên tập con không rõ là tập nào
    còn tệ hơn không có thống kê, vì nó vẫn trông như một con số.
    """
    if counts is None:
        return None
    counted = embeddings.count_tokens([c.content for c in chunks])
    if counted is None:  # pragma: no cover - provider khai có limit mà không đếm
        logger.warning("Provider %s có giới hạn token nhưng không đếm được", embeddings.name)
        return None
    counts.extend(counted)
    return counts


def _delete_stale(
    retriever: QdrantDenseRetriever,
    doc_id: str,
    previous: DocState | None,
    new_count: int,
) -> int:
    """Xoá point của bản cũ mà bản mới không còn dùng tới.

    Thứ tự cố ý là **upsert trước, xoá sau**: nếu tiến trình chết ở giữa thì tài
    liệu vẫn có mặt trong index (thừa vài chunk cũ) chứ không biến mất. Chạy lại
    sẽ dọn nốt. Xoá trước rồi chết là mất hẳn tài liệu khỏi index.
    """
    if previous is None or previous.n_chunks <= new_count:
        return 0
    stale = _stale_point_ids(doc_id, previous.n_chunks, new_count)
    deleted = retriever.delete_points(stale)
    logger.debug("%s: xoá %d chunk thừa của bản cũ", doc_id, deleted)
    return deleted


def _with_progress(documents: Sequence[Document], *, enabled: bool) -> Sequence[Document]:
    if not enabled:
        return documents
    try:
        from tqdm import tqdm  # type: ignore[import-untyped]
    except ImportError:  # pragma: no cover - tqdm nằm ở extra `pipeline`
        return documents
    return list(tqdm(documents, desc="index", unit="doc"))


def _chunk_preview(chunker: Chunker, documents: Sequence[Document], limit: int) -> list[Chunk]:
    chunker.prepare(len(documents))
    preview: list[Chunk] = []
    for doc in documents[:limit]:
        preview.extend(chunker.chunk([doc]))
    return preview


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build index Qdrant từ corpus (idempotent, chạy lại được)"
    )
    parser.add_argument("--config", type=Path, default=Path("configs/indexing/baseline.yaml"))
    parser.add_argument("--name", help="Ghi đè `name` trong config")
    parser.add_argument("--collection", help="Ghi đè tên collection")
    parser.add_argument(
        "--max-documents", type=int, help="Chỉ index N tài liệu đầu (theo doc_id) — dùng để thử"
    )
    parser.add_argument(
        "--recreate", action="store_true", help="Xoá collection và build lại từ đầu"
    )
    parser.add_argument(
        "--allow-mixed",
        action="store_true",
        help="Cho phép ghi đè lên index có fingerprint khác (mặc định là chặn)",
    )
    parser.add_argument(
        "--no-verify-hash",
        action="store_true",
        help="Bỏ qua đối chiếu sha256 với manifest (không khuyến khích)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ chunk và in thống kê, không chạm Qdrant",
    )
    parser.add_argument("--report", type=Path, help="Ghi báo cáo JSON ra file")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    # qdrant-client log mỗi request qua httpx ở mức INFO — với vài nghìn chunk
    # thì phần tóm tắt cuối cùng trôi mất giữa hàng trăm dòng "HTTP/1.1 200 OK".
    logging.getLogger("httpx").setLevel(logging.WARNING)

    config = load_index_config(
        args.config,
        name=args.name,
        collection=args.collection,
        max_documents=args.max_documents,
    )

    if args.dry_run:
        return _run_dry(config, verify_hash=not args.no_verify_hash)

    settings = get_settings()
    api_key = settings.qdrant_api_key.get_secret_value() if settings.qdrant_api_key else None
    report = build_index(
        config,
        qdrant_url=settings.qdrant_url,
        qdrant_api_key=api_key,
        recreate=args.recreate,
        allow_mixed=args.allow_mixed,
        verify_hash=not args.no_verify_hash,
        progress=True,
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report.to_json(), encoding="utf-8")
        logger.info("Đã ghi báo cáo %s", args.report)
    return 0


def _run_dry(config: IndexConfig, *, verify_hash: bool, preview_docs: int = 3) -> int:
    """Chunk vài tài liệu rồi in thống kê — kiểm tra config mà không cần Qdrant."""
    documents = load_documents(
        config.manifest_path,
        config.corpus_dir,
        languages=config.languages,
        doc_types=config.doc_types,
        max_documents=config.max_documents,
        verify_hash=verify_hash,
    )
    embeddings = config.build_embeddings()
    chunker = config.build_chunker(embeddings)
    chunks = _chunk_preview(chunker, documents, preview_docs)
    lengths = [len(c.content) for c in chunks]
    logger.info("DRY RUN — không ghi gì vào Qdrant")
    logger.info("  fingerprint  %s", config.fingerprint[:16])
    logger.info("  collection   %s (sẽ ghi vào đây khi chạy thật)", config.collection_name)
    logger.info("  chunker      %s", chunker.name)
    logger.info("  tài liệu     %d trong corpus, thử chunk %d", len(documents), preview_docs)
    logger.info("  chunk        %d từ %d tài liệu đầu", len(chunks), preview_docs)
    for key, value in _chunk_length_stats(lengths).items():
        logger.info("    %-6s %.0f ký tự", key, value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
