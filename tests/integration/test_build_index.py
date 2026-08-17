"""W1-08 — `build_index` chạy thật trên Qdrant.

Ba nhóm kiểm tra ứng với ba tầng idempotent mô tả ở đầu `build_index.py`:

* `TestIdempotent` — chạy hai lần, số point không đổi (tầng 1: point ID xác định).
* `TestDocumentChanges` — tài liệu ngắn lại / biến mất khỏi manifest thì chunk cũ
  phải đi theo (tầng 2). Đây là tầng mà chỉ dựa vào point ID xác định sẽ **không**
  bắt được, và nó im lặng làm hỏng eval.
* `TestConfigGuard` — đổi config rồi ghi đè vào cùng collection thì phải dừng
  (tầng 3).

Toàn bộ dùng `HashingEmbeddingProvider` để không phải tải model — thứ đang được
kiểm tra là logic đồng bộ index, không phải chất lượng vector.
"""

from __future__ import annotations

import hashlib
import os
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

pytest.importorskip("qdrant_client", reason="cần extra `qdrant`: uv sync --extra qdrant")

from pipeline.corpus.manifest import CorpusEntry, write_manifest
from pipeline.indexing.build_index import BuildReport, IndexState, build_index
from pipeline.indexing.config import IndexConfig
from rag_core.embedding import HashingEmbeddingProvider
from rag_core.retrieval.qdrant_store import QdrantDenseRetriever, chunk_point_id
from rag_core.schemas import DocType, Language

pytestmark = pytest.mark.integration

QDRANT_URL = os.getenv("QDRANT_URL", "http://127.0.0.1:6333")

# Đủ dài để cắt ra nhiều chunk với chunk_size=200.
_PARAGRAPH = (
    "Ngân sách nhà nước năm 2024 tăng chi đầu tư công cho hạ tầng giao thông "
    "tại các tỉnh miền Trung và đồng bằng sông Cửu Long. "
)


def _write_corpus(root: Path, bodies: dict[str, str]) -> Path:
    """Ghi corpus + manifest, trả về đường dẫn manifest."""
    entries = []
    for doc_id, text in bodies.items():
        # Gắn `doc_id` vào nội dung: manifest từ chối hai tài liệu trùng sha256,
        # và đó là hành vi đúng (xem `validate_manifest`).
        payload = f"{doc_id}. {text}".encode()
        path = root / f"{doc_id}.txt"
        path.write_bytes(payload)
        entries.append(
            CorpusEntry(
                doc_id=doc_id,
                relative_path=path.name,
                source_url=f"https://example.org/{doc_id}",
                license="CC BY 4.0",
                sha256=hashlib.sha256(payload).hexdigest(),
                bytes=len(payload),
                source="test",
                lang=Language.VI,
                doc_type=DocType.DEV_REPORT,
            )
        )
    manifest = root / "manifest.csv"
    write_manifest(manifest, entries)
    return manifest


@pytest.fixture
def workspace(tmp_path: Path) -> Iterator[tuple[Path, str]]:
    """Thư mục làm việc + tên collection dùng một lần rồi xoá."""
    collection = f"test_build_{uuid.uuid4().hex[:8]}"
    yield tmp_path, collection
    client = _store(collection).client
    if client.collection_exists(collection):
        client.delete_collection(collection)


def _store(collection: str) -> QdrantDenseRetriever:
    return QdrantDenseRetriever(
        HashingEmbeddingProvider(dimension=64), collection=collection, url=QDRANT_URL
    )


def _config(root: Path, collection: str, **overrides: object) -> IndexConfig:
    base: dict[str, object] = {
        "name": collection,
        "collection": collection,
        "manifest_path": root / "manifest.csv",
        "corpus_dir": root,
        "embedding_model": "hashing:64",
        "use_cache": False,
        "state_dir": root / "state",
        "chunking": {"strategy": "fixed", "chunk_size": 200, "chunk_overlap": 20},
    }
    base.update(overrides)
    return IndexConfig.model_validate(base)


def _build(
    config: IndexConfig, *, recreate: bool = False, allow_mixed: bool = False
) -> BuildReport:
    return build_index(config, qdrant_url=QDRANT_URL, recreate=recreate, allow_mixed=allow_mixed)


class TestFirstBuild:
    def test_writes_chunks_and_reports_them(self, workspace: tuple[Path, str]) -> None:
        root, collection = workspace
        _write_corpus(root, {"d-1": _PARAGRAPH * 20, "d-2": _PARAGRAPH * 10})
        report = _build(_config(root, collection))

        assert report.n_documents == 2
        assert report.n_chunks_written > 0
        assert report.collection_count == report.n_chunks_written

    def test_records_state_for_every_document(self, workspace: tuple[Path, str]) -> None:
        root, collection = workspace
        _write_corpus(root, {"d-1": _PARAGRAPH * 20, "d-2": _PARAGRAPH * 10})
        config = _config(root, collection)
        report = _build(config)

        state = IndexState.load(config.state_path)
        assert state is not None
        assert set(state.documents) == {"d-1", "d-2"}
        assert state.total_chunks == report.collection_count
        assert state.fingerprint == config.fingerprint

    def test_neighbor_context_is_the_only_thing_that_inflates_embedded_text(
        self, workspace: tuple[Path, str]
    ) -> None:
        """Đo trực tiếp chi phí của `neighbor_context_chars` — cùng corpus, hai config.

        Cố ý so hai lần chạy với nhau thay vì so với một hằng số: tỉ lệ tuyệt đối
        phụ thuộc `chunk_size` (đệm 100 ký tự vào chunk 1500 khác hẳn vào chunk
        300), nên một ngưỡng cứng sẽ vỡ mỗi lần đổi tham số mà không có lỗi thật.
        Chỉ số này là thứ dùng để đọc con số `1.24x` của baseline `W1-13`.
        """
        root, collection = workspace
        _write_corpus(root, {"d-1": _PARAGRAPH * 20})
        base = _config(root, collection)
        with_context = _config(
            root,
            collection,
            chunking={
                "strategy": "fixed",
                "chunk_size": 200,
                "chunk_overlap": 20,
                "neighbor_context_chars": 100,
            },
        )

        plain = _build(base)
        padded = _build(with_context, recreate=True)

        assert padded.context_inflation > plain.context_inflation
        # Không bật thì phần text đem embed xấp xỉ đúng bằng text gốc.
        assert 0.95 < plain.context_inflation < 1.1


class TestIdempotent:
    def test_second_run_writes_nothing_and_count_is_stable(
        self, workspace: tuple[Path, str]
    ) -> None:
        """Yêu cầu DoD của `W1-08`: chạy 2 lần không sinh bản trùng."""
        root, collection = workspace
        _write_corpus(root, {"d-1": _PARAGRAPH * 20, "d-2": _PARAGRAPH * 10})
        config = _config(root, collection)

        first = _build(config)
        second = _build(config)

        assert second.n_documents_skipped == 2
        assert second.n_chunks_written == 0
        assert second.collection_count == first.collection_count

    def test_stays_stable_even_without_state_file(self, workspace: tuple[Path, str]) -> None:
        """Xoá state thì phải index lại, nhưng số point vẫn y nguyên.

        Đây là chỗ point ID xác định (UUIDv5 của `chunk_id`) làm việc: không có
        nó, mất state là nhân đôi collection.
        """
        root, collection = workspace
        _write_corpus(root, {"d-1": _PARAGRAPH * 20})
        config = _config(root, collection)
        first = _build(config)

        config.state_path.unlink()
        second = _build(config)

        assert second.n_documents_indexed == 1
        assert second.collection_count == first.collection_count


class TestDocumentChanges:
    def test_shrinking_a_document_removes_its_orphan_chunks(
        self, workspace: tuple[Path, str]
    ) -> None:
        """Tài liệu ngắn lại → chunk đuôi của bản cũ phải bị xoá.

        Point ID xác định **không** cứu được ca này: `d-1::00030` của bản cũ
        không bị upsert nào ghi đè, nó cứ nằm đó và retriever vẫn trả về.
        """
        root, collection = workspace
        _write_corpus(root, {"d-1": _PARAGRAPH * 30})
        config = _config(root, collection)
        first = _build(config)

        _write_corpus(root, {"d-1": _PARAGRAPH * 5})
        second = _build(config)

        assert second.n_stale_points_deleted > 0
        assert second.collection_count < first.collection_count
        assert second.collection_count == second.n_chunks_written

    def test_orphan_point_is_really_gone_from_qdrant(self, workspace: tuple[Path, str]) -> None:
        root, collection = workspace
        _write_corpus(root, {"d-1": _PARAGRAPH * 30})
        config = _config(root, collection)
        first = _build(config)
        last_index = first.collection_count - 1
        orphan_id = chunk_point_id(f"d-1::{last_index:05d}")

        _write_corpus(root, {"d-1": _PARAGRAPH * 5})
        _build(config)

        store = _store(collection)
        assert store.client.retrieve(collection, ids=[orphan_id]) == []

    def test_growing_a_document_keeps_all_chunks(self, workspace: tuple[Path, str]) -> None:
        root, collection = workspace
        _write_corpus(root, {"d-1": _PARAGRAPH * 5})
        config = _config(root, collection)
        first = _build(config)

        _write_corpus(root, {"d-1": _PARAGRAPH * 30})
        second = _build(config)

        assert second.n_stale_points_deleted == 0
        assert second.collection_count > first.collection_count

    def test_document_dropped_from_manifest_leaves_the_index(
        self, workspace: tuple[Path, str]
    ) -> None:
        """Index phải là ảnh của manifest hiện tại, không phải hợp của mọi lần chạy."""
        root, collection = workspace
        _write_corpus(root, {"d-1": _PARAGRAPH * 10, "d-2": _PARAGRAPH * 10})
        config = _config(root, collection)
        _build(config)

        _write_corpus(root, {"d-1": _PARAGRAPH * 10})
        second = _build(config)

        assert second.n_documents_removed == 1
        state = IndexState.load(config.state_path)
        assert state is not None
        assert set(state.documents) == {"d-1"}
        assert second.collection_count == state.total_chunks

    def test_editing_content_replaces_chunks_not_appends(self, workspace: tuple[Path, str]) -> None:
        root, collection = workspace
        _write_corpus(root, {"d-1": _PARAGRAPH * 10})
        config = _config(root, collection)
        first = _build(config)

        _write_corpus(root, {"d-1": "Nội dung hoàn toàn khác. " * 100})
        second = _build(config)

        assert second.n_documents_indexed == 1
        assert second.collection_count == second.n_chunks_written
        assert second.collection_count != first.collection_count


class TestConfigGuard:
    def test_refuses_to_mix_two_configs_in_one_collection(
        self, workspace: tuple[Path, str]
    ) -> None:
        """Hai fingerprint trong một collection = mọi số eval sau đó vô nghĩa."""
        root, collection = workspace
        _write_corpus(root, {"d-1": _PARAGRAPH * 10})
        _build(_config(root, collection))

        other = _config(
            root,
            collection,
            chunking={"strategy": "fixed", "chunk_size": 400, "chunk_overlap": 20},
        )
        with pytest.raises(RuntimeError, match="fingerprint"):
            _build(other)

    def test_recreate_wipes_and_rebuilds(self, workspace: tuple[Path, str]) -> None:
        root, collection = workspace
        _write_corpus(root, {"d-1": _PARAGRAPH * 10})
        _build(_config(root, collection))

        other = _config(
            root,
            collection,
            chunking={"strategy": "fixed", "chunk_size": 400, "chunk_overlap": 20},
        )
        report = _build(other, recreate=True)
        assert report.collection_count == report.n_chunks_written

    def test_allow_mixed_is_an_explicit_opt_in(self, workspace: tuple[Path, str]) -> None:
        root, collection = workspace
        _write_corpus(root, {"d-1": _PARAGRAPH * 10})
        _build(_config(root, collection))

        other = _config(
            root,
            collection,
            chunking={"strategy": "fixed", "chunk_size": 400, "chunk_overlap": 20},
        )
        report = _build(other, allow_mixed=True)
        assert report.n_documents_indexed == 1

    def test_state_out_of_sync_with_qdrant_triggers_full_rebuild(
        self, workspace: tuple[Path, str]
    ) -> None:
        """State nói một đằng, Qdrant một nẻo → tin Qdrant và index lại.

        Xảy ra thật khi ai đó chạy `make down-clean` mà quên xoá `.cache/`.
        """
        root, collection = workspace
        _write_corpus(root, {"d-1": _PARAGRAPH * 10})
        config = _config(root, collection)
        _build(config)

        store = _store(collection)
        store.client.delete_collection(collection)

        second = _build(config)
        assert second.n_documents_indexed == 1
        assert second.n_documents_skipped == 0


class TestIntegrityStopsTheJob:
    def test_corrupt_corpus_file_aborts_before_touching_qdrant(
        self, workspace: tuple[Path, str]
    ) -> None:
        from pipeline.indexing.corpus_loader import CorpusIntegrityError

        root, collection = workspace
        _write_corpus(root, {"d-1": _PARAGRAPH * 10})
        (root / "d-1.txt").write_bytes(b"Noi dung khac han.")

        with pytest.raises(CorpusIntegrityError):
            _build(_config(root, collection))
