"""Cấu hình một lần build index — tiền thân của `RagBundle`.

Toàn bộ thứ quyết định **nội dung** của một index nằm gọn trong một file YAML:
corpus nào, chunk thế nào, embed bằng model gì, ghi vào collection nào. Lý do
gom lại một chỗ thay vì rải thành cờ dòng lệnh:

* Một lần chạy eval phải tái lập được từ một file. `--chunk-size 1000
  --overlap 100 --model ...` gõ tay lần sau sẽ lệch, và không ai biết là đã lệch.
* `fingerprint` băm đúng những trường ảnh hưởng tới vector. Đây là thứ để trả
  lời "index đang nằm trong Qdrant có phải do config này sinh ra không" — câu
  hỏi bắt buộc trước khi so hai con số eval với nhau.
* W4 sẽ đóng gói chính những trường này thành `RagBundle` có version. Định hình
  đúng từ bây giờ thì lúc đó chỉ là thêm chữ ký và metadata, không phải viết lại.

Cố ý **không** đưa `device` và `batch_size` vào `fingerprint`: chạy trên GPU thuê
hay trên laptop phải ra cùng một index về mặt logic, nếu không thì mọi lần đổi
máy đều buộc build lại toàn bộ. Chúng vẫn được ghi vào manifest để tra cứu.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag_core.chunking import Chunker, ChunkingConfig, SQLiteChunkCache, build_chunker
from rag_core.chunking.cache import CachedChunker
from rag_core.embedding import EmbeddingProvider, build_embedding_provider

if TYPE_CHECKING:
    from rag_core.retrieval.qdrant_store import QdrantDenseRetriever

__all__ = ["IndexConfig", "load_index_config"]


class IndexConfig(BaseModel):
    """Mô tả đầy đủ một lần build index."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    """Tên run — dùng cho tên collection mặc định, file state và tên báo cáo."""

    collection: str = Field(default="", description="Rỗng thì suy ra từ `name`")

    # ------------------------------------------------------------ nguồn
    manifest_path: Path = Path("data/corpus_manifest.csv")
    corpus_dir: Path = Path("data/corpus")
    languages: tuple[str, ...] = ()
    """Lọc theo ngôn ngữ; rỗng nghĩa là lấy hết."""
    doc_types: tuple[str, ...] = ()
    max_documents: int | None = Field(default=None, ge=1)

    # ------------------------------------------------------------ chunking
    chunking: ChunkingConfig = ChunkingConfig()

    # ------------------------------------------------------------ embedding
    embedding_model: str = "bkai-foundation-models/vietnamese-bi-encoder"
    embedding_device: str = "auto"
    embedding_batch_size: int = Field(default=32, ge=1)
    embedding_max_batch_tokens: int | None = Field(default=None, ge=1)
    """Trần token mỗi forward pass, chỉ có ý nghĩa với provider cửa sổ dài.

    `embedding_batch_size` một mình không chặn được VRAM khi cửa sổ là 8192:
    16 câu × 8192 token = 131k token và OOM ngay. `None` = dùng mặc định của
    provider. Là knob **tốc độ/bộ nhớ** nên nằm ngoài `fingerprint`, cùng lý do
    như `device` và `batch_size`.
    """
    embedding_normalize: bool = True
    embedding_kwargs: dict[str, Any] = Field(default_factory=dict)
    """`query_prefix` / `document_prefix` cho model bất đối xứng (E5, BGE)."""

    # ------------------------------------------------------------ ghi
    upsert_batch_size: int = Field(default=128, ge=1)
    use_cache: bool = True
    cache_path: Path = Path(".cache/chunks.sqlite3")
    state_dir: Path = Path(".cache/index_state")

    @field_validator("embedding_device")
    @classmethod
    def _check_device(cls, value: str) -> str:
        allowed = {"auto", "cpu", "cuda", "mps"}
        if value not in allowed:
            raise ValueError(f"embedding_device phải thuộc {sorted(allowed)}, nhận {value!r}")
        return value

    # ------------------------------------------------------------ dẫn xuất

    @property
    def collection_name(self) -> str:
        return self.collection or f"rag_{self.name}"

    @property
    def state_path(self) -> Path:
        return self.state_dir / f"{self.name}.json"

    @property
    def fingerprint(self) -> str:
        """Băm những trường quyết định nội dung vector.

        Không gồm `device`, `batch_size`, `upsert_batch_size`, đường dẫn cache —
        chúng đổi tốc độ chứ không đổi kết quả. Có gồm bộ lọc corpus vì chúng
        quyết định **tài liệu nào** có mặt trong index.
        """
        payload = {
            "chunking": json.loads(self.chunking.model_dump_json()),
            "embedding_model": self.embedding_model,
            "embedding_normalize": self.embedding_normalize,
            "embedding_kwargs": self.embedding_kwargs,
            "languages": sorted(self.languages),
            "doc_types": sorted(self.doc_types),
            "max_documents": self.max_documents,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------ factory

    def build_embeddings(self) -> EmbeddingProvider:
        extra: dict[str, Any] = dict(self.embedding_kwargs)
        if self.embedding_max_batch_tokens is not None:
            extra["max_batch_tokens"] = self.embedding_max_batch_tokens
        return build_embedding_provider(
            self.embedding_model,
            device=self.embedding_device,
            batch_size=self.embedding_batch_size,
            normalize=self.embedding_normalize,
            **extra,
        )

    def build_chunker(self, embeddings: EmbeddingProvider) -> Chunker:
        chunker = build_chunker(self.chunking, embeddings)
        if not self.use_cache:
            return chunker
        return CachedChunker(chunker, SQLiteChunkCache(self.cache_path))

    def build_retriever(
        self,
        embeddings: EmbeddingProvider,
        *,
        url: str,
        api_key: str | None = None,
    ) -> QdrantDenseRetriever:
        from rag_core.retrieval.qdrant_store import QdrantDenseRetriever

        return QdrantDenseRetriever(
            embeddings,
            collection=self.collection_name,
            url=url,
            api_key=api_key,
        )


def load_index_config(path: str | Path, **overrides: Any) -> IndexConfig:
    """Đọc config YAML. `overrides` để CLI ghi đè vài trường mà không sửa file."""
    import yaml

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"Không thấy config index tại {source}. Mẫu có sẵn ở `configs/indexing/baseline.yaml`."
        )
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{source} phải là một mapping YAML, đọc được {type(raw).__name__}")
    raw.update({k: v for k, v in overrides.items() if v is not None})
    return IndexConfig.model_validate(raw)
