"""Build index cho Qdrant: corpus → chunk → embed → collection.

Cố ý **không** re-export `build_index` ở đây. Gói cha nạp sẵn module con thì
`python -m pipeline.indexing.build_index` chạy module hai lần và Python cảnh báo
`RuntimeWarning: found in sys.modules after import of package`. Ai cần nó thì
`from pipeline.indexing.build_index import build_index`.
"""

from .config import IndexConfig, load_index_config
from .corpus_loader import CorpusIntegrityError, load_documents, select_entries

__all__ = [
    "CorpusIntegrityError",
    "IndexConfig",
    "load_documents",
    "load_index_config",
    "select_entries",
]
