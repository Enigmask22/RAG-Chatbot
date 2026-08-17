"""Sinh bản nháp golden set bằng LLM, từ chunk thật trong index.

Không re-export `generate` ở đây: gói cha nạp sẵn module con thì
`python -m pipeline.goldenset.generate` chạy module hai lần và Python cảnh báo.
"""

from .dedupe import DedupeResult, deduplicate_drafts
from .sampling import ChunkGroup, sample_groups
from .schema import DraftProvenance, GoldenDraft, drafts_summary, load_drafts, write_drafts

__all__ = [
    "ChunkGroup",
    "DedupeResult",
    "DraftProvenance",
    "GoldenDraft",
    "deduplicate_drafts",
    "drafts_summary",
    "load_drafts",
    "sample_groups",
    "write_drafts",
]
