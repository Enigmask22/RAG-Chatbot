"""Khử câu hỏi trùng ý trong bản nháp.

Vì sao phải khử: model được đưa 300 chunk khác nhau vẫn sinh ra hàng chục câu
dạng "Tỷ lệ nghèo ở Việt Nam năm 2020 là bao nhiêu?". Chúng không sai, nhưng
chúng làm **cùng một phép đo bị đếm nhiều lần** — nhóm nào model thích viết sẽ
chi phối con số tổng, và bảng breakdown theo nhóm thành vô nghĩa.

Cách làm, theo thứ tự rẻ trước:

1. **Trùng hệt** sau khi chuẩn hoá (hạ chữ, bỏ dấu câu, gom khoảng trắng).
2. **Jaccard trên tập token** vượt ngưỡng. Bắt được đúng kiểu trùng mà LLM hay
   tạo ra: cùng câu, đảo trật tự vài từ, thêm bớt hư từ.

Cố ý **không** dùng embedding làm mặc định. Hai lý do: hàm này phải chạy được
trong unit test mà không cần model, và ngưỡng cosine trên câu hỏi ngắn rất khó
đặt — 0,85 vừa bỏ sót vừa gộp nhầm hai câu khác nghĩa. Ai cần thì truyền
`EmbeddingProvider` vào để chạy thêm một lượt ngữ nghĩa, có ý thức.

Khi hai câu trùng nhau, giữ câu **nào có nhiều thông tin hơn** (nhiều chunk liên
quan hơn, rồi tới trích dẫn kiểm chứng được, rồi tới câu dài hơn), không phải
câu đến trước.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from rag_core.embedding.base import EmbeddingProvider

from .schema import GoldenDraft, normalize_for_dedupe

__all__ = ["DedupeResult", "deduplicate_drafts", "jaccard", "tokenize"]

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> frozenset[str]:
    return frozenset(_TOKEN_RE.findall(normalize_for_dedupe(text)))


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    union = len(left | right)
    return len(left & right) / union if union else 0.0


@dataclass(frozen=True)
class DedupeResult:
    kept: list[GoldenDraft]
    removed: list[tuple[GoldenDraft, str]]
    """Cặp (bản bị bỏ, `query_id` của bản được giữ) — để kiểm lại quyết định."""

    @property
    def n_removed(self) -> int:
        return len(self.removed)


def _richness(draft: GoldenDraft) -> tuple[int, int, int]:
    """Thứ tự ưu tiên khi hai câu trùng nhau. Lớn hơn là giữ."""
    return (
        len(draft.query.relevant_chunk_ids),
        1 if draft.provenance.quotes_verified else 0,
        len(draft.query.query),
    )


def deduplicate_drafts(
    drafts: Sequence[GoldenDraft],
    *,
    jaccard_threshold: float = 0.8,
    embeddings: EmbeddingProvider | None = None,
    cosine_threshold: float = 0.93,
) -> DedupeResult:
    """Bỏ câu trùng ý, giữ bản giàu thông tin nhất trong mỗi nhóm.

    So sánh chỉ diễn ra **trong cùng một nhóm truy vấn**: một câu factoid và một
    câu multi_hop dùng chung phần lớn từ vựng vẫn là hai phép đo khác nhau, gộp
    lại là làm hỏng chính bảng breakdown mà eval sinh ra để đọc.
    """
    if not drafts:
        return DedupeResult(kept=[], removed=[])

    ordered = sorted(drafts, key=_richness, reverse=True)
    kept: list[GoldenDraft] = []
    kept_tokens: list[frozenset[str]] = []
    kept_exact: dict[tuple[str, str], str] = {}
    removed: list[tuple[GoldenDraft, str]] = []

    for draft in ordered:
        category = draft.query.category.value
        exact_key = (category, draft.dedupe_key)
        if exact_key in kept_exact:
            removed.append((draft, kept_exact[exact_key]))
            continue

        tokens = tokenize(draft.query.query)
        duplicate_of: str | None = None
        for other, other_tokens in zip(kept, kept_tokens, strict=True):
            if other.query.category is not draft.query.category:
                continue
            if jaccard(tokens, other_tokens) >= jaccard_threshold:
                duplicate_of = other.query.query_id
                break

        if duplicate_of is not None:
            removed.append((draft, duplicate_of))
            continue

        kept.append(draft)
        kept_tokens.append(tokens)
        kept_exact[exact_key] = draft.query.query_id

    if embeddings is not None:
        kept, extra_removed = _semantic_pass(kept, embeddings, cosine_threshold)
        removed.extend(extra_removed)

    logger.info("Khử trùng: giữ %d, bỏ %d trong %d bản nháp", len(kept), len(removed), len(drafts))
    return DedupeResult(kept=kept, removed=removed)


def _semantic_pass(
    drafts: list[GoldenDraft],
    embeddings: EmbeddingProvider,
    threshold: float,
) -> tuple[list[GoldenDraft], list[tuple[GoldenDraft, str]]]:
    """Lượt hai: gộp câu khác từ nhưng cùng nghĩa. Chỉ chạy khi được yêu cầu."""
    if len(drafts) < 2:
        return drafts, []

    matrix = np.asarray(
        embeddings.embed_documents([d.query.query for d in drafts]), dtype=np.float64
    )
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0.0] = 1.0
    unit = matrix / norms[:, None]
    similarity = unit @ unit.T

    kept: list[GoldenDraft] = []
    kept_indices: list[int] = []
    removed: list[tuple[GoldenDraft, str]] = []
    for i, draft in enumerate(drafts):
        duplicate_of: str | None = None
        for j in kept_indices:
            if (
                drafts[j].query.category is draft.query.category
                and float(similarity[i, j]) >= threshold
            ):
                duplicate_of = drafts[j].query.query_id
                break
        if duplicate_of is None:
            kept.append(draft)
            kept_indices.append(i)
        else:
            removed.append((draft, duplicate_of))
    return kept, removed
