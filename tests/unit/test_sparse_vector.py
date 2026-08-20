"""Test cho `rag_core.embedding.sparse.SparseVector`.

Sparse vector đi qua ba tầng (provider → Qdrant upsert → retriever) và mỗi tầng
có quy ước riêng về thứ tự và về số 0. Sai bất biến ở đây không thành lỗi — nó
thành "sparse retrieval chạy nhưng tệ", đúng loại hỏng đắt nhất để tìm.
"""

from __future__ import annotations

from typing import Any

import pytest

from rag_core.embedding.sparse import SparseVector


class TestInvariants:
    def test_indices_must_be_strictly_increasing(self) -> None:
        with pytest.raises(ValueError, match="tăng nghiêm ngặt"):
            SparseVector(indices=(5, 3), values=(1.0, 1.0))

    def test_duplicate_index_is_rejected(self) -> None:
        """Trùng index = hai trọng số cho cùng một token; tầng dưới sẽ chọn bừa."""
        with pytest.raises(ValueError, match="tăng nghiêm ngặt"):
            SparseVector(indices=(3, 3), values=(1.0, 2.0))

    def test_length_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="cùng độ dài"):
            SparseVector(indices=(1, 2), values=(1.0,))

    def test_zero_value_is_rejected(self) -> None:
        """Giữ entry bằng 0 làm phồng payload và làm phép đếm token khớp sai."""
        with pytest.raises(ValueError, match="phải dương"):
            SparseVector(indices=(1,), values=(0.0,))

    def test_negative_value_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="phải dương"):
            SparseVector(indices=(1,), values=(-0.5,))

    def test_negative_index_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="không âm"):
            SparseVector(indices=(-1,), values=(1.0,))

    def test_empty_is_valid(self) -> None:
        """Rỗng = "đã tính, không token nào dương". Khác `None` = "không hỗ trợ"."""
        empty = SparseVector(indices=(), values=())
        assert len(empty) == 0
        assert empty.as_dict() == {}

    def test_is_immutable(self) -> None:
        from dataclasses import FrozenInstanceError

        vec = SparseVector.from_weights({1: 1.0})
        with pytest.raises(FrozenInstanceError):
            vec.indices = (2,)  # type: ignore[misc]


class TestFromWeights:
    def test_sorts_and_drops_non_positive(self) -> None:
        vec = SparseVector.from_weights({5: 0.2, 1: 0.9, 3: 0.0, 7: -1.0})
        assert vec.indices == (1, 5)
        assert vec.values == (0.9, 0.2)

    def test_empty_mapping(self) -> None:
        assert len(SparseVector.from_weights({})) == 0

    def test_all_non_positive_gives_empty_not_error(self) -> None:
        assert SparseVector.from_weights({1: 0.0, 2: -3.0}) == SparseVector((), ())

    def test_casts_keys_to_int(self) -> None:
        """Token id đến từ numpy nên là `np.int64`, không phải `int` thuần."""
        import numpy as np

        raw: dict[Any, Any] = {np.int64(4): np.float32(0.5)}
        vec = SparseVector.from_weights(raw)
        assert vec.indices == (4,)
        assert isinstance(vec.indices[0], int)
        assert isinstance(vec.values[0], float)


class TestDot:
    def test_hand_computed(self) -> None:
        a = SparseVector.from_weights({1: 2.0, 3: 4.0, 5: 6.0})
        b = SparseVector.from_weights({3: 0.5, 5: 0.25, 9: 100.0})
        # chỉ 3 và 5 chung: 4*0.5 + 6*0.25 = 3.5
        assert a.dot(b) == pytest.approx(3.5)

    def test_disjoint_is_zero(self) -> None:
        a = SparseVector.from_weights({1: 1.0, 2: 1.0})
        b = SparseVector.from_weights({3: 1.0, 4: 1.0})
        assert a.dot(b) == 0.0

    def test_symmetric(self) -> None:
        a = SparseVector.from_weights({1: 2.0, 7: 3.0})
        b = SparseVector.from_weights({7: 5.0, 8: 1.0})
        assert a.dot(b) == b.dot(a)

    def test_empty_operand(self) -> None:
        a = SparseVector.from_weights({1: 1.0})
        assert a.dot(SparseVector((), ())) == 0.0

    def test_merge_walk_does_not_skip_the_last_shared_index(self) -> None:
        """Vòng lặp hợp nhất dừng khi *một* con trỏ hết — dễ sai ở phần tử cuối."""
        a = SparseVector.from_weights({1: 1.0, 2: 1.0, 100: 3.0})
        b = SparseVector.from_weights({50: 1.0, 100: 5.0})
        assert a.dot(b) == pytest.approx(15.0)


class TestRepresentations:
    def test_as_qdrant_uses_lists(self) -> None:
        vec = SparseVector.from_weights({2: 0.5, 9: 0.1})
        assert vec.as_qdrant() == {"indices": [2, 9], "values": [0.5, 0.1]}

    def test_top_ranks_by_value_then_index(self) -> None:
        vec = SparseVector.from_weights({1: 0.5, 2: 0.9, 3: 0.5})
        assert vec.top(2) == ((2, 0.9), (1, 0.5))

    def test_top_tie_break_is_deterministic(self) -> None:
        """Bằng điểm thì lấy index nhỏ hơn — report phải tái lập được."""
        vec = SparseVector.from_weights({9: 0.5, 4: 0.5})
        assert vec.top(1) == ((4, 0.5),)

    def test_top_more_than_available(self) -> None:
        assert len(SparseVector.from_weights({1: 1.0}).top(10)) == 1
