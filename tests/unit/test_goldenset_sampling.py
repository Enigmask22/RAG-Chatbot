"""W1-10 — chọn mẫu chunk làm nguyên liệu sinh câu hỏi.

Bộ lọc ở đây quyết định chất lượng golden set nhiều hơn cả prompt. Bản `.txt`
của World Bank giữ nguyên bố cục PDF hai cột, nên có rất nhiều đoạn *trông như*
văn xuôi mà thật ra là hai cột bị đan xen từng dòng.
"""

from __future__ import annotations

import pytest

from pipeline.eval.golden import QueryCategory
from pipeline.goldenset.sampling import (
    ChunkRef,
    gutter_ratio,
    is_prose_like,
    mean_words_per_line,
    plan_groups,
)

# Văn xuôi thật: dòng dài, không có máng phân cột.
PROSE = "\n".join(
    [
        "Tỷ lệ nghèo đa chiều của Việt Nam đã giảm đáng kể trong giai đoạn 2016 đến 2020,",
        "từ mức 9,2 phần trăm xuống còn 4,4 phần trăm theo số liệu của Tổng cục Thống kê.",
        "Kết quả này phản ánh tác động của tăng trưởng kinh tế và các chương trình mục tiêu",
        "quốc gia về giảm nghèo bền vững được triển khai trên phạm vi cả nước trong kỳ.",
        "Tuy nhiên khoảng cách giữa khu vực thành thị và nông thôn vẫn còn khá lớn hiện nay.",
        "Các tỉnh miền núi phía Bắc tiếp tục là nơi tập trung phần lớn hộ nghèo của cả nước.",
    ]
)

# Hai cột PDF bị đan xen: câu nào cũng có khoảng trắng dài ở giữa.
TWO_COLUMN = "\n".join(
    [
        "ividual indexes on new orders, output,                minus the number of existing firms",
        "employment, suppliers delivery times and the          suspending their operations; net entry",
        "stock of items purchased by manufacturers.            is the sum of the three components.",
        "A reading above 50 indicates expansion of             SA means seasonally adjusted while NSA",
        "the manufacturing sector compared with the            means not seasonally adjusted here.",
        "previous month of the reporting period now.           LHS is left hand scale in the charts.",
    ]
)

# Chú thích biểu đồ: chữ cái nhiều, nhưng dòng rất ngắn.
CHART_LEGEND = "\n".join(
    ["Government consumption", "Private consumption", "GDP growth", "Domestic demand"] * 12
)


class TestGutterRatio:
    def test_prose_has_no_gutters(self) -> None:
        assert gutter_ratio(PROSE) == 0.0

    def test_two_column_text_is_almost_all_gutters(self) -> None:
        assert gutter_ratio(TWO_COLUMN) > 0.9

    def test_empty_text(self) -> None:
        assert gutter_ratio("   \n  ") == 0.0


class TestMeanWordsPerLine:
    def test_prose_lines_are_long(self) -> None:
        assert mean_words_per_line(PROSE) > 12

    def test_chart_legend_lines_are_short(self) -> None:
        assert mean_words_per_line(CHART_LEGEND) < 4

    def test_empty_text(self) -> None:
        assert mean_words_per_line("") == 0.0


class TestIsProseLike:
    def test_accepts_real_prose(self) -> None:
        assert is_prose_like(PROSE)

    def test_rejects_two_column_interleaving(self) -> None:
        """Đây là ca quan trọng nhất: văn bản trộn cột qua được MỌI bộ lọc theo
        tỉ lệ chữ cái vì nó gồm toàn từ tiếng Anh hợp lệ. Chỉ máng phân cột lộ ra.
        """
        assert not is_prose_like(TWO_COLUMN)

    def test_rejects_chart_legend(self) -> None:
        assert not is_prose_like(CHART_LEGEND)

    def test_rejects_short_text(self) -> None:
        assert not is_prose_like("Một câu quá ngắn để hỏi.")

    def test_rejects_number_table(self) -> None:
        table = "\n".join(["2016 2017 2018 2019 2020 9,2 8,1 6,7 5,5 4,4 100,0 98,3"] * 12)
        assert not is_prose_like(table)

    def test_thresholds_are_tunable(self) -> None:
        assert is_prose_like(TWO_COLUMN, max_gutter_ratio=1.0, min_words_per_line=0.0)


def _refs(doc_id: str, count: int, lang: str = "vi") -> list[ChunkRef]:
    return [
        ChunkRef(
            chunk_id=f"{doc_id}::{i:05d}",
            doc_id=doc_id,
            lang=lang,
            point_id=f"{doc_id}-{i}",
        )
        for i in range(count)
    ]


class TestPlanGroups:
    def test_multi_hop_gets_two_chunks_from_the_same_document(self) -> None:
        plan = plan_groups(_refs("d1", 40) + _refs("d2", 40), {QueryCategory.MULTI_HOP: 4})
        assert plan
        for category, refs in plan:
            assert category is QueryCategory.MULTI_HOP
            assert len(refs) == 2
            assert len({r.doc_id for r in refs}) == 1

    def test_aggregation_gets_three_chunks(self) -> None:
        plan = plan_groups(_refs("d1", 40), {QueryCategory.AGGREGATION: 3})
        assert all(len(refs) == 3 for _, refs in plan)

    def test_factoid_gets_one_chunk(self) -> None:
        plan = plan_groups(_refs("d1", 40), {QueryCategory.FACTOID: 5})
        assert all(len(refs) == 1 for _, refs in plan)

    def test_skips_the_front_matter_of_each_document(self) -> None:
        """Bìa, trang bản quyền, lời cảm ơn, mục lục là văn xuôi hợp lệ nên bộ
        lọc hình thức cho qua — nhưng câu hỏi sinh từ đó chỉ đo được khả năng
        tìm lại dòng "© 2022 International Bank for Reconstruction"."""
        plan = plan_groups(_refs("d1", 40), {QueryCategory.FACTOID: 10}, skip_leading_chunks=6)
        indices = [int(r.chunk_id.split("::")[1]) for _, refs in plan for r in refs]
        assert min(indices) >= 6

    def test_never_reuses_a_chunk_across_groups(self) -> None:
        plan = plan_groups(
            _refs("d1", 60) + _refs("d2", 60),
            {QueryCategory.FACTOID: 10, QueryCategory.MULTI_HOP: 10},
        )
        all_ids = [r.chunk_id for _, refs in plan for r in refs]
        assert len(all_ids) == len(set(all_ids))

    def test_spreads_across_documents_instead_of_draining_one(self) -> None:
        """Bốc ngẫu nhiên toàn cục thì tài liệu dài chiếm phần lớn mẫu, và golden
        set hoá ra chỉ đo được vài tài liệu."""
        refs = _refs("dai", 500) + _refs("ngan-a", 30) + _refs("ngan-b", 30)
        plan = plan_groups(refs, {QueryCategory.FACTOID: 12})
        docs = {r.doc_id for _, group in plan for r in group}
        assert docs == {"dai", "ngan-a", "ngan-b"}

    def test_same_seed_gives_the_same_plan(self) -> None:
        """Không cố định seed thì hai lần sinh nháp không so được với nhau, và
        tiền API đã tiêu là bỏ phí."""
        refs = _refs("d1", 60) + _refs("d2", 60)
        first = plan_groups(refs, {QueryCategory.FACTOID: 8}, seed=7)
        second = plan_groups(refs, {QueryCategory.FACTOID: 8}, seed=7)
        assert [r.chunk_id for _, g in first for r in g] == [
            r.chunk_id for _, g in second for r in g
        ]

    def test_different_seed_gives_a_different_plan(self) -> None:
        refs = _refs("d1", 60) + _refs("d2", 60)
        first = plan_groups(refs, {QueryCategory.FACTOID: 8}, seed=7)
        second = plan_groups(refs, {QueryCategory.FACTOID: 8}, seed=8)
        assert [r.chunk_id for _, g in first for r in g] != [
            r.chunk_id for _, g in second for r in g
        ]

    def test_filters_by_language(self) -> None:
        refs = _refs("vi-doc", 40, lang="vi") + _refs("en-doc", 40, lang="en")
        plan = plan_groups(refs, {QueryCategory.FACTOID: 6}, languages=["en"])
        assert {r.lang for _, g in plan for r in g} == {"en"}

    def test_raises_when_language_filter_matches_nothing(self) -> None:
        with pytest.raises(ValueError, match="languages"):
            plan_groups(_refs("d1", 40), {QueryCategory.FACTOID: 2}, languages=["fr"])

    def test_stops_instead_of_looping_when_corpus_is_too_small(self) -> None:
        """Corpus không đủ chunk thì trả ít hơn hạn mức và ghi cảnh báo — không
        được quay vòng vô hạn hay dùng lại chunk cũ."""
        plan = plan_groups(_refs("d1", 10), {QueryCategory.FACTOID: 100})
        assert 0 < len(plan) <= 4
