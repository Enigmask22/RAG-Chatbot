"""`W4-01` — `RagBundle`, và chủ yếu là những thứ nó phải TỪ CHỐI.

Một bundle hợp lệ không chứng minh được gì: round-trip xanh chỉ nói rằng pydantic
hoạt động. Giá trị của hạng mục này nằm ở tập những cấu hình **không** đóng gói
được, vì mỗi cái trong đó là một cách hệ thống có thể đi ra production mà không
ai đo nó.

Ba nhóm, xếp theo mức nguy hiểm giảm dần:

1. **Bundle mô tả thiếu hệ thống của chính nó** — trường có mặc định ở chỗ không
   được phép có mặc định. Đây là nhóm nguy hiểm nhất vì nó không bao giờ đỏ ở
   đâu cả: hệ thống chạy, chỉ là chạy bằng hằng số nằm trong mã serving.
2. **Bundle nói dối** — checksum, tên thư mục, semver.
3. **Bundle chưa được đo** — thiếu `eval`, thiếu `evaluated_with_generator`.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from rag_core.bundle import (
    BundleChecksumError,
    BundleComponents,
    BundleValidationError,
    ChunkingComponent,
    EmbeddingComponent,
    EvalReport,
    GateRecord,
    GateStatus,
    GenerationComponent,
    IndexComponent,
    PromptComponent,
    RagBundle,
    RerankComponent,
    RetrievalComponent,
    compute_checksum,
    latest_bundle,
    list_bundles,
    load_bundle,
    parse_semver,
    save_bundle,
)

# ---------------------------------------------------------------------------
# Fixture: bundle thật của điểm vận hành `rc50` + Contextual Retrieval
# ---------------------------------------------------------------------------


def make_bundle(**overrides: Any) -> RagBundle:
    """Số ở đây là số **đã đo thật** (`exp-002`), không phải giá trị bịa.

    Cố ý: một fixture toàn `"x"` và `1` sẽ đi qua mọi phép kiểm hình dạng mà
    không bao giờ chạm vào ràng buộc giữa các trường (`top_n ≤ candidates`,
    `chunk_overlap < chunk_size`) — bài học lặp lại năm lần trong dự án này.
    """
    base: dict[str, Any] = {
        "bundle_version": "1.0.0",
        "created_at": datetime(2026, 9, 3, 10, 0, tzinfo=UTC),
        "git_sha": "e833015",
        "components": BundleComponents(
            chunking=ChunkingComponent(
                strategy="hybrid",
                chunk_size=1000,
                chunk_overlap=100,
                contextual=True,
                chunking_fingerprint="c7ca3e6fc4da29a5",
            ),
            embedding=EmbeddingComponent(model="BAAI/bge-m3", dim=1024, normalize=True),
            index=IndexComponent(
                backend="qdrant",
                collection="rag_bgem3_ctx",
                fingerprint="a" * 64,
                n_chunks=15814,
                n_documents=60,
            ),
            retrieval=RetrievalComponent(mode="dense", top_k=50),
            rerank=RerankComponent(
                model="BAAI/bge-reranker-v2-m3", candidates=50, top_n=6, max_length=512
            ),
            prompt=PromptComponent(id="grounded_qa_vi_en", version=1, hash="deadbeef"),
            generation=GenerationComponent(
                primary="deepseek-chat", fallback=None, max_tokens=1024, temperature=0.0
            ),
        ),
        "eval": EvalReport(
            golden_set="golden_v1",
            n_queries=209,
            evaluated_with_generator="deepseek-chat@2026-09",
            retrieval_metrics={"ndcg@10": 0.6888, "hit_rate@5": 0.8086},
        ),
        "gate": GateRecord(status=GateStatus.NOT_RUN),
    }
    base.update(overrides)
    return RagBundle(**base)


def make_bundle_retrieval_only(**overrides: Any) -> RagBundle:
    """Bundle của **hôm nay**: đo được truy hồi, chưa có tầng sinh.

    Dựng lại qua `RagBundle(...)` chứ không qua `model_copy`: `model_copy`
    **không chạy validator**, nên một fixture dựng bằng nó sẽ tạo ra được đúng
    những object mà schema từ chối — và test sẽ kiểm một thứ không tồn tại.
    """
    full = make_bundle()
    base: dict[str, Any] = {
        "bundle_version": full.bundle_version,
        "created_at": full.created_at,
        "git_sha": full.git_sha,
        "components": full.components.model_copy(update={"prompt": None, "generation": None}),
        "eval": full.eval,
        "gate": full.gate,
    }
    base.update(overrides)
    return RagBundle(**base)


# ---------------------------------------------------------------------------
# 1. ⭐⭐ Bundle phải mô tả ĐỦ hệ thống — không mặc định ở chỗ ảnh hưởng kết quả
# ---------------------------------------------------------------------------

#: Những trường mà **thiếu là lỗi**. Danh sách này là phát biểu kiến trúc của cả
#: hạng mục: mỗi mặc định thêm vào đây là một hằng số cấu hình chuyển từ artifact
#: sang mã serving, tức một thứ mà gate không gác và rollback không lấy lại được.
REQUIRED_NO_DEFAULT = [
    (ChunkingComponent, "contextual"),
    (ChunkingComponent, "chunking_fingerprint"),
    (EmbeddingComponent, "normalize"),
    (EmbeddingComponent, "dim"),
    (IndexComponent, "collection"),
    (IndexComponent, "fingerprint"),
    (RetrievalComponent, "mode"),
    (RetrievalComponent, "top_k"),
    (RerankComponent, "model"),
    (RerankComponent, "top_n"),
    (RerankComponent, "candidates"),
    (RerankComponent, "max_length"),
    (GenerationComponent, "primary"),
    (GenerationComponent, "temperature"),
    (GenerationComponent, "max_tokens"),
    (PromptComponent, "hash"),
    (EvalReport, "evaluated_with_generator"),
]


@pytest.mark.parametrize(("model", "field"), REQUIRED_NO_DEFAULT)
def test_field_that_changes_behaviour_has_no_default(model: type[BaseModel], field: str) -> None:
    """⭐⭐ Phép kiểm quan trọng nhất của file, và nó không kiểm hành vi runtime.

    Nếu một trường ở đây có mặc định thì bundle im lặng về nó, serving lấp bằng
    hằng số của mình, và hai lần deploy cùng một bundle trên hai image khác nhau
    cho hai hệ thống khác nhau — không có gì đỏ, không có gì trong log.
    """
    assert model.model_fields[field].is_required(), (
        f"`{model.__name__}.{field}` có mặc định. Trường ảnh hưởng kết quả mà có "
        "mặc định = hằng số cấu hình sống trong mã serving thay vì trong artifact."
    )


def test_absent_rerank_means_disabled_not_default() -> None:
    """`rerank=None` là "tắt rerank", **không** phải "dùng reranker mặc định".

    Không ép được bằng type — `None` chỉ là `None`. Ghim ở đây để bất kỳ ai định
    thêm `RerankComponent()` làm mặc định phải đọc lý do trước: một bundle đo
    không rerank rồi chạy có rerank là bundle mà mọi số đo của nó đều sai.
    """
    assert BundleComponents.model_fields["rerank"].default is None
    bundle = make_bundle()
    stripped = bundle.model_copy(
        update={"components": bundle.components.model_copy(update={"rerank": None})}
    )
    assert stripped.components.rerank is None


def test_retrieval_only_bundle_is_a_legal_state() -> None:
    """Hôm nay tầng sinh chưa tồn tại (`W4-08`/`W4-11`), nên bundle phải **khai
    thiếu** được — nhét `prompt.hash = "todo"` vào rồi ký lên đó là một artifact
    bịa mang chữ ký hợp lệ, tệ hơn một artifact khai thiếu."""
    bundle = make_bundle_retrieval_only()
    assert bundle.serves_generation is False
    assert bundle.signed().checksum is not None


def test_full_bundle_serves_generation() -> None:
    assert make_bundle().serves_generation is True


def test_generation_metrics_without_a_generation_stack_are_rejected() -> None:
    """⭐ Chỗ giữ cho `generation=None` là khai thiếu chứ không phải lỗ hổng.

    Không có phép kiểm này thì một bundle đo `faithfulness` bằng một prompt nào
    đó, không ghi lại prompt ấy, rồi `W5` so nó với bundle khác và quy toàn bộ
    chênh lệch cho retrieval.
    """
    with pytest.raises(ValidationError, match="thiếu prompt, generation"):
        make_bundle_retrieval_only(
            eval=EvalReport(
                golden_set="golden_v1",
                n_queries=209,
                evaluated_with_generator="deepseek-chat@2026-09",
                retrieval_metrics={"ndcg@10": 0.6888},
                generation_metrics={"faithfulness": 0.94},
            )
        )


def test_unknown_field_is_rejected_not_ignored() -> None:
    """Trường gõ sai tên bị nuốt = một cấu hình tưởng đã bật mà chưa bao giờ bật."""
    with pytest.raises(ValidationError, match="extra"):
        RetrievalComponent(mode="dense", top_k=50, candidate_k=20)  # type: ignore[call-arg]


def test_bundle_is_frozen() -> None:
    """`verify_checksum()` chứng nhận nội dung tại một thời điểm; sửa được sau đó
    thì lời chứng nhận hết giá trị ngay dòng kế tiếp."""
    bundle = make_bundle()
    with pytest.raises(ValidationError):
        bundle.bundle_version = "9.9.9"


# ---------------------------------------------------------------------------
# 2. Bundle nói dối
# ---------------------------------------------------------------------------


def test_round_trip_through_json(tmp_path: Path) -> None:
    saved = save_bundle(make_bundle(), tmp_path)
    assert load_bundle(saved) == make_bundle().signed()


def test_signing_is_idempotent() -> None:
    once = make_bundle().signed()
    assert once.signed().checksum == once.checksum


def test_tampered_manifest_is_rejected(tmp_path: Path) -> None:
    manifest = save_bundle(make_bundle(), tmp_path)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["components"]["rerank"]["top_n"] = 20
    manifest.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(BundleChecksumError, match="không khớp"):
        load_bundle(manifest)


def test_deleting_a_defaulted_field_is_also_caught(tmp_path: Path) -> None:
    """⭐ Checksum bắt cả việc **xoá** một trường có mặc định.

    Xoá `notes` khỏi manifest thì pydantic lấp lại bằng `None` và validate xanh —
    nhưng payload đem băm đã khác, nên chữ ký lệch. Nếu không có tính chất này
    thì mọi trường optional là một chỗ sửa được mà checksum không thấy.
    """
    manifest = save_bundle(make_bundle(notes="điểm vận hành rc50"), tmp_path)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    del raw["notes"]
    manifest.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(BundleChecksumError):
        load_bundle(manifest)


def test_unsigned_bundle_is_rejected_with_a_different_message(tmp_path: Path) -> None:
    """ "Chưa ký" và "sai chữ ký" là hai lỗi khác nhau và hai cách sửa khác nhau."""
    manifest = tmp_path / "rag-bundle-v1.0.0" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(make_bundle().model_dump_json(), encoding="utf-8")

    with pytest.raises(BundleChecksumError, match="chưa ký"):
        load_bundle(manifest)


def test_reformatting_the_manifest_does_not_break_the_checksum(tmp_path: Path) -> None:
    """Checksum chứng nhận **nội dung**, không chứng nhận byte.

    Đây là lựa chọn có chủ đích, không phải chỗ lỏng: manifest đi qua git, CI và
    `docker cp`, nên một chữ ký vỡ vì thụt lề sẽ bị người ta tắt đi trong tuần
    đầu tiên. Ghim lại để lựa chọn ấy là *lựa chọn*, không phải tai nạn.
    """
    manifest = save_bundle(make_bundle(), tmp_path)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    manifest.write_text(
        json.dumps(raw, indent=8, sort_keys=True, ensure_ascii=True), encoding="utf-8"
    )
    load_bundle(manifest)  # không ném


def test_directory_version_must_match_manifest(tmp_path: Path) -> None:
    """⭐ Cách một bản rollback đi nhầm chỗ mà checksum vẫn khớp hoàn toàn."""
    save_bundle(make_bundle(), tmp_path)
    (tmp_path / "rag-bundle-v1.0.0").rename(tmp_path / "rag-bundle-v1.3.2")

    with pytest.raises(BundleValidationError, match="thư mục nói version"):
        load_bundle(tmp_path / "rag-bundle-v1.3.2")


def test_checksum_input_must_not_contain_the_checksum() -> None:
    """Băm cả `checksum` thì hàm băm phụ thuộc kết quả của chính nó."""
    with pytest.raises(BundleValidationError, match="không được nằm trong"):
        compute_checksum({"bundle_version": "1.0.0", "checksum": "sha256:x"})


def test_existing_version_is_not_overwritten(tmp_path: Path) -> None:
    """Bất biến là một tính chất của hệ thống file, không phải của schema."""
    save_bundle(make_bundle(), tmp_path)
    with pytest.raises(BundleValidationError, match="đã tồn tại"):
        save_bundle(make_bundle(notes="sửa nhẹ"), tmp_path)


# ---------------------------------------------------------------------------
# 3. Bundle chưa được đo
# ---------------------------------------------------------------------------


def test_eval_report_is_mandatory() -> None:
    """Không đo thì không đóng gói được, nên không deploy được. Toàn bộ luận điểm
    "eval trước tối ưu" nằm ở chỗ trường này không có mặc định."""
    assert RagBundle.model_fields["eval"].is_required()


def test_evaluated_with_generator_required() -> None:
    """`W5-07`/`G5` dựa vào trường này để gate chỉ so like-for-like."""
    with pytest.raises(ValidationError, match="evaluated_with_generator"):
        EvalReport(  # type: ignore[call-arg]
            golden_set="golden_v1", n_queries=209, retrieval_metrics={"ndcg@10": 0.68}
        )


def test_empty_generator_string_is_rejected() -> None:
    """Chuỗi rỗng đi qua "trường bắt buộc" mà không mang thông tin nào."""
    with pytest.raises(ValidationError):
        EvalReport(
            golden_set="golden_v1",
            n_queries=209,
            evaluated_with_generator="",
            retrieval_metrics={"ndcg@10": 0.68},
        )


def test_eval_with_no_metrics_at_all_is_rejected() -> None:
    """Khối eval rỗng đi qua mọi phép kiểm hình thức mà không chứng nhận gì."""
    with pytest.raises(ValidationError, match="ít nhất một metric"):
        EvalReport(golden_set="golden_v1", n_queries=209, evaluated_with_generator="deepseek-chat")


def test_gate_is_mandatory_and_not_run_is_a_real_state() -> None:
    """ "Chưa chạy gate" khác "gate PASS". Để `gate` optional là trộn hai thứ đó."""
    assert RagBundle.model_fields["gate"].is_required()
    assert make_bundle().gate.status is GateStatus.NOT_RUN


def test_decided_gate_must_name_its_champion() -> None:
    """PASS/FAIL là phán quyết **so với** một bản, không phải thuộc tính tự thân."""
    with pytest.raises(ValidationError, match="champion_compared"):
        GateRecord(status=GateStatus.PASS)


# ---------------------------------------------------------------------------
# 4. Semver — thứ tự, không chỉ định dạng
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("version", ["1", "1.0", "1.0.0.0", "v1.0.0", "01.0.0", "1.0.0-"])
def test_invalid_semver_is_rejected(version: str) -> None:
    with pytest.raises(BundleValidationError, match="semver"):
        parse_semver(version)


def test_ten_sorts_after_nine() -> None:
    """Sắp theo chuỗi thì `1.10.0 < 1.9.0`, và "bản trước đó" trỏ nhầm bundle.
    Lỗi này xuất hiện ở lần release thứ mười — lâu sau khi hết ai kiểm bằng mắt."""
    assert parse_semver("1.10.0") > parse_semver("1.9.0")


def test_prerelease_sorts_before_the_release() -> None:
    assert parse_semver("1.0.0-rc.1") < parse_semver("1.0.0")
    assert parse_semver("1.0.0-rc.2") > parse_semver("1.0.0-rc.1")
    assert parse_semver("1.0.0-alpha") < parse_semver("1.0.0-beta")


def test_build_metadata_is_ignored_in_ordering() -> None:
    """Semver quy định `build` không tham gia so sánh."""
    assert parse_semver("1.0.0+abc") == parse_semver("1.0.0+xyz")


def test_listing_is_sorted_by_semver_not_by_filename(tmp_path: Path) -> None:
    for version in ("1.9.0", "1.10.0", "1.0.0", "2.0.0-rc.1", "2.0.0"):
        save_bundle(make_bundle(bundle_version=version), tmp_path)
    assert [item.bundle_version for item in list_bundles(tmp_path)] == [
        "1.0.0",
        "1.9.0",
        "1.10.0",
        "2.0.0-rc.1",
        "2.0.0",
    ]


def test_latest_of_an_empty_root_is_none(tmp_path: Path) -> None:
    assert latest_bundle(tmp_path) is None


# ---------------------------------------------------------------------------
# 5. Ràng buộc giữa các trường — thứ mà fixture "toàn 1" không bao giờ chạm tới
# ---------------------------------------------------------------------------


def test_top_n_above_candidates_is_rejected() -> None:
    with pytest.raises(ValidationError, match="không thể trả về nhiều hơn"):
        RerankComponent(model="m", candidates=6, top_n=50, max_length=512)


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValidationError, match="vòng lặp vô hạn"):
        ChunkingComponent(
            strategy="hybrid",
            chunk_size=1000,
            chunk_overlap=1000,
            contextual=False,
            chunking_fingerprint="x",
        )


def test_naive_datetime_is_rejected() -> None:
    """Bundle sinh trên GPU thuê, đọc trên máy khác múi giờ; mốc không offset là
    mốc không so được, và `W4-02` chọn bản rollback theo thứ tự thời gian."""
    with pytest.raises(ValidationError, match="timezone"):
        make_bundle(created_at=datetime(2026, 9, 3, 10, 0))


@pytest.mark.parametrize("sha", ["not-a-sha", "ABCDEF1", "abc", "g" * 8])
def test_git_sha_must_look_like_one(sha: str) -> None:
    with pytest.raises(ValidationError, match="git_sha"):
        make_bundle(git_sha=sha)


# ---------------------------------------------------------------------------
# TD-36 + TD-38 — chữ ký sống sót qua việc schema mọc thêm trường
# ---------------------------------------------------------------------------


def _sign_raw(payload: dict[str, Any]) -> dict[str, Any]:
    """Ký một payload JSON thô, đúng như một pipeline **phiên bản cũ** đã làm."""
    body = {k: v for k, v in payload.items() if k != "checksum"}
    return {**body, "checksum": compute_checksum(body)}


def test_a_bundle_signed_before_a_field_existed_still_verifies(tmp_path: Path) -> None:
    """⭐⭐ `TD-36`. Đây là lỗi vô hiệu hoá chính cơ chế nó bảo vệ.

    Dựng lại **đúng** một manifest do pipeline cũ sinh: bỏ hẳn khoá
    `components.embedding.revision` (trường mới, có mặc định) rồi ký lên payload
    thiếu nó — byte-for-byte giống thứ tồn tại trước khi trường ấy ra đời.

    Bản đầu băm lại **model đã validate**, nên pydantic lấp `revision=None` lúc
    đọc, payload đem băm đổi theo, và một bundle không ai chạm vào trở thành
    "sai chữ ký". Hệ quả: mọi lần mở rộng schema làm hỏng toàn bộ bundle đã phát
    hành, tức chữ ký chỉ dùng được cho tới lần đổi schema đầu tiên.
    """
    directory = tmp_path / "rag-bundle-v1.0.0"
    directory.mkdir()
    raw = json.loads(make_bundle().model_dump_json())
    del raw["components"]["embedding"]["revision"]
    (directory / "manifest.json").write_text(
        json.dumps(_sign_raw(raw), ensure_ascii=False), encoding="utf-8"
    )

    loaded = load_bundle(directory)
    assert loaded.components.embedding.revision is None


def test_reformatting_a_manifest_still_does_not_break_the_signature(tmp_path: Path) -> None:
    """Tính chất **cũ** phải sống sót qua cách sửa `TD-36`: thụt lề và thứ tự khoá
    là chuyện hiển thị, không phải nội dung. Mất nó thì chữ ký bị tắt trong tuần
    đầu (`W4-01` §4)."""
    manifest = save_bundle(make_bundle(), tmp_path)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    manifest.write_text(
        json.dumps(dict(reversed(list(raw.items()))), indent=8, ensure_ascii=False),
        encoding="utf-8",
    )
    assert load_bundle(manifest).bundle_version == "1.0.0"


def test_tampering_is_still_caught_after_the_td36_fix(tmp_path: Path) -> None:
    """Nửa còn lại của đánh đổi: nới cho schema mọc **không** được nới cho sửa tay.

    `test_deleting_a_defaulted_field_is_also_caught` ở trên đã ghim ca xoá; đây
    là ca đổi **giá trị**, và nó phải vỡ vì `raw` đổi mà `checksum` thì không.
    """
    manifest = save_bundle(make_bundle(), tmp_path)
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["eval"]["retrieval_metrics"]["ndcg@10"] = 0.99
    manifest.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(BundleChecksumError):
        load_bundle(manifest)


def test_the_bundle_records_which_retriever_was_measured() -> None:
    """⭐ `TD-38`. Một chuỗi thay cho năm trường, vì quy ước đặt tên của `rag_core`
    đã gom sẵn **đúng** những cần điều khiển làm đổi kết quả — `rrf1`, `c20`,
    `L512`, `float16` — và cố ý bỏ những cái không (`batch_size`)."""
    name = "reranked[qdrant-hybrid:rag_bgem3_ctx:rrf1-c20]:BAAI/bge-reranker-v2-m3@cuda:L512:float16:n50"
    bundle = make_bundle(
        components=make_bundle().components.model_copy(update={"retriever_name": name})
    )
    assert bundle.components.retriever_name == name


def test_the_real_sample_bundle_carries_its_retriever_name() -> None:
    """Bundle mẫu thật trong repo, không phải fixture: nó là thứ `W4-03` nạp lúc
    khởi động, nên nếu nó thiếu trường này thì phép kiểm danh tính im lặng tắt."""
    bundle = load_bundle(Path(__file__).resolve().parents[2] / "bundles" / "rag-bundle-v0.1.0")
    assert bundle.components.retriever_name is not None
    assert "float16" in bundle.components.retriever_name
