"""Artifact ngữ cảnh phải sống sót khi mất laptop — nó là thứ duy nhất tốn tiền thật.

Mọi thứ khác trong repo sinh lại được từ mã nguồn và corpus. `contexts.jsonl`
thì không: **~$5,90 tiền API**, và trước hạng mục này nó tồn tại đúng một bản
trên đúng một máy (DVC remote của dự án trỏ vào `D:/dvc-remote/` — cùng ổ đĩa
với repo, tức bản sao chứ không phải bản lưu).

Ba nhóm, và nhóm cuối là nhóm đáng giá nhất vì lỗi ở đó **im lặng tuyệt đối**.
"""

from __future__ import annotations

import gzip
import subprocess
from pathlib import Path

import pytest

from pipeline.indexing.backup_contexts import BACKED_UP, backup, verify
from pipeline.indexing.build_index import _resolve_contexts_path

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def artifacts(tmp_path: Path) -> Path:
    for name in BACKED_UP:
        (tmp_path / name).write_text(
            '{"chunk_id": "doc::00000", "cfg": "c0ffee", "context": "Đoạn này nằm trong..."}\n',
            encoding="utf-8",
        )
    return tmp_path


# ---------------------------------------------------------------------------
# 1. Clone mới chỉ có bản nén — và nó phải đủ
# ---------------------------------------------------------------------------


def test_a_fresh_clone_with_only_the_compressed_file_still_builds(tmp_path: Path) -> None:
    """⭐⭐ Đúng tình huống mà cả cơ chế này tồn tại vì nó.

    Người clone repo về không có bản thô 12 MB (nó nằm trong `.gitignore`). Nếu
    `_resolve_contexts_path` không tìm tới `.gz` thì build đòi sinh lại ngữ cảnh
    — tức đòi trả **$5,90** cho thứ đã nằm sẵn trong repo.
    """
    (tmp_path / "contexts.jsonl.gz").write_bytes(gzip.compress(b'{"context": "x"}\n'))
    assert _resolve_contexts_path(tmp_path / "contexts.jsonl").name == "contexts.jsonl.gz"


def test_the_plain_file_wins_when_both_exist(tmp_path: Path) -> None:
    """Máy đang sinh ngữ cảnh có cả hai; bản thô là bản mới hơn."""
    (tmp_path / "contexts.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "contexts.jsonl.gz").write_bytes(gzip.compress(b"{}\n"))
    assert _resolve_contexts_path(tmp_path / "contexts.jsonl").name == "contexts.jsonl"


def test_missing_both_says_how_much_regenerating_costs(tmp_path: Path) -> None:
    """Thông báo lỗi phải nêu **giá**.

    "Không thấy file" dẫn người đọc tới việc chạy lại job. Nếu nguyên nhân thật
    là `.gitignore` cấu hình sai thì họ vừa trả $5,90 cho một lỗi cấu hình.
    """
    with pytest.raises(FileNotFoundError, match=r"\$5,90"):
        _resolve_contexts_path(tmp_path / "contexts.jsonl")


def test_the_error_also_points_at_gitignore(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="gitignore"):
        _resolve_contexts_path(tmp_path / "contexts.jsonl")


# ---------------------------------------------------------------------------
# 2. Sao lưu và phát hiện trôi
# ---------------------------------------------------------------------------


def test_compressing_the_same_content_twice_gives_identical_bytes(artifacts: Path) -> None:
    """⚠️ Mặc định `gzip` ghi thời điểm nén vào header, nên nén lại **cùng nội
    dung** cho ra byte khác và git thấy file "đổi" mỗi lần chạy — nhiễu trong
    lịch sử commit của một file nhị phân 1,8 MB."""
    backup(artifacts)
    first = (artifacts / "contexts.jsonl.gz").read_bytes()
    backup(artifacts)
    assert (artifacts / "contexts.jsonl.gz").read_bytes() == first


def test_a_fresh_backup_verifies_clean(artifacts: Path) -> None:
    backup(artifacts)
    assert verify(artifacts) == []


def test_regenerating_contexts_without_re_running_backup_is_caught(artifacts: Path) -> None:
    """⭐ Cách hỏng thật: sinh lại ngữ cảnh, quên `make ctx-backup`, và git giữ
    một bản cũ mà không gì báo."""
    backup(artifacts)
    (artifacts / "contexts.jsonl").write_text('{"context": "đã sinh lại"}\n', encoding="utf-8")
    problems = verify(artifacts)
    assert any("quên `make ctx-backup`" in problem for problem in problems)


def test_a_corrupt_backup_is_caught_even_without_the_plain_file(artifacts: Path) -> None:
    """Ca duy nhất mà tiền thật sự có nguy cơ mất, nên nó phải phát hiện được
    trên một clone mới — nơi bản thô không tồn tại để đối chiếu."""
    backup(artifacts)
    (artifacts / "contexts.jsonl").unlink()
    (artifacts / "contexts.jsonl.gz").write_bytes(gzip.compress(b'{"context": "khac"}\n'))
    assert any("bản lưu đã hỏng" in problem for problem in verify(artifacts))


def test_a_clone_with_only_the_backup_verifies_clean(artifacts: Path) -> None:
    """⚠️ Thiếu bản thô **không** phải lỗi — đó đúng là trạng thái của clone mới."""
    backup(artifacts)
    for name in BACKED_UP:
        (artifacts / name).unlink()
    assert verify(artifacts) == []


def test_never_backed_up_is_reported_differently_from_corrupt(tmp_path: Path) -> None:
    """ "Chưa từng sao lưu" và "bản lưu hỏng" đòi hai phản ứng khác nhau."""
    (tmp_path / "contexts.jsonl").write_text("{}\n", encoding="utf-8")
    assert any("chưa có bản sao lưu" in problem for problem in verify(tmp_path))


# ---------------------------------------------------------------------------
# 3. ⭐⭐ `.gitignore` thật sự cho file vào — lỗi ở đây im lặng tuyệt đối
# ---------------------------------------------------------------------------


def _ignored(relative: str) -> bool:
    """⚠️ Chỉ dùng cho chiều **khẳng định** (file này có bị loại không).

    `git check-ignore -q` trả 0 khi path khớp **bất kỳ** luật nào — kể cả một
    dòng phủ định `!`. Nên nó **không** trả lời được câu "file này có vào git
    được không", và dùng nhầm nó cho chiều kia sẽ cho một test luôn xanh.
    """
    return (
        subprocess.run(
            ["git", "check-ignore", "-q", relative], cwd=REPO, capture_output=True
        ).returncode
        == 0
    )


def _tracked(relative: str) -> bool:
    return (
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative], cwd=REPO, capture_output=True
        ).returncode
        == 0
    )


@pytest.mark.parametrize(
    "relative",
    [f"data/contexts/{name}{suffix}" for name in BACKED_UP for suffix in (".gz", ".sha256")],
)
def test_the_backup_files_are_actually_in_git(relative: str) -> None:
    """⭐⭐ Test đáng giá nhất file này, vì lỗi nó bắt **không có triệu chứng nào**.

    `.gitignore` không cho re-include một file nằm trong *thư mục* đã bị loại:
    với `data/contexts/` thì mọi dòng `!data/contexts/...` bên dưới đơn giản là
    **không có tác dụng** — không lỗi, không cảnh báo, `git add` lặng lẽ bỏ qua,
    và repo trông như đã sao lưu trong khi chưa hề. Dạng đúng là `data/contexts/*`.

    Tôi mắc đúng lỗi này khi viết hạng mục, và phép kiểm đầu tiên tôi viết cho nó
    (`git check-ignore -q`) **cũng sai** — lệnh ấy trả 0 cả khi path khớp một
    dòng phủ định. Nên phép kiểm phải hỏi thẳng câu cuối cùng: file có nằm trong
    git hay không.
    """
    assert _tracked(relative), (
        f"{relative} không nằm trong git — kiểm xem luật có phải dạng "
        "`data/contexts/*` (loại nội dung) chứ không phải `data/contexts/` (loại thư mục)"
    )


@pytest.mark.parametrize(
    "relative",
    [
        "data/contexts/contexts.jsonl",  # bản thô 12 MB
        "data/contexts/requests-b8.jsonl.gz",  # gói request, sinh lại miễn phí
        "data/contexts/contexts-b8.failures.jsonl",
    ],
)
def test_the_regenerable_artifacts_stay_out_of_git(relative: str) -> None:
    """Đối chứng. Không có nhóm này thì `data/contexts/*` nới ra quá tay cũng
    làm nhóm trên xanh — và 285 MB gói request đi vào lịch sử git vĩnh viễn."""
    assert _ignored(relative)


def test_the_committed_backup_matches_what_is_on_disk() -> None:
    """Ghim rằng bản đang nằm trong git **là** bản đã sinh ra index đang chạy."""
    problems = verify(REPO / "data" / "contexts")
    assert problems == [], "\n".join(problems)
