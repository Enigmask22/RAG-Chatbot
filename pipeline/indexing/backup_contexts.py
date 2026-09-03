"""Sao lưu artifact ngữ cảnh vào git — thứ duy nhất trong repo tốn tiền thật.

## Vì sao cần một file riêng cho việc `gzip`

Không phải vì nén khó. Vì **đường mất dữ liệu ở đây không giống mọi đường khác
trong dự án**, và không có gì trong repo nói ra điều đó.

Mọi artifact khác của `data/contexts/` — gói request 285 MB, các lượt dry-run —
sinh lại **miễn phí** trong vài giây từ corpus và mã nguồn. `.gitignore` loại cả
thư mục vì luật ấy đúng với chúng. Nhưng `contexts.jsonl` sinh lại tốn **~$5,90
tiền API thật**, và nó bị luật kia quét nhầm: cùng thư mục, cùng đuôi file, kinh
tế ngược hẳn.

Hệ quả trước khi có file này: artifact tồn tại **đúng một bản, trên đúng một
laptop**. DVC remote của dự án trỏ vào `D:/dvc-remote/` — cùng ổ đĩa với repo,
tức là một bản sao chứ không phải một bản lưu.

## Vì sao commit vào git chứ không đẩy lên DVC

Nén còn **1,8 MB** (12,1 → 1,8; các câu ngữ cảnh lặp lại rất nhiều cụm định vị
giống nhau nên nó nén cực tốt). Ở kích thước đó, git là chỗ lưu **ít cách hỏng
nhất**: không cần remote, không cần credential, không cần một lệnh `pull` mà
người clone có thể quên. Một clone mới build được index ngay.

Corpus thì ngược lại: 60 tài liệu PDF, không nén được, và **tải lại được miễn
phí** bằng `scripts/fetch_corpus.py` — nên nó thuộc về DVC, và nó đang ở đó.

## Hai điều dễ làm sai

* **`mtime` trong header gzip.** Mặc định `gzip` ghi thời điểm nén vào file, nên
  nén lại **cùng một nội dung** cho ra **byte khác nhau** và git thấy file "đổi"
  mỗi lần chạy. Đặt `mtime=0` để phép nén là hàm thuần của nội dung.
* **Bản nén trôi khỏi bản thô.** Nếu ngữ cảnh được sinh lại mà quên chạy lại
  target này thì git giữ một bản cũ và không gì báo. `--check` so băm của bản
  thô với sidecar `.sha256`, và `make ctx-verify` gọi nó.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import logging
from collections.abc import Sequence
from pathlib import Path

__all__ = ["BACKED_UP", "backup", "main", "verify"]

logger = logging.getLogger(__name__)

CONTEXTS_DIR = Path("data/contexts")

BACKED_UP = (
    "contexts.jsonl",
    "contexts-single-probe.jsonl",
)
"""Chỉ hai file, và cả hai đều là **tiền đã tiêu**.

* `contexts.jsonl` — 15.814 ngữ cảnh của điểm vận hành, ~$5,90.
* `contexts-single-probe.jsonl` — 860 ngữ cảnh chế độ đơn, là **bằng chứng** cho
  phát hiện "prompt gộp v1 đánh rơi danh tính tài liệu ở 30% chunk" (`W3-04`
  §6ter). Không có nó thì con số ấy trong báo cáo không kiểm lại được.

⚠️ Cố tình **không** gồm `contexts-b8.failures.jsonl` (response thô của lô bị từ
chối) và `contexts-b8-promptv1.jsonl` (lượt chạy đã bỏ): chúng cũng tốn tiền,
nhưng cả hai đã bị thay thế bởi artifact cuối và giữ lại chỉ tạo thêm chỗ để ai
đó nạp nhầm.
"""


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _pack(source: Path, target: Path) -> int:
    """Nén xác định: cùng nội dung ⇒ cùng byte, để git không thấy nhiễu."""
    raw = source.read_bytes()
    with (
        target.open("wb") as handle,
        gzip.GzipFile(filename="", mode="wb", compresslevel=9, fileobj=handle, mtime=0) as packed,
    ):
        packed.write(raw)
    return target.stat().st_size


def backup(directory: Path = CONTEXTS_DIR, names: Sequence[str] = BACKED_UP) -> list[Path]:
    written: list[Path] = []
    for name in names:
        source = directory / name
        if not source.exists():
            logger.warning("bỏ qua %s — không có trên đĩa", source)
            continue
        raw = source.read_bytes()
        packed = directory / f"{name}.gz"
        sidecar = directory / f"{name}.sha256"
        size = _pack(source, packed)
        sidecar.write_text(f"{_digest(raw)}  {name}\n", encoding="utf-8")
        logger.info(
            "%s: %.1f MB → %.2f MB (%.0f×)", name, len(raw) / 1e6, size / 1e6, len(raw) / size
        )
        written.extend((packed, sidecar))
    return written


def verify(directory: Path = CONTEXTS_DIR, names: Sequence[str] = BACKED_UP) -> list[str]:
    """Trả về danh sách vấn đề. Rỗng = bản trong git khớp bản trên đĩa.

    Ba câu hỏi khác nhau, và chúng cần ba câu trả lời khác nhau:

    * **thiếu bản nén** — sao lưu chưa từng chạy;
    * **bản thô lệch sidecar** — ngữ cảnh đã sinh lại mà quên `make ctx-backup`;
    * **bản nén giải ra không khớp sidecar** — bản trong git hỏng, và đây là ca
      duy nhất mà tiền thật sự có nguy cơ mất.

    ⚠️ Không có bản thô trên đĩa **không** phải lỗi: một clone mới chỉ có bản nén,
    và đó đúng là trạng thái mà cả cơ chế này nhắm tới.
    """
    problems: list[str] = []
    for name in names:
        source = directory / name
        packed = directory / f"{name}.gz"
        sidecar = directory / f"{name}.sha256"

        if not packed.exists() or not sidecar.exists():
            problems.append(f"{name}: chưa có bản sao lưu trong git (chạy `make ctx-backup`)")
            continue

        expected = sidecar.read_text(encoding="utf-8").split()[0]
        if _digest(gzip.decompress(packed.read_bytes())) != expected:
            problems.append(f"{name}: bản nén trong git KHÔNG khớp sidecar — bản lưu đã hỏng")
        if source.exists() and _digest(source.read_bytes()) != expected:
            problems.append(
                f"{name}: bản trên đĩa lệch bản đã lưu — "
                "ngữ cảnh sinh lại mà quên `make ctx-backup`"
            )
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=CONTEXTS_DIR)
    parser.add_argument("--check", action="store_true", help="Chỉ kiểm, không ghi.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.check:
        problems = verify(args.dir)
        for problem in problems:
            logger.error("✗ %s", problem)
        if not problems:
            logger.info("✓ bản sao lưu ngữ cảnh khớp (%s)", ", ".join(BACKED_UP))
        return 1 if problems else 0

    backup(args.dir)
    logger.info("Nhớ `git add data/contexts/*.gz data/contexts/*.sha256` rồi commit.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
