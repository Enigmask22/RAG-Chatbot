"""Đối chiếu thứ DVC đang theo dõi với thứ manifest khai báo.

Từ `W1-09` corpus có **hai** cơ chế versioning chồng lên nhau, và đó là chủ ý:

* `data/corpus_manifest.csv` — sha256 từng tài liệu, kèm nguồn và giấy phép. Đây
  là thứ ép quy tắc cứng #3, và là đường phục hồi độc lập: `scripts/fetch_corpus.py`
  tải lại được từ World Bank rồi so sha256.
* `data/corpus.dvc` — một md5 cho cả thư mục, trỏ tới nội dung nằm ở DVC remote.
  Đây là đường phục hồi khi URL gốc chết hoặc khi tài liệu **không** tải tự động
  được (nguồn (b) văn bản pháp luật và (c) báo cáo HOSE phải chọn tay).

Hai nguồn sự thật cho cùng một tập dữ liệu là một nguy cơ có thật: chỉ cần ai đó
thêm file vào `data/corpus/` rồi `dvc add` mà quên cập nhật manifest, hoặc ngược
lại, là từ đó trở đi "corpus" nghĩa là hai tập khác nhau tuỳ người hỏi ai. Không
có lỗi nào nổ ra — build index đọc theo manifest nên bỏ qua file lạ, còn
`dvc status` thì sạch. Module này biến sự lệch đó thành một lỗi tường minh.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from pipeline.corpus.manifest import load_manifest

__all__ = [
    "NON_DOCUMENT_FILES",
    "DvcOut",
    "DvcTrackingError",
    "TrackingReport",
    "load_dvc_out",
    "verify_corpus_tracking",
]

NON_DOCUMENT_FILES: frozenset[str] = frozenset({".gitkeep"})
"""File nằm trong `data/corpus/` nhưng cố ý **không** có trong manifest.

`.gitkeep` giữ thư mục tồn tại trong git khi nội dung do DVC quản. Nó được DVC
đếm vào `nfiles`, nên nếu không trừ ra thì phép so số lượng luôn lệch 1 và cách
sửa dễ nhất — nới điều kiện thành "xấp xỉ" — sẽ làm test mất hết tác dụng.
"""


class DvcTrackingError(RuntimeError):
    """DVC và manifest không nói cùng một chuyện về corpus."""


class DvcOut(BaseModel):
    """Một entry `outs` trong file `.dvc`.

    Chỉ nhận những khoá cần dùng nhưng **không** `extra="forbid"`: format file
    `.dvc` do DVC định nghĩa và nó có thêm khoá theo phiên bản (`files`,
    `remote`, `push`...). Khoá lạ ở đây là chuyện bình thường, không phải lỗi
    cấu hình của dự án — khác hẳn config do người viết, nơi khoá lạ gần như
    luôn là lỗi chính tả.
    """

    model_config = ConfigDict(frozen=True)

    path: str
    hash_md5: str = Field(alias="md5")
    size: int
    nfiles: int


def load_dvc_out(path: str | Path) -> DvcOut:
    """Đọc file `.dvc` và trả về entry `outs` duy nhất.

    Raises:
        DvcTrackingError: file không đọc được, hoặc có 0 / nhiều hơn 1 `outs`.
    """
    p = Path(path)
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DvcTrackingError(
            f"Không thấy {p}. Corpus chưa được DVC theo dõi — chạy `dvc add data/corpus`."
        ) from exc
    except yaml.YAMLError as exc:
        raise DvcTrackingError(f"{p} không phải YAML hợp lệ: {exc}") from exc

    if not isinstance(raw, dict):
        raise DvcTrackingError(f"{p} phải là mapping YAML, nhận được {type(raw).__name__}.")

    outs = raw.get("outs")
    if not isinstance(outs, list) or len(outs) != 1:
        n = len(outs) if isinstance(outs, list) else 0
        raise DvcTrackingError(
            f"{p} phải có đúng 1 entry `outs`, thấy {n}. "
            "File .dvc của một thư mục corpus không nên gộp nhiều output."
        )

    return DvcOut.model_validate(outs[0])


class TrackingReport(BaseModel):
    """Kết quả đối chiếu, dùng được cho cả log lẫn báo cáo."""

    model_config = ConfigDict(frozen=True)

    dvc_md5: str
    dvc_nfiles: int
    dvc_size: int
    manifest_documents: int
    non_document_files: int

    @property
    def tracked_documents(self) -> int:
        """Số file DVC đếm được mà đáng lẽ phải là tài liệu."""
        return self.dvc_nfiles - self.non_document_files

    def summary(self) -> str:
        return (
            f"DVC {self.dvc_md5[:12]}… · {self.dvc_nfiles} file "
            f"({self.tracked_documents} tài liệu + {self.non_document_files} phụ trợ) · "
            f"{self.dvc_size / 1_048_576:.1f} MiB · manifest {self.manifest_documents} tài liệu"
        )


def verify_corpus_tracking(
    dvc_file: str | Path = "data/corpus.dvc",
    manifest_path: str | Path = "data/corpus_manifest.csv",
    corpus_dir: str | Path | None = None,
) -> TrackingReport:
    """So số tài liệu DVC theo dõi với số entry trong manifest.

    Cố ý **không** băm lại nội dung: sha256 từng tài liệu đã được
    `pipeline.indexing.corpus_loader.iter_documents` kiểm ở mỗi lần build index,
    và md5 thư mục đã được `dvc status` kiểm. Việc còn thiếu chỉ là phép so số
    lượng giữa hai cơ chế đó — chỗ duy nhất mà cả hai đều mù.

    Args:
        dvc_file: đường dẫn file `.dvc` của thư mục corpus.
        manifest_path: đường dẫn manifest CSV.
        corpus_dir: nếu truyền, kiểm luôn rằng mọi `relative_path` trong manifest
            thật sự nằm trên đĩa. Bỏ trống khi corpus chưa `dvc pull` về.

    Raises:
        DvcTrackingError: số lượng lệch, hoặc thiếu file mà manifest khai.
    """
    out = load_dvc_out(dvc_file)
    entries = load_manifest(manifest_path)

    root = Path(corpus_dir) if corpus_dir is not None else None
    if root is not None:
        missing = [e.relative_path for e in entries if not (root / e.relative_path).is_file()]
        if missing:
            shown = ", ".join(missing[:5])
            raise DvcTrackingError(
                f"Manifest khai {len(missing)} tài liệu không có trên đĩa: {shown}"
                f"{'…' if len(missing) > 5 else ''}. Chạy `dvc pull` hoặc `make corpus`."
            )
        extra = sorted(
            p.name
            for p in root.rglob("*")
            if p.is_file()
            and p.name not in NON_DOCUMENT_FILES
            and p.relative_to(root).as_posix() not in {e.relative_path for e in entries}
        )
        if extra:
            shown = ", ".join(extra[:5])
            raise DvcTrackingError(
                f"{len(extra)} file trong {root} không có trong manifest: {shown}"
                f"{'…' if len(extra) > 5 else ''}. "
                "Build index sẽ bỏ qua chúng (nó đọc theo manifest) nên `dvc push` "
                "vẫn đem chúng lên remote như một phần của corpus. Thêm vào "
                "manifest qua `scripts/fetch_corpus.py`, hoặc xoá đi."
            )

    non_doc = _count_non_document_files(root) if root is not None else len(NON_DOCUMENT_FILES)
    report = TrackingReport(
        dvc_md5=out.hash_md5,
        dvc_nfiles=out.nfiles,
        dvc_size=out.size,
        manifest_documents=len(entries),
        non_document_files=non_doc,
    )

    if report.tracked_documents != report.manifest_documents:
        raise DvcTrackingError(
            f"DVC theo dõi {report.tracked_documents} tài liệu nhưng manifest khai "
            f"{report.manifest_documents}. Hai cơ chế versioning đã lệch nhau: "
            "`dvc add data/corpus` đã chạy trên một tập file khác với tập mà "
            "`scripts/fetch_corpus.py` ghi vào manifest. "
            f"({report.summary()})"
        )
    return report


def _count_non_document_files(root: Path) -> int:
    """Đếm file phụ trợ thật sự có trên đĩa, thay vì giả định là đủ cả."""
    return sum(1 for p in root.rglob("*") if p.is_file() and p.name in NON_DOCUMENT_FILES)


def main() -> int:
    """CLI: `python -m pipeline.corpus.dvc_state`. Trả 0 nếu khớp, 1 nếu lệch."""
    import argparse
    import logging

    parser = argparse.ArgumentParser(
        description="Đối chiếu corpus mà DVC theo dõi với manifest.",
    )
    parser.add_argument("--dvc-file", default="data/corpus.dvc")
    parser.add_argument("--manifest", default="data/corpus_manifest.csv")
    parser.add_argument(
        "--skip-disk",
        action="store_true",
        help="Chỉ so `.dvc` với manifest, không cần corpus đã `dvc pull` về.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("dvc_state")

    corpus_dir = None if args.skip_disk else Path(args.dvc_file).parent / "corpus"
    try:
        report = verify_corpus_tracking(args.dvc_file, args.manifest, corpus_dir)
    except DvcTrackingError as exc:
        log.error("LỆCH: %s", exc)
        return 1
    log.info("KHỚP: %s", report.summary())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
