"""Bộ parse có đổi nội dung corpus không, và đổi bao nhiêu. `W3-01`.

Chạy: `make loader-probe`

**Câu hỏi.** Cho tới hết `W2`, `Document.content` là `payload.decode("utf-8")`
— hàm đồng nhất. Manifest ghim `sha256` của byte, nên ghim byte cũng là ghim
nội dung, nên `TextSpan` của `golden_v1` an toàn. `W3-01` đặt một bộ parse vào
giữa. Câu hỏi không phải "docling có tốt không" mà là **"nếu corpus hiện tại đi
qua docling thì nhãn còn đúng không"** — và đó là câu trả lời bằng số, không
phải bằng phán đoán.

Ba phép đo:

1. **Đồng nhất byte** — bao nhiêu trong 60 tài liệu ra đúng văn bản cũ.
2. **Lệch bao nhiêu** — tỉ lệ độ dài, và tỉ lệ **dòng** còn nguyên vẹn.

   ⚠️ Lần đầu tôi đo bằng `SequenceMatcher.ratio()` trên nguyên hai chuỗi. Nó
   không sai, nó **không chạy xong**: thuật toán là bậc hai và tài liệu ở đây
   tới ~500 KB, tức một tài liệu đã hàng phút và 60 tài liệu thì không về. So
   theo dòng bằng `Counter` là tuyến tính và trả lời đúng câu đang hỏi —
   "docling giữ lại được bao nhiêu phần văn bản" — nên đổi sang đó.
3. **Nhãn nào chết** — với mỗi span của `golden_v1`, văn bản tại đúng offset ấy
   trong bản parse mới còn là văn bản cũ không. Đây mới là con số quan trọng:
   (1) và (2) nói tài liệu đổi, còn (3) nói **bằng chứng** đổi.

⚠️ docling không nhận `.txt` (`InputFormat` không có), nên phép đo phải đổi đuôi
thành `.md` rồi mới đưa vào. Đó cũng chính là kịch bản đáng lo thật: cùng một
nội dung, một lần vào repo dưới đuôi `.txt` và một lần dưới `.md`, sẽ cho hai
`Document.content` khác nhau.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import sys
import tempfile
from collections import Counter
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "packages") not in sys.path:  # chạy được kể cả khi chưa `uv sync`
    sys.path.insert(0, str(_REPO_ROOT / "packages"))

from rag_core.loaders import load_document  # noqa: E402

DEFAULT_MANIFEST = _REPO_ROOT / "data" / "corpus_manifest.csv"
DEFAULT_CORPUS = _REPO_ROOT / "data" / "corpus"
DEFAULT_GOLDEN = _REPO_ROOT / "data" / "golden" / "golden_v1.jsonl"


def _corpus_files(manifest: Path, corpus_dir: Path) -> list[tuple[str, Path]]:
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    files = [(row["doc_id"], corpus_dir / row["relative_path"]) for row in rows]
    return [(doc_id, path) for doc_id, path in files if path.is_file()]


def _spans_by_doc(golden: Path) -> dict[str, list[tuple[int, int]]]:
    """`{doc_id: [(start, end), …]}` từ golden set, bỏ qua bản ghi không có span."""
    spans: dict[str, list[tuple[int, int]]] = {}
    if not golden.is_file():
        return spans
    with golden.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for evidence in record.get("relevant_spans", []) or []:
                doc_id = evidence.get("doc_id")
                start, end = evidence.get("start"), evidence.get("end")
                if doc_id and isinstance(start, int) and isinstance(end, int):
                    spans.setdefault(doc_id, []).append((start, end))
    return spans


def _line_survival(baseline: str, parsed: str) -> float:
    """Tỉ lệ dòng không rỗng của bản gốc còn xuất hiện nguyên vẹn ở bản parse.

    Đếm theo bội (`Counter`) chứ không theo tập: một dòng lặp 5 lần mà bản parse
    chỉ giữ 2 thì mất 3, và dùng tập sẽ báo là còn nguyên.
    """
    left = Counter(line for line in baseline.splitlines() if line.strip())
    if not left:
        return 0.0
    right = Counter(line for line in parsed.splitlines() if line.strip())
    kept = sum((left & right).values())
    return kept / sum(left.values())


def probe(
    manifest: Path = DEFAULT_MANIFEST,
    corpus_dir: Path = DEFAULT_CORPUS,
    golden: Path = DEFAULT_GOLDEN,
    limit: int | None = None,
) -> dict[str, object]:
    files = _corpus_files(manifest, corpus_dir)[:limit]
    if not files:
        raise SystemExit(f"Không thấy tài liệu nào từ {manifest}")

    spans = _spans_by_doc(golden)
    identical = 0
    ratios: list[float] = []
    similarities: list[float] = []
    spans_checked = 0
    spans_surviving = 0
    failures: list[str] = []

    with tempfile.TemporaryDirectory(prefix="loader-probe-") as tmp:
        staging = Path(tmp)
        for doc_id, path in files:
            baseline = path.read_bytes().decode("utf-8")
            # Đổi đuôi sang `.md` vì docling không nhận `.txt`.
            renamed = staging / (path.stem + ".md")
            shutil.copyfile(path, renamed)
            try:
                parsed = load_document(renamed).text
            except Exception as exc:
                failures.append(f"{doc_id}: {type(exc).__name__} {exc}")
                continue

            if parsed == baseline:
                identical += 1
            ratios.append(len(parsed) / len(baseline) if baseline else 0.0)
            similarities.append(_line_survival(baseline, parsed))

            for start, end in spans.get(doc_id, []):
                spans_checked += 1
                if baseline[start:end] and parsed[start:end] == baseline[start:end]:
                    spans_surviving += 1

    return {
        "documents": len(files),
        "parsed": len(ratios),
        "failed": failures,
        "identical": identical,
        "length_ratio_mean": statistics.fmean(ratios) if ratios else 0.0,
        "length_ratio_min": min(ratios) if ratios else 0.0,
        "length_ratio_max": max(ratios) if ratios else 0.0,
        "similarity_mean": statistics.fmean(similarities) if similarities else 0.0,
        "similarity_min": min(similarities) if similarities else 0.0,
        "spans_checked": spans_checked,
        "spans_surviving": spans_surviving,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    result = probe(args.manifest, args.corpus, args.golden, args.limit)
    total, parsed = result["documents"], result["parsed"]
    print(f"tài liệu           : {total}  (parse được {parsed})")
    print(f"đồng nhất byte     : {result['identical']}/{parsed}")
    print(
        f"tỉ lệ độ dài       : tb {result['length_ratio_mean']:.4f}  "
        f"[{result['length_ratio_min']:.4f}, {result['length_ratio_max']:.4f}]"
    )
    print(
        f"dòng còn nguyên    : tb {result['similarity_mean']:.4f}  "
        f"min {result['similarity_min']:.4f}"
    )
    checked = result["spans_checked"]
    if checked:
        survived = result["spans_surviving"]
        print(f"span golden sống   : {survived}/{checked}  ({survived / checked:.1%})")
    else:
        print("span golden sống   : không có span nào để kiểm")
    for failure in result["failed"]:
        print(f"  ✗ {failure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
