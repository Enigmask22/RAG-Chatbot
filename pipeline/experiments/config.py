"""Ma trận thí nghiệm: YAML → danh sách ô, kiểm hết trước khi chạy ô nào.

## Vì sao ma trận là một **danh sách khối** chứ không phải một tích Descartes

Cách hiển nhiên là khai mỗi chiều một lần rồi `itertools.product` tất cả. Nó sai
ở đây vì không gian tham số **không phải hình hộp**: `k`/`candidate_k` chỉ có
nghĩa với nhánh `hybrid`, `rerank_candidates` chỉ có nghĩa với `reranked`. Tích
đầy đủ sinh ra `dense × k=1` — một ô mà `build_branch` từ chối (đúng ra phải từ
chối, xem `W2-03`).

Còn hai đường thoát và cả hai đều tệ hơn:

* **Sinh hết rồi lọc ô không hợp lệ.** Lúc đó "12 tổ hợp" trong DoD `W2-08` trở
  thành một con số không đoán được từ file config, và một ô *bị lọc vì gõ sai
  tên tham số* trông giống hệt một ô *bị lọc vì không hợp lệ về ngữ nghĩa*. Đó
  lại đúng là chế độ hỏng im lặng mà `W2-06` vừa dọn.
* **Cho phép `null` để đánh dấu "chiều này không áp dụng".** Đọc được, nhưng số
  ô sinh ra vẫn là tích, và người đọc file phải tự nhân trong đầu để biết grid
  bao nhiêu ô.

Nên: `matrix` là **danh sách khối**, mỗi khối là một tích nhỏ *đồng nhất về
nhánh*. Đúng cấu trúc `matrix.include` của GitHub Actions, và vì cùng lý do.

## `options` luôn là **list**, kể cả khi chỉ một giá trị

`rerank_dtype: [float16]` dài hơn `rerank_dtype: float16` một cặp ngoặc, nhưng
nó xoá một câu hỏi khỏi đầu người đọc: mọi thứ trong `options` là một chiều, và
số ô của khối là tích các độ dài. Cho phép cả scalar lẫn list thì phải đọc kiểu
của từng giá trị mới biết grid to bằng bao nhiêu.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rag_core.retrieval import HYBRID_OPTIONS, RERANK_OPTIONS, check_branch_options
from rag_core.schemas import RetrievalMode

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

__all__ = [
    "BRANCH_OPTIONS",
    "OPTION_TOKENS",
    "SPEED_ONLY_OPTIONS",
    "ExperimentCell",
    "ExperimentConfig",
    "MatrixBlock",
    "expand",
    "load_experiment_config",
]

#: Mọi tham số nhánh mà một khối được phép quét. Lấy **từ `rag_core`** chứ không
#: liệt kê lại: thêm một tham số cho `hybrid` mà quên ở đây thì file config sẽ
#: báo "tham số không hợp lệ" cho một tham số hợp lệ.
BRANCH_OPTIONS = HYBRID_OPTIONS | RERANK_OPTIONS

#: Viết tắt để sinh `run_name` đọc được. Tham số không có trong đây vẫn dùng
#: được — nó chỉ vào tên dưới dạng đầy đủ, dài hơn chứ không sai.
OPTION_TOKENS: dict[str, str] = {
    "k": "k",
    "candidate_k": "ck",
    "weights": "w",
    "base": "on",
    "reranker_model": "m",
    "rerank_candidates": "rc",
    "rerank_top_n": "n",
    "rerank_max_length": "len",
    "rerank_batch_size": "bs",
    "rerank_device": "dev",
    "rerank_activation": "act",
    "rerank_dtype": "dt",
}

#: Tham số mà **dự án này đã tuyên bố là không đổi kết quả**, nên quét nó sinh ra
#: hai dòng bảng `W2-08` chắc chắn giống nhau trong phạm vi nhiễu — và một dòng
#: như thế đọc y như một phát hiện.
#:
#: Đây là lập luận **nhất quán**, không phải một khẳng định thực nghiệm mới:
#: `IndexConfig.fingerprint` cố ý loại `batch_size` với đúng lý lẽ đó từ `W1-08`,
#: nên dùng nó làm chiều ablation là tự mâu thuẫn với một quyết định đã ghi.
#: Vẫn cho phép list một phần tử — ghim một giá trị khác mặc định là chuyện khác.
#:
#: ⚠️ `rerank_device` **không** nằm trong đây, và đó là chỗ dễ nhầm: nó *trông*
#: như knob tốc độ, nhưng `rerank_dtype` mặc định là `auto` = fp16 trên CUDA và
#: fp32 ở nơi khác, nên đổi device **đổi cả dtype**, mà `W2-05` đo được fp16 đổi
#: top-1 ở 1/60 câu. Quét device là quét dtype một cách vô tình.
SPEED_ONLY_OPTIONS = frozenset({"rerank_batch_size"})


class MatrixBlock(BaseModel):
    """Một tích Descartes nhỏ, đồng nhất về nhánh truy hồi."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    label: str = ""
    """Tiền tố cho `run_name` của cả khối. Rỗng thì tên bắt đầu từ tên index."""

    index_config: tuple[Path, ...] = Field(min_length=1)
    retrieval_mode: tuple[RetrievalMode, ...] = Field(min_length=1)
    options: dict[str, tuple[Any, ...]] = Field(default_factory=dict)

    @field_validator("options")
    @classmethod
    def _known_and_non_empty(cls, value: dict[str, tuple[Any, ...]]) -> dict[str, tuple[Any, ...]]:
        unknown = sorted(value.keys() - BRANCH_OPTIONS)
        if unknown:
            raise ValueError(
                f"Tham số nhánh không hợp lệ: {unknown}. Hợp lệ: {sorted(BRANCH_OPTIONS)}."
            )
        for name, values in value.items():
            if not values:
                # Cùng lý lẽ với `MetadataFilter` từ chối `[]` ở `W2-06`: một list
                # rỗng đọc như "không quét chiều này" nhưng tích Descartes với nó
                # cho **không ô nào**, tức grid im lặng biến thành rỗng.
                raise ValueError(
                    f"options.{name} = [] làm tích Descartes ra 0 ô, không phải "
                    f"'bỏ qua chiều này'. Bỏ hẳn khoá đó nếu không muốn quét."
                )
            if name in SPEED_ONLY_OPTIONS and len(values) > 1:
                raise ValueError(
                    f"options.{name} có {len(values)} giá trị, nhưng {name} là knob "
                    f"tốc độ — `IndexConfig.fingerprint` cố ý loại nó vì nó không đổi "
                    f"kết quả. Quét nó sinh ra các dòng bảng giống nhau trong phạm vi "
                    f"nhiễu, đọc y như một phát hiện. Ghim một giá trị thì dùng list "
                    f"một phần tử."
                )
        return value

    def cells(self) -> Iterator[tuple[Path, RetrievalMode, dict[str, Any]]]:
        """Tích Descartes của khối, theo thứ tự khai báo trong YAML.

        Thứ tự **cố định** vì nó là thứ tự chạy, và thứ tự chạy là thứ resume
        phải khớp lại được sau khi crash.
        """
        names = sorted(self.options)
        value_lists = [self.options[name] for name in names]
        for index_config, mode, combo in itertools.product(
            self.index_config, self.retrieval_mode, itertools.product(*value_lists)
        ):
            yield index_config, mode, dict(zip(names, combo, strict=True))


class ExperimentConfig(BaseModel):
    """Toàn bộ một lần chạy grid — tiền thân phía *đo* của `RagBundle`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1)
    matrix: tuple[MatrixBlock, ...] = Field(min_length=1)

    run_prefix: str = ""
    """Tiền tố cho **mọi** `run_name` của grid.

    Tồn tại vì lần `--dry-run` đầu tiên trên config thật đã đâm ngay vào kiểm
    quyền sở hữu của preflight: ô `bgem3-sparse` trùng tên với báo cáo `W2-03` đã
    công bố trong `plans/reports/runs/`, và không có tiền tố thì grid sẽ ghi đè
    bằng chứng của một hạng mục đã xong.

    Cách khác là để grid **dùng lại** những lần chạy cũ trùng tên. Tệ hơn: state
    không sở hữu chúng nên không biết chúng được sinh bằng tham số nào, tức không
    kiểm được là chúng có khớp ô hiện tại hay không — đúng cái `fingerprint` tồn
    tại để trả lời.
    """

    golden: Path = Path("data/golden/golden_v1.jsonl")
    top_k: int = Field(default=20, ge=1)
    min_overlap_ratio: float = Field(default=0.5, gt=0.0, le=1.0)

    out_dir: Path = Path("plans/reports/runs")
    state_dir: Path = Path(".cache/experiments")

    tracking_uri: str | None = None
    """URI MLflow. `None` = không theo dõi (grid vẫn chạy và vẫn ghi file)."""

    experiment: str = ""
    """Tên experiment trên MLflow. Rỗng thì dùng `name`."""

    tags: dict[str, str] = Field(default_factory=dict)

    @property
    def state_path(self) -> Path:
        return self.state_dir / f"{self.name}.json"

    @property
    def mlflow_experiment(self) -> str:
        return self.experiment or self.name


class ExperimentCell(BaseModel):
    """Một ô của grid = một lần chạy eval = một dòng của bảng `W2-08`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_name: str
    index_config: Path
    retrieval_mode: RetrievalMode
    branch_options: dict[str, Any] = Field(default_factory=dict)
    top_k: int
    min_overlap_ratio: float
    golden: Path

    @model_validator(mode="after")
    def _branch_accepts_options(self) -> ExperimentCell:
        """Ô không hợp lệ nổ **lúc mở file config**, không phải lúc chạy tới nó.

        Đây là toàn bộ lý do `check_branch_options` được tách ra khỏi
        `build_branch` ở `W2-07`: câu trả lời "nhánh này có nhận tham số này
        không" trước đây chỉ lấy được bằng cách *dựng* retriever, mà dựng nhánh
        `reranked` nạp một cross-encoder 2,2 GB. Một grid 12 ô mà ô thứ 11 gõ sai
        tên tham số thì phải chết ở giây thứ nhất, không phải ở phút thứ 33.
        """
        check_branch_options(self.retrieval_mode, self.branch_options)
        return self

    def fingerprint(self, *, index_fingerprint: str, golden_digest: str) -> str:
        """Danh tính của ô — thứ resume so, thay vì so tên file báo cáo.

        Nhận `index_fingerprint` và `golden_digest` từ ngoài **có chủ đích**: kết
        quả của một ô không chỉ phụ thuộc những gì viết trong ô. Build lại
        `bgem3.yaml` với `chunk_size` khác thì mọi ô đọc index đó phải chạy lại,
        và đổi golden set thì cả grid phải chạy lại — nhưng chuỗi YAML của ô
        không đổi một ký tự nào trong cả hai trường hợp.

        Resume theo tên file báo cáo là cái bẫy ở đây: `bgem3-hybrid-k1` tồn tại
        nên bỏ qua, và con số cũ đi vào bảng `W2-08` dưới nhãn của tham số mới.
        """
        payload = {
            "run_name": self.run_name,
            "index_config": str(self.index_config).replace("\\", "/"),
            "index_fingerprint": index_fingerprint,
            "retrieval_mode": self.retrieval_mode.value,
            "branch_options": self.branch_options,
            "top_k": self.top_k,
            "min_overlap_ratio": self.min_overlap_ratio,
            "golden": str(self.golden).replace("\\", "/"),
            "golden_digest": golden_digest,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _token(name: str, value: Any) -> str:
    prefix = OPTION_TOKENS.get(name, name)
    if isinstance(value, bool):
        return f"{prefix}{int(value)}"
    if isinstance(value, (list, tuple)):
        return prefix + "x".join(_scalar(v) for v in value)
    return f"{prefix}{_scalar(value)}"


def _scalar(value: Any) -> str:
    text = str(value)
    # Tên model đi vào `run_name` dưới dạng phần cuối đường dẫn HF, vì
    # `BAAI/bge-reranker-v2-m3` có dấu `/` và sẽ thành thư mục con.
    text = text.rsplit("/", 1)[-1]
    return "".join(ch if ch.isalnum() else "" for ch in text)


def _run_name(
    prefix: str,
    label: str,
    index_config: Path,
    mode: RetrievalMode,
    options: dict[str, Any],
    varying: frozenset[str],
) -> str:
    """Tên chỉ mang những chiều **thật sự biến thiên** trong khối.

    `rerank_dtype: [float16]` là một giá trị ghim, giống nhau ở mọi ô của khối,
    nên đưa nó vào tên chỉ làm tên dài thêm mà không phân biệt được ô nào với ô
    nào: `rr-bgem3-reranked-ondense-rc20-devcuda-dtfloat16` so với
    `rr-bgem3-reranked-ondense-rc20`.

    Không mất thông tin: **mọi** tham số, kể cả ghim, vẫn vào `config.branch_options`
    của báo cáo và vào param MLflow. ⚠️ Nhưng có một hệ quả phải biết — đổi một
    giá trị ghim (`float16` → `float32`) **không đổi tên**, nên file báo cáo cũ bị
    ghi đè. `fingerprint` đổi nên resume vẫn chạy lại đúng ô đó; chỉ là con số cũ
    không còn nằm cạnh con số mới để so. Muốn giữ cả hai thì cho dtype thành một
    chiều hai giá trị, hoặc đổi `label`.
    """
    parts = [prefix, label, index_config.stem, mode.value]
    parts += [_token(name, options[name]) for name in sorted(options) if name in varying]
    return "-".join(part for part in parts if part)


def expand(config: ExperimentConfig) -> tuple[ExperimentCell, ...]:
    """YAML → danh sách ô, **không chạm đĩa và không chạm mạng**.

    Thuần có chủ đích: `W2-08` cần biết grid có bao nhiêu ô và tên gì *trước khi*
    Qdrant chạy hay model được tải, và test "expand đúng số tổ hợp" của DoD phải
    chạy trong vài milli-giây.

    Đổi lại, `fingerprint` của ô cần thêm hai thứ chỉ đọc được từ đĩa
    (`index_fingerprint`, `golden_digest`) — nên chúng là tham số của
    `ExperimentCell.fingerprint`, do preflight cấp. Xem `runner.py`.
    """
    cells: list[ExperimentCell] = []
    seen: dict[str, ExperimentCell] = {}
    for block in config.matrix:
        varying = frozenset(name for name, values in block.options.items() if len(values) > 1)
        for index_config, mode, options in block.cells():
            run_name = _run_name(
                config.run_prefix, block.label, index_config, mode, options, varying
            )
            cell = ExperimentCell(
                run_name=run_name,
                index_config=index_config,
                retrieval_mode=mode,
                branch_options=options,
                top_k=config.top_k,
                min_overlap_ratio=config.min_overlap_ratio,
                golden=config.golden,
            )
            clash = seen.get(run_name)
            if clash is not None:
                # Hai ô cùng tên = hai ô ghi lên **cùng** ba file báo cáo, và ô
                # sau thắng. Bảng `W2-08` vẫn có đủ số dòng, chỉ là hai dòng mang
                # cùng một con số. Không có gì báo lỗi, nên phải nổ ở đây.
                raise ValueError(
                    f"Hai ô sinh ra cùng `run_name` {run_name!r}:\n"
                    f"  (1) {clash.retrieval_mode.value} {clash.branch_options}\n"
                    f"  (2) {mode.value} {options}\n"
                    "Chúng sẽ ghi lên cùng file báo cáo và ô sau xoá ô trước. "
                    "Đặt `label` khác nhau cho hai khối để tách tên."
                )
            seen[run_name] = cell
            cells.append(cell)
    return tuple(cells)


def load_experiment_config(path: str | Path, **overrides: Any) -> ExperimentConfig:
    """Đọc config YAML của thí nghiệm. `overrides` để CLI ghi đè mà không sửa file."""
    import yaml

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"Không thấy config thí nghiệm tại {source}. "
            "Mẫu có sẵn ở `configs/eval/exp-001-retrieval.yaml`."
        )
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{source} phải là một mapping YAML, đọc được {type(raw).__name__}")
    raw.update({key: value for key, value in overrides.items() if value is not None})
    return ExperimentConfig.model_validate(raw)


def golden_digest(path: Path) -> str:
    """Băm nội dung golden set. Đổi golden set = mọi ô phải chạy lại.

    Băm **nội dung** chứ không phải mtime hay đường dẫn: `goldenset-freeze` ghi
    lại cùng một đường dẫn, và `TD-13` (review lại bằng người) sẽ đổi nội dung
    file mà không đổi tên. Nếu resume nhìn tên thì grid sau `TD-13` sẽ bỏ qua
    toàn bộ ô cũ và bảng trộn hai golden set.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cell_table(cells: Sequence[ExperimentCell]) -> str:
    """Bảng Markdown của grid — in ra ở `--dry-run` để soi trước khi chạy."""
    lines = ["| # | run_name | index | nhánh | tham số |", "|---|---|---|---|---|"]
    for number, cell in enumerate(cells, start=1):
        options = (
            ", ".join(f"{k}={cell.branch_options[k]}" for k in sorted(cell.branch_options)) or "—"
        )
        lines.append(
            f"| {number} | `{cell.run_name}` | `{cell.index_config.stem}` "
            f"| {cell.retrieval_mode.value} | {options} |"
        )
    return "\n".join(lines)
