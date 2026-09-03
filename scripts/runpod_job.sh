#!/usr/bin/env bash
# `W0-08` — đưa job W3-04 lên GPU thuê và kéo kết quả về, không mang theo bí mật.
#
# Bốn lệnh, chạy theo thứ tự:
#
#   ./scripts/runpod_job.sh bundle    # dựng dist/runpod-job.tar.gz
#   ./scripts/runpod_job.sh verify    # quét bí mật — CHẠY TRƯỚC KHI ĐẨY
#   ./scripts/runpod_job.sh push      # đẩy lên pod qua runpodctl (hoặc scp)
#   ./scripts/runpod_job.sh pull      # kéo contexts.jsonl + run-report.json về
#
# Quy tắc cứng #2 được ép ở ba chỗ, không phải bằng trí nhớ:
#   1. Gói dựng bằng `git archive HEAD` → không thể chứa file bị .gitignore.
#   2. `verify` quét tên cấm + mẫu bí mật, kể cả bên trong file .gz.
#   3. `run_on_pod.sh` không đọc biến môi trường API key nào (có test ghim).
#
# KHÔNG có lệnh nào ở đây copy .env, cấu hình git credential, hay đăng nhập
# dịch vụ nào trên pod. Nếu bạn thấy mình cần thêm, dừng lại và đọc lại quy tắc.
set -euo pipefail

cd "$(dirname "$0")/.."

BUNDLE="${BUNDLE:-dist/runpod-job.tar.gz}"
REQUESTS="${REQUESTS:-data/contexts/requests.jsonl.gz}"
PY="${PY:-uv run python}"

usage() { sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

cmd_bundle() {
  [ -f "$REQUESTS" ] || {
    echo "Chưa có $REQUESTS. Chạy trước:" >&2
    echo "  make ctx-prepare" >&2
    exit 1
  }
  $PY - "$BUNDLE" "$REQUESTS" <<'PYEOF'
import sys
from pathlib import Path
from pipeline.indexing.job_bundle import build_bundle
out = build_bundle(Path(sys.argv[1]), requests_path=Path(sys.argv[2]))
print(f"{out}  ({out.stat().st_size / 1e6:.1f} MB)")
PYEOF
}

cmd_verify() {
  [ -f "$BUNDLE" ] || { echo "Chưa có $BUNDLE — chạy `bundle` trước." >&2; exit 1; }
  $PY - "$BUNDLE" <<'PYEOF'
import sys
from pathlib import Path
from pipeline.indexing.job_bundle import scan_bundle
scan = scan_bundle(Path(sys.argv[1]))
print(scan.summary())
sys.exit(0 if scan.clean else 2)
PYEOF
}

# Đẩy bằng runpodctl: nó dùng mã một lần, không cần khoá SSH cấu hình sẵn và
# không để lại credential nào trên pod. Nếu dùng scp thì phải là keypair RIÊNG
# cho RunPod (`ssh-keygen -t ed25519 -f ~/.ssh/runpod_ed25519`), không dùng lại
# khoá GitHub.
cmd_push() {
  cmd_verify
  command -v runpodctl >/dev/null || {
    echo "Thiếu runpodctl. Cài: https://github.com/runpod/runpodctl" >&2
    echo "Hoặc: scp -i ~/.ssh/runpod_ed25519 $BUNDLE root@<POD_IP>:/workspace/" >&2
    exit 1
  }
  echo "==> Chạy lệnh receive HIỆN RA BÊN DƯỚI ở trên pod:"
  runpodctl send "$BUNDLE"
}

cmd_pull() {
  mkdir -p data/contexts
  echo "Trên pod chạy:  runpodctl send job/contexts.jsonl"
  echo "Rồi ở đây chạy: runpodctl receive <MÃ>"
  echo "Đặt file về:    data/contexts/contexts.jsonl"
}

case "${1:-}" in
  bundle) cmd_bundle ;;
  verify) cmd_verify ;;
  push)   cmd_push ;;
  pull)   cmd_pull ;;
  *)      usage ;;
esac
