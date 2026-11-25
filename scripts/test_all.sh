#!/usr/bin/env bash
# 一键跑全部测试: 任一失败则整体失败 (CI 与本地共用同一入口)
# 用法: bash scripts/test_all.sh   (可用 WMFLOW_PY 指定解释器)
set -u
DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$DIR"
PY="${WMFLOW_PY:-python3}"

TESTS=(
  tests/test_smoke.py
  tests/test_netspec.py
  tests/test_components.py
  tests/test_custom_recipes.py
  tests/test_train_custom.py
  tests/test_train_val.py
  tests/test_webui_auth.py
)

fail=0
for t in "${TESTS[@]}"; do
  echo "===== $t ====="
  "$PY" "$t" || { echo "FAIL: $t"; fail=1; }
done

if [ "$fail" -ne 0 ]; then
  echo "部分测试失败"
  exit 1
fi
echo "ALL TESTS PASSED"