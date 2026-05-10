#!/usr/bin/env bash
# 학습 도중 매 ckpt마다 호출되는 "빠른 평가" — 1~3 rollout × 3 trial.
# 목적: 점수 회귀 즉시 발견. 100 rollout 정식 평가는 eval_100.sh 사용.
#
# 사용:
#   AIC_SUBMIT_TAG=v1-tmp bash scripts/eval_quick.sh quicktest 2
#
# 출력:
#   logs/eval_quick/<tag>/<run>/scoring.yaml
#   logs/eval_quick/<tag>/summary.csv
#   stdout: 평균/min/max + 직전 best 대비 delta

set -euo pipefail

TAG="${1:?usage: eval_quick.sh <tag> [N=2]}"
N="${2:-2}"
SUBMIT_TAG="${AIC_SUBMIT_TAG:-v1}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${ROOT}/logs/eval_quick/${TAG}_$(date +%H%M%S)"
mkdir -p "$RESULTS_DIR"

echo "[eval_quick] tag=${TAG} N=${N} image=aic-submit:${SUBMIT_TAG}"

for i in $(seq 1 "$N"); do
  RUN_DIR="${RESULTS_DIR}/run_$(printf %02d "$i")"
  mkdir -p "$RUN_DIR"
  AIC_RESULTS_DIR="${RUN_DIR}" \
  AIC_SUBMIT_TAG="${SUBMIT_TAG}" \
    docker compose \
      -f /workspace/aic/docker/docker-compose.yaml \
      -f "${ROOT}/docker/eval-override.yaml" \
      up --abort-on-container-exit --exit-code-from eval eval model \
      > "${RUN_DIR}/compose.log" 2>&1 || echo "[WARN] run $i exit non-zero"

  docker compose \
    -f /workspace/aic/docker/docker-compose.yaml \
    -f "${ROOT}/docker/eval-override.yaml" \
    down --remove-orphans >/dev/null 2>&1 || true
done

# 집계
python3 "${ROOT}/scripts/aggregate_scoring.py" \
  --dir "$RESULTS_DIR" --out "${RESULTS_DIR}/summary.csv" --policy "${TAG}"

# best 대비 delta (logs/eval_quick/best.csv가 있으면)
BEST="${ROOT}/logs/eval_quick/best.csv"
if [[ -f "$BEST" ]]; then
  echo "[eval_quick] comparing to $BEST"
  python3 - <<EOF
import csv, statistics
def avg(p):
    with open(p) as f:
        rows = [float(r['total']) for r in csv.DictReader(f) if r.get('total')]
    return statistics.mean(rows) if rows else 0.0
cur  = avg("${RESULTS_DIR}/summary.csv")
best = avg("${BEST}")
delta = cur - best
flag = "🟢" if delta >= -1 else ("🟡" if delta > -3 else "🔴")
print(f"{flag} current={cur:.2f}  best={best:.2f}  delta={delta:+.2f}")
EOF
else
  echo "[eval_quick] no baseline best.csv yet"
fi

echo "[eval_quick] done → ${RESULTS_DIR}/summary.csv"
