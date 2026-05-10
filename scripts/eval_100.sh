#!/usr/bin/env bash
# 우리 제출 컨테이너(aic-submit)를 N회 평가 → CSV 집계.
# Phase 7 (ablation/tune) 전용. baseline은 run_baseline.sh 참조.
#
# 사용:
#   AIC_SUBMIT_TAG=v1-abc1234 bash scripts/eval_100.sh hybrid_v1 100 4
#
#   인자:
#     1: TAG     — 결과 디렉토리 이름 (실험 이름)
#     2: N       — rollout 개수 (기본 100)
#     3: PARALLEL— 동시 실행 개수 (기본 4) — vast.ai L4 ×4 가정

set -euo pipefail

TAG="${1:?usage: eval_100.sh <experiment_tag> [N=100] [PARALLEL=4]}"
N="${2:-100}"
PARALLEL="${3:-4}"
SUBMIT_TAG="${AIC_SUBMIT_TAG:-v1}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${ROOT}/logs/eval/${TAG}_$(date +%Y%m%d_%H%M)"
mkdir -p "$RESULTS_DIR"
echo "[eval_100] results: $RESULTS_DIR"
echo "[eval_100] image:   aic-submit:${SUBMIT_TAG}"
echo "[eval_100] N=${N} parallel=${PARALLEL}"

# 세마포어 (mkfifo)
SEM=$(mktemp -u /tmp/aic_sem.XXXXXX)
mkfifo "$SEM"
exec 3<>"$SEM"
rm "$SEM"
for ((i=0; i<PARALLEL; i++)); do echo >&3; done

run_one() {
  local i="$1"
  local run_id; run_id="${TAG}_run$(printf %04d "$i")"
  local run_dir="${RESULTS_DIR}/${run_id}"
  mkdir -p "$run_dir"

  AIC_RESULTS_DIR="${run_dir}" \
  AIC_SUBMIT_TAG="${SUBMIT_TAG}" \
    docker compose \
      -p "aic_${i}" \
      -f /workspace/aic/docker/docker-compose.yaml \
      -f "${ROOT}/docker/eval-override.yaml" \
      up --abort-on-container-exit --exit-code-from eval eval model \
      > "${run_dir}/compose.log" 2>&1 \
      || echo "[WARN] ${run_id} non-zero exit"

  docker compose \
    -p "aic_${i}" \
    -f /workspace/aic/docker/docker-compose.yaml \
    -f "${ROOT}/docker/eval-override.yaml" \
    down --remove-orphans >/dev/null 2>&1 || true
}

for i in $(seq 1 "$N"); do
  read -u 3
  (
    run_one "$i"
    echo >&3
  ) &
done
wait

# 집계
python3 "${ROOT}/scripts/aggregate_scoring.py" \
  --dir "$RESULTS_DIR" \
  --out "${RESULTS_DIR}/summary.csv" \
  --policy "${TAG}"

echo "=== done. summary: ${RESULTS_DIR}/summary.csv ==="
