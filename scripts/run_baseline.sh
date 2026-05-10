#!/usr/bin/env bash
# 토킷의 example policy를 N회 반복 실행 → scoring.yaml 수집 → CSV 집계.
#
# 사용:
#   bash scripts/run_baseline.sh CheatCode 5
#   bash scripts/run_baseline.sh RunACT 5
#   bash scripts/run_baseline.sh WaveArm 3
#
# 출력: logs/baseline/<POLICY>/<run_id>/scoring.yaml + summary.csv

set -euo pipefail

POLICY="${1:?usage: run_baseline.sh <CheatCode|RunACT|WaveArm|GentleGiant|SpeedDemon|WallToucher|WallPresser> [N]}"
N_REPEATS="${2:-5}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${ROOT}/logs/baseline/${POLICY}"
mkdir -p "$RESULTS_DIR"

# CheatCode 만 ground_truth=true
GT="false"
[[ "$POLICY" == "CheatCode" ]] && GT="true"

echo "=== baseline: ${POLICY} × ${N_REPEATS} runs ==="
for i in $(seq 1 "$N_REPEATS"); do
  RUN_ID="${POLICY}_run${i}_$(date +%s)"
  RUN_DIR="${RESULTS_DIR}/${RUN_ID}"
  mkdir -p "$RUN_DIR"

  echo "--- ${RUN_ID} ---"

  AIC_POLICY_MODULE="aic_example_policies.ros.${POLICY}" \
  AIC_GROUND_TRUTH="${GT}" \
  AIC_RESULTS_DIR="${RUN_DIR}" \
    docker compose \
      -f /workspace/aic/docker/docker-compose.yaml \
      -f "${ROOT}/docker/baseline-override.yaml" \
      up --abort-on-container-exit --exit-code-from eval eval model \
      > "${RUN_DIR}/compose.log" 2>&1 \
      || echo "[WARN] ${RUN_ID} non-zero exit (logs: ${RUN_DIR}/compose.log)"

  docker compose \
    -f /workspace/aic/docker/docker-compose.yaml \
    -f "${ROOT}/docker/baseline-override.yaml" \
    down --remove-orphans >/dev/null 2>&1 || true
done

# 집계
python3 "${ROOT}/scripts/aggregate_scoring.py" \
  --dir "$RESULTS_DIR" \
  --out "${RESULTS_DIR}/summary.csv" \
  --policy "${POLICY}"

echo "=== done. summary: ${RESULTS_DIR}/summary.csv ==="
