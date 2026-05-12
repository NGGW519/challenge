#!/usr/bin/env bash
# 단일 인스턴스 안에서 eval + model 두 컨테이너 모두 aic_eval image / host network 로 띄움.
# 토킷 docker-compose 가 internal network + 컨테이너 호스트네임(eval:7447) 으로
# 동작하는 구조라, host pixi env 로 model 을 띄우면 zenoh router 등록이 어긋난다
# (2026-05-11 lifecycle ACTIVATE 실패 진단). 이 스크립트는 두 컨테이너를 같은
# aic_eval image 안에서 실행해 토킷 의도와 일치시킨다.
#
# 사용:
#   bash scripts/run_baseline_local.sh CheatCode 5
#   bash scripts/run_baseline_local.sh RunACT 3
#
# 출력: logs/baseline/<POLICY>/<run_id>/{scoring.yaml, eval.log, model.log}

set -euo pipefail

POLICY="${1:?usage: run_baseline_local.sh <CheatCode|RunACT|WaveArm|GentleGiant|SpeedDemon|WallToucher|WallPresser> [N]}"
N_REPEATS="${2:-5}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_BASE="${ROOT}/logs/baseline/${POLICY}"
mkdir -p "$RESULTS_BASE"

# aic_eval image (eval + model 둘 다 동일)
IMG="${AIC_EVAL_IMAGE:-ghcr.io/intrinsic-dev/aic/aic_eval:latest}"

# CheatCode 만 ground_truth=true
GT="false"
[[ "$POLICY" == "CheatCode" ]] && GT="true"

# zenoh router endpoint — host network 이므로 localhost
ROUTER_ADDR="${AIC_ROUTER_ADDR:-localhost:7447}"

# ACL / passwd — 토킷 entrypoint가 AIC_ENABLE_ACL=false 시 PASSWD 무시.
# 명시적으로 false 두고 PASSWD도 빈 값으로 두 컨테이너에 일관 주입 (대칭성 확보).
# 평가 환경(ACL on)을 재현하려면 AIC_ENABLE_ACL=true + PASSWD 둘 다 설정.
AIC_ENABLE_ACL="${AIC_ENABLE_ACL:-false}"
AIC_EVAL_PASSWD="${AIC_EVAL_PASSWD:-}"
AIC_MODEL_PASSWD="${AIC_MODEL_PASSWD:-}"

cleanup() {
  docker rm -f aic_eval aic_model >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "=== baseline (local single-host): ${POLICY} × ${N_REPEATS} runs ==="
echo "    image=${IMG}  router=${ROUTER_ADDR}  gt=${GT}"

for i in $(seq 1 "$N_REPEATS"); do
  RUN_ID="${POLICY}_run${i}_$(date +%s)"
  RUN_DIR="${RESULTS_BASE}/${RUN_ID}"
  mkdir -p "$RUN_DIR"
  echo "--- ${RUN_ID} ---"

  # 이전 잔여 컨테이너 정리
  cleanup

  # 1) eval = zenoh router + Gazebo + aic_engine
  #    토킷 entrypoint 가 first arg 부터 launch arg 로 받음
  docker run -d --rm --name aic_eval \
    --gpus all --network host \
    -e RMW_IMPLEMENTATION=rmw_zenoh_cpp \
    -e ZENOH_ROUTER_CHECK_ATTEMPTS=-1 \
    -e AIC_ENABLE_ACL="${AIC_ENABLE_ACL}" \
    -e AIC_EVAL_PASSWD="${AIC_EVAL_PASSWD}" \
    -e AIC_MODEL_PASSWD="${AIC_MODEL_PASSWD}" \
    -v "${RUN_DIR}:/root/aic_results" \
    "${IMG}" \
    gazebo_gui:=false \
    launch_rviz:=false \
    ground_truth:=${GT} \
    start_aic_engine:=true \
    shutdown_on_aic_engine_exit:=true \
    model_discovery_timeout_seconds:=600 \
    > "${RUN_DIR}/eval.docker.log" 2>&1 &

  # eval 컨테이너 안의 zenoh router 가 7447 listen 시작할 때까지 대기
  # (entrypoint 가 router 띄운 직후 launch 진행 — 1~2초면 충분하나 안전하게 12초)
  for s in $(seq 1 12); do
    if docker exec aic_eval bash -lc "ss -ltn 2>/dev/null | grep -q ':7447'" 2>/dev/null; then
      break
    fi
    sleep 1
  done

  # 2) model = 같은 image 의 ros2 run aic_model
  #    entrypoint override + 직접 setup.bash + zenoh config override
  docker run --rm --name aic_model \
    --gpus all --network host \
    -e RMW_IMPLEMENTATION=rmw_zenoh_cpp \
    -e ZENOH_ROUTER_CHECK_ATTEMPTS=-1 \
    -e AIC_ENABLE_ACL="${AIC_ENABLE_ACL}" \
    -e AIC_EVAL_PASSWD="${AIC_EVAL_PASSWD}" \
    -e AIC_MODEL_PASSWD="${AIC_MODEL_PASSWD}" \
    --entrypoint "" \
    "${IMG}" \
    bash -lc "
      set -e
      source /ws_aic/install/setup.bash
      # zenoh router 연결 + shared_memory off (다른 컨테이너이므로 SHM 불가).
      # I3 분석(2026-05-12): 토킷 aic_model entrypoint는 ZENOH_SESSION_CONFIG_URI를
      # 설정하지 않는다. ZENOH_CONFIG_OVERRIDE 만으로 router 연결 충분.
      export ZENOH_CONFIG_OVERRIDE='connect/endpoints=[\"tcp/${ROUTER_ADDR}\"];transport/shared_memory/enabled=false'
      exec ros2 run aic_model aic_model --ros-args \
        -p use_sim_time:=true \
        -p policy:=aic_example_policies.ros.${POLICY}
    " \
    > "${RUN_DIR}/model.log" 2>&1 || echo "[WARN] model exit non-zero (see ${RUN_DIR}/model.log)"

  # eval 종료 대기 (shutdown_on_aic_engine_exit=true 라 engine 끝나면 컨테이너 종료)
  docker wait aic_eval >/dev/null 2>&1 || true
  docker logs aic_eval > "${RUN_DIR}/eval.log" 2>&1 || true

  cleanup
done

# scoring.yaml 집계
python3 "${ROOT}/scripts/aggregate_scoring.py" \
  --dir "$RESULTS_BASE" \
  --out "${RESULTS_BASE}/summary.csv" \
  --policy "${POLICY}"

echo "=== done. summary: ${RESULTS_BASE}/summary.csv ==="
