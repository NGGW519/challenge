#!/usr/bin/env bash
# vast.ai 인스턴스가 갑자기 죽어도 자동 재개되는 학습 wrapper.
# 매 epoch end마다 ckpt를 R2에 push하고, crash 시 sleep 후 재시도.
#
# 사용:
#   bash scripts/train_with_resume.sh act      # ACT 학습
#   bash scripts/train_with_resume.sh yolo     # 포트 검출

set -euo pipefail

WHAT="${1:?usage: train_with_resume.sh <act|yolo>}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="${ROOT}/logs/train"
mkdir -p "$LOGDIR"

case "$WHAT" in
  act)
    CMD=(python3 "${ROOT}/training/train_act.py" --resume \
          --output_dir "${ROOT}/models/act_v1" \
          --remote "r2:${R2_BUCKET:-aic-ckpts}/act_v1/")
    ;;
  yolo)
    CMD=(yolo train \
          model=yolov8m.pt \
          data="${ROOT}/training/yolov8_port.yaml" \
          imgsz=1024 epochs=100 batch=32 amp=true \
          project="${ROOT}/models/port_detector_v1" \
          name=run0 \
          resume=true)
    ;;
  *)
    echo "unknown: $WHAT" >&2; exit 2 ;;
esac

ATTEMPT=0
while true; do
  ATTEMPT=$((ATTEMPT+1))
  TS="$(date +%Y%m%d_%H%M%S)"
  LOG="${LOGDIR}/${WHAT}_attempt${ATTEMPT}_${TS}.log"
  echo "[train] attempt ${ATTEMPT} at ${TS} → ${LOG}"
  if "${CMD[@]}" 2>&1 | tee "$LOG"; then
    echo "[train] success"
    break
  fi
  echo "[train] attempt ${ATTEMPT} failed; sleeping 30s before retry"
  sleep 30
done

# 최종 ckpt 백업
if [[ -n "${R2_BUCKET:-}" ]]; then
  case "$WHAT" in
    act)  rclone copy "${ROOT}/models/act_v1/" "r2:${R2_BUCKET}/act_v1/" --progress ;;
    yolo) rclone copy "${ROOT}/models/port_detector_v1/" "r2:${R2_BUCKET}/port_detector_v1/" --progress ;;
  esac
fi
