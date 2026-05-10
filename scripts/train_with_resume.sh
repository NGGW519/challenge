#!/usr/bin/env bash
# vast.ai 인스턴스가 갑자기 죽어도 자동 재개되는 학습 wrapper.
# 매 ckpt마다 HF Hub로 push하고, crash 시 sleep 후 재시도.
#
# 사용:
#   bash scripts/train_with_resume.sh act      # ACT 학습
#   bash scripts/train_with_resume.sh yolo     # 포트 검출
#
# 환경변수:
#   HF_CKPT_REPO   — model repo (기본 nggw519/aic-ckpts)
#   HF_TOKEN       — onstart.sh에서 이미 login되어 있으면 생략 가능

set -euo pipefail

WHAT="${1:?usage: train_with_resume.sh <act|yolo>}"
HF_REPO="${HF_CKPT_REPO:-nggw519/aic-ckpts}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="${ROOT}/logs/train"
mkdir -p "$LOGDIR"

case "$WHAT" in
  act)
    CMD=(python3 "${ROOT}/training/train_act.py" --resume \
          --output_dir "${ROOT}/models/act_v1" \
          --remote "hf://${HF_REPO}/act_v1")
    SUBPATH="act_v1"
    LOCAL="${ROOT}/models/act_v1"
    ;;
  yolo)
    CMD=(yolo train \
          model=yolov8m.pt \
          data="${ROOT}/training/yolov8_port.yaml" \
          imgsz=1024 epochs=100 batch=32 amp=true \
          project="${ROOT}/models/port_detector_v1" \
          name=run0 \
          resume=true)
    SUBPATH="port_detector_v1"
    LOCAL="${ROOT}/models/port_detector_v1"
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

# 최종 ckpt 백업 (학습 스크립트가 매 ckpt push했지만 종합본도 한 번 더)
if [[ -d "$LOCAL" ]] && command -v huggingface-cli >/dev/null 2>&1; then
  echo "[train] final upload to hf://${HF_REPO}/${SUBPATH}"
  huggingface-cli upload "$HF_REPO" "$LOCAL" "$SUBPATH" \
      --repo-type=model \
      --commit-message="final ${WHAT} ${TS}" || \
      echo "[train] WARN: final upload failed (학습 자체는 성공)"
fi
