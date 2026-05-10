#!/usr/bin/env bash
# vast.ai 인스턴스 부팅 직후 실행될 스크립트.
# vast.ai console > Instance > Edit Image Config > On-start script 에 등록하거나,
# ssh 직후 수동으로 한 번 실행.
#
# 가정: image = ghcr.io/nggw519/aic-dev:latest (curl, git, pixi, huggingface-cli, aws 포함)
# 환경변수 (vast.ai env 또는 .env 파일):
#   HF_TOKEN                 — HuggingFace Hub write 토큰 (필수)
#   HF_CKPT_REPO             — ckpt repo (기본 nggw519/aic-ckpts)
#   HF_DATASET_REPO          — dataset repo (기본 nggw519/aic-datasets)
#   GIT_USER_NAME, GIT_USER_EMAIL  — git 커밋 author (선택)

set -euo pipefail

WORKDIR=/workspace
HF_CKPT_REPO="${HF_CKPT_REPO:-nggw519/aic-ckpts}"
HF_DATASET_REPO="${HF_DATASET_REPO:-nggw519/aic-datasets}"
mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "[onstart] $(date) starting bootstrap"

# 0. base CUDA 이미지에 빠진 필수 도구 설치
#    (curl/git은 이미 있다고 가정 — apt만 확장)
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y -q --no-install-recommends \
    python3-pip python3-venv tmux htop \
    ca-certificates build-essential lsb-release \
    libgl1 libglib2.0-0 \
  || echo "[onstart] WARN: some apt packages failed (계속 진행)"

# huggingface_hub[cli]는 pip로만 받을 수 있음
pip install --no-cache-dir --quiet 'huggingface_hub[cli]>=0.23' \
  || echo "[onstart] WARN: huggingface_hub install failed"

# 1. 토킷 + 작업 저장소 clone
if [[ ! -d aic ]]; then
  git clone https://github.com/intrinsic-dev/aic.git aic
fi
if [[ ! -d aic_work ]]; then
  git clone https://github.com/NGGW519/challenge.git aic_work
fi

# 2. git config (선택)
if [[ -n "${GIT_USER_NAME:-}" ]]; then
  git -C aic_work config user.name  "$GIT_USER_NAME"
  git -C aic_work config user.email "${GIT_USER_EMAIL:-noreply@example.com}"
fi

# 3. HuggingFace login (R2 대신 HF Hub 단독 백업)
if [[ -n "${HF_TOKEN:-}" ]]; then
  huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
  echo "[onstart] huggingface login ok"
else
  echo "[onstart] WARN: HF_TOKEN not set — ckpt push/pull will be skipped"
fi

# 4. Pixi 환경 준비 (aic 토킷 기반)
cd "$WORKDIR/aic"
if ! pixi info >/dev/null 2>&1; then
  pixi install --frozen
fi

# 5. aic_work Python deps (옵션 — Pixi 외부, dev 작업용)
cd "$WORKDIR/aic_work"
pip install --no-cache-dir -e .[train,dev] 2>/dev/null || \
    echo "[onstart] WARN: pip install partial (이건 ROS 환경 안 들어가도 OK)"

# 6. 최신 ckpt 풀 (있으면)
if [[ -n "${HF_TOKEN:-}" ]]; then
  mkdir -p "$WORKDIR/aic_work/checkpoints"
  huggingface-cli download --repo-type=model "$HF_CKPT_REPO" \
      --local-dir "$WORKDIR/aic_work/checkpoints" 2>/dev/null \
    || echo "[onstart] no ckpts in $HF_CKPT_REPO yet (first run)"
fi

# 7. tmux 세션 미리 띄우기 (사용자가 ssh 후 attach)
if ! tmux has-session -t main 2>/dev/null; then
  tmux new-session -d -s main -n shell -c "$WORKDIR/aic_work" "bash -l"
  tmux new-window  -t main   -n train -c "$WORKDIR/aic_work" "bash -l"
  tmux new-window  -t main   -n eval  -c "$WORKDIR/aic_work" "bash -l"
  tmux new-window  -t main   -n gpu                          "watch -n 2 nvidia-smi"
  tmux new-window  -t main   -n disk                         "watch -n 5 df -h /workspace"
fi

echo "[onstart] $(date) bootstrap done. Attach with: tmux attach -t main"
