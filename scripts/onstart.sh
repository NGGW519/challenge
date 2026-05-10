#!/usr/bin/env bash
# vast.ai 인스턴스 부팅 직후 실행될 스크립트.
# vast.ai console > Instance > Edit Image Config > On-start script 에 등록하거나,
# ssh 직후 수동으로 한 번 실행.
#
# 가정: image = ghcr.io/nggw519/aic-dev:latest (curl, git, pixi, rclone, aws 포함)
# 환경변수 (vast.ai env 또는 .env 파일):
#   R2_ACCESS_KEY, R2_SECRET_KEY, R2_ENDPOINT, R2_BUCKET     — Cloudflare R2
#   HF_TOKEN                                                  — HuggingFace
#   GIT_USER_NAME, GIT_USER_EMAIL                              — git 커밋 author (선택)

set -euo pipefail

WORKDIR=/workspace
mkdir -p "$WORKDIR"
cd "$WORKDIR"

echo "[onstart] $(date) starting bootstrap"

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

# 3. rclone 설정 (R2)
if [[ -n "${R2_ACCESS_KEY:-}" ]]; then
  mkdir -p ~/.config/rclone
  cat > ~/.config/rclone/rclone.conf <<EOF
[r2]
type = s3
provider = Cloudflare
access_key_id = ${R2_ACCESS_KEY}
secret_access_key = ${R2_SECRET_KEY}
endpoint = ${R2_ENDPOINT}
acl = private
EOF
  echo "[onstart] rclone configured"
else
  echo "[onstart] WARN: R2_ACCESS_KEY not set — skipping rclone setup"
fi

# 4. HuggingFace login
if [[ -n "${HF_TOKEN:-}" ]]; then
  huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
  echo "[onstart] huggingface login ok"
fi

# 5. Pixi 환경 준비 (aic 토킷 기반)
cd "$WORKDIR/aic"
if ! pixi info >/dev/null 2>&1; then
  pixi install --frozen
fi

# 6. aic_work Python deps (옵션 — Pixi 외부, dev 작업용)
cd "$WORKDIR/aic_work"
pip install --no-cache-dir -e .[train,dev] 2>/dev/null || echo "[onstart] WARN: pip install partial (이건 ROS 환경 안 들어가도 OK)"

# 7. 최신 ckpt 풀 (있다면)
if [[ -n "${R2_BUCKET:-}" ]]; then
  rclone sync "r2:${R2_BUCKET}/latest/" "$WORKDIR/aic_work/checkpoints/" --progress || true
fi

# 8. tmux 세션 미리 띄우기 (사용자가 ssh 후 attach)
if ! tmux has-session -t main 2>/dev/null; then
  tmux new-session -d -s main -n shell -c "$WORKDIR/aic_work" "bash -l"
  tmux new-window  -t main   -n train -c "$WORKDIR/aic_work" "bash -l"
  tmux new-window  -t main   -n eval  -c "$WORKDIR/aic_work" "bash -l"
  tmux new-window  -t main   -n gpu                          "watch -n 2 nvidia-smi"
  tmux new-window  -t main   -n disk                         "watch -n 5 df -h /workspace"
fi

echo "[onstart] $(date) bootstrap done. Attach with: tmux attach -t main"
