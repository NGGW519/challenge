#!/usr/bin/env bash
# vast.ai 인스턴스 부팅 직후 실행될 스크립트.
# vast.ai console > Instance > Edit Image Config > On-start script 에 등록하거나,
# ssh 직후 수동으로 한 번 실행.
#
# 가정: image = ghcr.io/nggw519/aic-dev:latest (curl, git, pixi, hf CLI, aws 포함)
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
    ca-certificates build-essential lsb-release gnupg \
    libgl1 libglib2.0-0 sudo \
  || echo "[onstart] WARN: some apt packages failed (계속 진행)"

# 0.5. Docker engine + nvidia-container-toolkit
#      이전 시도 vast/00_setup.sh 패턴.
#      Phase 1 평가 = docker compose 의존이므로 docker 동작이 필수.
#      vast.ai 인스턴스가 --privileged 모드여야 iptables 권한 통과.
if ! command -v docker >/dev/null 2>&1; then
  echo "[onstart] installing docker engine"
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
  chmod a+r /etc/apt/keyrings/docker.asc
  codename="$(. /etc/os-release && echo "${VERSION_CODENAME:-jammy}")"
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $codename stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update -q
  apt-get install -y -q --no-install-recommends \
    docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin \
    || echo "[onstart] WARN: docker apt install failed"
fi

if ! command -v nvidia-ctk >/dev/null 2>&1; then
  echo "[onstart] installing nvidia-container-toolkit"
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -q
  apt-get install -y -q nvidia-container-toolkit \
    || echo "[onstart] WARN: nvidia-container-toolkit install failed"
  nvidia-ctk runtime configure --runtime=docker 2>&1 | tail -3 || true
fi

# dockerd 시작 (vast.ai 컨테이너엔 systemd 없으니 직접 띄움)
if ! docker info >/dev/null 2>&1; then
  echo "[onstart] starting dockerd in background"
  mkdir -p /var/log
  nohup dockerd > /var/log/dockerd.log 2>&1 &
  for i in $(seq 1 20); do
    sleep 2
    if docker info >/dev/null 2>&1; then
      echo "[onstart] dockerd ready"; break
    fi
  done
  if ! docker info >/dev/null 2>&1; then
    echo "[onstart] WARN: dockerd not responding. Check /var/log/dockerd.log."
    echo "[onstart]       이 인스턴스가 --privileged 모드 아닐 가능성."
  fi
fi
docker info 2>&1 | grep -iE "runtime|nvidia" | head -3 || true

# huggingface_hub 1.x — CLI는 `hf` 명령으로 함께 설치됨 (extras 불필요)
pip install --no-cache-dir --quiet 'huggingface_hub>=1.0' \
  || echo "[onstart] WARN: huggingface_hub install failed"

# Pixi v0.63.2 — 토킷이 .github/workflows/pixi.yml에서 명시한 정확한 버전.
# 다른 버전(특히 0.68+)은 `preview = ["pixi-build"]` 해시 mismatch로 install 실패.
PIXI_VERSION="${PIXI_VERSION:-v0.63.2}"
INSTALLED_PIXI_VERSION=""
if command -v pixi >/dev/null 2>&1; then
  INSTALLED_PIXI_VERSION="v$(pixi --version 2>/dev/null | awk '{print $2}')"
fi
if [[ "$INSTALLED_PIXI_VERSION" != "$PIXI_VERSION" ]]; then
  echo "[onstart] installing pixi ${PIXI_VERSION} (was: ${INSTALLED_PIXI_VERSION:-none})"
  curl -fsSL https://pixi.sh/install.sh | bash -s -- --version "$PIXI_VERSION" --no-modify-path
fi
export PATH="$HOME/.pixi/bin:$PATH"
if ! grep -q '\.pixi/bin' ~/.bashrc 2>/dev/null; then
  echo 'export PATH="$HOME/.pixi/bin:$PATH"' >> ~/.bashrc
fi

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

# 3. HuggingFace login (R2 대신 HF Hub 단독 백업).
#    huggingface_hub 1.x부터 CLI 이름이 `huggingface-cli` → `hf` 로 변경.
if [[ -n "${HF_TOKEN:-}" ]]; then
  hf auth login --token "$HF_TOKEN" --add-to-git-credential
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

# 6. 최신 ckpt 풀 (있으면) — hf CLI 사용 (huggingface_hub 1.x)
if [[ -n "${HF_TOKEN:-}" ]]; then
  mkdir -p "$WORKDIR/aic_work/checkpoints"
  hf download "$HF_CKPT_REPO" \
      --repo-type model \
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
