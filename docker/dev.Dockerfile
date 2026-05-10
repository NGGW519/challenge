# Dev image — 학습 / 데이터 수집 / 평가 sandbox.
# vast.ai 인스턴스에서 이 이미지로 컨테이너를 띄운 뒤
# /workspace에 aic + aic_work를 mount/clone하여 작업.
#
# 빌드:
#   docker build -t ghcr.io/nggw519/aic-dev:latest -f docker/dev.Dockerfile .
#   docker push ghcr.io/nggw519/aic-dev:latest
#
# 실행 (vast.ai):
#   - On-Demand instance, image: ghcr.io/nggw519/aic-dev:latest
#   - On-start cmd: bash /workspace/aic_work/scripts/onstart.sh

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV TZ=Asia/Seoul

# 1. 시스템 패키지
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl wget git rsync openssh-client \
        build-essential cmake pkg-config \
        python3 python3-pip python3-venv \
        software-properties-common lsb-release gnupg \
        tmux htop nvtop unzip jq \
        ffmpeg libgl1 libglib2.0-0 libegl1 libxkbcommon0 \
        libxcb1 libdbus-1-3 \
    && rm -rf /var/lib/apt/lists/*

# 2. Pixi (ROS 2 Kilted via robostack)
RUN curl -fsSL https://pixi.sh/install.sh | bash -s -- --no-modify-path
ENV PATH="/root/.pixi/bin:${PATH}"

# 3. rclone (R2 / S3 백업)
RUN curl -fsSL https://rclone.org/install.sh | bash

# 4. AWS CLI v2 (ECR push 용 — submit 단계)
RUN curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscli.zip \
    && unzip -q /tmp/awscli.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/awscli.zip /tmp/aws

# 5. HuggingFace CLI
RUN pip install --no-cache-dir huggingface_hub[cli]

WORKDIR /workspace

# Pixi 환경은 인스턴스 안에서 prepare (lock 동기화 위해)
# pixi.toml은 aic 토킷 안의 것을 재사용하므로 여기서 안 굽는다.

CMD ["bash"]
