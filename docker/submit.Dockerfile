# Submission image — AWS ECR로 push될 최종 컨테이너.
# 공식 평가 컨테이너(aic_eval)와 zenoh 통신.
#
# 빌드 (저장소 루트 = /home/nggw/challenge/ 또는 vast.ai /workspace 에서):
#   docker build \
#     -f aic_work/docker/submit.Dockerfile \
#     -t aic-submit:v1 \
#     .
#
# 검증:
#   docker compose \
#     -f aic/docker/docker-compose.yaml \
#     -f aic_work/docker/eval-override.yaml \
#     up
#
# 푸시:
#   docker tag aic-submit:v1 <ECR_URI>:v1-$(git rev-parse --short HEAD)
#   docker push <ECR_URI>:v1-...

# 베이스: 토킷의 aic_model 컨테이너 베이스를 그대로 쓰는 것이 안전.
# 만약 그 베이스가 ghcr에 공개되어 있지 않다면 multi-stage로 직접 빌드.
ARG BASE_IMAGE=ghcr.io/intrinsic-dev/aic/aic_model:latest
FROM ${BASE_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# 추가 시스템 라이브러리 (ultralytics / cv2 / torch 런타임 의존)
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 작업 폴더 (베이스 이미지가 정해놓은 위치를 따른다고 가정)
WORKDIR /workspace

# --- 우리 패키지 복사 ---
# (저장소 루트에서 빌드한다고 가정. context = /home/nggw/challenge)
COPY aic_work/aic_model_pkg /workspace/src/aic_model_pkg
COPY aic_work/pyproject.toml /workspace/pyproject.toml

# --- 모델 가중치 ---
# aic_model_pkg/aic_model_pkg/weights/ 안에 미리 .pt 파일들을 둔 상태에서 빌드.
# (위 COPY로 함께 들어옴)

# --- Python deps (베이스에 이미 ROS+pixi가 있으니 추가 ML deps만) ---
RUN pip install --no-cache-dir \
        torch==2.3.* torchvision==0.18.* \
        ultralytics==8.2.* \
        opencv-python-headless==4.9.* \
        numpy==1.26.* \
        transforms3d==0.4.* \
        einops==0.7.*

# --- colcon 빌드 (aic_model_pkg) ---
# 베이스 이미지가 ROS 2 Kilted을 /opt/ros/kilted 에 올려두었다고 가정.
# 만약 Pixi env 기반이면 다음 ENV/CMD를 그에 맞춰 조정.
RUN bash -lc "source /opt/ros/kilted/setup.bash 2>/dev/null || true; \
              cd /workspace && \
              colcon build --packages-select aic_model_pkg --merge-install --symlink-install"

# --- 환경변수 (베이스가 이미 zenoh 설정을 했더라도 명시) ---
ENV RMW_IMPLEMENTATION=rmw_zenoh_cpp
ENV ZENOH_ROUTER_CHECK_ATTEMPTS=-1
ENV AIC_MODEL_PASSWD=CHANGE_IN_PROD

# --- 진입점 ---
# 베이스 이미지가 이미 entrypoint를 가지고 있을 수 있음.
# 우리 정책을 -p policy:= 로 주입.
CMD ["bash", "-lc", \
     "source /opt/ros/kilted/setup.bash 2>/dev/null || true; \
      source /workspace/install/setup.bash; \
      ros2 run aic_model aic_model \
           -p policy:=aic_model_pkg.HybridPolicy \
           -p use_sim_time:=true"]
