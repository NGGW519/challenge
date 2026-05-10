# 17. Phase 8 — 제출 (Docker → ECR → Portal)

> **목표**: Phase 7에서 확정된 정책을 공식 평가 컨테이너 인터페이스에 맞춰 패키징, ECR에 push, 제출 포털에 OCI URI 등록.
>
> **기간**: 0.5~1일
>
> **결과물**: 제출 포털 "Submitted" 상태 + 평가 결과 확인.

---

## 1. 제출 인프라 사전 점검

조직에서 받은 온보딩 이메일에 다음 정보가 있어야 한다:
- AWS Access Key ID + Secret (팀별 IAM)
- ECR URI (예: `973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team_name>`)
- 제출 포털 URL + 로그인 자격증명

체크리스트:
- [ ] AWS CLI 설치 (`aws --version`)
- [ ] AWS 자격증명 등록: `aws configure --profile aic`
- [ ] ECR 인증 가능 확인
- [ ] 제출 포털 접속 가능

---

## 2. 컨테이너 빌드

### 2.1 Dockerfile 작성

토킷이 제공하는 `aic/docker/aic_model/Dockerfile`을 베이스로 시작. 우리 패키지를 추가 install:

`aic_work/docker/submit.Dockerfile`:

```dockerfile
# Base: 토킷의 aic_model Dockerfile을 변형 — 같은 base image 사용 권장
FROM ghcr.io/intrinsic-dev/aic/aic_model_base:latest AS base
# (만약 base image가 별도 제공 안 되면 Ubuntu 24.04 + Pixi + ROS 2 Kilted 설치)

WORKDIR /workspace

# 우리 패키지
COPY aic_work/aic_model_pkg /workspace/src/aic_model_pkg
COPY aic/aic_model           /workspace/src/aic_model
COPY aic/aic_interfaces      /workspace/src/aic_interfaces
COPY pixi.toml pixi.lock     /workspace/

# 의존성
RUN pixi install --frozen

# colcon build
RUN pixi run -e prod \
    bash -lc "source /opt/ros/kilted/setup.bash && \
              colcon build --packages-select aic_interfaces aic_model aic_model_pkg --merge-install"

# weights는 패키지 안에 이미 포함되어 있어야 함 (aic_model_pkg/weights/)

# 환경변수
ENV RMW_IMPLEMENTATION=rmw_zenoh_cpp
ENV ZENOH_ROUTER_CHECK_ATTEMPTS=-1
ENV AIC_MODEL_PASSWD=CHANGE_IN_PROD

# 진입 명령
CMD ["pixi", "run", "-e", "prod", "bash", "-lc", \
     "source /workspace/install/setup.bash && \
      ros2 run aic_model aic_model -p policy:=aic_model_pkg.HybridPolicy -p use_sim_time:=true"]
```

> 실제 base image / Dockerfile은 토킷 제공 `docker/aic_model/Dockerfile` 내용을 그대로 따른다. 위는 개념 — 우리 코드 추가만 하면 됨.

### 2.2 빌드

```bash
cd /home/nggw/challenge   # 또는 vast.ai /workspace
docker build \
    -f aic_work/docker/submit.Dockerfile \
    -t aic-submit:v1 \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    .
docker images aic-submit
```

빌드 시간: 처음 30~60분 (ROS, Pixi cache), 이후 layer cache로 5~10분.

### 2.3 weights 사이즈 점검

```bash
docker run --rm aic-submit:v1 ls -lh /workspace/src/aic_model_pkg/aic_model_pkg/weights/
```

목표 image total: ≤ 10GB. ACT v2 (~150MB), YOLOv8m (~25MB), 합쳐도 ROS+Pixi base가 5~7GB라 충분.

---

## 3. 로컬 검증 (제출 전 필수)

### 3.1 docker-compose로 실행

`aic_work/docker/submit-compose.yaml`:

```yaml
name: aic-submit
services:
  eval:
    image: ghcr.io/intrinsic-dev/aic/aic_eval:latest
    command: gazebo_gui:=false launch_rviz:=false ground_truth:=false start_aic_engine:=true shutdown_on_aic_engine_exit:=true model_discovery_timeout_seconds:=600
    gpus: all
    networks: [default]
    environment:
      AIC_EVAL_PASSWD: CHANGE_IN_PROD
      AIC_MODEL_PASSWD: CHANGE_IN_PROD
    volumes:
      - ./logs:/root/aic_results
  model:
    image: aic-submit:v1
    gpus: all
    networks: [default]
    environment:
      RMW_IMPLEMENTATION: rmw_zenoh_cpp
      ZENOH_ROUTER_CHECK_ATTEMPTS: -1
      AIC_ROUTER_ADDR: eval:7447
      AIC_MODEL_PASSWD: CHANGE_IN_PROD

networks:
  default:
    internal: true
```

```bash
docker compose -f aic_work/docker/submit-compose.yaml up
```

trial 3개 모두 완료 후 `aic_work/logs/scoring.yaml` 확인.

### 3.2 sanity points

- [ ] Tier 1 = 1 (lifecycle 통과)
- [ ] 3 trial 모두 완료 (180초 timeout 안에)
- [ ] contact penalty = 0
- [ ] 점수 ≥ 250 (보수적)

위 4개 모두 통과하지 않으면 push 금지. 회귀 디버그.

### 3.3 vast.ai L4 24GB 환경에서도 검증

학습은 A100/L40S에서 했더라도 평가는 공식 환경 = L4 24GB. vast.ai L4 인스턴스를 별도로 띄워 동일 docker compose 실행 → 점수 비교 (편차 ≥ 10점이면 GPU mismatch 의심).

---

## 4. ECR 푸시

### 4.1 인증 + 태깅

```bash
# AWS CLI v2
aws ecr get-login-password --region us-east-1 --profile aic | \
    docker login --username AWS --password-stdin \
    973918476471.dkr.ecr.us-east-1.amazonaws.com

# 태깅 (commit SHA 포함 권장)
SHA=$(git -C /home/nggw/challenge/aic_work rev-parse --short HEAD)
TAG="v1-${SHA}"

docker tag aic-submit:v1 \
    973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team_name>:${TAG}
```

> **불변 태그**: 한 번 push한 태그는 덮어쓸 수 없다. 매 제출마다 새 태그 (v1, v2, ... 또는 SHA).

### 4.2 푸시

```bash
docker push 973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team_name>:${TAG}
```

업로드 시간: 1~10GB → 5~30분 (vast.ai 대역폭 의존).

### 4.3 검증

```bash
aws ecr describe-images \
    --repository-name aic-team/<team_name> \
    --image-ids imageTag=${TAG} \
    --profile aic
```

`imageDigest`, `imagePushedAt` 확인.

---

## 5. 제출 포털 등록

1. 제출 포털 로그인 (팀 리더 = 사용자 본인 자격증명)
2. "AI for Industry Challenge" 선택
3. "Submit" → "Qualification Phase"
4. OCI Image URI 입력:
   ```
   973918476471.dkr.ecr.us-east-1.amazonaws.com/aic-team/<team_name>:v1-abc1234
   ```
5. (선택) 제출 메모 — 어떤 변경인지 한 줄 메모 기록 추천

상태 흐름: Submitted → Queued → Running → Finished/Failed (5~15분).

---

## 6. 모니터링 + 결과 확인

- 포털 "My Submissions" 대시보드에서 실시간 상태
- Finished 후 "View Score" → 300점 만점 점수 확인
- Failed 시: 로그 다운로드 → 디버그
  - 가장 흔한 fail: `model_discovery_timeout` → import 무겁거나 weight load 느림 → lazy load 검토
  - `ros2 daemon` 충돌 → entrypoint에서 `ros2 daemon start` 명시

---

## 7. 회귀 발생 시 빠른 재제출

```bash
# 1. 코드 fix
# 2. 로컬 검증 (3 trial)
# 3. SHA 갱신 → 재태그
SHA=$(git rev-parse --short HEAD)
TAG="v1-${SHA}"
docker build -t aic-submit:${TAG} -f aic_work/docker/submit.Dockerfile .
docker tag aic-submit:${TAG} <ECR_URI>:${TAG}
docker push <ECR_URI>:${TAG}
# 4. 포털에서 새 URI로 재제출
```

---

## 8. 위반 회피 최종 체크 (Code Auditor 페르소나 — A9)

`aic_work/scripts/audit_pre_submit.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
SRC="aic_work/aic_model_pkg/"

# 금지 패턴 검색
echo "== forbidden ROS calls =="
grep -rE "/scoring/|/gazebo/|/gz_server/|/clock|/model_states" "$SRC" || echo "  (clean)"

echo "== gz topic pub teleport =="
grep -rE "gz topic pub|gz model --pose|GazeboServiceClient" "$SRC" || echo "  (clean)"

echo "== ground_truth subscribe in eval mode =="
grep -rE "/scoring/tf|ground_truth_callback" "$SRC" || echo "  (only training code allowed)"

echo "== external network =="
grep -rE "requests\.get|urlopen|socket\.connect|http://|https://" "$SRC" || echo "  (clean)"

echo "== params modify =="
grep -rE "set_parameters|set_parameters_atomically" "$SRC" || echo "  (clean)"

echo "== lifecycle violations =="
grep -rE "publish.*before active|on_configure.*publish" "$SRC" || echo "  (manual check needed)"

echo "✅ pre-submit audit done"
```

수동 점검:
- [ ] aic_model lifecycle 60초 이내 configure
- [ ] /insert_cable 받기 전 motion 명령 발행 없음
- [ ] AIC_ENABLE_ACL=true 시도 (옵션이지만 권장)
- [ ] Dockerfile에 secrets 미포함

---

## 9. 제출 후 후속 작업

- 점수 결과를 `aic_work/strategy/submissions.md`에 기록:
  ```
  ## 2026-XX-XX v1-abc1234
  - score: 287 / 300
  - duration: trial1 23s / trial2 19s / trial3 28s
  - notes: ACT v2 + tuned StageC, contact 0, force 0
  ```
- best 점수 갱신 시 git tag (`git tag submission-v1`).

---

## 10. 완료 기준

- [ ] 로컬 docker compose 검증 통과 (3 trial 합 ≥ 250)
- [ ] vast.ai L4에서 동일 검증
- [ ] ECR push 성공 (digest 기록)
- [ ] 포털 "Submitted" 상태 확인
- [ ] 결과 점수 ≥ 280 / 300 (목표)
- [ ] submissions.md 갱신
