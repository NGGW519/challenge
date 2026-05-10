# 02. vast.ai 인프라 설정

> **목표**: 학습 / 데이터 수집 / 평가용 GPU 인스턴스를 즉시 spin-up 하고, 영속 데이터(ckpt, dataset, docker image)는 인스턴스 휘발성과 분리해 보관.
> **예산**: 무제한 (단, 불필요한 idle 인스턴스 정지 — 비용보다 시간 가치).

---

## 1. 작업 유형별 인스턴스 사양

| 작업 | 추천 GPU | RAM | 디스크 | 시간당 대략 비용 (참고) |
|---|---|---|---|---|
| 데이터 수집 (Gazebo CheatCode 텔레옵) | RTX 4090 24GB | 64GB+ | 200GB | ~$0.6 |
| ACT 학습 (chunk=100, BS=64) | L40S 48GB / A100 40GB | 96GB+ | 500GB | ~$0.9 / ~$1.2 |
| 대규모 ACT 학습 (DR + multi-sim) | A100 80GB / H100 80GB | 128GB+ | 1TB | ~$1.7 / ~$3.0 |
| Port detector (YOLOv8m) fine-tune | RTX 4090 24GB | 64GB | 200GB | ~$0.6 |
| 평가 일괄 (100 rollouts × N variants) | **L4 24GB** (← eval과 동일) | 96GB+ | 200GB | ~$0.4 |
| 도메인 랜덤화 multi-sim (Isaac+MuJoCo+Gazebo) | A100 80GB ×1 또는 L40S ×2 | 128GB+ | 1TB | ~$1.7 |

> **중요**: 평가 인스턴스는 반드시 **L4 24GB** (조직 측 평가 환경과 동일). 동일 GPU에서 통과해야 진짜 통과한 것.

### 1.1 vast.ai 인스턴스 검색 필터

```
GPU: RTX 4090 / L40S / A100 80GB / L4
CUDA: 12.4+ (ROS 2 Kilted Pixi env과 호환)
DLPerf: ≥ 30 (4090 기준)
Disk: ≥ 200 GB
RAM: ≥ 64 GB (Gazebo + ROS 2 + Pixi)
Inet up/down: ≥ 200 Mbps
Reliability: ≥ 0.99
Price/Hour: 무제한 (정렬은 dlperf/$)
Image: nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04   # ← Ubuntu 24.04 + CUDA 12.1 조합은 공식 태그 없음
On-demand interruptible? On-demand 권장 (학습은 짤리면 손해)
```

### 1.2 추천 base image

vast.ai 마켓의 PyTorch / Ubuntu 24.04 + CUDA 12.1 이미지를 시작점으로 쓰되, 우리는 ROS 2가 필요하므로 사용자 정의 Dockerfile 빌드 필요. (`docker/aic_dev/Dockerfile` — 02-A 참조).

---

## 2. 영속 스토리지 전략 (vast.ai 인스턴스는 휘발성)

vast.ai 인스턴스는 host 다운/계약 종료/사용자 stop으로 사라질 수 있다. 절대로 인스턴스 로컬에만 데이터를 두지 말 것.

> **결정 (2026-05-10)**: 단순함을 위해 **HuggingFace Hub 단독 백업**. R2/S3는 사용 안 함.
> - 사용자가 관리할 자격증명 1개 (HF token)로 끝.
> - private repo는 무료, 큰 파일도 git LFS로 자동 처리.
> - push 속도가 R2 대비 느릴 수 있으나 (HF는 git LFS 단일 push) 우리 ckpt 단위(<200MB)에선 충분.

### 2.1 2-tier 스토리지

| Tier | 용도 | 매체 | 동기화 |
|---|---|---|---|
| **Tier 0 (Hot)** | 현재 학습 중 active dataset, 현재 ckpt | 인스턴스 NVMe | — |
| **Tier 1 (Cold)** | 최신/best ckpt, dataset 원본, 평가 결과 | HuggingFace Hub (private repo) | `huggingface-cli upload`, 매 ckpt마다 |

### 2.2 HuggingFace Hub 셋업

권장 repo 2개 (둘 다 private):
- `nggw519/aic-ckpts` (model repo) — ACT/YOLOv8 가중치, 평가 결과
- `nggw519/aic-datasets` (dataset repo) — LeRobot v2.1 데이터셋, port_detection 합성 데이터

```bash
# 인스턴스 부팅 시 (onstart.sh가 자동 실행)
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

# repo 한 번만 생성 (사용자 로컬 또는 첫 인스턴스에서)
huggingface-cli repo create aic-ckpts    --type=model   --private
huggingface-cli repo create aic-datasets --type=dataset --private
```

### 2.3 자동 ckpt 푸시

학습 코드에 hook 추가:
```python
# train_act.py — push_ckpt()
from huggingface_hub import HfApi
HfApi().upload_folder(
    repo_id="nggw519/aic-ckpts",
    folder_path=str(ckpt_dir),
    path_in_repo=f"act_v1/",
    repo_type="model",
    commit_message=f"step {step}",
)
```

명령행 등가:
```bash
huggingface-cli upload nggw519/aic-ckpts ./models/act_v1/ act_v1/ --repo-type=model
```

매 ckpt(`step_00050000.pt` 등)마다 push. 가중치 ~150MB이라 5–30초 소요.

---

## 3. 인스턴스 부팅 자동화

### 3.1 vast.ai On-Start 스크립트

vast.ai console → instance create 시 "On-start script" 필드에 다음을 넣는다 (또는 `vastai start instance --onstart-cmd '...'`):

```bash
#!/bin/bash
set -e

# 1. apt 업데이트 + 기본 도구
export DEBIAN_FRONTEND=noninteractive
apt update && apt install -y \
  git curl wget rsync tmux htop nvtop unzip \
  build-essential python3-pip python3-venv \
  software-properties-common lsb-release

# 2. Pixi 설치
curl -fsSL https://pixi.sh/install.sh | bash
echo 'export PATH="$HOME/.pixi/bin:$PATH"' >> ~/.bashrc

# 3. HuggingFace CLI (R2 대신)
pip install -q huggingface_hub[cli]

# 4. 작업 디렉토리 동기화
mkdir -p /workspace
cd /workspace
git clone https://github.com/intrinsic-dev/aic.git aic
git clone https://github.com/NGGW519/challenge.git aic_work

# 5. HF 로그인 (HF_TOKEN 환경변수 사용)
huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential

# 6. 최신 ckpt 풀 (있으면)
mkdir -p /workspace/aic_work/checkpoints
huggingface-cli download --repo-type=model nggw519/aic-ckpts \
    --local-dir /workspace/aic_work/checkpoints || true

# 7. tmux 시작
tmux new-session -d -s main "bash -l"
echo "Bootstrap complete. Attach via: tmux attach -t main"
```

### 3.2 secrets 관리

비밀 키는 절대 git에 커밋 금지. 권장: vast.ai env vars로 다음 둘만 주입.
- `HF_TOKEN` — HuggingFace Hub write 토큰
- (선택) `GIT_USER_NAME`, `GIT_USER_EMAIL` — git 커밋 author

---

## 4. ROS 2 Kilted + Pixi 환경 부팅

vast.ai 인스턴스에 ROS 2 + 의존성을 매번 설치하지 말고, 미리 빌드한 Docker image 사용.

### 4.1 dev 이미지 (학습/데이터 수집용)

`/home/nggw/challenge/aic_work/docker/dev.Dockerfile`로 작성:

```dockerfile
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PIXI_HOME=/opt/pixi
ENV PATH="${PIXI_HOME}/bin:${PATH}"

RUN apt update && apt install -y \
    git curl wget rsync tmux htop nvtop \
    build-essential python3-pip python3-venv \
    software-properties-common lsb-release \
    libgl1 libglib2.0-0 libegl1 \
 && rm -rf /var/lib/apt/lists/*

# Pixi
RUN curl -fsSL https://pixi.sh/install.sh | bash -s -- --no-modify-path
ENV PATH="/root/.pixi/bin:${PATH}"

# ROS 2 Kilted (apt route as fallback; primary는 Pixi의 robostack)
# Pixi가 robostack channel로 ROS 2 Kilted를 가져옴 → 별도 apt ros 설정 불필요

WORKDIR /workspace
COPY pixi.toml pixi.lock ./
RUN pixi install --frozen

ENV LANG=C.UTF-8
SHELL ["/bin/bash", "-c"]
CMD ["bash"]
```

빌드 & push (개발 환경 또는 GitHub Actions):
```bash
docker build -t ghcr.io/nggw519/aic-dev:latest -f docker/dev.Dockerfile .
docker push ghcr.io/nggw519/aic-dev:latest
```

vast.ai에서 instance type "On-Demand", template image: `ghcr.io/nggw519/aic-dev:latest`로 띄우면 5분 내 작업 가능.

### 4.2 evaluation-clone 이미지 (제출 직전 검증용)

Phase 8에서 다룬다 (`17_phase8_submission.md`). 공식 평가 컨테이너 + 우리 모델 컨테이너를 함께 띄워 동일 인터페이스 통과 확인.

---

## 5. tmux 세션 레이아웃 (사용자 작업 효율)

ssh 후 `tmux attach -t main` → 다음 윈도우 자동 생성하는 dotfile:

```bash
# ~/.tmux.conf 일부
new-window -n train  'cd /workspace/aic_work && bash'
new-window -n eval   'cd /workspace/aic_work && bash'
new-window -n gpu    'watch -n 2 nvidia-smi'
new-window -n disk   'watch -n 5 df -h /workspace'
new-window -n logs   'tail -F /workspace/aic_work/logs/*.log'
```

---

## 6. 휘발성 대응 (자동 복구)

### 6.1 학습 잡: torchrun + auto-resume

```bash
# scripts/train_with_resume.sh
while true; do
  pixi run python training/train_act.py \
    --resume_from_latest \
    --ckpt_dir /workspace/aic_work/checkpoints \
    && break
  echo "[$(date)] training crashed, sleeping 30s..."
  sleep 30
done
```

(systemd 단위가 더 robust하나, vast.ai 컨테이너에선 supervisord 또는 단순 while loop 추천.)

### 6.2 인스턴스 종료 → 새 인스턴스에서 재개

1. 새 인스턴스 spin-up (동일 image)
2. on-start 스크립트가 자동으로 `huggingface-cli download nggw519/aic-ckpts` 풀
3. `scripts/train_with_resume.sh` 실행 → 직전 ckpt에서 재개

손실: 마지막 ckpt 이후 작업 (`ckpt_every` 단위, 기본 5,000 step ≈ 25분).
더 짧은 손실을 원하면 `ckpt_every`를 줄이되 push 빈도와의 trade-off 고려.

---

## 7. 평가 동등성 검증

vast.ai에서 학습한 정책이 공식 평가 환경(L4 24GB)에서 동일한 점수를 내는지가 핵심. Phase 7 끝, Phase 8에서 반드시 다음을 실행:

```bash
# vast.ai에서 L4 24GB 인스턴스 띄우고:
cd /workspace/aic
pixi run -e dev colcon build --packages-select aic_model
docker compose -f docker/docker-compose.yaml build model
docker compose -f docker/docker-compose.yaml up
# scoring.yaml 비교
```

학습-평가 GPU mismatch (A100→L4 등)로 인한 점수 회귀가 5점 이상이면 → 학습 인스턴스도 L4로 전환해 fine-tune.

---

## 8. 비용 모니터링 (참고)

비용 무제한이지만 idle 인스턴스는 정지 — 시간이 곧 가치. 일일 점검:

```bash
# vast-cli (로컬에서)
vastai show instances | awk '{print $1,$3,$8,$NF}'  # id, gpu, status, $/h
```

idle 시간 > 30분이면 자동 정지 cron (옵션):
```bash
*/15 * * * * if [[ $(uptime | awk '{print $11}' | tr -d ,) > 0 ]]; then ...; fi
```

(현실적으론 사용자가 매일 vastai 콘솔에서 확인하는 편이 편함.)

---

## 9. 체크리스트 (이 문서가 끝났을 때 갖추어져야 함)

- [ ] vast.ai 계정 + 잔고 충전
- [ ] HuggingFace Hub 토큰 발급 (write 권한) → `HF_TOKEN`
- [ ] HF private repo 2개 생성: `nggw519/aic-ckpts` (model) + `nggw519/aic-datasets` (dataset)
- [ ] HF_TOKEN을 vast.ai env var로 등록
- [ ] `aic_work/docker/dev.Dockerfile` 작성 + ghcr.io에 push
- [ ] `aic_work/scripts/onstart.sh` 작성 + vast.ai 템플릿에 등록
- [ ] 첫 인스턴스 띄워 5분 내 `pixi run colcon build` 통과 확인
