# aic_work — AI for Industry Challenge 작업 디렉토리

이 디렉토리는 [AI for Industry Challenge](https://www.intrinsic.ai/events/ai-for-industry-challenge)
(intrinsic-dev/aic) 출전을 위한 **참가자 측 작업물**이다.

- **공식 토킷**: `/home/nggw/challenge/aic/` (읽기 전용)
- **이 디렉토리**: `/home/nggw/challenge/aic_work/` (작업 + GitHub 동기화)
- **원격**: https://github.com/NGGW519/challenge.git (`main`)

## 디렉토리 구조

```
aic_work/
├── strategy/           # 전략 문서 (00 ~ 17)
├── docker/             # dev / submit Dockerfile, compose override
├── scripts/            # 자동화 (vast.ai 부팅, 평가, 학습 wrapper)
├── data/               # 데이터 수집 / 변환 스크립트
├── training/           # 학습 스크립트 (ACT, YOLOv8)
├── aic_model_pkg/      # 제출용 ROS 2 Python 패키지 (Hybrid policy)
├── pyproject.toml      # Python 의존성 (Pixi 외부 fallback)
└── README.md
```

`datasets/`, `models/`, `logs/`, `aic_results/`는 git ignore. 큰 파일은
Cloudflare R2 또는 HuggingFace Hub에 보관 (`strategy/02_vastai_setup.md` 참조).

## 빠른 시작 (vast.ai 인스턴스 부팅 후)

```bash
# 1. dev image로 인스턴스 띄우기 (vast.ai console)
#    image: ghcr.io/nggw519/aic-dev:latest  (직접 빌드 후 push 필요)
#    또는 nvidia/cuda:12.1.1-cudnn8-devel-ubuntu24.04 + onstart.sh

# 2. 부팅 후 (인스턴스 안에서)
cd /workspace
ls aic aic_work    # 둘 다 clone 되어 있어야 함

# 3. 베이스라인 평가 (Phase 1)
cd aic_work
bash scripts/run_baseline.sh CheatCode 5
bash scripts/run_baseline.sh RunACT 5

# 4. 결과 집계
ls logs/baseline/CheatCode/summary.csv
```

## 전략 문서 인덱스

[strategy/00_master_plan.md](strategy/00_master_plan.md)부터 시작.

- [01](strategy/01_agent_team.md) Claude 서브에이전트 페르소나
- [02](strategy/02_vastai_setup.md) vast.ai 인프라
- [10](strategy/10_phase1_baseline.md) Phase 1 — Baseline 재현
- [11](strategy/11_phase2_data_collection.md) Phase 2 — 데이터 수집
- [12](strategy/12_phase3_act_training.md) Phase 3 — ACT 학습
- [13](strategy/13_phase4_port_detection.md) Phase 4 — 포트 검출
- [14](strategy/14_phase5_hybrid_policy.md) Phase 5 — Hybrid 정책
- [15](strategy/15_phase6_domain_rand.md) Phase 6 — Domain Randomization
- [16](strategy/16_phase7_ablation_tune.md) Phase 7 — Ablation / Tune
- [17](strategy/17_phase8_submission.md) Phase 8 — 제출

## 사용자가 별도로 준비할 항목

코드만으로는 부족하고, 사용자가 직접 셋업해야 하는 것 ([02_vastai_setup.md](strategy/02_vastai_setup.md) §9 체크리스트):

- [ ] vast.ai 계정 + 잔고 충전
- [ ] Cloudflare R2 (또는 AWS S3) 버킷 1개 (이름 예: `aic-ckpts`)
- [ ] R2 access key → vast.ai env var (`R2_ACCESS_KEY`, `R2_SECRET_KEY`)
- [ ] HuggingFace Hub 계정 + private repo (데이터셋/가중치 백업)
- [ ] AWS ECR 자격증명 (조직 측 온보딩 이메일에서)
- [ ] 제출 포털 계정 / URL 확인
