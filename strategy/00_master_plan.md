# AIC 챌린지 — 280+/300 마스터 전략

> 목표: **3개 trial 평균 ≥ 93.3점, 즉 최소 280점 (이상적으로 285~290+)**
> 작업자: **1인 단독** (팀원 없음)
> 인프라: **vast.ai 클라우드 GPU** (예산 제약 없음 — 필요시 즉시 인스턴스 spin-up)
> 마감일: **변경됨** — 일정 페이스보다 품질 우선. 이 문서의 "예상 기간"은 단순 작업량 추정일뿐 실제 deadline과 무관.
> 작업 디렉토리: `/home/nggw/challenge/aic_work/`
> 토킷 원본: `/home/nggw/challenge/aic/` (수정 금지, 읽기 전용 참조)
> 원격 백업: `https://github.com/NGGW519/challenge.git` (브랜치 `main`)

---

## 1. 점수 산수: 어디서 어떻게 점수가 나오는가

| 항목 | 점수 | 트라이얼당 영향 | 280+ 목표 시 필요값 |
|---|---|---|---|
| Tier 1 — Validity | 0 / 1 | 모든 trial 필수 | **1/1 (3 trial 모두)** |
| Tier 2 — Smoothness (jerk) | 0 ~ 6 | 부드러운 모션 | **≥ 4** (jerk < 17 m/s³) |
| Tier 2 — Duration | 0 ~ 12 | 5초→12, 60초→0 | **≥ 8** (≤ 24초) |
| Tier 2 — Efficiency | 0 ~ 6 | 직선경로 | **≥ 3** (1.5× 직거리 이내) |
| Tier 2 — Force penalty | 0 / -12 | 20N×1초 초과 | **0 (피해야)** |
| Tier 2 — Contact penalty | 0 / -24 | 인클로저/보드 접촉 | **0 (절대 회피)** |
| Tier 3 — 정확 삽입 | 75 | 핵심 점수 | **3/3 trial에서 75점** |
| Tier 3 — 오삽입 | -12 | 잘못된 포트 | **0건** |

**핵심 인사이트**:
- 280+의 거의 전부는 **Tier 3 정확 삽입(75×3 = 225)** 에 달려 있다.
- 나머지 **55+ 점**은 Tier 1 (3) + Tier 2 보너스 합 (avg 18~20)/trial 필요.
- **단 1회의 contact penalty(-24)** 가 트라이얼 1개를 망가뜨리고 ~93점에서 ~70점으로 추락시킨다 → **충돌 회피가 최고 우선순위**.

### 점수 시뮬레이션 (목표 마진)
- 보수적 목표: 3 trial × (1 + 75 + 4 + 8 + 3) = **273점** ← 일단 이 라인 확보
- 도전 목표: 3 trial × (1 + 75 + 5 + 10 + 4) = **285점**
- 상위 목표: 3 trial × (1 + 75 + 6 + 12 + 5) = **297점**

---

## 2. 평가 환경 핵심 사실

- **3개 trial 고정**: SFP×2 (Trial 1,2) + SC×1 (Trial 3)
- **각 trial 180초 timeout**, but Tier 2에서는 5초가 만점, 60초가 0점
- 로봇은 처음부터 플러그를 잡고 있고, 포트 근처(수 cm)에서 시작
- 그립 오차 ±2mm, ±0.04 rad 존재 → robust 정책 필요
- 태스크 보드 위치/방향, NIC/SC 레일 슬라이드 위치 매 trial 무작위
- 평가 로봇: UR5e + 손목 F/T + 3개 카메라(좌/중/우, 1152×1024 @ 20FPS)
- 컨트롤러: Cartesian/Joint **임피던스 제어** (10–30Hz 명령 → 500Hz 보간)

---

## 3. 전략 골자: 3중 안전망 (Triple-Safety) Hybrid Policy

각 trial 내에서 **3 단계 정책**을 순차 실행:

### Stage A — Coarse Approach (시각 정렬)
- 입력: 3개 카메라 이미지 + (옵션) joint state
- 모듈: **Port Detection Network** (YOLOv8 / RT-DETR fine-tune)
- 출력: 카메라 좌표계 → 베이스 좌표계로 포트 진입점 3D 좌표
- 동작: TCP를 포트 entrance 위 ~3cm로 이동 (저강성 임피던스)
- 시간 예산: 3~6초

### Stage B — Vision-Servoed Alignment
- 입력: 중앙 카메라(혹은 stereo) + F/T 센서
- 모듈: **ACT (LeRobot) 또는 Diffusion Policy** 로 학습된 IL 모델
- 출력: 카르테시안 trust-region 명령 (작은 step)
- 동작: 깊이 0~5mm 내에서 plug-port 미세 정렬, 진입 직전 상태로 수렴
- 시간 예산: 3~6초

### Stage C — Compliant Insertion (force-guided)
- 입력: F/T 센서 (주) + camera 보조
- 모듈: **결정론적 spiral-search + admittance**
  - z축 천천히 하강 (≤0.5mm/step)
  - F_z > threshold → 약간 후퇴 후 spiral search
  - F_xy → admittance (lateral compliance)로 자가 정렬
- 출력: Cartesian impedance 명령 (낮은 stiffness 50~150 N/m, 적절한 damping)
- 시간 예산: 3~10초

이 hybrid 구조가 단일 end-to-end IL 정책보다 더 robust한 이유:
1. **충돌 -24점 회피**: Stage A에서 명시적 안전 마진을 확보 (low stiffness + bbox check).
2. **삽입 75점 확보**: Stage C가 잔여 ±5mm 오차를 force feedback으로 흡수.
3. **시간 단축**: Stage A,B에서 빠르게 close하고 Stage C에서만 신중.

---

## 4. 전체 파이프라인 (Phases)

| Phase | 내용 | 산출물 | 예상 기간(=계산 시간) |
|---|---|---|---|
| 0. Infra | vast.ai 인스턴스 셋업, Docker 검증 | working dev box | 0.5d |
| 1. Baseline Repro | CheatCode/RunACT 평가, scoring 베이스라인 측정 | baseline 점수표 | 1d |
| 2. Data Collection | CheatCode 텔레옵 → 시연 1000+ rollouts, randomization | LeRobot dataset | 2d |
| 3. ACT Fine-tune | LeRobot ACT를 자체 데이터로 학습 | ACT v1 weights | 2d |
| 4. Port Detection | YOLOv8/RT-DETR fine-tune (자체 라벨) | detector v1 | 1.5d |
| 5. Hybrid Policy | Stage A/B/C 통합, force-guided insertion | hybrid_policy v1 | 2d |
| 6. Domain Rand | Isaac/MuJoCo/Gazebo 3-way training | robust ACT v2 | 2d |
| 7. Ablation+Tune | 100 rollout 평가, hyperparam tune | final policy | 2d |
| 8. Submit | Docker 패키징, ECR push, validate | 제출 이미지 | 0.5d |

---

## 5. 실패 모드 & 방어책 매트릭스

| 실패 모드 | 발생 점수 | 방어책 |
|---|---|---|
| 정확한 포트가 아닌 인접 포트에 삽입 | -12점 | Port detection을 좌표 + class로 학습 (port_id 지정 task와 매칭) |
| TCP가 인클로저에 접촉 | -24점 | 모든 motion 명령을 safe-bbox 클램프 (Stage A) + impedance 낮춤 |
| 삽입 시 과도한 force(>20N×1s) | -12점 | F/T 모니터링 → 임계 시 vertical step 멈춤, lateral compliance 활성 |
| 시간 초과(>60s) | -12점 (Tier2) + 부분 Tier3 | 각 stage timeout 설정, 실패 시 fallback (다음 stage forced) |
| Jerk 폭발 | -6점 | 명령 간 low-pass filter, 가속도 제한 |
| 그립 오차로 정렬 실패 | -75점 부분만 | TCP 좌표만 신뢰하지 말고 **camera-in-hand visual servoing** 사용 |
| Sim-to-sim 차이 | 평가 시 점수 폭락 | Isaac+MuJoCo+Gazebo 3중 학습, augmentation 강화 |

---

## 6. 평가 환경 핵심 수치 (출처: aic/docs)

- **공식 평가 인스턴스 (조직 측):** 64 vCPU / 256 GiB RAM / 1× NVIDIA L4 (24 GiB VRAM) / CUDA 13.0 / Driver 580.126.09
- **공식 평가 시뮬레이터:** Gazebo (ROS 2 Kilted Kaiju)
- **카메라:** 1152×1024 @ 20 FPS × 3 (좌/중/우 손목 장착)
- **F/T 센서:** ATI AXIA80-M20 (`/fts_broadcaster/wrench`)
- **컨트롤러:** UR5e + Robotiq Hand-E + Cartesian/Joint impedance (~500Hz 보간)
- **로봇 home pose:** `home_joint_positions` (`sample_config.yaml`)
- **각 trial timeout:** 180초 (config) / Tier 2 점수는 5초 만점, 60초 0점
- **NIC 레일 슬라이드:** [-0.0215, +0.0234] m, 회전 ±10°
- **SC 레일 슬라이드:** [-0.06, +0.055] m

## 7. 핵심 원칙 (Operating Principles)

1. **재학습 가능성 보장**: 모든 학습 스크립트, 데이터셋 분할, seed를 git으로 관리. 어떤 인스턴스에서든 동일 결과 재현되어야 함.
2. **vast.ai는 휘발성**: 인스턴스가 갑자기 종료될 수 있다고 가정 → 체크포인트는 매 epoch 마다 외부 스토리지(rclone S3 또는 HuggingFace Hub)에 푸시.
3. **단일 진실의 원천**: 학습/평가/서빙 모두 동일한 `aic_model/policy.py` 인터페이스 사용. 평가 컨테이너에서 동작하는 코드만이 "최종 제출".
4. **가능한 한 빨리 end-to-end 1회**: Phase 1~5의 부실한 v0를 먼저 통과시키고 → 거기서부터 개선. 각 Phase를 완벽하게 끝낸 후 다음으로 넘어가지 않음.
5. **점수 회귀 모니터링**: 매 변경마다 100-rollout 평가 → 직전 best 대비 회귀 시 즉시 롤백.

## 8. 다른 문서 인덱스 (이 디렉토리 내)

- `01_agent_team.md` — Claude 서브에이전트 역할 분담 (1인 작업자가 활용할 AI 페르소나들)
- `02_vastai_setup.md` — vast.ai 인스턴스 선택, Docker, ROS 2, 영속 스토리지
- `10_phase1_baseline.md` — Baseline 재현 (CheatCode/RunACT 평가)
- `11_phase2_data_collection.md` — 시연 데이터 수집 (CheatCode 기반 자동 텔레옵)
- `12_phase3_act_training.md` — LeRobot ACT 학습
- `13_phase4_port_detection.md` — 포트 검출 모델 (YOLOv8/RT-DETR fine-tune)
- `14_phase5_hybrid_policy.md` — Stage A/B/C 통합 hybrid policy
- `15_phase6_domain_rand.md` — 도메인 랜덤화 (Isaac/MuJoCo/Gazebo 3중)
- `16_phase7_ablation_tune.md` — 100 rollout 평가 & 튜닝
- `17_phase8_submission.md` — Docker 패키징 → ECR 푸시 → 포털 제출
