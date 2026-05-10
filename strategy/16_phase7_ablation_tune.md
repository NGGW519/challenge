# 16. Phase 7 — Ablation & Hyperparam Tune

> **목표**: Phase 1~6의 결과를 통합해 100-rollout 통계 평가, 핵심 하이퍼파라미터 그리드 서치, 회귀 디버깅. 최종 제출 직전 마지막 점수 짜내기 단계.
>
> **기간**: 1.5~2일 (vast.ai L4 24GB ×2 또는 ×4 병렬)
>
> **결과물**:
> - `aic_work/models/final/` — 최종 가중치 + cfg
> - `aic_work/strategy/ablation_report.md` — 결정 근거
> - 자체 평균 점수 ≥ 270 / 300

---

## 1. 평가 자동화

### 1.1 100-rollout 평가 스크립트

`aic_work/scripts/eval_100.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
POLICY="${1:?usage: eval_100.sh <policy> [N=100] [tag]}"
N="${2:-100}"
TAG="${3:-$(date +%Y%m%d_%H%M)}"
RESULTS_DIR="/workspace/aic_work/logs/eval/${POLICY}_${TAG}"
mkdir -p "$RESULTS_DIR"

# 병렬 트랙 (4 동시)
PARALLEL=4
SEM=$(mktemp -u /tmp/sem.XXXXXX)
mkfifo "$SEM"; exec 3<>"$SEM"; rm "$SEM"
for ((i=0;i<PARALLEL;i++)); do echo >&3; done

for i in $(seq 1 "$N"); do
  read -u3
  (
    RUN="${POLICY}_run${i}"
    AIC_RESULTS_DIR="${RESULTS_DIR}/${RUN}" \
      docker compose \
        -f /workspace/aic/docker/docker-compose.yaml \
        -f /workspace/aic_work/docker/eval-override.yaml \
        up --abort-on-container-exit --exit-code-from eval eval model \
        > "${RESULTS_DIR}/${RUN}.log" 2>&1 || true
    echo >&3
  ) &
done
wait

pixi run python /workspace/aic_work/scripts/aggregate_scoring.py \
  --dir "$RESULTS_DIR" --out "${RESULTS_DIR}/summary.csv"

pixi run python /workspace/aic_work/scripts/visualize_eval.py \
  --csv "${RESULTS_DIR}/summary.csv" \
  --out "${RESULTS_DIR}/report.html"
```

### 1.2 시각화 (`visualize_eval.py`)

생성 항목:
- 점수 히스토그램 (300점 만점)
- Tier별 분해 박스플롯
- contact penalty / force penalty 발생률
- 시간 분포 (duration, jerk)
- per-trial 비교 (sfp1 vs sfp2 vs sc)
- failure 케이스 timeline (logs 링크)

---

## 2. Ablation 매트릭스

| 변수 | 후보 값 | 선택 후보 |
|---|---|---|
| ACT chunk_size | {50, 100, 150} | 100 (default) |
| ACT backbone | {resnet18, resnet34} | resnet18 |
| Action space | {cartesian_delta, joint_delta} | cartesian_delta |
| Stage A timeout | {3s, 5s, 8s} | 5s |
| Stage B timeout | {5s, 8s, 12s} | 8s |
| Stage C spiral 시작 반경 | {0, 1mm, 3mm} | 0 |
| Stage C 진행 step | {0.3mm, 0.5mm, 1mm} | 0.5mm |
| Cartesian stiffness (xy) | {300, 500, 800} | 800 |
| Cartesian stiffness (z) | {800, 1500, 2500} | 1500 |
| Force threshold (forward stop) | {6N, 8N, 12N} | 8N |
| Force threshold (back-off) | {12N, 15N, 18N} | 15N |
| Watchdog window | {0.5s, 0.8s, 1.0s} | 0.8s |
| Detector model | {YOLOv8m, RT-DETR-r18} | YOLOv8m |
| Camera input | {3 cams, center only} | 3 cams |

**전략**: 시간 제약상 grid 전체는 비효율. **one-at-a-time** + 가장 영향 큰 3개만 grid:
1. Stage C 파라미터 (impedance, force threshold) — 직접 tier 3 점수 영향
2. ACT chunk_size — convergence 속도
3. Detector accuracy threshold — Stage A fail rate

각 변경마다 50 rollout (시간 절약), 최종 후보는 100 rollout.

### 2.1 ablation 진행 순서

```
day 1 AM: Stage C parameter sweep (3×3=9 변형 × 50 rollout = 450 trial)
day 1 PM: ACT chunk_size A/B/C (3 변형 × 50 rollout)
day 2 AM: Detector threshold + model A/B (4 변형 × 50)
day 2 PM: 최종 best combo × 100 rollout 두 번 (재현성 확인)
```

각 ablation 결과는 `aic_work/strategy/ablation_report.md`에 추가:
```markdown
## A1: Stage C stiffness (xy=800 vs 500 vs 300)
- xy=800: avg 254 (n=50, std=18), force_pen 1×, contact 0
- xy=500: avg 247 (n=50, std=22), force_pen 0×, contact 1
- xy=300: avg 230 (n=50, std=27), 정렬 실패 빈도 ↑
**선택**: xy=800
```

---

## 3. 점수 회귀 디버깅 워크플로우

평가 결과에서 -2점 이상 회귀 발견 시:

1. **failure case 분류**:
   ```python
   # logs/eval/<run>/summary.csv 분석
   failures = df[df["total"] < 90]
   failures["fail_type"] = failures.apply(classify, axis=1)
   # types: detect_miss, alignment_timeout, insertion_force, contact, ...
   ```

2. **케이스별 영상 재생**:
   - aic_engine은 rosbag 기록 가능 → 실패 trial bag 재생 → RViz 시각화
   - 가장 빈번한 failure mode 찾기

3. **fix → 재평가**:
   - fix는 가능한 한 작은 단위 (1 변수 변경)
   - 50 rollout 재평가
   - 회귀 사라지면 100으로 검증

---

## 4. 후처리 점수 짜내기

### 4.1 jerk 최소화 (smoothness ↑)

ACT 출력 action에 low-pass 추가:
```python
filtered = alpha * raw + (1 - alpha) * prev_filtered
# alpha = 0.7 → 가벼운 smoothing
```

또는 cubic spline interpolation으로 명령 spike 제거.

### 4.2 efficiency ↑ (직선 경로)

- Stage A 종료 후 Stage B 시작 시 잠깐 hover → 회귀 동작 발생 가능 → 즉시 transition.
- Stage B 동안 ACT가 backtracking할 수 있음 → action history 모니터링, 큰 dx, dy 부호 반전 시 limit.

### 4.3 duration ↓ (≤ 24초로)

- Stage A: detection이 첫 frame에 잘 되면 즉시 transition (avg → median 사용해서 latency 감소)
- Stage B: convergence 기준 더 느슨하게 (5mm → 3mm threshold 완화 안 함; 3→2mm) 시간 ↑ 위험. tradeoff.
- Stage C: spiral 너무 천천히 안 돌게 — 시간 5초 안에 ≥1 cycle.

### 4.4 contact 0 절대 사수

contact -24는 1번 발생만으로도 trial 점수 70 → 46. 다음 hard guard:

```python
def _check_contact_risk(self, obs):
    """현재 TCP 위치가 enclosure walls bbox 안에 있는지 검사."""
    # task_board pose에서 enclosure 평면 4개 계산
    # TCP가 5mm 이내면 강제로 stop & retreat
```

---

## 5. 최종 제출 후보 선정

```
candidates = [
   ("act_v1_baseline",    Phase 3 weights),
   ("act_v2_multi_sim",   Phase 6 weights),
   ("act_v2 + tuned_stageC", v2 + Stage C 튜닝),
   ("act_v2 + spline_smoothing", v2 + post-smooth),
]
```

각 후보에 100 rollout × 2회 (총 200회). 평균 점수 + 분산 + 최악 값을 모두 비교 → 가장 안정적인 것 선택.

| 후보 | 평균 | std | min | contact 횟수 | force 횟수 |
|---|---|---|---|---|---|
| act_v2_multi_sim | 256 | 14 | 220 | 0 | 1 |
| **act_v2 + tuned_C** | **272** | **9** | **245** | **0** | **0** |
| act_v2 + smoothing | 268 | 11 | 230 | 0 | 1 |

→ tuned_C 채택.

---

## 6. ablation_report.md 템플릿

```markdown
# Ablation Report

## 평가 기준
- Eval GPU: vast.ai L4 24GB (공식과 동일)
- N rollouts: 100 (final), 50 (sweep)
- Trial 분포: sfp1×33, sfp2×34, sc×33

## 결과 요약 표
| 후보 | 평균 | std | tier 1 | tier 2 mean | tier 3 mean | C-pen | F-pen |
|---|---|---|---|---|---|---|---|
...

## 결정 근거 (decisions)
1. ACT chunk_size=100 — chunk=50은 alignment 빈번한 재추론 비용, 150은 over-commit.
2. Stage C stiffness xy=800 — 500 이하에서 lateral overshoot로 contact 위험.
...

## 미해결 이슈
- SC trial에서 occasional alignment failure (~10%). Phase 6 데이터에서 SC 비중 늘리는 것이 다음 우선순위였으나 시간 부족으로 보류.
```

---

## 7. 완료 기준

- [ ] 100-rollout 평가 자동화 동작
- [ ] 시각화 리포트 (HTML) 생성
- [ ] 최종 후보 선정 (1개) — 100 rollout × 2회 평균 ≥ 270
- [ ] contact penalty 발생률 0% (200 rollout 중)
- [ ] ablation_report.md 작성
- [ ] `aic_work/models/final/` 디렉토리 정리 (weights + cfg + stats)
