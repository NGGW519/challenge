# 10. Phase 1 — Baseline 재현

> **목표**: 토킷에 포함된 예제 정책(CheatCode, RunACT, WaveArm 등)을 그대로 실행하여:
> 1. 평가 파이프라인 동작 확인
> 2. 점수 상한(CheatCode = ~290+)과 하한(WaveArm = ~3) 측정
> 3. 우리 자체 평가 자동화 스크립트 검증
>
> **기간**: 0.5~1일 (vast.ai L4 24GB 1대)
>
> **선행조건**: `02_vastai_setup.md`의 dev 이미지로 인스턴스 부팅 완료.

---

## 1. 사전 점검

```bash
cd /workspace/aic
pixi --version              # 0.30+
docker --version            # 24+
nvidia-smi                  # CUDA 12+
ros2 --help 2>/dev/null || echo "ros2 in pixi env"
pixi run ros2 --help        # 정상 출력
```

문제 시 → 인스턴스 image가 `ghcr.io/nggw519/aic-dev:latest`인지 확인.

---

## 2. aic_eval 컨테이너 풀

```bash
docker pull ghcr.io/intrinsic-dev/aic/aic_eval:latest
docker images | grep aic_eval
```

크기 ~6-8GB 예상. 실패 시 `docker login ghcr.io` (PAT 필요할 수 있음 — 공개 이미지면 로그인 불필요).

---

## 3. 5개 예제 정책 — 점수 분포 측정

각 정책을 3 trial × 5 반복 = 15 rollout 돌려 평균/최대/최소를 표로 정리.

### 3.1 자동 실행 스크립트

`aic_work/scripts/run_baseline.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

POLICY="${1:?usage: run_baseline.sh <CheatCode|RunACT|WaveArm|GentleGiant|SpeedDemon|WallToucher|WallPresser>}"
N_REPEATS="${2:-5}"
RESULTS_DIR="${RESULTS_DIR:-/workspace/aic_work/logs/baseline/${POLICY}}"
mkdir -p "$RESULTS_DIR"

# CheatCode requires ground_truth=true
GT_FLAG="ground_truth:=false"
if [[ "$POLICY" == "CheatCode" ]]; then
  GT_FLAG="ground_truth:=true"
fi

for i in $(seq 1 "$N_REPEATS"); do
  RUN_ID="${POLICY}_run${i}_$(date +%s)"
  echo "=== ${RUN_ID} ==="
  AIC_RESULTS_DIR="${RESULTS_DIR}/${RUN_ID}" \
  AIC_POLICY_MODULE="aic_example_policies.ros.${POLICY}" \
    docker compose -f /workspace/aic/docker/docker-compose.yaml up \
      --abort-on-container-exit \
      --exit-code-from eval \
      eval model \
      || echo "[WARN] run ${RUN_ID} non-zero exit"
  cp -r "${RESULTS_DIR}/${RUN_ID}/scoring.yaml" "${RESULTS_DIR}/${RUN_ID}.yaml" 2>/dev/null || true
done

# Aggregate
pixi run python /workspace/aic_work/scripts/aggregate_scoring.py \
  --dir "$RESULTS_DIR" \
  --out "${RESULTS_DIR}/summary.csv"
```

### 3.2 docker-compose 오버라이드

기본 `docker-compose.yaml`은 image `my-solution:v1` (참가자 자기 이미지)를 model로 띄움. baseline은 example_policies가 들어있는 dev 이미지를 쓰도록 override 파일 작성:

`aic_work/docker/baseline-override.yaml`:
```yaml
services:
  model:
    image: ghcr.io/nggw519/aic-dev:latest
    command: >
      bash -lc "pixi run ros2 run aic_model aic_model
                 -p policy:=${AIC_POLICY_MODULE}
                 -p use_sim_time:=true"
    environment:
      AIC_POLICY_MODULE: ${AIC_POLICY_MODULE:-aic_example_policies.ros.CheatCode}
  eval:
    command: >
      gazebo_gui:=false launch_rviz:=false
      ground_truth:=${AIC_GROUND_TRUTH:-false}
      start_aic_engine:=true
      shutdown_on_aic_engine_exit:=true
      model_discovery_timeout_seconds:=600
    volumes:
      - ${AIC_RESULTS_DIR:-./aic_results}:/root/aic_results
```

실행:
```bash
docker compose \
  -f /workspace/aic/docker/docker-compose.yaml \
  -f /workspace/aic_work/docker/baseline-override.yaml \
  up
```

### 3.3 scoring.yaml 집계

`aic_work/scripts/aggregate_scoring.py`:

```python
#!/usr/bin/env python3
"""scoring.yaml 묶음 → CSV 통계."""
import argparse
import csv
import statistics
from pathlib import Path

import yaml

KEYS = [
    "trial_id",
    "tier_1_validity",
    "smoothness_score",
    "duration_score",
    "efficiency_score",
    "force_penalty",
    "contact_penalty",
    "tier_3_insertion",
    "total",
]

def flatten(scoring):
    """공식 스키마는 trials list. 정확한 필드명은 실제 출력 후 확정."""
    rows = []
    for trial in scoring.get("trials", []):
        row = {
            "trial_id": trial.get("trial_id"),
            "tier_1_validity": trial.get("tier_1_validity"),
            "smoothness_score": trial.get("tier_2", {}).get("smoothness_score"),
            "duration_score": trial.get("tier_2", {}).get("duration_score"),
            "efficiency_score": trial.get("tier_2", {}).get("efficiency_score"),
            "force_penalty": trial.get("tier_2", {}).get("force_penalty"),
            "contact_penalty": trial.get("tier_2", {}).get("contact_penalty"),
            "tier_3_insertion": trial.get("tier_3_insertion"),
            "total": trial.get("total"),
        }
        rows.append(row)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    all_rows = []
    for run in sorted(Path(args.dir).glob("*/scoring.yaml")):
        with run.open() as f:
            data = yaml.safe_load(f)
        for r in flatten(data):
            r["run"] = run.parent.name
            all_rows.append(r)

    if not all_rows:
        print("no scoring.yaml found")
        return

    fields = ["run"] + KEYS
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    totals = [r["total"] for r in all_rows if r["total"] is not None]
    print(f"runs: {len(all_rows)}")
    print(f"mean total: {statistics.mean(totals):.2f}")
    print(f"median total: {statistics.median(totals):.2f}")
    print(f"stdev total: {statistics.stdev(totals):.2f}" if len(totals) > 1 else "")
    print(f"min/max: {min(totals):.2f} / {max(totals):.2f}")

if __name__ == "__main__":
    main()
```

> **주의**: scoring.yaml 실제 키명은 `aic_scoring/` 코드를 참조해야 정확. 위 스크립트는 가정이므로 첫 실행 후 실제 키와 맞춰 수정.

---

## 4. 측정 대상 정책 & 기대 점수

| 정책 | 기대 평균 점수 / 300 | 의의 |
|---|---|---|
| CheatCode (gt:=true) | 270~290 | 상한 확인 — 우리가 도달해야 할 천장 |
| RunACT (사전학습 ckpt) | 80~150 | IL 베이스라인 — 우리 ACT가 이걸 넘어야 함 |
| WaveArm | ~3 | 하한 + 평가 파이프라인 sanity |
| GentleGiant | ~10 | smoothness 만점, insertion 0 |
| SpeedDemon | -10 ~ 0 | force penalty 발동 |
| WallToucher | -20 ~ 0 | contact penalty -24 발동 |
| WallPresser | -10 ~ 0 | force penalty -12 발동 |

CheatCode는 학습 시(or scoring 분석 시) ground_truth=true 활용 — 이 정책은 평가 환경에선 작동하지 않으니 절대 제출하지 말 것.

---

## 5. 결과물

`aic_work/logs/baseline/` 아래 다음 파일들:

```
baseline/
├── CheatCode/summary.csv
├── RunACT/summary.csv
├── WaveArm/summary.csv
├── GentleGiant/summary.csv
├── SpeedDemon/summary.csv
├── WallToucher/summary.csv
└── WallPresser/summary.csv
```

추가로 `aic_work/strategy/baseline_report.md` (수동 작성):
- 각 정책 평균/min/max/stdev
- 발견한 이슈 (예: 디스커버리 타임아웃, RTF 저하, F/T spike)
- 우리 전략에 대한 시사점

---

## 6. 디버깅 가이드 (자주 만나는 문제)

| 증상 | 원인 | 해결 |
|---|---|---|
| `model_discovery_timeout` (30초) | aic_model 노드 active 못 감 | dev 이미지에서 pixi env activated 확인. import 무거우면 lazy load. |
| RTF < 0.1 | GPU 미사용 | `nvidia-smi`로 컨테이너에 GPU 보이는지 확인. `--gpus all` 누락? |
| Zenoh 경고 (`shared memory monitor`) | 무해 | 무시 |
| scoring.yaml 비어있음 | engine이 비정상 종료 | logs에서 `aic_engine` 컨테이너 stderr 확인 |
| trial 1 통과 후 trial 2 hang | 시뮬레이터 reset 실패 | gz topic list로 contact 토픽 stuck 확인 |

`troubleshooting.md` 참조.

---

## 7. 의사결정 로그 (작성하면서 채울 것)

- **결정**: ACT baseline은 RunACT의 사전학습 ckpt를 그대로 쓸지, fine-tune부터 시작할지
- **결정**: 첫 학습 데이터셋은 CheatCode 텔레옵 N rollouts → N=?
- **결정**: 평가 GPU를 vast.ai L4 (공식 동일)와 4090 (빠름) 중 어디서 fix할지

---

## 8. 완료 기준

- [ ] 7개 예제 정책 모두 1회 이상 실행 성공
- [ ] CheatCode 평균 ≥ 270 확인 (이게 안 되면 셋업 자체 문제)
- [ ] RunACT 평균 측정값 기록 (우리 ACT 목표치 설정용)
- [ ] aggregate_scoring.py로 CSV 자동 생성
- [ ] baseline_report.md 작성
