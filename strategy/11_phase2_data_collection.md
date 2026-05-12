# 11. Phase 2 — 시연 데이터 수집

> ## ⚠️ 이전 v2 회귀(3/300)의 핵심 교훈 (2026-05-11 추가)
>
> 이전 시도에서 같은 토킷·CheatCode로 데이터 수집했지만 학습이 marginal-mean
> collapse로 망가졌다. 사용자가 직접 rosbag 재생해보니 **"port 앞으로 출발해서
> 채 도착도 못 했는데 영상이 끝남"** — 즉 trajectory가 잘려 학습 데이터의
> 대부분이 "정지 또는 approach 초반"이었음.
>
> **원인**: `timeout 600` (10분)이 한 ep(3 trial = 20–60분)에 비해 짧았고,
> 성공 여부를 사후 검증하지 않은 채 모든 frame을 dataset에 넣음.
>
> **반드시 지켜야 할 sanity layer (코드로 강제)**:
> 1. `episode_timeout_s ≥ 1800` (`data/collect_demos.py` default).
> 2. 매 ep마다 `data/validate_episode.py` 호출:
>    - `scoring.yaml` 의 trial total ≥ 50 (부분 삽입까지 허용)
>    - rosbag duration ≥ 20s (잘리지 않았는지)
>    - 실패 ep는 `raw_episodes/_failed/` 로 격리 — dataset에 안 들어감
> 3. 누적 success rate < 0.8 이면 collection 즉시 abort (잘못된 데이터 대량 생성 차단).
> 4. `bag_to_lerobot.py` 가 ep별 `motion_frame_ratio` 검사 — `< 0.3` 인 ep 제외.
> 5. 매 N=10 ep 마다 1개를 mp4로 추출해 HF dataset `_samples/` 에 push.
>    사용자가 콘솔에서 한 번 보고 OK 사인 → 다음 batch.
>
> Phase 2를 시작할 때 위 5개가 코드에 살아있는지 audit 필수.

### Marginal-mean collapse 근본 원인 (Day 5 진단 결론)
이전 시도의 진단 (3 증거):
1. **시간 불변성**: 추론 step 0~580 내내 raw action 변동 < 1%
2. **공간 불변성**: trial 1/2/3 (다른 케이블/target) 모두 동일한 raw action
3. **통계 일치**: raw action ≈ 학습 데이터의 `action_mean`

→ Policy가 관측 무시하고 marginal mean만 출력. MSE 최소화 솔루션.

**원인**: 학습 데이터의 `action.std ≈ 1.2 cm/s` (CheatCode descent step 0.5mm/0.05s
= 1 cm/s. 다른 단계도 비슷). 너무 작은 std → ACT가 conditional prediction
포기하고 평균만 학습.

**Phase 2에서 추가 방어**:
- `bag_to_lerobot.py --action-amplify N` 옵션으로 action vector를 N배 증폭해
  학습 분포의 std를 인위적으로 키울 수 있음 (default N=1.0, 권장 4.0 — 이전 시도 RAW_ACTION_SCALE).
- 또는 `--action-aggregate K` 로 K 미래 step의 action을 합쳐 큰 step으로 변환
  (실효 속도 ↑).
- 학습 후 ACT inference에서도 amplify 적용 가능 — 단, 학습/추론이 일관돼야 함.


> **목표**: ACT/Diffusion Policy 학습용 LeRobot 데이터셋 v2.1 포맷으로 **1000+ rollouts** 수집.
> 시연자: **CheatCode 정책** (ground_truth=true) — 사람 텔레옵보다 빠르고 일관성 있음.
> 다양성: 매 rollout마다 task_board pose / 컴포넌트 위치 / 광원 / 카메라 noise 무작위화.
>
> **기간**: 1.5~2일 (vast.ai 4090 1대)

---

## 1. 왜 CheatCode 텔레옵?

| 옵션 | 장점 | 단점 |
|---|---|---|
| 사람 텔레옵 (RViz) | 다양한 전략 | 1인이 1000회는 비현실적 |
| **CheatCode 자동 텔레옵** | 무한 반복, 일관된 라벨, sim-only | 한 가지 전략만 학습됨 → noise/randomization 필수 |
| RL exploration | 새로운 행동 발견 | 학습 매우 비쌈, 챌린지 일정 압박 |

**선택**: CheatCode 자동 텔레옵 + heavy randomization. 부족하면 Phase 6에서 RL fine-tune 추가.

---

## 2. 데이터셋 분포

| Trial 시나리오 | rollouts | 비고 |
|---|---|---|
| Trial 1 (NIC SFP) | 400 | 5개 NIC card 모두 균등 |
| Trial 2 (NIC SFP) | 400 | 다른 NIC slot |
| Trial 3 (SC) | 200 | sc_rail_0/1 모두 |
| Edge cases | 100 | 그립 오차 max, 카메라 noise heavy |
| **합계** | **1100** | |

각 rollout: 평균 ~10초 시연 × 20Hz observation = ~200 timestep.

---

## 3. LeRobot 데이터셋 스키마 (v2.1)

```
aic_work/datasets/aic_v0/
├── meta/
│   ├── info.json          # codebase_version="v2.1", fps=20, robot_type="ur5e"
│   ├── episodes.jsonl     # 한 줄/episode
│   ├── tasks.jsonl        # task_index ↔ description
│   └── stats.json         # mean/std for normalization
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet  # state, action 시계열
│       └── ...
└── videos/
    └── chunk-000/
        ├── observation.images.left/episode_000000.mp4
        ├── observation.images.center/episode_000000.mp4
        └── observation.images.right/episode_000000.mp4
```

### 3.1 feature 정의

```python
features = {
    "observation.images.left":   {"dtype": "video", "shape": (480, 640, 3), "names": ["height","width","channels"]},
    "observation.images.center": {"dtype": "video", "shape": (480, 640, 3), "names": ["height","width","channels"]},
    "observation.images.right":  {"dtype": "video", "shape": (480, 640, 3), "names": ["height","width","channels"]},
    "observation.state":         {"dtype": "float32", "shape": (12,), "names": ["q0","q1","q2","q3","q4","q5","gripper","fx","fy","fz","tx","ty"]},
    "observation.wrench":        {"dtype": "float32", "shape": (6,), "names": ["fx","fy","fz","tx","ty","tz"]},
    "action":                    {"dtype": "float32", "shape": (7,), "names": ["dx","dy","dz","drx","dry","drz","gripper"]},
    "action.absolute_pose":      {"dtype": "float32", "shape": (7,), "names": ["x","y","z","qx","qy","qz","qw"]},  # 디버그용
    "task_index":                {"dtype": "int64", "shape": (1,)},
    "episode_index":             {"dtype": "int64", "shape": (1,)},
    "frame_index":               {"dtype": "int64", "shape": (1,)},
    "timestamp":                 {"dtype": "float32", "shape": (1,)},
}
```

> 이미지를 1152×1024 원본으로 저장하면 디스크 폭발 (1100 ep × 200 frame × 3 cam ≈ 660K frame). 480×640으로 다운샘플 → 학습/추론도 동일 사이즈.

### 3.2 task description

```json
{"task_index": 0, "task": "Insert SFP plug into nic_card_0 sfp_port_0"}
{"task_index": 1, "task": "Insert SFP plug into nic_card_1 sfp_port_0"}
...
{"task_index": 5, "task": "Insert SC plug into sc_port_0"}
{"task_index": 6, "task": "Insert SC plug into sc_port_1"}
```

ACT는 task description text-conditioning을 지원 (LeRobot의 `language` feature).

---

## 4. 데이터 수집 파이프라인 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│ generate_episodes.py (orchestrator)                     │
│  for i in range(N):                                     │
│    1. 무작위 scene config 생성 → trial_${i}.yaml       │
│    2. docker compose up (eval + cheat_code_recorder)    │
│    3. 종료 후 bag → LeRobot parquet 변환                │
│    4. videos 인코딩 + checksum                          │
│    5. R2/HF Hub 업로드                                  │
└─────────────────────────────────────────────────────────┘
```

### 4.1 무작위 scene 생성기

`aic_work/data/randomize_scene.py`:

```python
"""sample_config.yaml을 베이스로 매번 다른 trial config를 생성."""
import argparse
import copy
import math
import random
from pathlib import Path
import yaml

def sample_nic_layout(rng):
    """5 rail 중 1~3개에 랜덤하게 NIC card 배치. 타겟 1개 선택."""
    n_present = rng.randint(1, 3)
    present_rails = rng.sample(range(5), n_present)
    target_rail = rng.choice(present_rails)
    out = {}
    for r in range(5):
        key = f"nic_rail_{r}"
        if r in present_rails:
            out[key] = {
                "entity_present": True,
                "entity_name": f"nic_card_{r}",
                "entity_pose": {
                    "translation": rng.uniform(-0.0215, 0.0234),
                    "roll": 0.0, "pitch": 0.0,
                    "yaw": rng.uniform(-0.1745, 0.1745),  # ±10°
                },
            }
        else:
            out[key] = {"entity_present": False}
    return out, target_rail

def sample_task_board_pose(rng, scenario):
    """Trial 1/2 vs Trial 3 base pose 다르게."""
    if scenario in ("sfp1", "sfp2"):
        x, y, yaw = 0.15, -0.2, math.pi
    else:
        x, y, yaw = 0.17, 0.0, 3.0
    return {
        "x": x + rng.uniform(-0.02, 0.02),
        "y": y + rng.uniform(-0.02, 0.02),
        "z": 1.14,
        "roll": rng.uniform(-0.05, 0.05),
        "pitch": rng.uniform(-0.05, 0.05),
        "yaw": yaw + rng.uniform(-0.1, 0.1),
    }

def sample_grip_offset(rng, plug_type):
    base_z = 0.04245 if plug_type == "sfp" else 0.04045
    return {
        "x": rng.uniform(-0.002, 0.002),  # ±2mm 그립 오차 시뮬
        "y": 0.015385 + rng.uniform(-0.002, 0.002),
        "z": base_z + rng.uniform(-0.002, 0.002),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["sfp1","sfp2","sc"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    base = yaml.safe_load(Path("/workspace/aic/aic_engine/config/sample_config.yaml").read_text())

    trial_key = {"sfp1":"trial_1","sfp2":"trial_2","sc":"trial_3"}[args.scenario]
    trial = copy.deepcopy(base["trials"][trial_key])
    trial["scene"]["task_board"]["pose"].update(sample_task_board_pose(rng, args.scenario))

    if args.scenario in ("sfp1","sfp2"):
        layout, target = sample_nic_layout(rng)
        trial["scene"]["task_board"].update(layout)
        trial["tasks"]["task_1"]["target_module_name"] = f"nic_card_mount_{target}"
        trial["scene"]["cables"]["cable_0"]["pose"]["gripper_offset"] = sample_grip_offset(rng, "sfp")
    else:
        trial["scene"]["cables"]["cable_1"]["pose"]["gripper_offset"] = sample_grip_offset(rng, "sc")

    out_cfg = {"scoring": base["scoring"], "robot": base["robot"], "trials": {trial_key: trial}}
    Path(args.out).write_text(yaml.dump(out_cfg, sort_keys=False))

if __name__ == "__main__":
    main()
```

### 4.2 CheatCode 데이터 레코더 (수정 정책)

CheatCode는 ground_truth TF로 정확한 plug-port 변환을 알기 때문에 직접 trajectory를 생성. 이 trajectory + observation을 rosbag2로 기록.

`aic_work/policies/cheat_code_recorder.py` (CheatCode 베이스로 hook 추가):

```python
"""CheatCode + rosbag2 record + LeRobot parquet 변환 hook."""
from aic_example_policies.ros.CheatCode import CheatCode
from rclpy.serialization import serialize_message
import rosbag2_py
import time

class CheatCodeRecorder(CheatCode):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        ts = int(time.time())
        self._writer = rosbag2_py.SequentialWriter()
        self._writer.open(
            rosbag2_py.StorageOptions(uri=f"/workspace/aic_work/raw_episodes/ep_{ts}", storage_id="mcap"),
            rosbag2_py.ConverterOptions("", "")
        )
        for tname, ttype in [
            ("/joint_states", "sensor_msgs/msg/JointState"),
            ("/fts_broadcaster/wrench", "geometry_msgs/msg/WrenchStamped"),
            ("/left_camera/image",  "sensor_msgs/msg/Image"),
            ("/center_camera/image","sensor_msgs/msg/Image"),
            ("/right_camera/image", "sensor_msgs/msg/Image"),
            ("/aic_controller/pose_commands", "aic_control_interfaces/msg/MotionUpdate"),
        ]:
            self._writer.create_topic(rosbag2_py.TopicMetadata(name=tname, type=ttype, serialization_format="cdr"))
        # 위는 토픽 중복 등록 방지를 위해 한 번만 - 실제 구현시 try/except
```

### 4.3 rosbag → LeRobot 변환

`aic_work/data/bag_to_lerobot.py`:
- mcap 읽기 → 시간 동기화 (가장 가까운 timestamp의 image+state+action)
- 20Hz로 리샘플 (이미지 캡처는 20FPS, state/action은 더 빠르므로 nearest)
- `lerobot.dataset.LeRobotDataset.create()`로 episode 추가
- 비디오는 ffmpeg `libx264 crf=23`으로 인코딩

```python
# 핵심 부분만
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset.create(
    repo_id="nggw519/aic_v0",
    fps=20,
    features=FEATURES,
    root="/workspace/aic_work/datasets/aic_v0",
)

for ep_id, bag_dir in enumerate(sorted(Path("raw_episodes").glob("ep_*"))):
    frames = align_frames(bag_dir)  # list of dicts at 20Hz
    for f in frames:
        ds.add_frame({
            "observation.images.left":  f["img_left"],
            "observation.images.center":f["img_center"],
            "observation.images.right": f["img_right"],
            "observation.state":        f["state"],
            "observation.wrench":       f["wrench"],
            "action":                   f["action_delta"],
            "task_index":               f["task_idx"],
        })
    ds.save_episode(task=task_str_for(ep_id))
ds.consolidate()
```

---

## 5. 도메인 랜덤화 (Phase 2 단계 — 가벼운 버전)

Phase 6에서 더 강력한 multi-sim DR을 적용하지만, Phase 2부터 다음은 적용:

| 변수 | 분포 | 근거 |
|---|---|---|
| Camera Gaussian noise σ | U(0, 5) (uint8) | 실제 Basler 노이즈 시뮬 |
| Camera salt&pepper | p=U(0, 0.005) | hot pixel |
| Image JPEG quality | U(70, 95) | 압축 손실 |
| Lighting intensity scale | U(0.7, 1.3) | Gazebo SDF light multiplier |
| F/T white noise σ | U(0.1, 0.5) N | sensor noise |
| F/T bias drift | U(-0.5, 0.5) N | warm-up drift |
| Joint encoder noise | U(0, 0.001) rad | encoder resolution |

이미지 noise는 학습 시 augmentation으로도 추가 (CPU에서 처리). Gazebo 측 SDF는 lighting만 수정.

---

## 6. 검증 (수집 데이터 sanity check)

```bash
pixi run python aic_work/scripts/check_dataset.py \
  --root aic_work/datasets/aic_v0 \
  --check_episodes 10
```

체크 항목:
- 각 ep frame 수 동일하게 정렬됨 (timestamp 정합)
- action ↔ state 차분이 0이 아님 (정지된 데이터 없음)
- 이미지 비어있지 않음 (mean > 5)
- task_index 분포 합산 = 비율 의도와 일치
- successful insertion 비율 ≥ 95% (CheatCode가 실패한 ep 제거)

실패 ep는 CSV에 기록 → 재수집 또는 제거.

---

## 7. 백업

```bash
# R2 업로드
rclone sync aic_work/datasets/aic_v0/ r2:aic-datasets/aic_v0/ --progress

# HuggingFace Hub (private)
hf upload nggw519/aic_v0 ./aic_work/datasets/aic_v0/ --repo-type dataset
```

R2: 빠른 vast.ai 풀용. HF Hub: 영구 백업 + 버전 관리.

---

## 8. 결정 로그

### 8.1 이미지 해상도: 480×640
- **고려**: 480×640 (학습 속도/메모리 ↑) vs 720×1280 (미세 정렬 유리)
- **결정**: 480×640
- **근거**: ALOHA/ACT 원본도 480×640 사용. 우리 24GB VRAM 에서 chunk_size=100 + 3 cam + batch=32 가 안전. ablation 으로 720×1280 비교는 Phase 7 까지 미룸.
- **트리거**: Phase 5 평가에서 Stage A detection mAP < 0.85 면 해상도 ↑ 고려.

### 8.2 Action space: Cartesian delta (xyz + rpy + grip)
- **고려**: Cartesian delta vs joint delta vs absolute pose
- **결정**: Cartesian velocity delta `[vx, vy, vz, wx, wy, wz, grip]` 7-D
- **근거**: 이전 시도 `scripts/rosbag_to_lerobot.py` 가 `ControllerState.tcp_velocity` 사용한 패턴 (RunACT.py:273 호환). Cartesian impedance controller 의 자연 입력. Phase 5 Stage C 의 admittance 와도 호환.
- **트리거**: Phase 5 에서 stage 전환 시 jerk 폭증 → joint delta 비교 ablation.

### 8.3 Trial 분포: SFP×0.36 + SFP×0.36 + SC×0.18 + edge×0.10
- **결정**: 4 시나리오 가중 (`DEFAULT_SCENARIO_DIST` in `collect_demos.py`)
- **근거**: 평가 trial 수 (SFP×2 + SC×1) = 2:1 비율. edge (grip ±3mm) 추가는 robustness 확보.
- **트리거**: Phase 5 에서 SC trial 만 평균 점수 ≥ 20pt 낮으면 SC 비율 ↑.

### 8.4 Episode timeout: 1800s (이전 600s 회귀 직접 대응)
- **결정**: `episode_timeout_s` default 1800. min_success_rate 0.8 abort gate.
- **근거**: Day 5 진단 — 이전 시도의 timeout 600s 가 한 ep 의 3 trial × 20분(CheatCode descent 느림)에 부족 → trajectory 잘림 → marginal-mean collapse. 1800s 면 충분 + abort gate 가 systemic 오류 시 즉시 stop.

### 8.5 검증 layer 5중 — sanity gate 코드화
- **결정**: validate_episode + motion_frame_ratio + min_success_rate + sample mp4 hook + action_amplify 옵션
- **근거**: 이전 시도가 모든 frame 을 학습에 넣어 정지 데이터로 collapse. 5중 검증 어느 하나라도 트립하면 자동 격리/중단.

### 8.6 백업: HuggingFace Hub 단독
- **결정**: ckpt+dataset 모두 `nggw519/aic-ckpts` + `nggw519/aic-datasets` private repo.
- **근거**: R2/S3 추가 자격증명 부담 vs HF 단독 단순성. 우리 작업 패턴 (작은 ckpt 자주 push, dataset 한 번 push)에서 속도 차이 결정적 아님 (2026-05-10 결정).

### 8.7 docker run 단일 호스트 패턴 (lifecycle 함정 회피)
- **결정**: `data/collect_demos.py` 의 docker compose → docker run 두 컨테이너 (같은 aic_eval image, host network) 로 변경 (D1).
- **근거**: 2026-05-11 lifecycle ACTIVATE 실패 → I3 분석에서 호스트 pixi env / dev image 분리 가 zenoh discovery 단절 원인. aic_eval image 통일 + host network 가 토킷 의도 환경과 가장 가깝다.

---

## 9. 완료 기준

- [ ] 1100+ episodes 수집
- [ ] LeRobot v2.1 dataset 생성 (`aic_work/datasets/aic_v0/`)
- [ ] R2 + HF Hub에 백업 완료
- [ ] sanity check 통과 (≥95% success)
- [ ] dataset card 작성 (HF 공개 페이지)
