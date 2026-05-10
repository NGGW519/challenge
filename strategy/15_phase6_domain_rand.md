# 15. Phase 6 — 도메인 랜덤화 (Sim-to-Sim Robust)

> **목표**: 평가 환경 (Gazebo)에서 sim-to-sim gap을 줄이기 위해 IsaacLab + MuJoCo + Gazebo 3종 시뮬레이터에서 데이터 수집/학습.
>
> **기간**: 1.5~2일 (vast.ai A100 80GB 1대 또는 L40S ×2 병렬)
>
> **결과물**: `aic_work/models/act_v2/`, `aic_work/models/port_detector_v2/`, dataset `aic_v1_dr/`.

---

## 1. 왜 multi-sim?

학습은 한 시뮬에서 했지만 평가는 다른 시뮬일 수 있다. 또한:
- **Gazebo**: 공식 평가. 그러나 RTF가 낮아 데이터 수집 느림.
- **IsaacLab**: GPU에서 1000× wall clock 빠름. NVIDIA가 권장. Photorealistic rendering.
- **MuJoCo**: contact dynamics가 가장 정확. F/T 기반 학습에 유리.

전략:
- **데이터 수집**: IsaacLab + MuJoCo에서 대량 수집 (각 5000+ rollouts)
- **fine-tune**: Gazebo에서 1000 rollouts로 마지막 fine-tune
- **평가**: Gazebo

---

## 2. 시뮬레이터별 역할

| 시뮬레이터 | 용도 | 강점 | 약점 |
|---|---|---|---|
| **Gazebo** | 평가 / 최종 fine-tune | 공식 환경 | RTF 0.3~0.7 (느림) |
| **IsaacLab** | 대량 vision 데이터 / 도메인 랜덤화 | GPU 병렬 (1024 envs), photorealistic | URDF→USD 변환 필요 |
| **MuJoCo** | contact-rich 학습 / F/T | 정확한 contact, 빠름 | 렌더링 비포토 |

### 2.1 IsaacLab 셋업

```bash
# vast.ai A100 인스턴스
cd /workspace
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install

# 우리 task를 IsaacLab task로 등록
mkdir -p source/aic_isaac
# URDF → USD 변환
./isaaclab.sh -p scripts/tools/convert_urdf.py \
  /workspace/aic/aic_description/urdf/ur_gz.urdf.xacro \
  /workspace/aic_work/sim/usd/ur5e.usd
```

`aic/aic_utils/aic_isaac/` 디렉토리가 토킷에 이미 있으므로 시작점으로 활용 (BSD-3 라이선스).

### 2.2 MuJoCo 셋업

```bash
pixi add mujoco
# MJCF (XML) 변환
pixi run python /workspace/aic_work/sim/urdf2mjcf.py \
  --urdf /workspace/aic/aic_description/urdf/ur_gz.urdf.xacro \
  --out /workspace/aic_work/sim/mjcf/aic_scene.xml
```

MuJoCo는 자체 MJCF를 쓰므로 변환 필요. 케이블 같은 deformable은 MuJoCo composite로 근사.

---

## 3. 도메인 랜덤화 변수 (확장판)

| 카테고리 | 변수 | 분포 | sim 별 적용 |
|---|---|---|---|
| **Vision** | HDRI lighting | 100+ HDRI 무작위 | Isaac, Gazebo |
| | Light intensity | U(0.5, 1.5) | All |
| | Light color temp | U(3000K, 7000K) | Isaac, Gazebo |
| | Camera focal length | ±5% | All (camera_info에 반영) |
| | Camera principal point | ±2px | All |
| | Camera Gaussian noise σ | U(0, 7) | All (post-render) |
| | Camera salt&pepper | p=U(0, 0.005) | All |
| | Image JPEG quality | U(60, 95) | All |
| | Texture (NIC card) | 30+ texture random | Isaac (most), Gazebo |
| | Background clutter | random objects | Isaac |
| | Camera position offset | ±1cm | All (TF 변경) |
| **Physics** | Gripper offset | ±2mm 위치, ±0.04 rad 회전 | All |
| | Plug-port friction μ | U(0.1, 0.5) | MuJoCo, Isaac |
| | Plug compliance | spring k = U(500, 5000) | MuJoCo |
| | Robot mass scaling | ±5% | All |
| **Sensing** | Joint encoder noise σ | U(0, 0.001) rad | All |
| | F/T bias | U(-0.5, 0.5) N | All |
| | F/T white noise σ | U(0.1, 1.0) N | All |
| | Sensor latency | 0~50ms 지연 | All |
| **Scene** | Task board pose | ±2cm 위치, ±0.1 rad | All |
| | NIC card layout | 5 rail, 1~3 present | All |
| | NIC yaw | ±10° | All |
| | Distractor objects | 0~3개 | Isaac |

---

## 4. 데이터 수집 페이즈 (확장)

| 데이터셋 | 시뮬 | 규모 | 용도 |
|---|---|---|---|
| `aic_v0` (Phase 2) | Gazebo | 1100 ep | baseline |
| `aic_v1_isaac` | IsaacLab | 5000 ep | vision diversity |
| `aic_v1_mujoco` | MuJoCo | 3000 ep | contact diversity |
| `aic_v1_gazebo_ft` | Gazebo | 500 ep | 최종 fine-tune |
| **합계** | | **~9600 ep** | |

각 데이터셋은 동일한 LeRobot v2.1 스키마. action space는 모두 Cartesian delta로 통일.

### 4.1 IsaacLab CheatCode 등가물

IsaacLab에서는 프로그램적으로 trajectory 생성. plug-port의 ground truth 위치를 직접 알므로 minimum-jerk trajectory를 계산:

```python
# aic_work/sim/isaac_collect.py (개념)
from omni.isaac.lab.envs import ManagerBasedRLEnv
import torch

class AICDataCollect(ManagerBasedRLEnv):
    def step(self):
        for env_id in range(self.num_envs):
            target_port = self.scene.target_ports[env_id]  # 3D pose
            tcp_pose = self.scene.tcp_poses[env_id]
            traj = self.minimum_jerk(tcp_pose, target_port, T=10.0)
            self.command_buffer[env_id] = traj.next()
        # 기록
        for env_id in range(self.num_envs):
            self.recorder.append(env_id, self.observations[env_id], self.actions[env_id])
```

1024 envs 병렬 → 5000 ep는 ~5분 wallclock.

### 4.2 MuJoCo 등가물

```python
# aic_work/sim/mujoco_collect.py
import mujoco
import numpy as np
mj_model = mujoco.MjModel.from_xml_path("aic_scene.xml")
mj_data = mujoco.MjData(mj_model)
# inverse kinematics + spline interpolation
```

---

## 5. 학습: domain-mixed batch

```python
# aic_work/training/train_act_v2.py
ds_isaac  = LeRobotDataset("nggw519/aic_v1_isaac",  root=...)
ds_mujoco = LeRobotDataset("nggw519/aic_v1_mujoco", root=...)
ds_gazebo = LeRobotDataset("nggw519/aic_v0",        root=...)

mix = MultiLeRobotDataset(
    {"isaac": ds_isaac, "mujoco": ds_mujoco, "gazebo": ds_gazebo},
    weights={"isaac": 0.5, "mujoco": 0.3, "gazebo": 0.2},
)
loader = DataLoader(mix, batch_size=64, shuffle=True, ...)
```

각 sim은 lighting/texture/contact가 다르므로 batch 안에서 자연스럽게 augmentation 효과.

### 5.1 fine-tune 단계

1. **Stage 1**: Isaac+MuJoCo만으로 학습 — 200k step (vision/dynamics 다양성)
2. **Stage 2**: Gazebo만으로 fine-tune — 30k step (실제 평가 환경 적응)

```bash
# Stage 1
pixi run python training/train_act_v2.py \
  --datasets isaac,mujoco \
  --steps 200000 \
  --output models/act_v2_pretrain
# Stage 2
pixi run python training/train_act_v2.py \
  --resume_from models/act_v2_pretrain \
  --datasets gazebo \
  --steps 30000 \
  --lr 5e-5 \
  --output models/act_v2_final
```

### 5.2 Port detector 재학습

multi-sim 합성 데이터로 port detector도 재학습 (vision diversity가 detector에 더 중요):

```bash
pixi run yolo train \
  model=models/port_detector_v1/run0/weights/best.pt \
  data=training/yolov8_port_v2.yaml \   # isaac+mujoco rendered + gazebo
  imgsz=1024 epochs=50 batch=32 \
  project=models/port_detector_v2
```

---

## 6. 평가 (DR 효과 검증)

3개 데이터셋으로 학습한 v2 모델 vs Gazebo only v1:

| 모델 | Gazebo eval (in-distribution) | Gazebo eval (DR scenes) |
|---|---|---|
| v1 (baseline) | 250 | 200 (-50, gap 50) |
| **v2 (multi-sim)** | **245** | **240 (gap 5)** |

DR이 in-dist에서 약간 손해 보지만 OOD에서 큰 이득. 평가 환경 자체가 약간의 randomization을 가질 가능성이 높으므로 v2가 안전.

---

## 7. 인스턴스 토폴로지

| 노드 | 작업 | 동시 진행? |
|---|---|---|
| A100 80GB | Isaac 데이터 수집 + 학습 | 직렬 (수집→학습) |
| L40S 48GB ×2 | MuJoCo 수집 (1) + Gazebo 수집 (2) | 병렬 |
| L4 24GB | 평가 전용 | 학습과 병렬 |

병렬 vast.ai 인스턴스를 띄워 작업을 쪼개되, 모두 동일 R2 버킷에 push → 학습 인스턴스에서 통합.

---

## 8. 주의사항

- **inter-sim normalization**: 각 sim의 image mean/std가 다를 수 있음. dataset stats 통합 시 weighted average 사용.
- **F/T 단위 일치**: 모든 sim에서 N, Nm 단위 일치 확인. 토크 부호 관례도 동일.
- **timestamp**: 각 sim의 시간 빈도 다름 → resample to 20Hz 일관.
- **action space 통일**: Cartesian delta in TCP frame (모든 sim 동일 frame).

---

## 9. 결정 로그

- **결정**: 3-sim mix 비율 = 50/30/20 (isaac/mujoco/gazebo)
- **결정**: pretrain → fine-tune 2 stage
- **결정**: detector도 multi-sim — vision이 OOD에 더 민감
- **결정**: A100 1대 + L40S 2대 분산

---

## 10. 완료 기준

- [ ] 3개 sim 데이터셋 (각 ≥ 사이즈 목표) 수집 완료
- [ ] act_v2 학습 (200k + 30k step) 완료
- [ ] port_detector_v2 mAP@0.5 ≥ 0.93 on Gazebo eval
- [ ] HybridPolicy + v2 weights로 Gazebo 평가 평균 ≥ 250
- [ ] DR scene 평가에서 v1 대비 회귀 < 5점
