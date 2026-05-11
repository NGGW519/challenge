# 14. Phase 5 — Hybrid Policy 통합 (Stage A + B + C)

> **목표**: Phase 3 (ACT) + Phase 4 (Port Detector)에 force-guided spiral search를 더해, 한 컨테이너에서 동작하는 `aic_model` 정책을 완성.
>
> **기간**: 1.5~2일
>
> **결과물**:
> - `aic_work/policies/HybridPolicy.py` — `Policy` 상속 클래스
> - `aic_work/aic_model_pkg/` — colcon 빌드 가능한 ROS 2 패키지
> - 자체 평가 평균 점수 ≥ 230 / 300

---

## 1. 정책 인터페이스 (재확인)

`aic_model.policy.Policy` 상속:

```python
def insert_cable(self, task, get_observation, move_robot, send_feedback):
    """
    task: aic_task_interfaces/msg/Task
        - cable_type: 'sfp_sc'
        - plug_type: 'sfp' or 'sc'
        - port_type: 'sfp' or 'sc'
        - target_module_name: e.g. 'nic_card_mount_0' or 'sc_port_1'
        - port_name: e.g. 'sfp_port_0' or 'sc_port_base'
        - time_limit: 180 (seconds)
    get_observation() → Observation (20 Hz까지 호출 가능)
        - images.{left,center,right} + camera_info
        - joint_state, gripper_state, wrench, controller_state
    move_robot(MotionUpdate | JointMotionUpdate)
    send_feedback(str)
    """
```

---

## 2. 상태 머신 (3 단계 + safety supervisor)

```
                  ┌──────────────────────────────────────────────┐
                  │                  WATCHDOG                    │
                  │  - 시간 초과 감시 (각 stage timeout)         │
                  │  - F/T 한계 (>20N, >2Nm) 감시 → emergency    │
                  │  - off-limit contact 감시 (TF 거리 추정)     │
                  └──────────────────────────────────────────────┘
                                       │
        ┌───────────────┐  detect ok    │  detect fail
        │   STAGE A     │ ────────────► [STAGE B]
        │ Coarse Visual │                            ▲
        │ Approach      │ ─── fallback ─► [DIRECT_FORCE]  
        └───────────────┘                            │
        ┌───────────────┐  conv ok     │             │
        │   STAGE B     │ ────────────► [STAGE C] ◄──┘
        │ ACT IL        │
        │ Alignment     │ ── timeout ─► [STAGE C with prior]
        └───────────────┘
        ┌───────────────┐  insertion event
        │   STAGE C     │ ────────────► [SUCCESS — return]
        │ Compliant     │
        │ Insertion     │ ── force> ─► [BACK_OFF + retry spiral]
        └───────────────┘
```

### 2.1 stage timeout / 예산

| Stage | 시간 예산 | rationale |
|---|---|---|
| Stage A | 5초 | duration 점수 만점 (5초)을 위해 시각 정렬은 빠르게 |
| Stage B | 8초 | ACT alignment, 5+ chunk 추론 가능 |
| Stage C | 15초 | force-guided가 가장 비싼 부분 |
| Total ideal | ~28초 | duration 점수 ~10/12 |
| Hard timeout | 60초 | 그 이상은 0 duration |

---

## 3. Stage A — Coarse Visual Approach

### 3.1 알고리즘

```python
def stage_A(self, task, get_obs, move_robot):
    self.feedback("stage_A: detect target port")
    detector = self.detector  # PortDetector from Phase 4

    # 1. 5 frame 평균으로 안정된 3D 위치 추정
    positions = []
    for _ in range(5):
        obs = get_obs()
        dets = detector.detect(obs.images.left, obs.images.center, obs.images.right)
        p_world = detector.get_target_port_3d(
            dets, task,
            K_left=obs.camera_info.left.K, T_left=obs.camera_tf.left,
            K_right=obs.camera_info.right.K, T_right=obs.camera_tf.right
        )
        if p_world is not None:
            positions.append(p_world)
        time.sleep(0.05)
    if len(positions) < 3:
        return False  # → DIRECT_FORCE fallback
    p_target = np.median(np.stack(positions), axis=0)

    # 2. 진입점 위 3cm 지점 = approach pose
    plug_axis = self.estimate_plug_axis(task)  # gripper TCP frame에서 plug 끝 방향
    approach_offset = -0.03 * plug_axis        # plug 길이 방향으로 3cm 후진
    p_approach = p_target + approach_offset

    # 3. 명령 발행: Cartesian impedance, low stiffness
    cmd = self._make_motion_update(
        target_position=p_approach,
        target_orientation=self.compute_orientation(plug_axis),
        stiffness=np.diag([300,300,300, 30,30,30]),  # 부드럽게
        damping=np.diag([60,60,60, 6,6,6]),
        velocity_limit=0.15,  # 15 cm/s
        frame="base_link",
    )
    move_robot(cmd)

    # 4. 도달 확인 (controller_state에서 tracking error)
    deadline = time.time() + 5.0
    while time.time() < deadline:
        obs = get_obs()
        if obs.controller_state.tracking_error < 0.005:  # 5mm
            return True
        time.sleep(0.05)
    return False  # 시간 초과 → 그래도 다음 stage 진행
```

### 3.2 Safety bbox clamp

모든 stage에서 발행하는 모든 명령은 다음으로 clamp:

```python
SAFE_BOX_BASE = {
    "x": (0.0, 0.7),
    "y": (-0.4, 0.4),
    "z": (1.05, 1.45),  # task_board top ~1.14
}
```

`HybridPolicy._safe_clamp(pose)` 헬퍼로 모든 send 직전 호출.

---

## 4. Stage B — ACT IL Alignment

### 4.1 알고리즘

```python
def stage_B(self, task, get_obs, move_robot):
    self.feedback("stage_B: ACT alignment")
    self.act.reset()

    deadline = time.time() + 8.0
    while time.time() < deadline:
        obs = get_obs()
        act_obs = self._build_act_observation(obs, task)
        action = self.act.select_action(act_obs)  # (7,) Cartesian delta + grip
        cmd = self._delta_to_motion_update(action, base_pose=obs.tcp_pose)
        cmd = self._safe_clamp(cmd)
        move_robot(cmd)

        # convergence: TCP가 plug-port 정렬 직전 상태로 수렴
        if self._is_aligned(obs, task):
            return True
        time.sleep(0.05)
    return False
```

### 4.2 _is_aligned 판정

```python
def _is_aligned(self, obs, task):
    """plug 끝(TCP에서 z축으로 연장)이 port 진입점에서 ±3mm 이내, 
    plug 축이 port 노말과 ±3° 이내인지."""
    plug_tip_world = self.compute_plug_tip(obs)
    port_pose_world = self._cached_port_pose  # Stage A에서 갱신
    pos_err = np.linalg.norm(plug_tip_world.translation - port_pose_world.translation)
    ang_err = self._angle_between_axes(plug_tip_world.z_axis, port_pose_world.z_axis)
    return pos_err < 0.003 and ang_err < np.deg2rad(3.0)
```

### 4.3 ACT 추론 동안 Cartesian delta는 수령 즉시 발행

ACT의 chunk_size=100 → 5초어치를 한번에 예측. 처음 20step (1초) 실행 후 새 obs로 재추론. (`act_inference.py`의 step counter 사용.)

---

## 5. Stage C — Compliant Force-Guided Insertion

### 5.1 알고리즘 (spiral + admittance)

```python
def stage_C(self, task, get_obs, move_robot):
    self.feedback("stage_C: compliant insertion")

    # 1. 시작 시 F/T tare는 안 됨 → 초기 wrench를 baseline으로 기억
    obs0 = get_obs()
    f_baseline = np.array([obs0.wrench.fx, obs0.wrench.fy, obs0.wrench.fz])

    deadline = time.time() + 15.0
    insertion_depth = 0.0
    spiral_phase = 0.0
    spiral_radius = 0.0

    target_depth = self._target_insertion_depth(task)  # SFP: 8mm, SC: 6mm

    while time.time() < deadline:
        obs = get_obs()
        f = np.array([obs.wrench.fx, obs.wrench.fy, obs.wrench.fz]) - f_baseline

        # 2. force-guided lateral compliance (admittance)
        lat_correction = -0.0001 * f[:2]  # F_xy → 약한 lateral motion
        lat_correction = np.clip(lat_correction, -0.001, 0.001)  # ±1mm/step

        # 3. spiral search (XY plane perpendicular to plug axis)
        spiral_phase += 0.3  # rad/step
        spiral_radius += 0.0001  # 0.1mm/step ≤ 5mm 한계
        spiral_radius = min(spiral_radius, 0.005)
        spiral_xy = spiral_radius * np.array([np.cos(spiral_phase), np.sin(spiral_phase)])

        # 4. 진행 결정 — Fz가 작으면 forward, 크면 hold
        if abs(f[2]) < 8.0:
            forward = 0.0005  # 0.5mm/step
        elif abs(f[2]) < 15.0:
            forward = 0.0   # hold
        else:
            forward = -0.001  # 1mm 후퇴
            spiral_radius += 0.0005  # spiral 확장

        delta_tcp = np.array([
            lat_correction[0] + spiral_xy[0],
            lat_correction[1] + spiral_xy[1],
            forward,
        ])
        # plug axis 방향으로 forward 매핑
        delta_world = self._tcp_delta_to_world(delta_tcp, obs)
        new_pose = obs.tcp_pose.translation + delta_world

        cmd = self._make_motion_update(
            target_position=new_pose,
            target_orientation=obs.tcp_pose.orientation,  # 회전 고정
            stiffness=np.diag([800,800,1500, 80,80,150]),  # 삽입은 z stiffer, lateral compliant
            damping=np.diag([100,100,150, 12,12,18]),
            frame="base_link",
        )
        cmd = self._safe_clamp(cmd)
        move_robot(cmd)

        # 5. 삽입 깊이 추정 (TCP 진행)
        insertion_depth += forward
        if insertion_depth >= target_depth:
            self.feedback("inserted")
            return True

        time.sleep(0.05)  # 20Hz

    return False
```

### 5.2 force 한계 가드 (force penalty -12 회피)

`watchdog` 별도 thread:
- F_total > 18N (1초 평균) → emergency_back_off()
- 1.0초 동안 |F| > 20N 누적 시 -12점이므로 0.8초에서 컷.

```python
def _watchdog(self, get_obs):
    high_force_t0 = None
    while not self._stop:
        obs = get_obs()
        f = np.linalg.norm([obs.wrench.fx, obs.wrench.fy, obs.wrench.fz])
        if f > 18.0:
            if high_force_t0 is None: high_force_t0 = time.time()
            if time.time() - high_force_t0 > 0.7:
                self._emergency = True; return
        else:
            high_force_t0 = None
        time.sleep(0.05)
```

---

## 6. ROS 2 패키지 구조

```
aic_work/aic_model_pkg/
├── package.xml
├── setup.py
├── setup.cfg
├── resource/
│   └── aic_model_pkg
└── aic_model_pkg/
    ├── __init__.py
    ├── HybridPolicy.py        # main policy
    ├── port_detector.py       # Phase 4
    ├── act_inference.py       # Phase 3
    ├── safety.py              # bbox clamp, watchdog
    ├── geometry.py            # TF utils, plug axis, triangulation
    ├── stage_a.py
    ├── stage_b.py
    └── stage_c.py
```

`setup.py`:
```python
from setuptools import setup, find_packages
package_name = "aic_model_pkg"
setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={"console_scripts": []},
)
```

빌드:
```bash
cd /workspace/aic
pixi run -e dev colcon build --packages-select aic_model_pkg --symlink-install
source install/setup.bash
ros2 run aic_model aic_model -p policy:=aic_model_pkg.HybridPolicy
```

---

## 7. 모델 weight 패키징

```
aic_work/aic_model_pkg/aic_model_pkg/weights/
├── act_v1.pt              # Phase 3 결과 (압축 후 ~150MB)
├── port_detector_v1.pt    # Phase 4 결과 (~25MB)
└── stats.json             # observation/action 정규화 통계
```

`HybridPolicy.__init__` 시 weight 로드 (모델 발견 30초 timeout 안에 들어와야 → lazy load 권장).

---

## 8. 자체 평가 (Phase 5 단계)

```bash
bash scripts/run_eval.sh HybridPolicy 30   # 30 rollouts × 3 trial
```

**기대 점수**: 평균 230~260 / 300 (Phase 6 DR 적용 전 baseline).
- Tier 1: 1
- Tier 2: 4 (smoothness) + 8 (duration) + 4 (efficiency) ≈ 16
- Tier 3: 75 in ~80% trials → 평균 60

미달 시:
- Stage A 검출 실패 → Phase 4 재학습 / 평면 fitting
- Stage B 정렬 실패 → Phase 3 데이터 늘리기
- Stage C 삽입 실패 → spiral 파라미터 튜닝, target_depth 재측정

---

## 9. 결정 로그

### 9.1 구조 결정 (Phase 5 시작 시)

- **결정**: Stage 분리 vs end-to-end ACT — 분리 채택 (안전성/디버깅 용이)
- **결정**: Stage C는 결정론적 + admittance — 학습 안 함 (설명가능성)
- **결정**: 검출 실패 시 직접 force fallback — 점수 0보다 부분 25점 노림
- **결정**: 모든 명령은 Cartesian (모드 전환 없음) — 단순화

### 9.2 SOTA 검증 결과 반영 (2026-05-11 A8 Researcher 11편 논문 비교)

검증 출처: `MEMORY.md` → `project_design_decisions.md`. 11편 비교 — ACT, IndustReal, ResiP, TacDiffusion, Comp-ACT, FARM, Insertion-Net, RFT-Insertion, USB-Insert, Sim-to-Real Cable, PEG-IN-HOLE compliance.

**고정한 디자인 (이미 코드 반영):**

| 결정 | 근거 | 코드 위치 |
|---|---|---|
| 3-stage hybrid (A: YOLOv8m + 스테레오, B: LeRobot ACT chunk=100/n_action=20/ResNet18 + 3 wrist + 12-D state, C: 결정론 spiral + admittance 20 Hz) | ACT 단독은 force-rich 마지막 5mm 약함 (TacDiffusion §4.2). 우리 데이터/시간 예산엔 chunk=100 + admittance 가 ROI 최대. | `HybridPolicy.{_stage_a, _stage_b, _stage_c}` |
| **B→C 전환 hysteresis** = `|F| > 5N AND z_dist ≤ 10mm AND 200ms hold` | ResiP/TacDiffusion 모두 명시적 전환 없으면 force buildup → −12N 페널티. | `HybridPolicy._should_transition_to_c`, `STAGE_TRANSITION_*` 상수 |
| port detector 보수적 임계 (conf ≥ 0.85, epipolar y ≤ 8 px), 불일치 시 None | tier-3 잘못된 포트 −12 vs Stage A fallback 25 점 (검출 실패해도 부분 보상) | `port_detector.PortDetector` |
| `CheatCodeRecorder` 가 `reference_profile` (approach/alignment/insertion → stiffness + desired_wrench 6-D) 를 task.yaml 에 기록 | Comp-ACT 13-D head 로 갈 때 demo 재수집 불필요 | `CheatCodeRecorder.py`, `validate_episode.py` |
| ACTPlus 패턴: EMA(α=0.4) + RAW_ACTION_SCALE=4.0 + 동적 time budget (0.85 × task.time_limit) + LOCAL_DIR > arg > HF fallback | 이전 시도에서 jerk 점수 만점 + marginal-mean collapse 회피 검증 | `motion.EmaSmoother`, `act_inference.resolve_ckpt_path`, `HybridPolicy._allocate_budget` |
| ACTPlus 패턴 v2 (2026-05-12 Q1): `ForceAttenuator` (12N × 0.4s sustained → ×0.35) + `InsertionStabilityDetector` (1.2s stable contact 조기 종료) | hard watchdog 보다 먼저 발동해 −12N 페널티 진입 자체 회피, duration score 보존 | `safety.{ForceAttenuator, InsertionStabilityDetector}`, `_stage_c` |

**조건부 업그레이드 트리거 (선제 적용 금지):**

| 옵션 | 트리거 측정값 | 난이도 | 기대 +pts |
|---|---|---|---|
| Comp-ACT head (action 7-D → 13-D: stiffness + desired_wrench 출력) | Phase 5 평가에서 trial-avg contact penalty ≥ −10 **OR** force penalty ≥ −6 | 3 | +20~30 |
| ResiP residual RL (Stage B freeze + PPO/SAC residual on last 5mm) | Phase 5 Stage C reach-rate < 80% **OR** mean spiral time ≥ 6s | 4 | +5~15 |
| IsaacLab supplementary training (Gazebo + IsaacLab + MuJoCo mix) | Phase 7 inter-trial variance σ ≥ 15 pts **OR** edge-case (grip ±3mm) success < 60% | 5 | +10 |

**열린 리스크 (현재 미조치, 측정 후 결정):**

- chunk_size=100 의 20 FPS contact-rich window 적합성 — Phase 3 v1 학습 후 ablation
- CheatCode demo jerk 천장 — 수집 후 측정, 50 m/s³ peak 초과 시 5차 trajectory 스무딩 추가
- 학습용 Gazebo vs 평가용 Gazebo 플러그인/솔버 버전 drift — Phase 6 에서 F/T noise σ 를 0.3~1.5 N 으로 (현재 0.1~0.5)

### 9.3 재호출 기준

위 SOTA-검증된 디자인은 Phase 1/3/5/7 평가가 예상 벗어난 패턴 (예: 단일 trial-avg 60 미만) 을 낼 때만 A8 Researcher 재호출. 그 외엔 측정 → 트리거 매칭 → 옵션 적용 순.

---

## 10. 완료 기준

- [ ] HybridPolicy.py 동작 (lifecycle 통과)
- [ ] aic_model_pkg colcon build 성공
- [ ] 30 rollout 평균 ≥ 230
- [ ] 충돌(-24) 0회 / force(-12) ≤ 1회
- [ ] 단일 trial 평균 시간 ≤ 30초
- [ ] R2/HF에 v1 weights 백업

---

## 11. 디버그 노트 — zenoh / ACL / lifecycle (2026-05-11)

### 11.1 호스트 pixi env 에서 lifecycle ACTIVATE 가 막힌 이유

증상:
```
[aic_model] on_configure(...)  ← OK
[aic_model] Policy.__init__()  ← OK
[ERROR] aic_model lifecycle is not in the active state
[WARN] No transition matching 4 found for current state unconfigured
```

원인 (`aic/docker/aic_eval/Dockerfile`, `aic_model/Dockerfile`, `aic_zenoh_config.json5` 정독 결과):

1. **토킷 평가 컨테이너의 zenoh router 는 컨테이너 호스트네임 `eval:7447` 로 listen.**
   `--network host` 로 띄우면 `localhost:7447` 로 노출되지만, 호스트 pixi env 에서 ROS 2 노드를 띄울 때 다음 두 env 가 누락돼 있었다.
   - `ZENOH_CONFIG_OVERRIDE='connect/endpoints=["tcp/localhost:7447"];transport/shared_memory/enabled=false'`
   - `ZENOH_SESSION_CONFIG_URI=/aic_zenoh_config.json5` (image 안 경로)
2. `aic_zenoh_config.json5` 의 `access_control.default_permission = "allow"` 라 ACL 자체는 막지 않지만, `transport/shared_memory/enabled=true` 가 호스트 ↔ 컨테이너 SHM 호환 안 되어 transport 가 깨질 수 있음.
3. router 와 model session 사이 RMW QoS 가 어긋나면 lifecycle service (`/aic_model/change_state`) 가 등록되지 않아 외부에서 transition 4 (ACTIVATE) 요청이 도달하지 않음.

### 11.2 해결책

**model 도 토킷 `aic_eval` image 안에서 실행.** 같은 image 의 `/ws_aic/install` 에 aic_model 노드가 빌드돼 있으므로 별도 빌드 불필요. 두 경로 모두 수정 완료:

- `scripts/run_baseline_local.sh` — host network 단일 인스턴스 패턴
- `docker/baseline-override.yaml`, `docker/collect-override.yaml` — compose 경로

핵심 env (제출 컨테이너도 동일):
```
RMW_IMPLEMENTATION=rmw_zenoh_cpp
ZENOH_ROUTER_CHECK_ATTEMPTS=-1
ZENOH_CONFIG_OVERRIDE=connect/endpoints=["tcp/<router>:7447"];transport/shared_memory/enabled=false
ZENOH_SESSION_CONFIG_URI=/aic_zenoh_config.json5
AIC_ROUTER_ADDR=<router>:7447
```

### 11.3 ACL on/off

토킷 entrypoint (`should_enable_acl`) 는 `AIC_ENABLE_ACL=true|1` 일 때만 ACL 활성. 그 외엔 PASSWD 환경변수가 있어도 무시. 자체 검증은 ACL off, 제출은 토킷이 ACL on 으로 띄우므로 `AIC_MODEL_PASSWD` 가 채워져야 함 (`audit_pre_submit.sh` Q3 검사 추가됨).

### 11.4 60초 lifecycle 예산

`aic_zenoh_config.json5` 의 `connect.timeout_ms.peer=-1`, `transport.unicast.open_timeout=60000`. 즉 model 컨테이너 entrypoint 가 시작된 뒤 60초 안에 `/aic_model` 노드가 `configure → active` 로 가야 평가가 진행. weight lazy load + `on_configure` 안에서 heavy I/O 금지.
