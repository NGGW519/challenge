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

- **결정**: Stage 분리 vs end-to-end ACT — 분리 채택 (안전성/디버깅 용이)
- **결정**: Stage C는 결정론적 + admittance — 학습 안 함 (설명가능성)
- **결정**: 검출 실패 시 직접 force fallback — 점수 0보다 부분 25점 노림
- **결정**: 모든 명령은 Cartesian (모드 전환 없음) — 단순화

---

## 10. 완료 기준

- [ ] HybridPolicy.py 동작 (lifecycle 통과)
- [ ] aic_model_pkg colcon build 성공
- [ ] 30 rollout 평균 ≥ 230
- [ ] 충돌(-24) 0회 / force(-12) ≤ 1회
- [ ] 단일 trial 평균 시간 ≤ 30초
- [ ] R2/HF에 v1 weights 백업
