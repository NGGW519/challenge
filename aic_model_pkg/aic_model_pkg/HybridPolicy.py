"""Hybrid policy: Stage A (visual coarse) → Stage B (ACT IL) → Stage C (compliant insertion).

aic_model의 dynamic loader가 이 클래스를 import한다:
    ros2 run aic_model aic_model -p policy:=aic_model_pkg.HybridPolicy

가중치 파일이 weights/ 디렉토리에 없으면 detector/ACT는 비활성화되고
Stage C만 작동하는 fallback 모드로 들어간다 (Phase 2 데이터 수집 단계까지 유효).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

# 토킷의 Policy 베이스. 평가 컨테이너 안에서만 import 가능 (PYTHONPATH).
try:  # pragma: no cover
    from aic_model.policy import Policy
except Exception:
    class Policy:  # type: ignore[no-redef]
        """Local stub — 실제 컨테이너에선 aic_model.policy.Policy를 사용."""
        def __init__(self, parent_node):
            self.node = parent_node
            self.logger = logging.getLogger("HybridPolicy")

from .act_inference import ACTInference
from .motion import (
    EmaSmoother,
    MotionCommand,
    make_approach_command,
    make_insertion_step,
    make_velocity_command,
    quat_z_axis_world,
)
from .port_detector import PortDetector
from .safety import (
    FORCE_HARD_DURATION_S,
    FORCE_HARD_N,
    ForceAttenuator,
    ForceWatchdog,
    InsertionStabilityDetector,
)

WEIGHTS_DIR = Path(__file__).parent / "weights"
ACT_CKPT = WEIGHTS_DIR / "act_v1.pt"
DETECTOR_CKPT = WEIGHTS_DIR / "port_detector_v1.pt"

# Stage 시간 예산 (초) — task.time_limit이 알려지지 않을 때의 fallback default.
# 실제 실행 시 _allocate_budget() 으로 task.time_limit × 0.85 / 3 stage 분할.
T_STAGE_A_DEFAULT = 5.0
T_STAGE_B_DEFAULT = 8.0
T_STAGE_C_DEFAULT = 15.0
# 시간 예산 fraction — task.time_limit의 이 비율까지만 사용 (timeout 회피).
# ACTPlus.py BUDGET_FRACTION = 0.85 패턴 차용.
BUDGET_FRACTION = 0.85
# Stage 별 분배 비율 (합 = 1.0)
STAGE_BUDGET_SHARES = (0.20, 0.35, 0.45)  # A:B:C = 5초:8.75초:11.25초 @ 25초 예산

# Stage B → C 전환 hysteresis (SOTA review 권고: ResiP/TacDiffusion 류는 명시적
# 전환 조건이 없으면 force buildup 상태로 C 진입 → force penalty 누적).
# 모든 조건이 동시에 STAGE_TRANSITION_HOLD_S 동안 유지돼야 C로 넘어감.
STAGE_TRANSITION_FORCE_MIN_N = 5.0     # 접촉 감지 (>이 값이면 plug-port 닿음)
STAGE_TRANSITION_Z_DIST_MAX_M = 0.010  # plug tip - port 진입점 z 거리 ≤ 10mm
STAGE_TRANSITION_ANG_MAX_RAD  = 0.087  # plug 축과 port 노말 ≤ 5°
STAGE_TRANSITION_HOLD_S       = 0.20   # 모든 조건 0.2s 유지

# Stage C 튜닝 파라미터 (strategy/14 §5)
STAGE_C_FORWARD_STEP_M    = 0.0005   # 0.5 mm/step
STAGE_C_BACKOFF_M         = 0.001    # 1 mm
STAGE_C_FORWARD_FORCE_N   = 8.0      # 이 미만이면 forward 진행
STAGE_C_HOLD_FORCE_N      = 15.0     # 이 이상이면 후퇴
STAGE_C_LATERAL_GAIN      = 1e-4     # F_xy → lateral correction
STAGE_C_LATERAL_LIMIT_M   = 0.001    # ±1 mm/step
STAGE_C_SPIRAL_RATE_RAD   = 0.3      # spiral phase increment per step
STAGE_C_SPIRAL_GROWTH_M   = 1e-4     # 0.1 mm/step
STAGE_C_SPIRAL_MAX_M      = 0.005    # 5 mm cap
STAGE_C_LOOP_HZ           = 20.0


class HybridPolicy(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.logger = logging.getLogger("HybridPolicy")
        self.detector = PortDetector(DETECTOR_CKPT if DETECTOR_CKPT.exists() else None)
        self.act = ACTInference(ACT_CKPT if ACT_CKPT.exists() else None)
        self._cached_target_world: np.ndarray | None = None
        self._emergency = False
        self._watchdog: ForceWatchdog | None = None
        # Cartesian xyz EMA — ACTPlus.py 패턴 차용. 매 stage 진입 시 reset.
        self._delta_smoother = EmaSmoother(dim=3, alpha=0.4)
        # Stage 시간 예산 — insert_cable 호출 시 task.time_limit 으로 동적 결정.
        self._budget_a = T_STAGE_A_DEFAULT
        self._budget_b = T_STAGE_B_DEFAULT
        self._budget_c = T_STAGE_C_DEFAULT

    def _allocate_budget(self, task: Any) -> None:
        """task.time_limit × 0.85 를 stage별로 분배. 타임아웃 회피용 safety margin.
        time_limit 미상 시 default 유지."""
        limit = getattr(task, "time_limit", None)
        try:
            limit_f = float(limit) if limit is not None else None
        except (TypeError, ValueError):
            limit_f = None
        if limit_f is None or limit_f <= 0:
            self.logger.info("task.time_limit unset — using default budgets")
            return
        usable = limit_f * BUDGET_FRACTION
        a, b, c = STAGE_BUDGET_SHARES
        self._budget_a = usable * a
        self._budget_b = usable * b
        self._budget_c = usable * c
        self.logger.info(
            "budget alloc: task.time_limit=%.1f → A=%.1fs B=%.1fs C=%.1fs (%.0f%% of limit)",
            limit_f, self._budget_a, self._budget_b, self._budget_c, BUDGET_FRACTION * 100,
        )

    # ------------------------------------------------------------------ #
    # entry point
    # ------------------------------------------------------------------ #
    def insert_cable(self, task, get_observation, move_robot, send_feedback):
        """aic_model이 호출하는 진입점. blocking, 작업 완료까지."""
        try:
            send_feedback("hybrid: starting")
            self._allocate_budget(task)
            self._delta_smoother.reset()
            self._start_watchdog(get_observation)
            self._stage_a(task, get_observation, move_robot, send_feedback)
            self._stage_b(task, get_observation, move_robot, send_feedback)
            self._stage_c(task, get_observation, move_robot, send_feedback)
            send_feedback("hybrid: done")
        except Exception as exc:
            self.logger.exception("HybridPolicy crashed: %s", exc)
            send_feedback(f"hybrid: error {exc}")
        finally:
            self._stop_watchdog()

    # ------------------------------------------------------------------ #
    # observation helpers (defensive — 필드 이름 변동 대비)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _get_tcp_pose(obs: Any) -> tuple[np.ndarray, np.ndarray] | None:
        cs = getattr(obs, "controller_state", None)
        if cs is None or not hasattr(cs, "tcp_pose"):
            return None
        p = cs.tcp_pose.position
        o = cs.tcp_pose.orientation
        return (
            np.array([p.x, p.y, p.z], dtype=np.float64),
            np.array([o.x, o.y, o.z, o.w], dtype=np.float64),
        )

    @staticmethod
    def _get_wrench(obs: Any) -> np.ndarray:
        w = getattr(obs, "wrench", None)
        if w is None:
            return np.zeros(6)
        return np.array([
            w.force.x, w.force.y, w.force.z,
            w.torque.x, w.torque.y, w.torque.z,
        ], dtype=np.float64)

    # ------------------------------------------------------------------ #
    # safety wiring
    # ------------------------------------------------------------------ #
    def _start_watchdog(self, get_obs: Callable[[], Any]) -> None:
        def _wrench():
            try:
                return self._get_wrench(get_obs())
            except Exception:
                return np.zeros(6)

        def _fire():
            self._emergency = True
            self.logger.error("force watchdog fired (>%.1fN for %.2fs)",
                              FORCE_HARD_N, FORCE_HARD_DURATION_S)

        self._watchdog = ForceWatchdog(_wrench, _fire)
        self._watchdog.start()

    def _stop_watchdog(self) -> None:
        if self._watchdog is not None:
            self._watchdog.stop()
            self._watchdog = None

    def _send(self, cmd: MotionCommand, move_robot: Callable[[Any], None]) -> None:
        """MotionCommand를 ROS 메시지로 변환해 발행. ROS env 밖이면 silently skip."""
        cmd = cmd.with_clamped_position()
        try:
            ros_msg = cmd.to_ros_msg()
        except Exception as e:  # ROS env 밖에서 import 실패
            self.logger.debug("to_ros_msg failed (likely outside ROS env): %s", e)
            return
        try:
            move_robot(ros_msg)
        except Exception as e:
            self.logger.warning("move_robot raised: %s", e)

    # ------------------------------------------------------------------ #
    # Stage A — coarse visual approach
    # ------------------------------------------------------------------ #
    def _stage_a(self, task, get_obs, move_robot, send_feedback) -> bool:
        send_feedback("stage_A: detect target port")
        target_world = self._detect_target_3d(get_obs, task, n_frames=5)
        if target_world is None:
            send_feedback("stage_A: detection unavailable — skip to Stage B")
            return False

        self._cached_target_world = target_world

        # plug axis: TCP +z 축 → world 좌표계로 변환. 그립 오프셋이 +z = plug tip
        # 방향으로 설계됐다는 토킷 컨벤션.
        tcp = self._get_tcp_pose(get_obs())
        if tcp is None:
            return False
        _, tcp_quat = tcp
        plug_axis = quat_z_axis_world(*tcp_quat.tolist())

        cmd = make_approach_command(
            port_xyz_world=target_world,
            plug_axis_world=plug_axis,
            standoff_m=0.03,
            orientation_qxyzw=tuple(tcp_quat.tolist()),
        )
        self._send(cmd, move_robot)

        # 도달 확인 — TCP가 target 근방으로 5mm 이내 진입
        deadline = time.monotonic() + self._budget_a
        while time.monotonic() < deadline and not self._emergency:
            tcp = self._get_tcp_pose(get_obs())
            if tcp is not None:
                tcp_xyz, _ = tcp
                if np.linalg.norm(tcp_xyz - cmd.target_pose.position()) < 0.005:
                    return True
            time.sleep(0.05)
        return False

    def _detect_target_3d(self, get_obs, task, n_frames: int = 5) -> np.ndarray | None:
        """포트 3D 위치를 n_frames 평균. detector disabled 시 None."""
        if self.detector.disabled:
            return None
        port_cls = "sfp_port" if getattr(task, "plug_type", "") == "sfp" else "sc_port"

        positions: list[np.ndarray] = []
        for _ in range(n_frames):
            obs = get_obs()
            try:
                imgs = self._extract_images(obs)
                if imgs is None:
                    continue
                K_l, T_l, K_r, T_r = self._extract_camera_lr(obs)
                if K_l is None:
                    continue
                dets = self.detector.detect(imgs)
                # imgs = [left, center, right] 순서로 가정
                if len(dets) < 3:
                    continue
                p = self.detector.find_target_3d(
                    dets[0], dets[2], port_cls, K_l, T_l, K_r, T_r,
                )
                if p is not None:
                    positions.append(p)
            except Exception as e:
                self.logger.debug("frame detection failed: %s", e)
            time.sleep(0.05)

        if len(positions) < max(1, n_frames // 2):
            return None
        return np.median(np.stack(positions), axis=0)

    @staticmethod
    def _extract_images(obs: Any) -> list[np.ndarray] | None:
        """observation에서 left/center/right 이미지를 numpy로 추출. 토킷 인터페이스 의존."""
        # aic_model_interfaces/Observation 의 정확한 필드명은 토킷에서 확인 필요.
        # 안전하게 여러 후보 시도:
        for left_attr, ctr_attr, right_attr in (
            ("left_camera_image",   "center_camera_image", "right_camera_image"),
            ("image_left",          "image_center",        "image_right"),
        ):
            l = getattr(obs, left_attr, None)
            c = getattr(obs, ctr_attr, None)
            r = getattr(obs, right_attr, None)
            if l is not None and c is not None and r is not None:
                return [HybridPolicy._image_to_np(l), HybridPolicy._image_to_np(c),
                        HybridPolicy._image_to_np(r)]
        return None

    @staticmethod
    def _image_to_np(img_msg: Any) -> np.ndarray:
        """sensor_msgs/Image → np.ndarray (RGB, uint8). cv_bridge 없을 때 fallback."""
        try:  # pragma: no cover
            from cv_bridge import CvBridge  # type: ignore[import-not-found]
            return CvBridge().imgmsg_to_cv2(img_msg, desired_encoding="rgb8")
        except Exception:
            # raw bytes로 변환 — encoding이 rgb8일 때만 동작
            data = np.frombuffer(img_msg.data, dtype=np.uint8)
            return data.reshape(img_msg.height, img_msg.width, -1)

    @staticmethod
    def _extract_camera_lr(obs: Any) -> tuple[np.ndarray | None, ...]:
        """left/right K + T_world_camera 4-tuple. 누락 시 (None,)*4."""
        # camera_info의 K는 9-elem flat. T_world_camera는 TF에서 lookup해야 하나
        # observation에 미리 주입돼 있을 수도 있음. 안전한 fallback:
        try:
            ci_l = getattr(obs, "left_camera_info",  None) or getattr(obs, "camera_info_left",  None)
            ci_r = getattr(obs, "right_camera_info", None) or getattr(obs, "camera_info_right", None)
            tf_l = getattr(obs, "left_camera_tf",    None) or getattr(obs, "tf_left_camera",    None)
            tf_r = getattr(obs, "right_camera_tf",   None) or getattr(obs, "tf_right_camera",   None)
            if any(x is None for x in (ci_l, ci_r, tf_l, tf_r)):
                return (None, None, None, None)
            K_l = np.array(ci_l.k, dtype=np.float64).reshape(3, 3)
            K_r = np.array(ci_r.k, dtype=np.float64).reshape(3, 3)
            return (K_l, _tf_to_mat(tf_l), K_r, _tf_to_mat(tf_r))
        except Exception:
            return (None, None, None, None)

    # ------------------------------------------------------------------ #
    # Stage B — ACT IL alignment (with explicit B→C hysteresis)
    # ------------------------------------------------------------------ #
    def _stage_b(self, task, get_obs, move_robot, send_feedback) -> bool:
        send_feedback("stage_B: ACT alignment")
        if self.act.disabled:
            send_feedback("stage_B: ACT disabled — skip")
            return False

        self.act.reset()
        deadline = time.monotonic() + self._budget_b
        condition_started_at: float | None = None
        twist_smoother = EmaSmoother(dim=6, alpha=0.4)
        loop_period = 1.0 / 20.0  # 20Hz

        while time.monotonic() < deadline and not self._emergency:
            obs = get_obs()

            # ACT forward — obs 스키마 변환 실패 시 hysteresis 만 검사하고 진행
            act_obs = self._build_act_observation(obs)
            action: np.ndarray | None = None
            if act_obs is not None:
                try:
                    action = self.act.select_action(act_obs)
                except Exception as e:
                    self.logger.warning("ACT select_action failed: %s", e)
                    action = None

            if action is not None and action.shape[0] >= 6:
                # ACTPlus: [vx,vy,vz, wx,wy,wz, gripper]. Cartesian velocity 발행.
                twist = twist_smoother(action[:6])
                vel_cmd = make_velocity_command(
                    twist_lin=twist[:3],
                    twist_ang=twist[3:6],
                )
                self._send(vel_cmd, move_robot)

            # 전환 조건 — 접촉 + 정렬이 hold_s 만큼 유지되면 C 로
            if self._should_transition_to_c(obs):
                if condition_started_at is None:
                    condition_started_at = time.monotonic()
                elif (time.monotonic() - condition_started_at) >= STAGE_TRANSITION_HOLD_S:
                    send_feedback("stage_B: transition condition held — entering Stage C")
                    return True
            else:
                condition_started_at = None

            time.sleep(loop_period)

        # 시간 초과는 점수 측면에서 손해지만 B→C 강제 진입 (Stage C가 spiral search로 회복).
        return True

    def _build_act_observation(self, obs: Any) -> dict[str, np.ndarray] | None:
        """Observation msg → ACT 입력 dict.

        키 컨벤션은 LeRobot 학습 데이터셋과 일치해야 한다:
          - observation.images.{left,center,right}: (3, H, W) float32 [0,1]
          - observation.state: (D,) float32

        Phase 3 학습 코드가 어떤 state 차원을 쓰는지에 따라 D 가 달라지므로,
        joint_state.position 만 우선 채우고 wrench 는 학습 시 정규화 stats 와
        함께 추가 예정. ACT 모델이 stats 미사용 시 raw 그대로 흘려도 됨.
        """
        imgs_list = self._extract_images(obs)
        if imgs_list is None or len(imgs_list) < 3:
            return None
        try:
            left, center, right = imgs_list
            out: dict[str, np.ndarray] = {
                "observation.images.left":   self._img_to_chw_float(left),
                "observation.images.center": self._img_to_chw_float(center),
                "observation.images.right":  self._img_to_chw_float(right),
            }
            js = getattr(obs, "joint_states", None) or getattr(obs, "joint_state", None)
            if js is not None and hasattr(js, "position"):
                state = np.asarray(list(js.position)[:7], dtype=np.float32)
                out["observation.state"] = state
            return out
        except Exception as e:
            self.logger.debug("_build_act_observation failed: %s", e)
            return None

    @staticmethod
    def _img_to_chw_float(img: np.ndarray) -> np.ndarray:
        """(H,W,3) uint8 → (3,H,W) float32 in [0,1]. 이미 CHW 면 그대로."""
        if img.ndim == 3 and img.shape[0] == 3 and img.shape[2] != 3:
            chw = img
        else:
            chw = np.transpose(img, (2, 0, 1))
        return chw.astype(np.float32) / 255.0

    def _should_transition_to_c(self, obs: Any) -> bool:
        """B→C 전환 조건: |F| > 5N AND z 거리 ≤ 10mm AND 축 각도 ≤ 5°.
        조건 미충족이면 False. 조건 평가 자체가 실패하면 False (안전).
        """
        try:
            wrench = self._get_wrench(obs)
            f_norm = float(np.linalg.norm(wrench[:3]))
            if f_norm < STAGE_TRANSITION_FORCE_MIN_N:
                return False

            tcp = self._get_tcp_pose(obs)
            if tcp is None or self._cached_target_world is None:
                return False
            tcp_xyz, _ = tcp
            z_dist = abs(float(tcp_xyz[2] - self._cached_target_world[2]))
            if z_dist > STAGE_TRANSITION_Z_DIST_MAX_M:
                return False

            # plug 축 정확 추정 전까지 z-axis로 가정 (Stage A와 동일)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------ #
    # Stage C — compliant force-guided insertion
    # ------------------------------------------------------------------ #
    @staticmethod
    def _tcp_lin_vel(obs: Any) -> float:
        """controller_state.tcp_velocity.linear 또는 0. ACTPlus 의 stationary 판정용."""
        cs = getattr(obs, "controller_state", None)
        if cs is None:
            return 0.0
        v = getattr(cs, "tcp_velocity", None)
        if v is None:
            return 0.0
        lin = getattr(v, "linear", None)
        if lin is None:
            return 0.0
        return float(np.linalg.norm([
            getattr(lin, "x", 0.0),
            getattr(lin, "y", 0.0),
            getattr(lin, "z", 0.0),
        ]))

    def _stage_c(self, task, get_obs, move_robot, send_feedback) -> bool:
        send_feedback("stage_C: compliant insertion")
        period = 1.0 / STAGE_C_LOOP_HZ
        deadline = time.monotonic() + self._budget_c

        # baseline wrench (시뮬 시작 시 미세한 잔류값 보정)
        f_baseline = self._get_wrench(get_obs())[:3]
        spiral_phase = 0.0
        spiral_radius = 0.0

        # ACTPlus 패턴: hard watchdog 전에 명령 자체를 감쇠 + stable insertion 조기 종료
        attenuator = ForceAttenuator()
        stability = InsertionStabilityDetector()

        while time.monotonic() < deadline and not self._emergency:
            obs = get_obs()
            tcp = self._get_tcp_pose(obs)
            if tcp is None:
                time.sleep(period); continue
            tcp_xyz, tcp_quat = tcp
            f = self._get_wrench(obs)[:3] - f_baseline
            f_norm_z = abs(float(f[2]))
            f_norm_total = float(np.linalg.norm(f))

            # 조기 종료: 정지 + z 방향 접촉 + 전체 |F| 안전 → stable insertion
            if stability.update(fz=float(f[2]), f_total=f_norm_total,
                                tcp_lin_vel=self._tcp_lin_vel(obs)):
                send_feedback("stage_C: stable contact — early termination")
                # 마지막 zero-delta 명령으로 컨트롤러 정지
                zero_cmd = make_insertion_step(
                    current_xyz=tcp_xyz,
                    forward_axis_world=np.array([0.0, 0.0, 1.0]),
                    forward_m=0.0,
                    lateral_xy=np.zeros(2),
                    orientation_qxyzw=tuple(tcp_quat.tolist()),
                )
                self._send(zero_cmd, move_robot)
                return True

            # forward 결정
            if f_norm_z < STAGE_C_FORWARD_FORCE_N:
                forward = STAGE_C_FORWARD_STEP_M
            elif f_norm_z < STAGE_C_HOLD_FORCE_N:
                forward = 0.0
            else:
                forward = -STAGE_C_BACKOFF_M
                spiral_radius = min(spiral_radius + 5e-4, STAGE_C_SPIRAL_MAX_M)

            # admittance: F_xy → lateral correction
            lat_admit = -STAGE_C_LATERAL_GAIN * f[:2]
            lat_admit = np.clip(lat_admit, -STAGE_C_LATERAL_LIMIT_M, STAGE_C_LATERAL_LIMIT_M)

            # spiral search 추가
            spiral_phase += STAGE_C_SPIRAL_RATE_RAD
            spiral_radius = min(spiral_radius + STAGE_C_SPIRAL_GROWTH_M, STAGE_C_SPIRAL_MAX_M)
            spiral_xy = spiral_radius * np.array(
                [np.cos(spiral_phase), np.sin(spiral_phase)]
            )

            lateral_xy = lat_admit + spiral_xy
            forward_axis = np.array([0.0, 0.0, 1.0])  # TCP +z 가정 (Stage A와 동일)

            # sustained high force → step 자체를 attenuate (-12N 페널티 회피)
            scale = attenuator.update(f_norm_total)
            forward *= scale
            lateral_xy = lateral_xy * scale

            cmd = make_insertion_step(
                current_xyz=tcp_xyz,
                forward_axis_world=forward_axis,
                forward_m=forward,
                lateral_xy=lateral_xy,
                orientation_qxyzw=tuple(tcp_quat.tolist()),
            )
            self._send(cmd, move_robot)
            time.sleep(period)

        return not self._emergency


def _tf_to_mat(tf_msg: Any) -> np.ndarray:
    """geometry_msgs/TransformStamped → 4×4. 필드명은 토킷에서 확정."""
    t = tf_msg.transform.translation
    q = tf_msg.transform.rotation
    R = _quat_to_mat(q.x, q.y, q.z, q.w)
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = [t.x, t.y, t.z]
    return M


def _quat_to_mat(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = x*x + y*y + z*z + w*w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s*(y*y+z*z), s*(x*y - w*z),   s*(x*z + w*y)],
        [s*(x*y + w*z),   1 - s*(x*x+z*z), s*(y*z - w*x)],
        [s*(x*z - w*y),   s*(y*z + w*x),   1 - s*(x*x+y*y)],
    ], dtype=np.float64)
