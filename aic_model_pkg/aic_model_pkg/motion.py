"""MotionUpdate 메시지 빌더 — ROS 의존성 없이 단위 테스트 가능한 추상화.

평가 컨테이너 안에서는 `aic_control_interfaces.msg.MotionUpdate`를 import해서
build_ros_msg()로 변환. 로컬에선 `MotionCommand` dataclass로 검증.

설계 의도:
  - 정책 코드(`HybridPolicy.py`)는 모두 `MotionCommand`를 만든다.
  - 하나의 헬퍼(`MotionUpdateBuilder`)가 ROS 메시지로 변환.
  - clamp / 안전 체크 / 빈도 제한 같은 로직을 ROS 의존성 없이 테스트.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .safety import clamp_position

# Cartesian impedance 권장 stiffness (단위: N/m, Nm/rad)
DEFAULT_CART_STIFFNESS = (300.0, 300.0, 800.0, 30.0, 30.0, 80.0)
DEFAULT_CART_DAMPING   = (60.0, 60.0, 100.0, 6.0, 6.0, 12.0)

# trajectory_generation_mode (aic_control_interfaces enum과 일치)
MODE_POSITION = 1
MODE_VELOCITY = 2

# pose_commands frame_id 후보
FRAME_BASE = "base_link"
FRAME_TCP  = "gripper/tcp"

# EMA 스무딩 (ACTPlus 패턴 차용 — 이전 시도에서 jerk 점수 만점 기여)
# α=0.4 → 새 명령에 40% 가중, 직전 출력에 60% 가중 → 자연스러운 저주파 통과.
DEFAULT_EMA_ALPHA = 0.4


@dataclass
class Pose:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0

    def position(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    def orientation(self) -> np.ndarray:
        return np.array([self.qx, self.qy, self.qz, self.qw], dtype=np.float64)


@dataclass
class MotionCommand:
    """ROS 의존성 없는 motion 명령 표현.

    `to_ros_msg()`는 평가 컨테이너 안에서만 호출 가능 (aic_control_interfaces import).
    로컬 테스트에선 dict/dataclass 비교로 검증.
    """
    target_pose: Pose
    frame_id: str = FRAME_BASE
    stiffness: Sequence[float] = field(default_factory=lambda: DEFAULT_CART_STIFFNESS)
    damping: Sequence[float]   = field(default_factory=lambda: DEFAULT_CART_DAMPING)
    velocity_limit: float | None = None
    mode: int = MODE_POSITION

    def __post_init__(self) -> None:
        if len(self.stiffness) != 6:
            raise ValueError(f"stiffness must be length-6, got {len(self.stiffness)}")
        if len(self.damping) != 6:
            raise ValueError(f"damping must be length-6, got {len(self.damping)}")
        if self.frame_id not in (FRAME_BASE, FRAME_TCP):
            raise ValueError(f"frame_id must be {FRAME_BASE!r} or {FRAME_TCP!r}, got {self.frame_id!r}")

    def with_clamped_position(self) -> MotionCommand:
        """SAFE_BOX_BASE로 위치 clamp한 새 MotionCommand 반환 (frame=base_link 일 때만)."""
        if self.frame_id != FRAME_BASE:
            return self  # TCP-relative는 그대로 (사용처에서 base 변환 후 clamp 권장)
        p = self.target_pose.position()
        clamped = clamp_position(p)
        if np.allclose(p, clamped):
            return self
        new_pose = Pose(*clamped, *self.target_pose.orientation())
        return MotionCommand(
            target_pose=new_pose,
            frame_id=self.frame_id,
            stiffness=self.stiffness,
            damping=self.damping,
            velocity_limit=self.velocity_limit,
            mode=self.mode,
        )

    def to_ros_msg(self) -> Any:
        """aic_control_interfaces/MotionUpdate 메시지 생성.

        ROS 환경에서만 호출 가능. 6×6 stiffness/damping 매트릭스는 row-major로
        평탄화한 36-elem float64 배열.
        """
        # late import — local test에선 import 실패해도 OK
        from aic_control_interfaces.msg import (  # type: ignore[import-not-found]
            MotionUpdate,
            TrajectoryGenerationMode,
        )
        from geometry_msgs.msg import Pose as RosPose  # type: ignore[import-not-found]

        msg = MotionUpdate()
        msg.header.frame_id = self.frame_id
        msg.pose = RosPose()
        msg.pose.position.x = float(self.target_pose.x)
        msg.pose.position.y = float(self.target_pose.y)
        msg.pose.position.z = float(self.target_pose.z)
        msg.pose.orientation.x = float(self.target_pose.qx)
        msg.pose.orientation.y = float(self.target_pose.qy)
        msg.pose.orientation.z = float(self.target_pose.qz)
        msg.pose.orientation.w = float(self.target_pose.qw)

        # 6×6 매트릭스는 대각만 채워서 36-elem flat
        msg.target_stiffness = _diag_36(self.stiffness)
        msg.target_damping   = _diag_36(self.damping)

        msg.trajectory_generation_mode.mode = (
            TrajectoryGenerationMode.MODE_POSITION if self.mode == MODE_POSITION
            else TrajectoryGenerationMode.MODE_VELOCITY
        )
        return msg


def _diag_36(values: Sequence[float]) -> list[float]:
    """길이 6 → 6×6 대각 매트릭스의 row-major 36-elem 평탄화."""
    if len(values) != 6:
        raise ValueError("expected 6 diagonal values")
    flat = [0.0] * 36
    for i, v in enumerate(values):
        flat[i * 6 + i] = float(v)
    return flat


def make_approach_command(
    port_xyz_world: np.ndarray,
    plug_axis_world: np.ndarray,
    standoff_m: float = 0.03,
    orientation_qxyzw: tuple[float, float, float, float] | None = None,
    stiffness: Sequence[float] = (300.0, 300.0, 300.0, 30.0, 30.0, 30.0),  # 부드럽게
    damping: Sequence[float] = (60.0, 60.0, 60.0, 6.0, 6.0, 6.0),
    velocity_limit: float = 0.15,  # 15 cm/s
) -> MotionCommand:
    """Stage A — 포트 진입점 위 standoff_m 만큼 후퇴한 approach pose."""
    axis = plug_axis_world / max(np.linalg.norm(plug_axis_world), 1e-9)
    target = port_xyz_world - standoff_m * axis
    qx, qy, qz, qw = orientation_qxyzw or (0.0, 0.0, 0.0, 1.0)
    cmd = MotionCommand(
        target_pose=Pose(*target, qx, qy, qz, qw),
        frame_id=FRAME_BASE,
        stiffness=stiffness,
        damping=damping,
        velocity_limit=velocity_limit,
        mode=MODE_POSITION,
    )
    return cmd.with_clamped_position()


class EmaSmoother:
    """지수 이동 평균 스무더 — Cartesian delta 또는 velocity 명령에 적용.

    ACTPlus.py의 EMA_ALPHA=0.4 패턴 차용 (이전 시도에서 jerk 점수 안정 기여).

    사용:
        ema = EmaSmoother(dim=3, alpha=0.4)
        smoothed = ema(np.array([0.1, 0.0, -0.05]))  # 첫 호출은 입력 그대로
        smoothed = ema(np.array([0.12, 0.01, -0.06]))  # 이후 EMA 적용
    """

    def __init__(self, dim: int, alpha: float = DEFAULT_EMA_ALPHA) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1]: {alpha}")
        self.dim = dim
        self.alpha = alpha
        self._state: np.ndarray | None = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.shape[0] != self.dim:
            raise ValueError(f"expected dim={self.dim}, got {x.shape[0]}")
        if self._state is None:
            self._state = x.copy()
        else:
            self._state = self.alpha * x + (1.0 - self.alpha) * self._state
        return self._state.copy()

    def reset(self) -> None:
        self._state = None


def make_insertion_step(
    current_xyz: np.ndarray,
    forward_axis_world: np.ndarray,
    forward_m: float,
    lateral_xy: np.ndarray,
    orientation_qxyzw: tuple[float, float, float, float],
    stiffness: Sequence[float] = (800.0, 800.0, 1500.0, 80.0, 80.0, 150.0),
    damping: Sequence[float] = (100.0, 100.0, 150.0, 12.0, 12.0, 18.0),
) -> MotionCommand:
    """Stage C — 한 step의 forward + lateral 합성. 회전은 고정."""
    axis = forward_axis_world / max(np.linalg.norm(forward_axis_world), 1e-9)
    delta = forward_m * axis + np.array([lateral_xy[0], lateral_xy[1], 0.0])
    target = current_xyz + delta
    cmd = MotionCommand(
        target_pose=Pose(*target, *orientation_qxyzw),
        frame_id=FRAME_BASE,
        stiffness=stiffness,
        damping=damping,
        mode=MODE_POSITION,
    )
    return cmd.with_clamped_position()


def quat_z_axis_world(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """orientation 쿼터니언(xyzw)의 회전행렬 3번째 열 = TCP +z 축의 world 방향.

    Stage A 가 plug 끝 방향을 추정할 때 사용. 그립 오프셋이 +z = plug tip 으로
    설계됐다는 토킷 컨벤션을 따른다.
    """
    n = qx * qx + qy * qy + qz * qz + qw * qw
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    s = 2.0 / n
    return np.array([
        s * (qx * qz + qw * qy),
        s * (qy * qz - qw * qx),
        1.0 - s * (qx * qx + qy * qy),
    ], dtype=np.float64)


def make_velocity_command(
    twist_lin: np.ndarray,
    twist_ang: np.ndarray,
    orientation_qxyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    stiffness: Sequence[float] = (100.0, 100.0, 100.0, 50.0, 50.0, 50.0),
    damping: Sequence[float] = (40.0, 40.0, 40.0, 15.0, 15.0, 15.0),
    max_lin: float = 0.15,
    max_ang: float = 0.5,
) -> MotionCommand:
    """Stage B — ACT 의 6D velocity twist 를 MODE_VELOCITY MotionCommand 로.

    회전은 명령에 사용되지 않으나 메시지 채움용으로 quat 그대로 전달.
    twist 값은 안전을 위해 component-wise clip.
    """
    lin = np.clip(np.asarray(twist_lin, dtype=np.float64).reshape(3), -max_lin, max_lin)
    ang = np.clip(np.asarray(twist_ang, dtype=np.float64).reshape(3), -max_ang, max_ang)
    # MotionCommand 는 pose 기반. velocity 모드일 때 target_pose 의 position 은
    # 컨트롤러가 무시하지만, 메시지 검증 용도로 0,0,0 사용.
    cmd = MotionCommand(
        target_pose=Pose(lin[0], lin[1], lin[2], *orientation_qxyzw),
        frame_id=FRAME_BASE,
        stiffness=stiffness,
        damping=damping,
        mode=MODE_VELOCITY,
    )
    # angular velocity 는 호출 측이 ros msg 변환 후 별도 채움 필요 — 임시 보관.
    cmd._twist_ang = ang  # type: ignore[attr-defined]
    cmd._twist_lin = lin  # type: ignore[attr-defined]
    return cmd
