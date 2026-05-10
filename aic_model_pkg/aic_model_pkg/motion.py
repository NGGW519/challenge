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
