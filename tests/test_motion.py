"""MotionCommand 빌더 단위 테스트 (ROS 의존성 없이 검증)."""

import numpy as np
import pytest

from aic_model_pkg import motion as M


def test_pose_helpers():
    p = M.Pose(0.1, 0.2, 0.3, 0, 0, 0, 1)
    np.testing.assert_array_almost_equal(p.position(), [0.1, 0.2, 0.3])
    np.testing.assert_array_almost_equal(p.orientation(), [0, 0, 0, 1])


def test_motion_command_validates_lengths():
    with pytest.raises(ValueError, match="stiffness"):
        M.MotionCommand(target_pose=M.Pose(), stiffness=[1, 2, 3])
    with pytest.raises(ValueError, match="damping"):
        M.MotionCommand(target_pose=M.Pose(), damping=[1, 2, 3])


def test_motion_command_validates_frame():
    with pytest.raises(ValueError, match="frame_id"):
        M.MotionCommand(target_pose=M.Pose(), frame_id="world")


def test_clamp_position_inside_safe_box():
    cmd = M.MotionCommand(target_pose=M.Pose(0.3, 0.0, 1.2))
    out = cmd.with_clamped_position()
    assert out is cmd  # clamp 발생 안 하면 동일 객체


def test_clamp_position_clamps():
    cmd = M.MotionCommand(target_pose=M.Pose(2.0, -1.0, 0.5))
    out = cmd.with_clamped_position()
    assert out is not cmd
    p = out.target_pose.position()
    from aic_model_pkg.safety import SAFE_BOX_BASE
    bx, by, bz = SAFE_BOX_BASE["x"], SAFE_BOX_BASE["y"], SAFE_BOX_BASE["z"]
    assert bx[0] <= p[0] <= bx[1]
    assert by[0] <= p[1] <= by[1]
    assert bz[0] <= p[2] <= bz[1]


def test_clamp_only_for_base_frame():
    """TCP frame은 clamp하지 않는다 (사용처에서 base 변환 후 clamp 권장)."""
    cmd = M.MotionCommand(target_pose=M.Pose(2, 2, 2), frame_id=M.FRAME_TCP)
    out = cmd.with_clamped_position()
    assert out is cmd  # 변경 없음


def test_diag_36_layout():
    flat = M._diag_36([1, 2, 3, 4, 5, 6])
    assert len(flat) == 36
    expected = np.diag([1, 2, 3, 4, 5, 6]).flatten().tolist()
    assert flat == expected


def test_diag_36_rejects_wrong_length():
    with pytest.raises(ValueError):
        M._diag_36([1, 2, 3])


def test_make_approach_command_standoff():
    """플러그 축 +z 방향이면 standoff만큼 -z 쪽으로 물러난 pose."""
    port = np.array([0.3, 0.0, 1.2])
    axis = np.array([0.0, 0.0, 1.0])  # +z
    cmd = M.make_approach_command(port, axis, standoff_m=0.03)
    p = cmd.target_pose.position()
    np.testing.assert_array_almost_equal(p, [0.3, 0.0, 1.17])
    assert cmd.frame_id == M.FRAME_BASE
    assert cmd.mode == M.MODE_POSITION


def test_make_approach_command_normalizes_axis():
    """축 벡터가 정규화 안 되어 있어도 동작."""
    cmd = M.make_approach_command(
        np.array([0.3, 0.0, 1.2]),
        np.array([0.0, 0.0, 5.0]),  # |axis| = 5
        standoff_m=0.03,
    )
    p = cmd.target_pose.position()
    np.testing.assert_array_almost_equal(p, [0.3, 0.0, 1.17])


def test_make_insertion_step_forward_and_lateral():
    """forward 1mm + lateral (0.5mm, 0) 합성."""
    current = np.array([0.3, 0.0, 1.2])
    axis = np.array([0.0, 0.0, 1.0])
    cmd = M.make_insertion_step(
        current, axis, forward_m=0.001,
        lateral_xy=np.array([0.0005, 0.0]),
        orientation_qxyzw=(0.0, 0.0, 0.0, 1.0),
    )
    p = cmd.target_pose.position()
    np.testing.assert_array_almost_equal(p, [0.3005, 0.0, 1.201])


def test_make_insertion_step_uses_z_stiff():
    """삽입은 z를 단단하게, xy는 부드럽게 — Stage C 권장 파라미터."""
    cmd = M.make_insertion_step(
        np.zeros(3), np.array([0, 0, 1.0]),
        forward_m=0.001, lateral_xy=np.zeros(2),
        orientation_qxyzw=(0, 0, 0, 1),
    )
    assert cmd.stiffness[2] > cmd.stiffness[0]  # z stiffer than x


def test_to_ros_msg_raises_outside_ros_env():
    """ROS 환경 밖에선 to_ros_msg가 ImportError로 실패해야 한다."""
    cmd = M.MotionCommand(target_pose=M.Pose())
    with pytest.raises((ImportError, ModuleNotFoundError)):
        cmd.to_ros_msg()
