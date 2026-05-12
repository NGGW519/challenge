"""ACT observation contract — HybridPolicy 와 training/train_act.py 가
같은 state_dim/layout 을 공유한다는 것을 강제하는 단위 테스트.

Day 5 진단 (3/300 회귀):
- state shape mismatch (26 vs 7) 가 정책을 marginal-mean collapse 로 몰았다.
- 학습과 추론의 observation contract 가 깨지면 Tier 3 점수 0.

이 테스트는 contract 변경 시 즉시 빨간불이 들어오게 한다.
"""

import numpy as np

from aic_model_pkg.HybridPolicy import HybridPolicy


# --------------------------------------------------------------------- #
# helpers — minimal stub objects
# --------------------------------------------------------------------- #
class _Vec3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


class _Wrench:
    def __init__(self, fx=0.0, fy=0.0, fz=0.0, tx=0.0, ty=0.0, tz=0.0):
        self.force = _Vec3(fx, fy, fz)
        self.torque = _Vec3(tx, ty, tz)


class _JointState:
    def __init__(self, position):
        self.position = position


class _Obs:
    def __init__(self, joint_pos, wrench, images_dict):
        self.joint_states = _JointState(joint_pos)
        self.wrench = wrench
        for k, v in images_dict.items():
            setattr(self, k, v)


def _stub_node():
    """HybridPolicy 가 super().__init__(parent_node) 만 필요로 함."""
    class _N:
        pass
    return _N()


def _hp() -> HybridPolicy:
    return HybridPolicy(_stub_node())


# --------------------------------------------------------------------- #
# contract — state dim / layout
# --------------------------------------------------------------------- #
def test_state_dim_is_13():
    """state vector dim = 13 (joint 7 + force 3 + torque 3). 변경 시 학습 코드도 같이 수정."""
    assert HybridPolicy.ACT_STATE_DIM == 13


def test_state_layout_documented():
    """layout 튜플이 의미있는 3개 슬라이스를 명시 — joint, force, torque."""
    layout = HybridPolicy.ACT_STATE_LAYOUT
    assert len(layout) == 3
    assert "joint" in layout[0]
    assert "force" in layout[1]
    assert "torque" in layout[2]


# --------------------------------------------------------------------- #
# behavioral — _build_act_observation
# --------------------------------------------------------------------- #
class _ImgMsg:
    """sensor_msgs/Image stub — height/width/data 만 가짐 (cv_bridge fallback 경로)."""
    def __init__(self, arr: np.ndarray):
        self.height = arr.shape[0]
        self.width = arr.shape[1]
        self.data = arr.tobytes()


def _make_obs_with_images() -> _Obs:
    """3 cam 이미지 + joint + wrench 채운 stub obs."""
    arr = (np.random.rand(120, 160, 3) * 255).astype(np.uint8)
    img_msg = _ImgMsg(arr)
    return _Obs(
        joint_pos=[0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7],
        wrench=_Wrench(fx=1.0, fy=2.0, fz=3.0, tx=0.1, ty=0.2, tz=0.3),
        images_dict={
            "left_camera_image":   img_msg,
            "center_camera_image": img_msg,
            "right_camera_image":  img_msg,
        },
    )


def test_obs_returns_expected_keys():
    hp = _hp()
    out = hp._build_act_observation(_make_obs_with_images())
    assert out is not None
    assert set(out.keys()) == {
        "observation.images.left",
        "observation.images.center",
        "observation.images.right",
        "observation.state",
    }


def test_obs_state_is_13_dim_float32():
    hp = _hp()
    out = hp._build_act_observation(_make_obs_with_images())
    assert out is not None
    state = out["observation.state"]
    assert state.shape == (HybridPolicy.ACT_STATE_DIM,)
    assert state.dtype == np.float32


def test_obs_state_joint_slice():
    hp = _hp()
    out = hp._build_act_observation(_make_obs_with_images())
    state = out["observation.state"]
    np.testing.assert_array_almost_equal(
        state[:7], [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7],
    )


def test_obs_state_wrench_slice():
    hp = _hp()
    out = hp._build_act_observation(_make_obs_with_images())
    state = out["observation.state"]
    np.testing.assert_array_almost_equal(state[7:10], [1.0, 2.0, 3.0])    # force
    np.testing.assert_array_almost_equal(state[10:13], [0.1, 0.2, 0.3])   # torque


def test_obs_images_chw_float():
    hp = _hp()
    out = hp._build_act_observation(_make_obs_with_images())
    for k in ("observation.images.left", "observation.images.center", "observation.images.right"):
        img = out[k]
        assert img.shape == (3, 120, 160), f"{k} shape mismatch"
        assert img.dtype == np.float32
        assert 0.0 <= float(img.min()) and float(img.max()) <= 1.0


def test_obs_missing_wrench_zero_padded():
    """wrench 누락 시 0 으로 채워야 함 (학습 시 normalize stats 가 동작하게)."""
    obs = _make_obs_with_images()
    obs.wrench = None
    hp = _hp()
    out = hp._build_act_observation(obs)
    state = out["observation.state"]
    np.testing.assert_array_almost_equal(state[7:13], [0.0] * 6)


def test_obs_missing_images_returns_none():
    """이미지 없으면 None — Stage B 가 fallback 으로 hysteresis 만 검사."""
    obs = _Obs(joint_pos=[0]*7, wrench=_Wrench(), images_dict={})
    hp = _hp()
    out = hp._build_act_observation(obs)
    assert out is None
