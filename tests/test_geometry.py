"""기본 기하 유틸 단위 테스트."""

import numpy as np

from aic_model_pkg import geometry as G


def test_quat_identity():
    R = G.quat_to_rot(0.0, 0.0, 0.0, 1.0)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-9)


def test_quat_roundtrip():
    rng = np.random.default_rng(42)
    for _ in range(20):
        q = rng.standard_normal(4)
        q /= np.linalg.norm(q)
        R = G.quat_to_rot(*q)
        q2 = np.array(G.rot_to_quat(R))
        # 부호 모호성 처리
        if np.dot(q, q2) < 0:
            q2 = -q2
        np.testing.assert_allclose(q, q2, atol=1e-6)


def test_make_invert_transform():
    rng = np.random.default_rng(0)
    R = G.quat_to_rot(0.1, 0.2, 0.3, 1.0)
    t = rng.standard_normal(3)
    T = G.make_transform(t, R)
    Tinv = G.invert_transform(T)
    np.testing.assert_allclose(Tinv @ T, np.eye(4), atol=1e-9)


def test_project_to_pixel_basic():
    K = np.array([[500.0, 0.0, 320.0],
                  [0.0,   500.0, 240.0],
                  [0.0,   0.0,   1.0]])
    # camera at world origin, looking down +Z
    T_cam_world = np.eye(4)
    p_world = np.array([0.0, 0.0, 1.0])  # 1m in front
    res = G.project_to_pixel(p_world, K, T_cam_world)
    assert res is not None
    u, v, depth = res
    assert abs(u - 320.0) < 1e-6
    assert abs(v - 240.0) < 1e-6
    assert abs(depth - 1.0) < 1e-9


def test_project_behind_camera_returns_none():
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    T = np.eye(4)
    assert G.project_to_pixel(np.array([0, 0, -0.5]), K, T) is None


def test_triangulate_two_views_synthetic():
    """두 카메라가 같은 점을 보는 경우 정확한 3D 복원."""
    K = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    p_world = np.array([0.05, 0.02, 1.0])
    # cam1: world frame == cam frame
    T_w_l = np.eye(4)
    # cam2: 10cm right (along +x in world)
    T_w_r = np.eye(4); T_w_r[0, 3] = 0.1
    # 픽셀 좌표 계산
    uv_l = G.project_to_pixel(p_world, K, G.invert_transform(T_w_l))
    uv_r = G.project_to_pixel(p_world, K, G.invert_transform(T_w_r))
    assert uv_l is not None and uv_r is not None
    p_est = G.triangulate_two_views(uv_l[:2], uv_r[:2], K, T_w_l, K, T_w_r)
    assert p_est is not None
    np.testing.assert_allclose(p_est, p_world, atol=1e-3)


def test_axis_angle():
    z = np.array([0, 0, 1.0])
    np.testing.assert_allclose(G.axis_angle(z, z), 0.0, atol=1e-9)
    np.testing.assert_allclose(G.axis_angle(z, -z), np.pi, atol=1e-9)
    np.testing.assert_allclose(G.axis_angle(z, np.array([1, 0, 0.0])), np.pi / 2, atol=1e-9)
