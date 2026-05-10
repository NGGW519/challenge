"""TF / 좌표 / 삼각측량 유틸."""

from __future__ import annotations

import numpy as np

try:  # ROS 환경이면 cv2 / transforms3d 사용 가능
    import cv2  # noqa: F401
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]


def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Quaternion → 3×3 rotation matrix (right-handed, scalar last)."""
    n = qx * qx + qy * qy + qz * qz + qw * qw
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = qx * qx * s, qy * qy * s, qz * qz * s
    xy, xz, yz = qx * qy * s, qx * qz * s, qy * qz * s
    wx, wy, wz = qw * qx * s, qw * qy * s, qw * qz * s
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def rot_to_quat(R: np.ndarray) -> tuple[float, float, float, float]:
    """3×3 rotation matrix → quaternion (qx, qy, qz, qw)."""
    t = float(np.trace(R))
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        return ((R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s, (R[1, 0] - R[0, 1]) * s, 0.25 / s)
    # 안정성 위해 가장 큰 대각 사용
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return (0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s)
    if R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return ((R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s)
    s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
    return ((R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s)


def make_transform(t: np.ndarray, R: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def invert_transform(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Tinv = np.eye(4)
    Tinv[:3, :3] = R.T
    Tinv[:3, 3] = -R.T @ t
    return Tinv


def project_to_pixel(p_world: np.ndarray, K: np.ndarray, T_cam_world: np.ndarray) -> tuple[float, float, float] | None:
    """Project 3D world point to pixel (u, v, depth_in_cam). Returns None if behind cam."""
    p_h = np.append(p_world, 1.0)
    p_cam = T_cam_world @ p_h
    if p_cam[2] <= 1e-3:
        return None
    u = K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]
    v = K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]
    return (float(u), float(v), float(p_cam[2]))


def triangulate_two_views(
    uv_left: tuple[float, float],
    uv_right: tuple[float, float],
    K_left: np.ndarray,
    T_world_left: np.ndarray,
    K_right: np.ndarray,
    T_world_right: np.ndarray,
) -> np.ndarray | None:
    """Two-view triangulation. T_world_X is X→world; we need X→cam = inverse."""
    if cv2 is None:
        return _triangulate_dlt(uv_left, uv_right, K_left, T_world_left, K_right, T_world_right)
    T_left_world  = invert_transform(T_world_left)
    T_right_world = invert_transform(T_world_right)
    P_l = K_left  @ T_left_world[:3]
    P_r = K_right @ T_right_world[:3]
    pts4 = cv2.triangulatePoints(P_l, P_r, np.array([[uv_left[0]], [uv_left[1]]], dtype=np.float64),
                                            np.array([[uv_right[0]], [uv_right[1]]], dtype=np.float64))
    if abs(pts4[3, 0]) < 1e-9:
        return None
    return (pts4[:3, 0] / pts4[3, 0]).astype(np.float64)


def _triangulate_dlt(uv_l, uv_r, K_l, T_w_l, K_r, T_w_r) -> np.ndarray | None:
    T_l_w = invert_transform(T_w_l)
    T_r_w = invert_transform(T_w_r)
    P1 = K_l @ T_l_w[:3]
    P2 = K_r @ T_r_w[:3]
    A = np.zeros((4, 4))
    A[0] = uv_l[0] * P1[2] - P1[0]
    A[1] = uv_l[1] * P1[2] - P1[1]
    A[2] = uv_r[0] * P2[2] - P2[0]
    A[3] = uv_r[1] * P2[2] - P2[1]
    _, _, vt = np.linalg.svd(A)
    X = vt[-1]
    if abs(X[3]) < 1e-9:
        return None
    return (X[:3] / X[3]).astype(np.float64)


def axis_angle(a: np.ndarray, b: np.ndarray) -> float:
    """두 단위 벡터 사이 각도 (rad). 입력이 단위벡터가 아니어도 정규화."""
    a = a / max(np.linalg.norm(a), 1e-9)
    b = b / max(np.linalg.norm(b), 1e-9)
    return float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))
