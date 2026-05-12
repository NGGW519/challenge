"""포트 자동 라벨링의 ROS-free 핵심 — 투영 + bbox 클립 + YOLO 형식.

`auto_label_ports.py` 가 ROS 노드 안에서 TF lookup 후 이 모듈의 함수들로
픽셀 좌표 + bbox 를 만든다. ROS 의존성 없이 단위 테스트 가능.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# 클래스 ID — `aic_model_pkg/port_detector.py` 의 CLASS_ID 와 1:1 일치해야 함.
CLASS_ID: dict[str, int] = {
    "sfp_port": 0,
    "sc_port": 1,
    "sfp_plug": 2,
    "sc_plug": 3,
    "nic_card": 4,
    "enclosure_edge": 5,
}

# 각 클래스의 픽셀 bbox 크기 (정렬된 카메라 ~50cm 거리 기준 추정).
DEFAULT_BBOX_SIZE_PX: dict[str, tuple[int, int]] = {
    "sfp_port": (24, 16),
    "sc_port":  (28, 18),
    "sfp_plug": (60, 30),
    "sc_plug":  (60, 30),
    "nic_card": (140, 90),
    "enclosure_edge": (200, 8),
}


@dataclass(frozen=True)
class YoloLabel:
    """YOLO format: cls_id + (cx, cy, w, h) 모두 [0,1] 정규화."""
    cls_id: int
    cx_norm: float
    cy_norm: float
    w_norm: float
    h_norm: float

    def to_line(self) -> str:
        return (f"{self.cls_id} {self.cx_norm:.6f} {self.cy_norm:.6f} "
                f"{self.w_norm:.6f} {self.h_norm:.6f}")


def project_point_to_pixel(
    p_world: np.ndarray,
    K: np.ndarray,
    T_cam_world: np.ndarray,
) -> tuple[float, float, float] | None:
    """world 3D 점 → 카메라 픽셀 (u, v, depth). 카메라 뒤면 None.

    K: 3x3 intrinsics, T_cam_world: 4x4 world→cam transform (cam frame 기준 world 점).
    """
    if K.shape != (3, 3):
        raise ValueError(f"K must be 3x3, got {K.shape}")
    if T_cam_world.shape != (4, 4):
        raise ValueError(f"T_cam_world must be 4x4, got {T_cam_world.shape}")
    p_h = np.append(np.asarray(p_world, dtype=np.float64), 1.0)
    p_cam = T_cam_world @ p_h
    z = float(p_cam[2])
    if z <= 1e-3:
        return None
    u = float(K[0, 0] * p_cam[0] / z + K[0, 2])
    v = float(K[1, 1] * p_cam[1] / z + K[1, 2])
    return (u, v, z)


def make_yolo_label(
    cls_name: str,
    u: float, v: float,
    img_h: int, img_w: int,
    bbox_size_px: tuple[int, int] | None = None,
) -> YoloLabel | None:
    """픽셀 (u,v) 중심 + bbox 크기 → YOLO 정규화 label. bbox 가 이미지 밖이면 None.

    이미지 안에 부분만 걸치면 클립한 결과로 label 생성 (단 크기가 1px 미만이면 None).
    """
    if cls_name not in CLASS_ID:
        raise ValueError(f"unknown class: {cls_name}")
    bw, bh = bbox_size_px or DEFAULT_BBOX_SIZE_PX[cls_name]

    # bbox 픽셀 좌표 (clip)
    x1 = max(0.0, u - bw / 2.0)
    x2 = min(float(img_w), u + bw / 2.0)
    y1 = max(0.0, v - bh / 2.0)
    y2 = min(float(img_h), v + bh / 2.0)
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None

    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return YoloLabel(
        cls_id=CLASS_ID[cls_name],
        cx_norm=cx, cy_norm=cy, w_norm=w, h_norm=h,
    )


def write_yolo_label_file(path, labels) -> None:
    """labels (list[YoloLabel]) → YOLO txt 파일 작성."""
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for label in labels:
            f.write(label.to_line() + "\n")
