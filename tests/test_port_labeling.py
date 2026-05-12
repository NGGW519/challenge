"""port_labeling 의 ROS-free 투영 + YOLO label 단위 테스트.

Phase 4 (port detector 학습) 진입 시 합성 라벨 정확성이 mAP 직접 결정.
이 테스트는 라벨링 핵심 수식을 동결한다.
"""

import math

import numpy as np
import pytest

from data import port_labeling as PL


# --------------------------------------------------------------------- #
# class id contract — port_detector.CLASS_ID 와 일치
# --------------------------------------------------------------------- #
def test_class_ids_match_port_detector():
    """port_detector.py 의 CLASS_ID 와 동일해야 한다 (학습/추론 mismatch 방지)."""
    from aic_model_pkg.port_detector import CLASS_ID as DETECTOR_CLS

    # 양쪽 모두 같은 키, 같은 값
    for name, expected_id in DETECTOR_CLS.items():
        assert PL.CLASS_ID.get(name) == expected_id, (
            f"class id mismatch for {name}: detector={expected_id}, "
            f"labeling={PL.CLASS_ID.get(name)}"
        )


def test_class_ids_are_unique():
    assert len(set(PL.CLASS_ID.values())) == len(PL.CLASS_ID)


# --------------------------------------------------------------------- #
# project_point_to_pixel
# --------------------------------------------------------------------- #
def _K(fx=500.0, fy=500.0, cx=320.0, cy=240.0):
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def test_project_origin_in_front():
    """카메라 좌표계 +z 방향 1m 점은 principal point 에 투영."""
    K = _K()
    T = np.eye(4)
    uv = PL.project_point_to_pixel(np.array([0.0, 0.0, 1.0]), K, T)
    assert uv is not None
    u, v, depth = uv
    assert math.isclose(u, 320.0, abs_tol=1e-6)
    assert math.isclose(v, 240.0, abs_tol=1e-6)
    assert math.isclose(depth, 1.0)


def test_project_behind_camera_returns_none():
    K = _K()
    T = np.eye(4)
    assert PL.project_point_to_pixel(np.array([0, 0, -0.5]), K, T) is None


def test_project_offset_point():
    """fx=500, point=(0.1, 0, 1) → u = 500*0.1/1 + cx = 50 + 320 = 370."""
    K = _K()
    T = np.eye(4)
    uv = PL.project_point_to_pixel(np.array([0.1, 0.0, 1.0]), K, T)
    assert uv is not None
    assert math.isclose(uv[0], 370.0, abs_tol=1e-6)
    assert math.isclose(uv[1], 240.0, abs_tol=1e-6)


def test_project_validates_shapes():
    with pytest.raises(ValueError):
        PL.project_point_to_pixel(np.zeros(3), np.zeros(2), np.eye(4))
    with pytest.raises(ValueError):
        PL.project_point_to_pixel(np.zeros(3), _K(), np.eye(3))


# --------------------------------------------------------------------- #
# make_yolo_label
# --------------------------------------------------------------------- #
def test_yolo_label_centered():
    """이미지 중심 픽셀 → cx/cy = 0.5."""
    label = PL.make_yolo_label("sfp_port", u=512.0, v=512.0,
                               img_h=1024, img_w=1024)
    assert label is not None
    assert math.isclose(label.cx_norm, 0.5, abs_tol=1e-6)
    assert math.isclose(label.cy_norm, 0.5, abs_tol=1e-6)
    bw, bh = PL.DEFAULT_BBOX_SIZE_PX["sfp_port"]
    assert math.isclose(label.w_norm, bw / 1024.0, abs_tol=1e-6)
    assert math.isclose(label.h_norm, bh / 1024.0, abs_tol=1e-6)
    assert label.cls_id == PL.CLASS_ID["sfp_port"]


def test_yolo_label_clipped_at_edge():
    """bbox 가 좌측 경계에 걸치면 클립해서 작아진다."""
    # u=5, bw=24, bh=16 → x1=-7 클립=0, x2=17. 폭 17 (절반은 음수)
    label = PL.make_yolo_label("sfp_port", u=5.0, v=500.0,
                               img_h=1024, img_w=1024)
    assert label is not None
    # 클립된 폭은 24보다 작아야 함
    assert label.w_norm < 24 / 1024.0


def test_yolo_label_outside_image_returns_none():
    """bbox 전부 밖이면 None."""
    label = PL.make_yolo_label("sfp_port", u=-100.0, v=-100.0,
                               img_h=1024, img_w=1024)
    assert label is None


def test_yolo_label_rejects_unknown_class():
    with pytest.raises(ValueError):
        PL.make_yolo_label("not_a_class", u=100, v=100,
                           img_h=1024, img_w=1024)


def test_yolo_label_to_line_format():
    label = PL.YoloLabel(cls_id=2, cx_norm=0.5, cy_norm=0.5,
                         w_norm=0.1, h_norm=0.2)
    line = label.to_line()
    parts = line.split()
    assert len(parts) == 5
    assert parts[0] == "2"
    assert float(parts[1]) == 0.5


# --------------------------------------------------------------------- #
# write_yolo_label_file
# --------------------------------------------------------------------- #
def test_write_yolo_file(tmp_path):
    labels = [
        PL.YoloLabel(0, 0.1, 0.2, 0.05, 0.04),
        PL.YoloLabel(1, 0.5, 0.5, 0.1, 0.08),
    ]
    out = tmp_path / "labels" / "ep_0.txt"
    PL.write_yolo_label_file(out, labels)
    assert out.exists()
    lines = out.read_text().strip().split("\n")
    assert len(lines) == 2
    assert lines[0].startswith("0 ")
    assert lines[1].startswith("1 ")
