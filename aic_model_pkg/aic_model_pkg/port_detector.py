"""포트 검출 + 3D 백프로젝션 (Stage A 비전 모듈).

학습된 YOLOv8 (또는 RT-DETR) 가중치는 weights/ 디렉토리에 둔다.
가중치 파일이 없으면 detector는 비활성화되고, Stage A는 fallback 경로로 빠진다.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import geometry as G

logger = logging.getLogger(__name__)

# 클래스 정의 (training/yolov8_port.yaml과 동일해야 함)
CLASS_NAMES = ("sfp_port", "sc_port", "sfp_plug", "sc_plug", "nic_card", "enclosure_edge")
CLASS_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


@dataclass
class Detection:
    cls: int
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2
    conf: float

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)

    @property
    def cls_name(self) -> str:
        return CLASS_NAMES[self.cls] if 0 <= self.cls < len(CLASS_NAMES) else f"cls{self.cls}"


# Tier 3 잘못된 포트(-12) 방어 — SOTA review 권고에 따라 conservative threshold.
DEFAULT_CONF_THRESHOLD = 0.85
# Stereo geometry consistency: left/right 매칭 시 epipolar y 차이 허용치 (px)
EPIPOLAR_Y_TOL_PX = 8.0


class PortDetector:
    """YOLOv8 wrapper. 가중치 누락 시 disabled=True."""

    def __init__(self, weights_path: str | Path | None = None, device: str = "cuda",
                 conf: float = DEFAULT_CONF_THRESHOLD) -> None:
        self.disabled = True
        self._model = None
        self.device = device
        self.conf = conf
        if weights_path and Path(weights_path).exists():
            try:
                from ultralytics import YOLO  # local import — heavy
                self._model = YOLO(str(weights_path))
                self.disabled = False
                logger.info("PortDetector loaded weights: %s", weights_path)
            except Exception as e:  # pragma: no cover
                logger.warning("PortDetector init failed (%s) — running disabled", e)
        else:
            logger.warning("PortDetector weights missing (%s) — Stage A will fallback", weights_path)

    def detect(self, images: Iterable[np.ndarray]) -> list[list[Detection]]:
        """Each image (H,W,3) RGB uint8. Returns per-image detection list."""
        if self.disabled or self._model is None:
            return [[] for _ in images]
        results = self._model.predict(list(images), imgsz=1024, conf=self.conf, iou=0.5,
                                      device=self.device, verbose=False)
        out: list[list[Detection]] = []
        for r in results:
            dets: list[Detection] = []
            if r.boxes is None:
                out.append(dets); continue
            for b in r.boxes:
                xyxy = b.xyxy.cpu().numpy().flatten()
                dets.append(Detection(
                    cls=int(b.cls.item()),
                    bbox=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                    conf=float(b.conf.item()),
                ))
            out.append(dets)
        return out

    def find_target_3d(
        self,
        dets_left: list[Detection],
        dets_right: list[Detection],
        target_cls_name: str,
        K_left: np.ndarray,
        T_world_left: np.ndarray,
        K_right: np.ndarray,
        T_world_right: np.ndarray,
    ) -> np.ndarray | None:
        """좌/우 카메라에서 target class의 bbox를 매칭, triangulate해 base_link 좌표 반환."""
        target_id = CLASS_ID.get(target_cls_name)
        if target_id is None:
            return None

        left_targets  = [d for d in dets_left  if d.cls == target_id]
        right_targets = [d for d in dets_right if d.cls == target_id]
        if not left_targets or not right_targets:
            return None

        # Stereo consistency: epipolar y(같은 horizontal line) 차이가 작은 쌍 우선.
        # left/right 카메라가 수평으로 배치되어 있다고 가정 → 같은 y 좌표.
        best_pair: tuple[Detection, Detection] | None = None
        best_score = float("inf")
        for l in left_targets:
            for r in right_targets:
                dy = abs(l.center[1] - r.center[1])
                if dy > EPIPOLAR_Y_TOL_PX:
                    continue
                # 점수 = epipolar 차이 - confidence 합 (작을수록 좋음)
                score = dy - 10.0 * (l.conf + r.conf)
                if score < best_score:
                    best_score = score
                    best_pair = (l, r)

        if best_pair is None:
            return None  # consistent 매칭 실패 — 잘못된 포트 위험 회피
        l, r = best_pair
        return G.triangulate_two_views(l.center, r.center, K_left, T_world_left, K_right, T_world_right)
