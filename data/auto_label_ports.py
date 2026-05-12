#!/usr/bin/env python3
"""Gazebo의 ground-truth TF에서 SFP/SC 포트 위치를 읽어 자동 YOLO 라벨 생성.

각 scene에서 3개 카메라 이미지를 캡처하고, 각 포트 (sfp_port_0..7, sc_port_0/1)와
nic_card 본체의 3D 위치를 카메라 intrinsics + extrinsics로 픽셀에 투영.
픽셀이 이미지 안이면 bbox 추정값 (포트 입구 크기에 따른 고정 박스)으로 라벨 생성.

사용 (ROS 2 Kilted env 안에서):
    pixi run python data/auto_label_ports.py \
        --scene_seeds 0..5000 \
        --out datasets/port_detection \
        --image_size 1024x1024 \
        --classes sfp_port,sc_port,nic_card

ground_truth=true 옵션으로 aic_engine을 띄운 상태에서 실행 (학습용 only — 평가에선 금지).

이 스크립트도 ROS deps가 있는 환경에서만 실제 동작. 로컬에선 --dry_run으로 검증.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger("auto_label_ports")

# 클래스 ID + bbox 크기 + YOLO 변환은 port_labeling.py 로 분리됨 (ROS-free 테스트 가능)
from data.port_labeling import (  # noqa: E402
    CLASS_ID,
    DEFAULT_BBOX_SIZE_PX as DEFAULT_BBOX_SIZE,
    make_yolo_label,
    project_point_to_pixel,
    write_yolo_label_file,
)

# 포트 경로 (TF frame 이름 기준)
PORT_FRAMES = {
    "sfp_port": [f"nic_card_{r}/sfp_port_{p}" for r in range(5) for p in range(8)],
    "sc_port":  [f"sc_mount_{r}/sc_port_{p}"  for r in range(2) for p in range(2)],
    "nic_card": [f"nic_card_{r}/base"         for r in range(5)],
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, help="datasets/port_detection 루트")
    p.add_argument("--scene_seeds", default="0..5000",
                   help="시나리오 seed 범위, 예: '0..5000' 또는 '100,200,300'")
    p.add_argument("--image_size", default="1024x1024", help="HxW (정사각 권장)")
    p.add_argument("--classes", default="sfp_port,sc_port,nic_card")
    p.add_argument("--split", default="80/10/10", help="train/val/test 비율")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args(argv)


def parse_seeds(spec: str) -> list[int]:
    if ".." in spec:
        a, b = spec.split("..")
        return list(range(int(a), int(b)))
    return [int(s) for s in spec.split(",") if s]


def parse_size(s: str) -> tuple[int, int]:
    h, _, w = s.partition("x")
    return int(h), int(w)


def parse_split(s: str) -> tuple[float, float, float]:
    parts = [float(x) for x in s.split("/")]
    if len(parts) != 3 or abs(sum(parts) - 100) > 1e-6:
        raise ValueError("split must be 3 ratios summing to 100, e.g. 80/10/10")
    return tuple(p / 100.0 for p in parts)  # type: ignore[return-value]


def write_yolo_label(label_path: Path, items: list[tuple[int, float, float, float, float]]) -> None:
    """items: list of (cls, cx_norm, cy_norm, w_norm, h_norm).
    레거시 호환을 위한 wrapper — 새 코드는 port_labeling.write_yolo_label_file 사용 권장.
    """
    label_path.parent.mkdir(parents=True, exist_ok=True)
    with label_path.open("w") as f:
        for cls, cx, cy, w, h in items:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def label_scene_dryrun(seed: int, image_size: tuple[int, int], classes: list[str]) -> dict:
    """ROS deps 없이 형식만 검증. 실제 라벨링은 ros 환경에서."""
    H, W = image_size
    items = []
    for cls in classes:
        cls_id = CLASS_ID[cls]
        bw, bh = DEFAULT_BBOX_SIZE[cls]
        # 가짜 위치: seed로 결정적
        cx = (seed * 7 % W) / W
        cy = (seed * 13 % H) / H
        items.append((cls_id, cx, cy, bw / W, bh / H))
    return {"image_shape": (H, W, 3), "labels": items}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    image_size = parse_size(args.image_size)
    seeds = parse_seeds(args.scene_seeds)
    classes = [c.strip() for c in args.classes.split(",")]
    train_r, val_r, test_r = parse_split(args.split)
    out = Path(args.out)

    # 디렉토리 트리 준비
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        logger.info("dry_run — generating dummy labels for first 3 seeds")
        for s in seeds[:3]:
            res = label_scene_dryrun(s, image_size, classes)
            logger.info("seed=%d → %d items", s, len(res["labels"]))
        return 0

    # 실제 라벨링은 ROS 2 환경에서.
    try:
        import rclpy  # noqa: F401
        # TODO: ros 노드 띄우고 → randomize_scene으로 새 trial 띄움 → camera 캡처 + TF lookup → 투영
        logger.error("실제 라벨링 코드는 ros2 노드를 띄워야 합니다. 다음 단계에서 채우세요.")
        return 2
    except ImportError as e:
        logger.error("rclpy not available: %s", e)
        return 3

    # (추후 채울 부분)
    summary = {"seeds": len(seeds), "classes": classes, "image_size": list(image_size),
               "split": [train_r, val_r, test_r]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
