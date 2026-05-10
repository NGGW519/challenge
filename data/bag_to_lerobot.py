#!/usr/bin/env python3
"""rosbag2(mcap) episode 묶음을 LeRobot v2.1 dataset으로 변환.

가정:
  - 입력: <bag_root>/ep_*/   (mcap 파일이 있는 episode 디렉토리들)
  - 각 ep는 다음 토픽을 기록:
      /joint_states                        sensor_msgs/JointState
      /fts_broadcaster/wrench              geometry_msgs/WrenchStamped
      /left_camera/image                   sensor_msgs/Image
      /center_camera/image                 sensor_msgs/Image
      /right_camera/image                  sensor_msgs/Image
      /aic_controller/pose_commands        aic_control_interfaces/MotionUpdate
      (선택) /scoring/insertion_event       std_msgs/String

  - 시간 정합: 20Hz로 리샘플 — 각 frame에서 가장 가까운 image timestamp.

ROS 2 Kilted env 안에서 실행 (rosbag2_py + cv_bridge):
    pixi run python data/bag_to_lerobot.py \
        --bag_root  raw_episodes \
        --out_root  datasets/aic_v0 \
        --repo_id   nggw519/aic_v0 \
        --image_size 480x640

이 스크립트는 ROS 2 Python 의존성 (rosbag2_py, sensor_msgs, cv_bridge)이 있는
환경에서만 실제로 동작한다. 로컬에서는 --dry_run으로 인터페이스만 검증.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("bag_to_lerobot")

DEFAULT_TOPICS = {
    "joint_state":   "/joint_states",
    "wrench":        "/fts_broadcaster/wrench",
    "image_left":    "/left_camera/image",
    "image_center":  "/center_camera/image",
    "image_right":   "/right_camera/image",
    "motion":        "/aic_controller/pose_commands",
    "insertion":     "/scoring/insertion_event",
}


@dataclass
class Frame:
    t: float
    images: dict[str, object]  # cv2 ndarray (lazy import) — typed as object to keep linters happy
    state: list[float]              # joint positions + gripper + wrench (state vector)
    wrench: list[float]
    action: list[float]             # 7D Cartesian delta + gripper
    task_index: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--bag_root",   required=True, help="ep_* 디렉토리들이 있는 루트")
    p.add_argument("--out_root",   required=True, help="datasets/aic_v0/ 등 출력 디렉토리")
    p.add_argument("--repo_id",    required=True, help="HF repo (예: nggw519/aic_v0)")
    p.add_argument("--fps",        type=int,   default=20)
    p.add_argument("--image_size", default="480x640", help="HxW")
    p.add_argument("--max_episodes", type=int, default=0, help="0 = 전부")
    p.add_argument("--dry_run", action="store_true", help="ROS deps 없이 인터페이스만 검증")
    return p.parse_args(argv)


def list_episodes(bag_root: Path, limit: int = 0) -> list[Path]:
    eps = sorted(p for p in bag_root.glob("ep_*") if p.is_dir())
    if limit > 0:
        eps = eps[:limit]
    return eps


def parse_image_size(s: str) -> tuple[int, int]:
    h, _, w = s.partition("x")
    return int(h), int(w)


# --------------------------------------------------------------------- #
# ROS-dependent path
# --------------------------------------------------------------------- #
def convert_episode(ep_dir: Path, fps: int, image_size: tuple[int, int]) -> list[Frame]:
    """rosbag2 mcap → list[Frame] (20Hz resampled)."""
    import cv2
    import rosbag2_py
    from cv_bridge import CvBridge
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    H, W = image_size
    bridge = CvBridge()

    storage = rosbag2_py.StorageOptions(uri=str(ep_dir), storage_id="mcap")
    converter = rosbag2_py.ConverterOptions("", "")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage, converter)

    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}

    # 시간순으로 모든 메시지 → topic별 timeline에 push
    timelines: dict[str, list[tuple[int, object]]] = {k: [] for k in DEFAULT_TOPICS.values()}
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic not in timelines or topic not in type_map:
            continue
        msg_cls = get_message(type_map[topic])
        msg = deserialize_message(data, msg_cls)
        timelines[topic].append((t_ns, msg))

    # 기준 시간: image_left 첫 메시지 ~ 마지막 메시지
    img_left = timelines[DEFAULT_TOPICS["image_left"]]
    if not img_left:
        return []
    t0 = img_left[0][0]
    t_end = img_left[-1][0]
    period_ns = int(1e9 / fps)

    def nearest(timeline, t_ns):
        if not timeline: return None
        # 이진 탐색 안 하고 단순 — episode당 ≤ 4000 samples
        best = min(timeline, key=lambda kv: abs(kv[0] - t_ns))
        return best[1]

    frames: list[Frame] = []
    t = t0
    prev_motion = None
    while t <= t_end:
        img_l = nearest(timelines[DEFAULT_TOPICS["image_left"]],   t)
        img_c = nearest(timelines[DEFAULT_TOPICS["image_center"]], t)
        img_r = nearest(timelines[DEFAULT_TOPICS["image_right"]],  t)
        js    = nearest(timelines[DEFAULT_TOPICS["joint_state"]],  t)
        wr    = nearest(timelines[DEFAULT_TOPICS["wrench"]],       t)
        mc    = nearest(timelines[DEFAULT_TOPICS["motion"]],       t) or prev_motion
        prev_motion = mc

        if any(x is None for x in (img_l, img_c, img_r, js, wr, mc)):
            t += period_ns; continue

        def im(msg):
            arr = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            arr = cv2.resize(arr, (W, H), interpolation=cv2.INTER_AREA)
            return arr

        joints = [*list(js.position[:6]), js.position[6] if len(js.position) > 6 else 0.0]
        wrench = [wr.wrench.force.x, wr.wrench.force.y, wr.wrench.force.z,
                  wr.wrench.torque.x, wr.wrench.torque.y, wr.wrench.torque.z]
        # action: 다음 스텝의 cartesian delta — 단순 placeholder
        # (정확하려면 mc.pose 와 직전 frame mc.pose의 difference로 구해야 함)
        action = [0.0] * 7

        frames.append(Frame(
            t=(t - t0) / 1e9,
            images={"left": im(img_l), "center": im(img_c), "right": im(img_r)},
            state=joints + wrench,
            wrench=wrench,
            action=action,
            task_index=0,  # 호출자가 ep 단위로 채워줘야 함
        ))
        t += period_ns

    return frames


# --------------------------------------------------------------------- #
# LeRobot writer
# --------------------------------------------------------------------- #
def write_lerobot(frames_per_ep: list[list[Frame]], out_root: Path, repo_id: str, fps: int) -> None:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    if not frames_per_ep or not frames_per_ep[0]:
        logger.warning("no frames to write")
        return

    sample = frames_per_ep[0][0]
    img_h, img_w, _ = sample.images["left"].shape
    state_dim = len(sample.state)
    action_dim = len(sample.action)

    features = {
        "observation.images.left":   {"dtype": "video", "shape": (img_h, img_w, 3),
                                      "names": ["height", "width", "channels"]},
        "observation.images.center": {"dtype": "video", "shape": (img_h, img_w, 3),
                                      "names": ["height", "width", "channels"]},
        "observation.images.right":  {"dtype": "video", "shape": (img_h, img_w, 3),
                                      "names": ["height", "width", "channels"]},
        "observation.state":         {"dtype": "float32", "shape": (state_dim,)},
        "observation.wrench":        {"dtype": "float32", "shape": (6,)},
        "action":                    {"dtype": "float32", "shape": (action_dim,)},
        "task_index":                {"dtype": "int64",  "shape": (1,)},
    }

    ds = LeRobotDataset.create(repo_id=repo_id, fps=fps, features=features, root=out_root)
    for ep_idx, frames in enumerate(frames_per_ep):
        for f in frames:
            ds.add_frame({
                "observation.images.left":   f.images["left"],
                "observation.images.center": f.images["center"],
                "observation.images.right":  f.images["right"],
                "observation.state":         f.state,
                "observation.wrench":        f.wrench,
                "action":                    f.action,
                "task_index":                [f.task_index],
            })
        ds.save_episode(task=f"insert_cable_ep_{ep_idx}")
    ds.consolidate()
    logger.info("wrote %d episodes → %s", len(frames_per_ep), out_root)


# --------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    bag_root = Path(args.bag_root)
    out_root = Path(args.out_root)
    image_size = parse_image_size(args.image_size)

    eps = list_episodes(bag_root, args.max_episodes)
    if not eps:
        logger.error("no ep_* dirs in %s", bag_root)
        return 1
    logger.info("found %d episodes (limit=%d)", len(eps), args.max_episodes)

    if args.dry_run:
        logger.info("dry_run — exit before ROS deps load")
        return 0

    frames_per_ep: list[list[Frame]] = []
    for ep in eps:
        try:
            f = convert_episode(ep, args.fps, image_size)
        except Exception as e:
            logger.warning("skip %s: %s", ep, e); continue
        if not f:
            logger.warning("no frames in %s", ep); continue
        frames_per_ep.append(f)

    write_lerobot(frames_per_ep, out_root, args.repo_id, args.fps)

    # 메타 정보 한 줄 요약
    summary = {
        "episodes": len(frames_per_ep),
        "frames":   sum(len(f) for f in frames_per_ep),
        "fps":      args.fps,
        "image_size": list(image_size),
        "repo_id":  args.repo_id,
    }
    (out_root / "conversion_summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
