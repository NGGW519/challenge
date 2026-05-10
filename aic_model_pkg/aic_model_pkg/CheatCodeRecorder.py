"""CheatCode + rosbag2 mcap 기록 — Phase 2 데이터 수집의 ground-truth 시연자.

# AIC-TRAINING-ONLY: 이 파일은 학습 데이터 수집 전용. 제출 컨테이너에 절대
# 포함하면 안 됨. (`/scoring/tf` ground-truth를 구독하므로 챌린지 규칙 위반.)
# - submit.Dockerfile은 .dockerignore로 이 파일 제외.
# - audit_pre_submit.sh는 위 마커를 인식해 검사 예외 처리.

토킷의 `aic_example_policies.ros.CheatCode`를 그대로 상속해 동일한 trajectory를
실행하면서, 모든 sensor/command 토픽을 rosbag2(mcap)에 기록한다.

활성 조건: 환경변수 `AIC_RECORD_BAG_DIR`가 설정돼 있어야 함. 미설정 시
super().insert_cable()만 호출 (= 평범한 CheatCode 동작).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml

# 토킷의 CheatCode를 베이스로. 평가 컨테이너 안에서만 실제 import.
try:  # pragma: no cover
    from aic_example_policies.ros.CheatCode import CheatCode  # type: ignore[import-not-found]
except Exception:
    class CheatCode:  # type: ignore[no-redef]
        """Local stub — 실제 환경에선 토킷의 CheatCode를 사용."""
        def __init__(self, parent_node):
            self._parent_node = parent_node

        def insert_cable(self, task, get_observation, move_robot, send_feedback):
            raise NotImplementedError("CheatCode stub — only available inside ROS env")

        def get_logger(self):
            return logging.getLogger("CheatCodeStub")


# 기록 대상 토픽 — sample_config.yaml의 scoring.topics 와 정합되게.
# (cable_type 등 task 메타는 별도 task.yaml로 저장)
RECORDED_TOPICS: tuple[tuple[str, str], ...] = (
    ("/joint_states",                    "sensor_msgs/msg/JointState"),
    ("/fts_broadcaster/wrench",          "geometry_msgs/msg/WrenchStamped"),
    ("/left_camera/image",               "sensor_msgs/msg/Image"),
    ("/center_camera/image",             "sensor_msgs/msg/Image"),
    ("/right_camera/image",              "sensor_msgs/msg/Image"),
    ("/left_camera/camera_info",         "sensor_msgs/msg/CameraInfo"),
    ("/center_camera/camera_info",       "sensor_msgs/msg/CameraInfo"),
    ("/right_camera/camera_info",        "sensor_msgs/msg/CameraInfo"),
    ("/aic_controller/pose_commands",    "aic_control_interfaces/msg/MotionUpdate"),
    ("/aic_controller/joint_commands",   "aic_control_interfaces/msg/JointMotionUpdate"),
    ("/aic_controller/controller_state", "aic_control_interfaces/msg/ControllerState"),
    ("/tf",                              "tf2_msgs/msg/TFMessage"),
    ("/tf_static",                       "tf2_msgs/msg/TFMessage"),
    ("/scoring/tf",                      "tf2_msgs/msg/TFMessage"),  # GT — 학습용만
    ("/scoring/insertion_event",         "std_msgs/msg/String"),
)


class CheatCodeRecorder(CheatCode):
    """CheatCode + 모든 토픽 rosbag2 mcap 기록.

    Lifecycle 설계:
      __init__:        super 초기화 (TF buffer 등)
      insert_cable:
        1. AIC_RECORD_BAG_DIR 읽기
        2. SequentialWriter 열고 토픽 등록
        3. 모든 RECORDED_TOPICS subscribe — callback이 write
        4. super().insert_cable() 실행 (CheatCode 본 trajectory)
        5. 종료 시 subscriber 제거 + writer close
    """

    def __init__(self, parent_node: Any) -> None:
        super().__init__(parent_node)
        self._logger = logging.getLogger("CheatCodeRecorder")
        self._writer: Any = None
        self._subs: list[Any] = []
        self._record_count: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    def insert_cable(self, task, get_observation, move_robot, send_feedback):  # type: ignore[override]
        bag_dir = os.environ.get("AIC_RECORD_BAG_DIR")
        recording = bag_dir is not None and bag_dir != ""
        if recording:
            try:
                self._open_writer(bag_dir, task)
                self._subscribe_all()
                self.get_logger().info(f"CheatCodeRecorder: recording → {bag_dir}")
            except Exception as e:
                self._logger.exception("recorder setup failed: %s", e)
                self._close_writer()
                recording = False
        try:
            return super().insert_cable(task, get_observation, move_robot, send_feedback)
        finally:
            if recording:
                self._close_writer()

    # ------------------------------------------------------------------ #
    def _open_writer(self, bag_dir: str, task: Any) -> None:
        from rosbag2_py import (  # type: ignore[import-not-found]
            ConverterOptions,
            SequentialWriter,
            StorageOptions,
            TopicMetadata,
        )

        bag_root = Path(bag_dir)
        bag_root.mkdir(parents=True, exist_ok=True)

        # task 메타데이터 저장 (post-hoc episode → task_index 매칭)
        meta = {
            "cable_type":         getattr(task, "cable_type", ""),
            "cable_name":         getattr(task, "cable_name", ""),
            "plug_type":          getattr(task, "plug_type", ""),
            "plug_name":          getattr(task, "plug_name", ""),
            "port_type":          getattr(task, "port_type", ""),
            "port_name":          getattr(task, "port_name", ""),
            "target_module_name": getattr(task, "target_module_name", ""),
        }
        (bag_root / "task.yaml").write_text(yaml.dump(meta, sort_keys=False))

        bag_path = bag_root / "bag"
        self._writer = SequentialWriter()
        self._writer.open(
            StorageOptions(uri=str(bag_path), storage_id="mcap"),
            ConverterOptions("", ""),
        )
        for tname, ttype in RECORDED_TOPICS:
            self._writer.create_topic(
                TopicMetadata(name=tname, type=ttype, serialization_format="cdr"),
            )

    def _subscribe_all(self) -> None:
        from rosidl_runtime_py.utilities import get_message  # type: ignore[import-not-found]

        node = self._parent_node
        for tname, ttype in RECORDED_TOPICS:
            try:
                msg_cls = get_message(ttype)
            except Exception as e:
                self._logger.warning("skip %s (%s): %s", tname, ttype, e)
                continue

            sub = node.create_subscription(
                msg_cls, tname, self._make_callback(tname), 10,
            )
            self._subs.append(sub)
            self._record_count[tname] = 0

    def _make_callback(self, topic: str):
        from rclpy.serialization import serialize_message  # type: ignore[import-not-found]

        node = self._parent_node

        def _cb(msg: Any) -> None:
            if self._writer is None:
                return
            try:
                t_ns = node.get_clock().now().nanoseconds
                self._writer.write(topic, serialize_message(msg), t_ns)
                self._record_count[topic] = self._record_count.get(topic, 0) + 1
            except Exception as e:
                self._logger.debug("write fail %s: %s", topic, e)
        return _cb

    def _close_writer(self) -> None:
        for s in self._subs:
            try:
                self._parent_node.destroy_subscription(s)
            except Exception:
                pass
        self._subs = []
        if self._writer is not None:
            try:
                # SequentialWriter는 GC 시 자동 close. 안전하게 None 처리.
                pass
            finally:
                self._writer = None

        # 통계 로깅
        if self._record_count:
            summary = ", ".join(f"{k}={v}" for k, v in sorted(self._record_count.items()))
            self._logger.info("recorded counts: %s", summary)
        self._record_count = {}
