"""Hybrid policy: Stage A (visual coarse) → Stage B (ACT IL) → Stage C (compliant insertion).

aic_model의 dynamic loader가 이 클래스를 import한다:
    ros2 run aic_model aic_model -p policy:=aic_model_pkg.HybridPolicy

가중치 파일이 weights/ 디렉토리에 없으면 detector/ACT는 비활성화되고
Stage C만 작동하는 fallback 모드로 들어간다 (Phase 2 데이터 수집 단계까지 유효).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

# 토킷의 Policy 베이스. 평가 컨테이너 안에서만 import 가능 (PYTHONPATH).
try:  # pragma: no cover
    from aic_model.policy import Policy
except Exception:
    class Policy:  # type: ignore[no-redef]
        """Local stub — 실제 컨테이너에선 aic_model.policy.Policy를 사용."""
        def __init__(self, parent_node):
            self.node = parent_node
            self.logger = logging.getLogger("HybridPolicy")

from .act_inference import ACTInference
from .port_detector import PortDetector
from .safety import (
    FORCE_HARD_DURATION_S,
    FORCE_HARD_N,
    ForceWatchdog,
    clamp_position,
)

WEIGHTS_DIR = Path(__file__).parent / "weights"
ACT_CKPT = WEIGHTS_DIR / "act_v1.pt"
DETECTOR_CKPT = WEIGHTS_DIR / "port_detector_v1.pt"

# Stage 시간 예산 (초)
T_STAGE_A = 5.0
T_STAGE_B = 8.0
T_STAGE_C = 15.0


class HybridPolicy(Policy):
    def __init__(self, parent_node):
        super().__init__(parent_node)
        self.logger = logging.getLogger("HybridPolicy")
        self.detector = PortDetector(DETECTOR_CKPT if DETECTOR_CKPT.exists() else None)
        self.act = ACTInference(ACT_CKPT if ACT_CKPT.exists() else None)
        self._cached_port_xyz: np.ndarray | None = None
        self._emergency = False
        self._watchdog: ForceWatchdog | None = None

    # ------------------------------------------------------------------ #
    # entry point
    # ------------------------------------------------------------------ #
    def insert_cable(self, task, get_observation, move_robot, send_feedback):
        """aic_model이 호출하는 진입점. blocking, 작업 완료까지."""
        try:
            send_feedback("hybrid: starting")
            self._start_watchdog(get_observation)

            # ----- Stage A: coarse visual approach -----
            self._stage_a(task, get_observation, move_robot, send_feedback)

            # ----- Stage B: ACT alignment -----
            self._stage_b(task, get_observation, move_robot, send_feedback)

            # ----- Stage C: compliant insertion -----
            self._stage_c(task, get_observation, move_robot, send_feedback)

            send_feedback("hybrid: done")
        except Exception as exc:  # 안전: 어떤 예외라도 정상 종료
            self.logger.exception("HybridPolicy crashed: %s", exc)
            send_feedback(f"hybrid: error {exc}")
        finally:
            self._stop_watchdog()

    # ------------------------------------------------------------------ #
    # safety wiring
    # ------------------------------------------------------------------ #
    def _start_watchdog(self, get_obs) -> None:
        def _wrench():
            obs = get_obs()
            try:
                w = obs.wrench  # geometry_msgs/Wrench fields
                return np.array([w.force.x, w.force.y, w.force.z,
                                 w.torque.x, w.torque.y, w.torque.z], dtype=np.float64)
            except Exception:
                return np.zeros(6)

        def _fire():
            self._emergency = True
            self.logger.error("force watchdog fired (>%.1fN for %.2fs)",
                              FORCE_HARD_N, FORCE_HARD_DURATION_S)

        self._watchdog = ForceWatchdog(_wrench, _fire)
        self._watchdog.start()

    def _stop_watchdog(self) -> None:
        if self._watchdog is not None:
            self._watchdog.stop()
            self._watchdog = None

    # ------------------------------------------------------------------ #
    # Stage A
    # ------------------------------------------------------------------ #
    def _stage_a(self, task, get_obs, move_robot, send_feedback) -> bool:
        send_feedback("stage_A: detect target port")
        if self.detector.disabled:
            send_feedback("stage_A: detector disabled — fallback to current TCP")
            return False
        # TODO: triangulation + plane fitting + safe approach pose 발행.
        # 1차 골조에서는 현재 TCP를 그대로 유지하고 Stage B로 진입한다.
        time.sleep(0.5)
        return True

    # ------------------------------------------------------------------ #
    # Stage B
    # ------------------------------------------------------------------ #
    def _stage_b(self, task, get_obs, move_robot, send_feedback) -> bool:
        send_feedback("stage_B: ACT alignment")
        if self.act.disabled:
            send_feedback("stage_B: ACT disabled — skip")
            return False
        # TODO: 20Hz loop, observation → action → MotionUpdate.
        # 1차 골조에서는 즉시 종료.
        return True

    # ------------------------------------------------------------------ #
    # Stage C
    # ------------------------------------------------------------------ #
    def _stage_c(self, task, get_obs, move_robot, send_feedback) -> bool:
        send_feedback("stage_C: compliant insertion (deterministic)")
        # TODO: spiral search + admittance, 자세한 알고리즘은
        #       strategy/14_phase5_hybrid_policy.md §5 참조.
        # 1차 골조에서는 단순 forward push 시도 후 종료 — 점수는 낮지만
        # lifecycle (Tier 1) 통과 검증용.
        deadline = time.monotonic() + min(T_STAGE_C, 5.0)
        while time.monotonic() < deadline and not self._emergency:
            try:
                obs = get_obs()
                tcp = obs.tcp_pose if hasattr(obs, "tcp_pose") else None
                if tcp is None:
                    time.sleep(0.05); continue
                # 1mm 전진 (TCP local z) — 실제 plug axis 매핑은 다음 라운드에서.
                target_xyz = np.array([tcp.position.x, tcp.position.y, tcp.position.z + 0.001])
                target_xyz = clamp_position(target_xyz)
                # MotionUpdate 발행은 다음 라운드에서 채운다 (msg 객체 import 필요).
                _ = target_xyz
            except Exception:
                pass
            time.sleep(0.05)
        return True
