"""안전 가드 — bbox clamp, force watchdog, contact risk 추정."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

# base_link 좌표계의 안전 작업 영역. task_board top ≈ z=1.14.
# 평가 시나리오의 task_board pose 분산을 감안해 약간 넉넉히.
SAFE_BOX_BASE: dict[str, tuple[float, float]] = {
    "x": (0.00, 0.75),
    "y": (-0.45, 0.45),
    "z": (1.04, 1.50),
}

# F/T 한계
FORCE_SOFT_N = 15.0   # 이 이상이면 진행 중단
FORCE_HARD_N = 18.0   # 이 이상 + 시간 초과 → emergency
FORCE_HARD_DURATION_S = 0.7  # 0.7s 견디면 emergency (스코어 컷 1.0s 이전)


def clamp_position(p: np.ndarray) -> np.ndarray:
    """3D position을 SAFE_BOX_BASE로 clamp."""
    out = p.copy()
    for i, axis in enumerate(("x", "y", "z")):
        lo, hi = SAFE_BOX_BASE[axis]
        out[i] = float(np.clip(out[i], lo, hi))
    return out


@dataclass
class ForceWatchdogState:
    over_since: float | None = None
    triggered: bool = False
    last_force_norm: float = 0.0


class ForceWatchdog:
    """별도 스레드에서 F/T를 감시. emergency 발생 시 콜백 호출."""

    def __init__(
        self,
        get_wrench: Callable[[], np.ndarray],
        on_emergency: Callable[[], None],
        soft_n: float = FORCE_SOFT_N,
        hard_n: float = FORCE_HARD_N,
        hard_duration_s: float = FORCE_HARD_DURATION_S,
        poll_hz: float = 50.0,
    ) -> None:
        self._get_wrench = get_wrench
        self._on_emergency = on_emergency
        self._soft = soft_n
        self._hard = hard_n
        self._dur = hard_duration_s
        self._period = 1.0 / poll_hz
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.state = ForceWatchdogState()

    def start(self) -> None:
        self._stop.clear()
        self.state = ForceWatchdogState()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                w = self._get_wrench()
            except Exception:
                time.sleep(self._period)
                continue
            f_norm = float(np.linalg.norm(w[:3]))
            self.state.last_force_norm = f_norm
            now = time.monotonic()
            if f_norm > self._hard:
                if self.state.over_since is None:
                    self.state.over_since = now
                elif (now - self.state.over_since) > self._dur:
                    if not self.state.triggered:
                        self.state.triggered = True
                        try:
                            self._on_emergency()
                        except Exception:
                            pass
            else:
                self.state.over_since = None
            time.sleep(self._period)


def is_above_force_soft(w: np.ndarray) -> bool:
    return float(np.linalg.norm(w[:3])) > FORCE_SOFT_N
