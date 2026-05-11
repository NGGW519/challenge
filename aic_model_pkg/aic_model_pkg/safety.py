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

# 2026-05-12 Q1: ACTPlus 패턴 차용 — soft attenuation + insertion stability.
# Hard watchdog 가 트리거되기 전에 명령 자체를 감쇠해 -12N 페널티 회피.
FORCE_WARN_N = 12.0          # 이 이상 sustained 면 명령 attenuate
FORCE_WARN_WINDOW_S = 0.4    # 이 시간 동안 유지 시 attenuation 적용
FORCE_ATTENUATION = 0.35     # attenuate 시 velocity / forward step 스케일

# 삽입 안정성 (Stage C 조기 종료) 임계값.
INSERTION_FZ_N = 3.0         # 이 이상 z 방향 접촉력 = 삽입 추정
INSERTION_STABLE_S = 1.2     # 이 시간 동안 stable 유지 시 종료
STATIONARY_LIN_VEL = 0.01    # TCP 선속도 이 미만 = 정지로 간주


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


class ForceAttenuator:
    """ACTPlus 패턴: |F| 가 FORCE_WARN_N 이상으로 FORCE_WARN_WINDOW_S 만큼
    지속되면 velocity / forward step 명령을 FORCE_ATTENUATION 으로 감쇠.

    Hard watchdog 보다 먼저 발동해 -12N (1초 누적) 페널티 진입 자체를 회피.
    시간은 monotonic 으로 측정. 외부에서 매 step 마다 update(f_norm) 호출.
    """

    def __init__(
        self,
        warn_n: float = FORCE_WARN_N,
        window_s: float = FORCE_WARN_WINDOW_S,
        attenuation: float = FORCE_ATTENUATION,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._warn = warn_n
        self._window = window_s
        self._atten = attenuation
        self._clock = clock
        self._over_since: float | None = None

    def update(self, f_norm: float) -> float:
        """현재 force 크기를 받아 attenuation 계수(0~1) 반환."""
        now = self._clock()
        if f_norm >= self._warn:
            if self._over_since is None:
                self._over_since = now
            elif (now - self._over_since) >= self._window:
                return self._atten
        else:
            self._over_since = None
        return 1.0

    def reset(self) -> None:
        self._over_since = None


class InsertionStabilityDetector:
    """ACTPlus 패턴: TCP 선속도 < STATIONARY_LIN_VEL AND |fz| > INSERTION_FZ_N
    AND |F| < FORCE_WARN_N 이 INSERTION_STABLE_S 만큼 지속되면 stable insertion.

    Stage C 가 spiral search 끝없이 도는 것 방지. update() 가 True 반환 시
    호출 측에서 zero command 발행 후 종료.
    """

    def __init__(
        self,
        fz_min_n: float = INSERTION_FZ_N,
        f_total_max_n: float = FORCE_WARN_N,
        lin_vel_max: float = STATIONARY_LIN_VEL,
        hold_s: float = INSERTION_STABLE_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._fz_min = fz_min_n
        self._f_max = f_total_max_n
        self._v_max = lin_vel_max
        self._hold = hold_s
        self._clock = clock
        self._stable_since: float | None = None

    def update(self, fz: float, f_total: float, tcp_lin_vel: float) -> bool:
        """현재 step 의 안정성 신호로 stable 여부 갱신. True 반환 시 종료 권장."""
        looks_inserted = (
            tcp_lin_vel < self._v_max
            and abs(fz) > self._fz_min
            and f_total < self._f_max
        )
        now = self._clock()
        if looks_inserted:
            if self._stable_since is None:
                self._stable_since = now
            elif (now - self._stable_since) >= self._hold:
                return True
        else:
            self._stable_since = None
        return False

    def reset(self) -> None:
        self._stable_since = None
