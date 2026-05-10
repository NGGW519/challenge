"""안전 가드 단위 테스트."""

import threading
import time

import numpy as np

from aic_model_pkg import safety as S


def test_clamp_position_inside_box():
    p = np.array([0.3, 0.0, 1.2])
    np.testing.assert_array_equal(S.clamp_position(p), p)


def test_clamp_position_clamps():
    p = np.array([2.0, -1.0, 0.5])
    out = S.clamp_position(p)
    bx, by, bz = S.SAFE_BOX_BASE["x"], S.SAFE_BOX_BASE["y"], S.SAFE_BOX_BASE["z"]
    assert bx[0] <= out[0] <= bx[1]
    assert by[0] <= out[1] <= by[1]
    assert bz[0] <= out[2] <= bz[1]


def test_force_watchdog_triggers_after_duration():
    """sustained 20N → 0.7s 후 콜백 호출."""
    high = np.array([25.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    fired = threading.Event()
    wd = S.ForceWatchdog(
        get_wrench=lambda: high,
        on_emergency=lambda: fired.set(),
        soft_n=15.0, hard_n=18.0, hard_duration_s=0.2, poll_hz=100,
    )
    wd.start()
    try:
        assert fired.wait(timeout=1.0), "watchdog should have fired"
        assert wd.state.triggered
    finally:
        wd.stop()


def test_force_watchdog_no_trigger_below_threshold():
    low = np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    fired = threading.Event()
    wd = S.ForceWatchdog(
        get_wrench=lambda: low,
        on_emergency=lambda: fired.set(),
        soft_n=15.0, hard_n=18.0, hard_duration_s=0.2, poll_hz=100,
    )
    wd.start()
    try:
        time.sleep(0.5)
        assert not fired.is_set()
    finally:
        wd.stop()


def test_is_above_force_soft():
    assert not S.is_above_force_soft(np.zeros(6))
    high = np.zeros(6); high[0] = 20.0
    assert S.is_above_force_soft(high)
