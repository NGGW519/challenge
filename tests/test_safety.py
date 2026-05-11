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


class _FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_force_attenuator_below_warn_returns_unity():
    clk = _FakeClock()
    att = S.ForceAttenuator(warn_n=12.0, window_s=0.4, attenuation=0.35, clock=clk)
    assert att.update(5.0) == 1.0
    clk.t = 1.0
    assert att.update(11.9) == 1.0


def test_force_attenuator_kicks_in_after_window():
    clk = _FakeClock()
    att = S.ForceAttenuator(warn_n=12.0, window_s=0.4, attenuation=0.35, clock=clk)
    clk.t = 0.0; assert att.update(13.0) == 1.0   # 첫 진입, 측정 시작
    clk.t = 0.3; assert att.update(13.0) == 1.0   # 아직 window 미달
    clk.t = 0.45; assert att.update(13.0) == 0.35  # window 초과 → attenuate


def test_force_attenuator_resets_on_drop():
    clk = _FakeClock()
    att = S.ForceAttenuator(warn_n=12.0, window_s=0.4, attenuation=0.35, clock=clk)
    clk.t = 0.0; att.update(13.0)
    clk.t = 0.5; att.update(13.0)  # attenuating
    clk.t = 0.6; assert att.update(5.0) == 1.0  # 떨어지면 즉시 해제
    clk.t = 0.61; assert att.update(13.0) == 1.0  # 다시 카운트 시작


def test_insertion_stability_requires_hold():
    clk = _FakeClock()
    det = S.InsertionStabilityDetector(
        fz_min_n=3.0, f_total_max_n=12.0, lin_vel_max=0.01,
        hold_s=1.2, clock=clk,
    )
    # stable signal 진입
    clk.t = 0.0; assert det.update(fz=-5.0, f_total=6.0, tcp_lin_vel=0.005) is False
    clk.t = 0.8; assert det.update(fz=-5.0, f_total=6.0, tcp_lin_vel=0.005) is False
    clk.t = 1.3; assert det.update(fz=-5.0, f_total=6.0, tcp_lin_vel=0.005) is True


def test_insertion_stability_resets_when_unstable():
    clk = _FakeClock()
    det = S.InsertionStabilityDetector(hold_s=1.0, clock=clk)
    clk.t = 0.0; det.update(fz=-5.0, f_total=6.0, tcp_lin_vel=0.005)
    clk.t = 0.5; det.update(fz=-5.0, f_total=6.0, tcp_lin_vel=0.005)
    clk.t = 0.6; assert det.update(fz=-5.0, f_total=6.0, tcp_lin_vel=0.05) is False
    # tcp 흔들리면 reset → 다음 stable 진입부터 다시 1초 카운트
    clk.t = 0.7; det.update(fz=-5.0, f_total=6.0, tcp_lin_vel=0.005)
    clk.t = 1.5; assert det.update(fz=-5.0, f_total=6.0, tcp_lin_vel=0.005) is False
    clk.t = 1.8; assert det.update(fz=-5.0, f_total=6.0, tcp_lin_vel=0.005) is True
