"""EMA 스무더 단위 테스트 — ACTPlus.py 패턴 차용한 jerk 완화 layer."""

import numpy as np
import pytest
from aic_model_pkg.motion import DEFAULT_EMA_ALPHA, EmaSmoother


def test_first_call_returns_input():
    """첫 호출은 state가 비어있어 입력을 그대로 반환 (warm-up 없음)."""
    ema = EmaSmoother(dim=3, alpha=0.4)
    x = np.array([0.1, -0.2, 0.05])
    out = ema(x)
    np.testing.assert_allclose(out, x)


def test_second_call_blends_prev_and_new():
    ema = EmaSmoother(dim=3, alpha=0.4)
    ema(np.array([0.0, 0.0, 0.0]))           # state = [0,0,0]
    out = ema(np.array([1.0, 0.0, 0.0]))     # 0.4*1 + 0.6*0 = 0.4
    np.testing.assert_allclose(out, [0.4, 0.0, 0.0])


def test_converges_to_constant_input():
    """일정 입력에 대해 출력은 점진적으로 그 값에 수렴."""
    ema = EmaSmoother(dim=1, alpha=0.5)
    target = 10.0
    out = 0.0
    for _ in range(30):
        out = float(ema(np.array([target]))[0])
    assert abs(out - target) < 0.01


def test_reset_clears_state():
    ema = EmaSmoother(dim=2, alpha=0.4)
    ema(np.array([1.0, 2.0]))
    ema.reset()
    out = ema(np.array([5.0, -5.0]))
    np.testing.assert_allclose(out, [5.0, -5.0])  # 첫 호출처럼 input 그대로


def test_alpha_one_is_passthrough():
    """alpha=1.0 → 스무딩 안 함, 항상 input 그대로."""
    ema = EmaSmoother(dim=2, alpha=1.0)
    out1 = ema(np.array([1.0, 1.0]))
    out2 = ema(np.array([5.0, -5.0]))
    np.testing.assert_allclose(out1, [1.0, 1.0])
    np.testing.assert_allclose(out2, [5.0, -5.0])


def test_validates_alpha_range():
    with pytest.raises(ValueError):
        EmaSmoother(dim=3, alpha=0.0)
    with pytest.raises(ValueError):
        EmaSmoother(dim=3, alpha=1.5)
    with pytest.raises(ValueError):
        EmaSmoother(dim=3, alpha=-0.1)


def test_validates_input_dim():
    ema = EmaSmoother(dim=3, alpha=0.4)
    with pytest.raises(ValueError):
        ema(np.array([1.0, 2.0]))  # dim=2 입력
    with pytest.raises(ValueError):
        ema(np.array([1.0, 2.0, 3.0, 4.0]))  # dim=4 입력


def test_default_alpha_matches_actplus():
    """기본값은 ACTPlus.py의 EMA_ALPHA = 0.4 와 일치해야 한다."""
    assert DEFAULT_EMA_ALPHA == 0.4


def test_jerk_reduction_property():
    """EMA가 입력 step의 jerk(2차 차분)을 감소시켜야 한다."""
    rng = np.random.default_rng(0)
    raw = rng.standard_normal((100, 3))  # noisy
    ema = EmaSmoother(dim=3, alpha=0.3)
    smoothed = np.stack([ema(x) for x in raw])

    def jerk(seq: np.ndarray) -> float:
        # 2차 차분 = jerk proxy
        d2 = np.diff(seq, n=2, axis=0)
        return float(np.linalg.norm(d2, axis=1).mean())

    assert jerk(smoothed) < jerk(raw), "EMA가 jerk를 줄여야 함"
