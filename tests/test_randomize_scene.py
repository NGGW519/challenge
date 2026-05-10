"""randomize_scene.py 단위 테스트."""

from pathlib import Path

import pytest

from data import randomize_scene as R


@pytest.fixture(scope="module")
def base_path() -> Path:
    return R._resolve_base_config(None)


def test_resolve_base_config(base_path: Path) -> None:
    assert base_path.exists()


@pytest.mark.parametrize("scenario", ["sfp1", "sfp2", "sc"])
def test_build_trial_keys(scenario: str, base_path: Path) -> None:
    cfg = R.build_trial(scenario, seed=1, base_path=base_path)
    assert "scoring" in cfg and "robot" in cfg and "trials" in cfg
    trial_key = next(iter(cfg["trials"]))
    trial = cfg["trials"][trial_key]
    assert "scene" in trial and "tasks" in trial


def test_grip_jitter_range(base_path: Path) -> None:
    """gripper offset이 ±2mm 이내여야 한다 (challenge spec)."""
    for seed in range(20):
        cfg = R.build_trial("sfp1", seed=seed, base_path=base_path, grip_jitter=0.002)
        cable = next(iter(cfg["trials"]["trial_1"]["scene"]["cables"].values()))
        off = cable["pose"]["gripper_offset"]
        # 베이스 z = 0.04245 (sfp). |z - base| ≤ 0.002.
        assert abs(off["z"] - 0.04245) <= 0.0021, off


def test_sfp1_vs_sfp2_differ(base_path: Path) -> None:
    """seed가 같아도 시나리오별로 다른 결과가 나와야 한다."""
    a = R.build_trial("sfp1", seed=1, base_path=base_path)["trials"]
    b = R.build_trial("sfp2", seed=1, base_path=base_path)["trials"]
    # trial_key 자체가 다름 (trial_1 vs trial_2)
    assert next(iter(a)) != next(iter(b))


def test_nic_layout_constraints(base_path: Path) -> None:
    """NIC 레일 슬라이드 ∈ [-0.0215, 0.0234], yaw ≤ 10°."""
    import math
    cfg = R.build_trial("sfp1", seed=42, base_path=base_path)
    tb = cfg["trials"]["trial_1"]["scene"]["task_board"]
    n_present = 0
    for r in range(5):
        rail = tb[f"nic_rail_{r}"]
        if rail.get("entity_present"):
            n_present += 1
            pose = rail["entity_pose"]
            assert -0.0215 <= pose["translation"] <= 0.0234, pose
            assert abs(pose["yaw"]) <= math.radians(10) + 1e-6, pose
    assert 1 <= n_present <= 3


def test_sc_exclusive(base_path: Path) -> None:
    """sc 시나리오에선 NIC 레일은 비어 있어야 한다."""
    cfg = R.build_trial("sc", seed=7, base_path=base_path)
    tb = cfg["trials"]["trial_3"]["scene"]["task_board"]
    for r in range(5):
        assert tb[f"nic_rail_{r}"]["entity_present"] is False
    sc_present = sum(1 for r in (0, 1) if tb[f"sc_rail_{r}"]["entity_present"])
    assert sc_present == 1
