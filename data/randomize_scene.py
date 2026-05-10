#!/usr/bin/env python3
"""sample_config.yaml을 토대로 무작위 trial config를 생성한다.

Phase 2 (데이터 수집) 와 Phase 6 (도메인 랜덤화) 양쪽에서 사용.

사용:
    python3 data/randomize_scene.py --scenario sfp1 --seed 42 --out trial.yaml
    python3 data/randomize_scene.py --scenario sfp2 --seed 100
    python3 data/randomize_scene.py --scenario sc   --seed 7  --grip-jitter 0.003
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import random
from pathlib import Path

import yaml

DEFAULT_BASE_CONFIG = Path("/workspace/aic/aic_engine/config/sample_config.yaml")
LOCAL_BASE_CONFIG   = Path("/home/nggw/challenge/aic/aic_engine/config/sample_config.yaml")

# 슬라이드 한계 (sample_config.yaml의 task_board_limits에서 확인)
NIC_RAIL_RANGE = (-0.0215, 0.0234)
NIC_YAW_DEG_RANGE = (-10.0, 10.0)
SC_RAIL_RANGE  = (-0.06,   0.055)
MOUNT_RAIL_RANGE = (-0.09425, 0.09425)


def _resolve_base_config(path: str | None) -> Path:
    if path:
        p = Path(path)
        if p.exists():
            return p
    env = os.environ.get("AIC_SAMPLE_CONFIG")
    if env and Path(env).exists():
        return Path(env)
    for candidate in (DEFAULT_BASE_CONFIG, LOCAL_BASE_CONFIG):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "base sample_config.yaml not found. Set --base, AIC_SAMPLE_CONFIG, or place it at "
        f"{DEFAULT_BASE_CONFIG} or {LOCAL_BASE_CONFIG}"
    )


def _sample_nic_layout(rng: random.Random) -> tuple[dict, int]:
    n_present = rng.randint(1, 3)
    present = rng.sample(range(5), n_present)
    target = rng.choice(present)
    out: dict = {}
    for r in range(5):
        key = f"nic_rail_{r}"
        if r in present:
            out[key] = {
                "entity_present": True,
                "entity_name": f"nic_card_{r}",
                "entity_pose": {
                    "translation": rng.uniform(*NIC_RAIL_RANGE),
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": math.radians(rng.uniform(*NIC_YAW_DEG_RANGE)),
                },
            }
        else:
            out[key] = {"entity_present": False}
    return out, target


def _sample_sc_layout(rng: random.Random) -> tuple[dict, int]:
    target_rail = rng.choice([0, 1])
    out: dict = {}
    for r in (0, 1):
        key = f"sc_rail_{r}"
        if r == target_rail:
            out[key] = {
                "entity_present": True,
                "entity_name": f"sc_mount_{r}",
                "entity_pose": {
                    "translation": rng.uniform(*SC_RAIL_RANGE),
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": math.radians(rng.uniform(-5.0, 5.0)),
                },
            }
        else:
            out[key] = {"entity_present": False}
    return out, target_rail


def _sample_task_board_pose(rng: random.Random, scenario: str) -> dict:
    if scenario in ("sfp1", "sfp2"):
        x, y, yaw = 0.15, -0.2, math.pi
    else:
        x, y, yaw = 0.17, 0.0, 3.0
    return {
        "x": x + rng.uniform(-0.02, 0.02),
        "y": y + rng.uniform(-0.02, 0.02),
        "z": 1.14,
        "roll":  rng.uniform(-0.05, 0.05),
        "pitch": rng.uniform(-0.05, 0.05),
        "yaw":   yaw + rng.uniform(-0.10, 0.10),
    }


def _sample_grip_offset(rng: random.Random, plug_type: str, jitter: float) -> dict:
    base_z = 0.04245 if plug_type == "sfp" else 0.04045
    return {
        "x": 0.0      + rng.uniform(-jitter, jitter),
        "y": 0.015385 + rng.uniform(-jitter, jitter),
        "z": base_z   + rng.uniform(-jitter, jitter),
    }


def _sample_filler_mounts(rng: random.Random) -> dict:
    """LC/SFP/SC mount slots — 시나리오 다양성 차원에서 일부만 채움."""
    out: dict = {}
    for slot in ("lc_mount_rail_0", "sfp_mount_rail_0", "sc_mount_rail_0",
                 "lc_mount_rail_1", "sfp_mount_rail_1", "sc_mount_rail_1"):
        present = rng.random() < 0.5
        if present:
            out[slot] = {
                "entity_present": True,
                "entity_name": slot.replace("_rail_", "_") + "_x",  # placeholder name
                "entity_pose": {
                    "translation": rng.uniform(*MOUNT_RAIL_RANGE),
                    "roll": 0.0, "pitch": 0.0,
                    "yaw":  math.radians(rng.uniform(-5.0, 5.0)),
                },
            }
        else:
            out[slot] = {"entity_present": False}
    return out


def build_trial(scenario: str, seed: int, base_path: Path, grip_jitter: float = 0.002) -> dict:
    # 시나리오를 seed에 섞어, 같은 seed로 sfp1/sfp2/sc가 다른 결과를 내도록.
    rng = random.Random(hash((scenario, seed)) & 0xFFFFFFFF)
    base = yaml.safe_load(base_path.read_text())

    trial_key = {"sfp1": "trial_1", "sfp2": "trial_2", "sc": "trial_3"}[scenario]
    trial = copy.deepcopy(base["trials"][trial_key])

    # task_board pose
    trial["scene"]["task_board"]["pose"] = _sample_task_board_pose(rng, scenario)

    # NIC / SC layout
    if scenario in ("sfp1", "sfp2"):
        nic, target = _sample_nic_layout(rng)
        trial["scene"]["task_board"].update(nic)
        # SC rails 비움 (NIC trial)
        for r in (0, 1):
            trial["scene"]["task_board"][f"sc_rail_{r}"] = {"entity_present": False}
        trial["tasks"]["task_1"]["target_module_name"] = f"nic_card_mount_{target}"
        cable_key = next(iter(trial["scene"]["cables"]))
        trial["scene"]["cables"][cable_key]["pose"]["gripper_offset"] = _sample_grip_offset(rng, "sfp", grip_jitter)
    else:  # sc
        # NIC rails 비움
        for r in range(5):
            trial["scene"]["task_board"][f"nic_rail_{r}"] = {"entity_present": False}
        sc, target = _sample_sc_layout(rng)
        trial["scene"]["task_board"].update(sc)
        trial["tasks"]["task_1"]["target_module_name"] = f"sc_port_{target}"
        cable_key = next(iter(trial["scene"]["cables"]))
        trial["scene"]["cables"][cable_key]["pose"]["gripper_offset"] = _sample_grip_offset(rng, "sc", grip_jitter)

    # mount fillers (배경 다양성)
    trial["scene"]["task_board"].update(_sample_filler_mounts(rng))

    return {"scoring": base["scoring"], "robot": base["robot"], "trials": {trial_key: trial}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["sfp1", "sfp2", "sc"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--base", default=None, help="path to sample_config.yaml")
    ap.add_argument("--out", default=None, help="output yaml path (default: stdout)")
    ap.add_argument("--grip-jitter", type=float, default=0.002,
                    help="gripper offset noise (m, ±). default ±2mm matches challenge spec.")
    args = ap.parse_args()

    base_path = _resolve_base_config(args.base)
    cfg = build_trial(args.scenario, args.seed, base_path, args.grip_jitter)
    text = yaml.dump(cfg, sort_keys=False, allow_unicode=True)
    if args.out:
        Path(args.out).write_text(text)
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
