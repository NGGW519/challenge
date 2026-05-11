"""validate_episode 의 scoring 기반 ep 검증 로직 단위 테스트.

이전 v2 회귀(3/300)의 핵심 원인 = "실패 trial이 dataset에 섞임" → 학습 collapse.
이 모듈이 그걸 막는 1차 방어선이므로 견고해야 한다.
"""

import shutil
from pathlib import Path

import pytest
import yaml

from data import validate_episode as VE


def _make_ep(tmp_path: Path, ep_name: str, scoring: dict | None,
             bag_duration_ns: int | None = None) -> Path:
    """fixture episode 디렉토리 생성."""
    ep = tmp_path / ep_name
    ep.mkdir()
    if scoring is not None:
        (ep / "scoring.yaml").write_text(yaml.dump(scoring))
    if bag_duration_ns is not None:
        bag_dir = ep / "bag"
        bag_dir.mkdir()
        (bag_dir / "metadata.yaml").write_text(yaml.dump({
            "rosbag2_bagfile_information": {
                "duration": {"nanoseconds": bag_duration_ns}
            }
        }))
    return ep


PERFECT_SCORING = {
    "trials": [
        {"trial_id": 1, "tier_3_insertion": 75, "total": 97.5},
        {"trial_id": 2, "tier_3_insertion": 75, "total": 97.0},
        {"trial_id": 3, "tier_3_insertion": 75, "total": 98.4},
    ]
}

# 이전 v2 회귀 패턴: 매우 낮은 점수 (approach 초반 cut)
COLLAPSED_SCORING = {
    "trials": [
        {"trial_id": 1, "tier_3_insertion": 0, "total": 1.0},
        {"trial_id": 2, "tier_3_insertion": 0, "total": 1.0},
        {"trial_id": 3, "tier_3_insertion": 0, "total": 1.0},
    ]
}

# 부분 삽입 (학습 신호로는 유효)
PARTIAL_SCORING = {
    "trials": [
        {"trial_id": 1, "tier_3_insertion": 50, "total": 65.0},
        {"trial_id": 2, "tier_3_insertion": 50, "total": 60.0},
        {"trial_id": 3, "tier_3_insertion": 50, "total": 62.0},
    ]
}


# --------------------------------------------------------------------- #
# happy path
# --------------------------------------------------------------------- #
def test_perfect_ep_is_valid(tmp_path):
    ep = _make_ep(tmp_path, "ep_perfect", PERFECT_SCORING,
                  bag_duration_ns=int(30e9))
    r = VE.validate(ep)
    assert r.valid
    assert r.tier_3_insertion == 75
    assert r.bag_duration_s == pytest.approx(30.0)


def test_partial_ep_is_valid_by_default(tmp_path):
    """부분 삽입 (Tier 3 = 50)도 학습 신호로 유효 — default 통과."""
    ep = _make_ep(tmp_path, "ep_partial", PARTIAL_SCORING,
                  bag_duration_ns=int(30e9))
    r = VE.validate(ep)
    assert r.valid


# --------------------------------------------------------------------- #
# failure cases (이전 v2 회귀 패턴)
# --------------------------------------------------------------------- #
def test_collapsed_ep_is_rejected(tmp_path):
    """이전 시도의 핵심 실패 패턴 — 모든 trial이 1점 (approach 초반 cut)."""
    ep = _make_ep(tmp_path, "ep_collapsed", COLLAPSED_SCORING,
                  bag_duration_ns=int(30e9))
    r = VE.validate(ep)
    assert not r.valid
    assert "trial total" in r.reason


def test_no_scoring_yaml_is_rejected(tmp_path):
    ep = _make_ep(tmp_path, "ep_no_yaml", None)
    r = VE.validate(ep)
    assert not r.valid
    assert "not found" in r.reason


def test_short_bag_is_rejected(tmp_path):
    """trajectory 잘림 — bag duration이 너무 짧음."""
    ep = _make_ep(tmp_path, "ep_short", PERFECT_SCORING,
                  bag_duration_ns=int(5e9))   # 5초만
    r = VE.validate(ep, min_bag_duration=20.0)
    assert not r.valid
    assert "bag duration" in r.reason


# --------------------------------------------------------------------- #
# strict flag
# --------------------------------------------------------------------- #
def test_partial_rejected_when_full_required(tmp_path):
    ep = _make_ep(tmp_path, "ep_partial2", PARTIAL_SCORING,
                  bag_duration_ns=int(30e9))
    r = VE.validate(ep, require_full_insertion=True)
    assert not r.valid
    assert "require-full-insertion" in r.reason


def test_perfect_passes_when_full_required(tmp_path):
    ep = _make_ep(tmp_path, "ep_perfect2", PERFECT_SCORING,
                  bag_duration_ns=int(30e9))
    r = VE.validate(ep, require_full_insertion=True)
    assert r.valid


# --------------------------------------------------------------------- #
# quarantine
# --------------------------------------------------------------------- #
def test_quarantine_moves_dir(tmp_path):
    ep = _make_ep(tmp_path, "ep_bad", COLLAPSED_SCORING,
                  bag_duration_ns=int(30e9))
    q_root = tmp_path / "_failed"
    dest = VE.quarantine(ep, q_root)
    assert dest.exists()
    assert not ep.exists()
    assert (dest / "scoring.yaml").exists()


def test_quarantine_overwrites_existing(tmp_path):
    """같은 ep_id로 두 번 quarantine해도 충돌 안 함."""
    ep1 = _make_ep(tmp_path, "ep_bad", COLLAPSED_SCORING)
    q_root = tmp_path / "_failed"
    VE.quarantine(ep1, q_root)

    ep2 = _make_ep(tmp_path, "ep_bad", COLLAPSED_SCORING)
    dest = VE.quarantine(ep2, q_root)
    assert dest.exists()


# --------------------------------------------------------------------- #
# alternative scoring schema (dict 형태)
# --------------------------------------------------------------------- #
def test_scoring_as_dict_format(tmp_path):
    """일부 토킷 버전은 trials를 dict로 출력 — 그것도 처리."""
    scoring = {
        "trials": {
            "trial_1": {"trial_id": 1, "tier_3_insertion": 75, "total": 97.0},
            "trial_2": {"trial_id": 2, "tier_3_insertion": 75, "total": 96.0},
        }
    }
    ep = _make_ep(tmp_path, "ep_dict", scoring, bag_duration_ns=int(30e9))
    r = VE.validate(ep)
    assert r.valid
