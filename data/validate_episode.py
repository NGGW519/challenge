#!/usr/bin/env python3
"""한 episode가 학습에 쓸 만한지 검증.

수집 즉시 호출되어 다음 두 기준을 확인:
  1. scoring.yaml 의 Tier 3 insertion == 75 (실제로 plug가 port에 들어감)
  2. (옵션) rosbag duration 이 최소 threshold 이상 — trajectory가 잘리지 않음

실패 episode 는 별도 디렉토리(`<raw_root>/_failed/`)로 격리.
이전 시도(v2 3/300 회귀)의 핵심 원인 = "approach 시작 직후 cut된 trajectory가
dataset 대부분을 차지 → ACT가 평균값(정지)으로 collapse" 를 방지.

사용:
    python3 data/validate_episode.py \\
        --ep-dir raw_episodes/ep_000123_sfp1_10000 \\
        --min-trial-score 50 \\
        --min-bag-duration 30
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger("validate_episode")


@dataclass
class ValidationResult:
    ep_id: str
    valid: bool
    tier_3_insertion: float | None
    trial_total: float | None
    bag_duration_s: float | None
    reason: str = ""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ep-dir", required=True, help="단일 episode 디렉토리")
    p.add_argument("--min-trial-score", type=float, default=50.0,
                   help="trial total이 이 미만이면 무효. 75 = 완전 삽입 + Tier2 만점, "
                        "50 정도면 부분 삽입까지 허용 (학습 신호로 유효).")
    p.add_argument("--require-full-insertion", action="store_true",
                   help="Tier 3 insertion == 75 인 trial만 허용 (가장 엄격).")
    p.add_argument("--min-bag-duration", type=float, default=20.0,
                   help="rosbag 총 시간 최소값(초). 너무 짧으면 trajectory가 잘림.")
    p.add_argument("--quarantine", default=None,
                   help="실패 ep를 옮길 디렉토리. 미지정 시 사실은 옮기지 않고 reason만 보고.")
    p.add_argument("--json-out", default=None, help="결과를 JSON으로 저장")
    return p.parse_args(argv)


def _read_scoring(ep_dir: Path) -> dict[str, Any] | None:
    candidates = [
        ep_dir / "scoring.yaml",
        ep_dir / "aic_results" / "scoring.yaml",
    ]
    for path in candidates:
        if path.exists():
            try:
                return yaml.safe_load(path.read_text())
            except yaml.YAMLError as e:
                logger.warning("yaml parse failed %s: %s", path, e)
                return None
    return None


def _trial_totals(scoring: dict[str, Any]) -> list[float]:
    trials = scoring.get("trials")
    if isinstance(trials, dict):
        items = list(trials.values())
    elif isinstance(trials, list):
        items = trials
    else:
        return []
    out: list[float] = []
    for t in items:
        v = t.get("total")
        if isinstance(v, (int, float)):
            out.append(float(v))
    return out


def _tier3_scores(scoring: dict[str, Any]) -> list[float]:
    trials = scoring.get("trials")
    if isinstance(trials, dict):
        items = list(trials.values())
    elif isinstance(trials, list):
        items = trials
    else:
        return []
    out: list[float] = []
    for t in items:
        v = t.get("tier_3_insertion")
        if v is None:
            v = (t.get("tier_3") or {}).get("insertion_score")
        if isinstance(v, (int, float)):
            out.append(float(v))
    return out


def _bag_duration(ep_dir: Path) -> float | None:
    """rosbag2 metadata.yaml 의 duration 추출."""
    for path in ep_dir.rglob("metadata.yaml"):
        try:
            data = yaml.safe_load(path.read_text()) or {}
            info = data.get("rosbag2_bagfile_information", {})
            dur_ns = info.get("duration", {}).get("nanoseconds")
            if dur_ns is not None:
                return float(dur_ns) / 1e9
        except yaml.YAMLError:
            continue
    return None


def validate(ep_dir: Path, *, min_trial_score: float = 50.0,
             require_full_insertion: bool = False,
             min_bag_duration: float = 20.0) -> ValidationResult:
    ep_id = ep_dir.name
    scoring = _read_scoring(ep_dir)
    if scoring is None:
        return ValidationResult(ep_id, False, None, None, None,
                                "scoring.yaml not found")

    totals = _trial_totals(scoring)
    tier3 = _tier3_scores(scoring)
    bag_dur = _bag_duration(ep_dir)

    if not totals:
        return ValidationResult(ep_id, False, None, None, bag_dur,
                                "no trials in scoring.yaml")

    avg_total = sum(totals) / len(totals)
    min_tier3 = min(tier3) if tier3 else 0.0

    if require_full_insertion and any(t < 75 for t in tier3):
        return ValidationResult(ep_id, False, min_tier3, avg_total, bag_dur,
                                f"--require-full-insertion: tier3 has {tier3}")

    if min(totals) < min_trial_score:
        return ValidationResult(ep_id, False, min_tier3, avg_total, bag_dur,
                                f"trial total {min(totals):.1f} < {min_trial_score}")

    if bag_dur is not None and bag_dur < min_bag_duration:
        return ValidationResult(ep_id, False, min_tier3, avg_total, bag_dur,
                                f"bag duration {bag_dur:.1f}s < {min_bag_duration}s")

    return ValidationResult(ep_id, True, min_tier3, avg_total, bag_dur, "ok")


def quarantine(ep_dir: Path, quarantine_root: Path) -> Path:
    quarantine_root.mkdir(parents=True, exist_ok=True)
    dest = quarantine_root / ep_dir.name
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(ep_dir), str(dest))
    return dest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ep_dir = Path(args.ep_dir)
    if not ep_dir.is_dir():
        logger.error("not a directory: %s", ep_dir)
        return 2

    result = validate(
        ep_dir,
        min_trial_score=args.min_trial_score,
        require_full_insertion=args.require_full_insertion,
        min_bag_duration=args.min_bag_duration,
    )

    payload = {
        "ep_id": result.ep_id,
        "valid": result.valid,
        "tier_3_insertion": result.tier_3_insertion,
        "trial_total_avg": result.trial_total,
        "bag_duration_s": result.bag_duration_s,
        "reason": result.reason,
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))

    if not result.valid and args.quarantine:
        dest = quarantine(ep_dir, Path(args.quarantine))
        logger.info("quarantined → %s", dest)

    return 0 if result.valid else 1


if __name__ == "__main__":
    sys.exit(main())
