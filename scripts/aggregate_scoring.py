#!/usr/bin/env python3
"""scoring.yaml 묶음을 CSV로 집계하고 요약 통계를 출력한다.

사용:
    python3 aggregate_scoring.py --dir logs/baseline/CheatCode --out summary.csv

scoring.yaml 실제 스키마는 aic_scoring 패키지에 의해 결정되므로,
여기서는 흔히 사용되는 필드들을 모두 추출하되 누락된 키는 None으로 둔다.
"""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path
from typing import Any

import yaml


def _safe_get(d: dict, *keys, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def parse_scoring(scoring: dict) -> list[dict]:
    """공식 스키마가 변할 수 있어 list/dict 양쪽을 견딘다."""
    rows: list[dict] = []

    trials = scoring.get("trials")
    if isinstance(trials, dict):
        # dict 형태일 가능성 (trial_1, trial_2, ...)
        items = list(trials.items())
    elif isinstance(trials, list):
        items = [(t.get("trial_id", i + 1), t) for i, t in enumerate(trials)]
    else:
        return rows

    for trial_id, t in items:
        rows.append({
            "trial_id": trial_id,
            "tier_1_validity":   _safe_get(t, "tier_1_validity") or _safe_get(t, "tier_1", "validity"),
            "smoothness":        _safe_get(t, "tier_2", "smoothness_score") or _safe_get(t, "smoothness_score"),
            "duration":          _safe_get(t, "tier_2", "duration_score")   or _safe_get(t, "duration_score"),
            "efficiency":        _safe_get(t, "tier_2", "efficiency_score") or _safe_get(t, "efficiency_score"),
            "force_penalty":     _safe_get(t, "tier_2", "force_penalty")    or _safe_get(t, "force_penalty"),
            "contact_penalty":   _safe_get(t, "tier_2", "contact_penalty")  or _safe_get(t, "contact_penalty"),
            "tier_3_insertion":  _safe_get(t, "tier_3_insertion")           or _safe_get(t, "tier_3", "insertion_score"),
            "total":             _safe_get(t, "total") or _safe_get(t, "trial_total"),
        })
    return rows


def collect(directory: Path) -> list[dict]:
    out: list[dict] = []
    for ypath in sorted(directory.rglob("scoring.yaml")):
        try:
            with ypath.open() as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            print(f"[skip] {ypath}: {e}")
            continue
        run_name = ypath.parent.name
        for r in parse_scoring(data):
            r["run"] = run_name
            r["yaml_path"] = str(ypath.relative_to(directory))
            out.append(r)
    return out


def summarize(rows: list[dict], policy: str) -> None:
    if not rows:
        print(f"[{policy}] no scoring.yaml found")
        return
    totals = [r["total"] for r in rows if isinstance(r.get("total"), (int, float))]
    contact_hits = sum(1 for r in rows if (r.get("contact_penalty") or 0) < 0)
    force_hits   = sum(1 for r in rows if (r.get("force_penalty")   or 0) < 0)
    print(f"=== {policy} summary (rows={len(rows)}) ===")
    if totals:
        print(f"  total: mean={statistics.mean(totals):.2f}  "
              f"median={statistics.median(totals):.2f}  "
              f"min={min(totals):.2f}  max={max(totals):.2f}  "
              f"stdev={statistics.stdev(totals):.2f}" if len(totals) > 1 else "")
    print(f"  contact_penalty hits: {contact_hits}/{len(rows)}")
    print(f"  force_penalty   hits: {force_hits}/{len(rows)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", default="run")
    args = ap.parse_args()

    rows = collect(Path(args.dir))
    if rows:
        fields = [
            "run", "yaml_path", "trial_id",
            "tier_1_validity", "smoothness", "duration", "efficiency",
            "force_penalty", "contact_penalty", "tier_3_insertion", "total",
        ]
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in fields})
        print(f"[ok] wrote {len(rows)} rows → {args.out}")
    summarize(rows, args.policy)


if __name__ == "__main__":
    main()
