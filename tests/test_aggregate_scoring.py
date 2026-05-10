"""scripts/aggregate_scoring.py end-to-end CLI 테스트.

fixture 3개 (perfect/contact/force) → CSV 집계 → 행 수, 페널티 카운트, 합계 검증.
"""

import csv
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "aggregate_scoring.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "scoring"


def _run(out_dir: Path) -> Path:
    out_csv = out_dir / "summary.csv"
    subprocess.run(
        [sys.executable, str(SCRIPT),
         "--dir", str(FIXTURES),
         "--out", str(out_csv),
         "--policy", "fixture"],
        check=True, cwd=ROOT,
    )
    return out_csv


def test_aggregate_produces_csv(tmp_path):
    out_csv = _run(tmp_path)
    assert out_csv.exists() and out_csv.stat().st_size > 0


def test_aggregate_row_count(tmp_path):
    out_csv = _run(tmp_path)
    with out_csv.open() as f:
        rows = list(csv.DictReader(f))
    # 3 fixture × 3 trial = 9
    assert len(rows) == 9


def test_aggregate_columns(tmp_path):
    out_csv = _run(tmp_path)
    with out_csv.open() as f:
        rdr = csv.DictReader(f)
        cols = rdr.fieldnames or []
    expected = {"run", "trial_id", "tier_1_validity", "smoothness", "duration",
                "efficiency", "force_penalty", "contact_penalty",
                "tier_3_insertion", "total"}
    missing = expected - set(cols)
    assert not missing, f"missing columns: {missing}"


def test_aggregate_penalty_detection(tmp_path):
    """contact -24 1건 + force -12 1건이 정확히 잡혀야 한다."""
    out_csv = _run(tmp_path)
    with out_csv.open() as f:
        rows = list(csv.DictReader(f))

    contact_hits = [r for r in rows if r["contact_penalty"] and float(r["contact_penalty"]) < 0]
    force_hits   = [r for r in rows if r["force_penalty"]   and float(r["force_penalty"])   < 0]
    assert len(contact_hits) == 1
    assert len(force_hits)   == 1
    assert contact_hits[0]["run"] == "run_contact"
    assert force_hits[0]["run"]   == "run_force"


def test_aggregate_perfect_run_total(tmp_path):
    """perfect run의 trial 1 total = 97.5."""
    out_csv = _run(tmp_path)
    with out_csv.open() as f:
        rows = list(csv.DictReader(f))
    perfect = [r for r in rows if r["run"] == "run_perfect"]
    totals = sorted(float(r["total"]) for r in perfect)
    assert totals == [97.0, 97.5, 98.4]


def test_aggregate_handles_missing_dir(tmp_path):
    """존재하지 않는 dir이면 정상 종료 + 빈 CSV (또는 미생성)."""
    out_csv = tmp_path / "summary.csv"
    bogus = tmp_path / "nonexistent"
    bogus.mkdir()
    result = subprocess.run(
        [sys.executable, str(SCRIPT),
         "--dir", str(bogus),
         "--out", str(out_csv),
         "--policy", "empty"],
        check=False, cwd=ROOT, capture_output=True, text=True,
    )
    # 비어있는 디렉토리는 에러 없이 종료
    assert result.returncode == 0


def test_aggregate_perfect_submission_total(tmp_path):
    """perfect run 3 trial 합 = 97.5 + 97.0 + 98.4 = 292.9 → 280+ 목표선 통과."""
    out_csv = _run(tmp_path)
    with out_csv.open() as f:
        rows = list(csv.DictReader(f))
    perfect = sum(float(r["total"]) for r in rows if r["run"] == "run_perfect")
    assert perfect == pytest.approx(292.9, rel=1e-3)
    assert perfect >= 280  # 우리 마스터 플랜 목표선
