"""data/collect_demos.py 단위 테스트 — dry_run 인터페이스 + 분포 샘플링."""

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data import collect_demos as CD  # noqa: E402


# --------------------------------------------------------------------- #
# distribution
# --------------------------------------------------------------------- #
def test_parse_dist_normalizes():
    d = CD.parse_dist("sfp1:1,sfp2:1,sc:2")
    assert d == pytest.approx({"sfp1": 0.25, "sfp2": 0.25, "sc": 0.5})


def test_parse_dist_default():
    d = CD.parse_dist(CD.DEFAULT_SCENARIO_DIST)
    assert sum(d.values()) == pytest.approx(1.0)
    assert set(d) == {"sfp1", "sfp2", "sc", "edge"}


def test_parse_dist_rejects_zero():
    with pytest.raises(ValueError):
        CD.parse_dist("sfp1:0")


def test_parse_dist_rejects_empty():
    with pytest.raises(ValueError):
        CD.parse_dist("")


def test_sample_scenario_distribution():
    """다수 샘플링하면 비율이 분포에 수렴해야 한다."""
    dist = {"a": 0.7, "b": 0.3}
    rng = random.Random(0)
    counts = {"a": 0, "b": 0}
    n = 5000
    for _ in range(n):
        counts[CD.sample_scenario(rng, dist)] += 1
    assert abs(counts["a"] / n - 0.7) < 0.03
    assert abs(counts["b"] / n - 0.3) < 0.03


# --------------------------------------------------------------------- #
# CLI dry-run
# --------------------------------------------------------------------- #
def _run_cli(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "data" / "collect_demos.py"), *args],
        capture_output=True, text=True, cwd=cwd, check=False,
    )


def test_dry_run_creates_summary(tmp_path):
    p = _run_cli([
        "--n", "3", "--dry-run",
        "--out", str(tmp_path),
        "--seed-shuffle", "1",
    ], cwd=ROOT)
    assert p.returncode == 0, f"stderr:\n{p.stderr}"

    summary = tmp_path / "collection_summary.json"
    assert summary.exists()
    data = json.loads(summary.read_text())
    assert data["n_total"] == 3
    assert data["n_success"] == 3
    assert data["n_failure"] == 0
    assert len(data["episodes"]) == 3


def test_dry_run_creates_scene_yaml_per_episode(tmp_path):
    p = _run_cli([
        "--n", "2", "--dry-run",
        "--out", str(tmp_path),
        "--seed-shuffle", "42",
    ], cwd=ROOT)
    assert p.returncode == 0, p.stderr

    ep_dirs = sorted(d for d in tmp_path.iterdir() if d.is_dir())
    assert len(ep_dirs) == 2
    for ep in ep_dirs:
        scene = ep / "scene.yaml"
        assert scene.exists() and scene.stat().st_size > 0


def test_dry_run_reproducible_with_seed(tmp_path):
    """동일 seed_base + seed_shuffle → 동일 시나리오 시퀀스."""
    cwd = ROOT
    out1 = tmp_path / "a"
    out2 = tmp_path / "b"
    common = ["--n", "5", "--dry-run", "--seed-base", "777", "--seed-shuffle", "13"]

    p1 = _run_cli([*common, "--out", str(out1)], cwd=cwd)
    p2 = _run_cli([*common, "--out", str(out2)], cwd=cwd)
    assert p1.returncode == 0 and p2.returncode == 0

    s1 = json.loads((out1 / "collection_summary.json").read_text())
    s2 = json.loads((out2 / "collection_summary.json").read_text())
    seq1 = [e["scenario"] for e in s1["episodes"]]
    seq2 = [e["scenario"] for e in s2["episodes"]]
    assert seq1 == seq2


def test_episode_id_format(tmp_path):
    p = _run_cli([
        "--n", "1", "--dry-run",
        "--out", str(tmp_path),
        "--seed-base", "9999",
        "--seed-shuffle", "0",
    ], cwd=ROOT)
    assert p.returncode == 0
    summary = json.loads((tmp_path / "collection_summary.json").read_text())
    ep_id = summary["episodes"][0]["ep_id"]
    # ep_<idx>_<scenario>_<seed>
    assert ep_id.startswith("ep_000000_")
    assert ep_id.endswith("_9999")
