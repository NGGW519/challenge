#!/usr/bin/env python3
"""Phase 2 Data Collection — N개 에피소드 자동 수집 orchestrator.

수행 절차 (각 episode):
  1. `randomize_scene.py`로 무작위 trial config 생성 → `<ep_dir>/scene.yaml`
  2. `docker compose up` (eval + model)
     - model 정책: `aic_model_pkg.CheatCodeRecorder` (ground_truth=true)
     - 정책이 자체 rosbag2 SequentialWriter로 토픽 기록 → `<ep_dir>/bag/`
  3. trial 종료 후 compose down + bag 검증
N 에피소드 완료 후:
  4. `bag_to_lerobot.py`로 LeRobot v2.1 변환
  5. `hf upload`로 HF Hub 업로드 (--push-hf 시) — huggingface_hub 1.x CLI

사용:
    # 동작 시뮬레이션 (실제 docker 호출 안 함)
    python3 data/collect_demos.py --n 10 --dry-run

    # 실제 수집 (vast.ai 인스턴스 안에서)
    python3 data/collect_demos.py \
        --n 1100 \
        --dist sfp1:0.36,sfp2:0.36,sc:0.18,edge:0.10 \
        --out raw_episodes \
        --convert-to-lerobot datasets/aic_v0 \
        --push-hf nggw519/aic-datasets

dist 형식: 'sfp1:0.4,sfp2:0.4,sc:0.2' — 합 1.0 기대 (내부 정규화).
edge 시나리오는 grip_jitter를 spec 한계까지 키운 stress 케이스.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

logger = logging.getLogger("collect_demos")

ROOT = Path(__file__).resolve().parent.parent
RANDOMIZE_SCRIPT = ROOT / "data" / "randomize_scene.py"

DEFAULT_SCENARIO_DIST = "sfp1:0.36,sfp2:0.36,sc:0.18,edge:0.10"


@dataclass
class EpisodeResult:
    ep_id: str
    scenario: str
    seed: int
    success: bool
    duration_s: float
    error: str | None = None


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10, help="총 에피소드 수")
    p.add_argument("--dist", default=DEFAULT_SCENARIO_DIST,
                   help="시나리오 분포 (sfp1:0.4,sfp2:0.4,sc:0.2 형식)")
    p.add_argument("--out", default="raw_episodes",
                   help="bag/scene.yaml/result.json 저장 루트")
    p.add_argument("--seed-base", type=int, default=10000,
                   help="seed = seed_base + i. 재현 가능한 데이터셋용")
    p.add_argument("--convert-to-lerobot", default=None, metavar="DEST",
                   help="수집 끝난 후 bag_to_lerobot.py 호출 → DEST에 LeRobot dataset 저장")
    p.add_argument("--push-hf", default=None, metavar="REPO_ID",
                   help="HF Hub dataset repo (예: nggw519/aic-datasets)에 업로드")
    p.add_argument("--policy-module", default="aic_model_pkg.CheatCodeRecorder",
                   help="model 컨테이너에서 띄울 정책. 기본은 우리 자체 recorder")
    p.add_argument("--ground-truth", default="true",
                   choices=["true", "false"], help="aic_eval ground_truth 옵션")
    p.add_argument("--episode-timeout-s", type=float, default=1800.0,
                   help="단일 episode 강제 종료 시각. "
                        "이전 시도에서 600s timeout이 짧아 trajectory가 잘려 학습 collapse를 유발 — "
                        "30분(1800s) default로 보수적. RTF가 좋으면 단축 가능.")
    p.add_argument("--min-success-rate", type=float, default=0.8,
                   help="누적 success rate가 이 미만이면 수집 즉시 중단 "
                        "(잘못된 데이터 다량 생성 방지). 기본 0.8.")
    p.add_argument("--sample-every", type=int, default=10,
                   help="N ep마다 1개를 mp4로 추출 + HF dataset에 업로드해 사용자 검증 유도. "
                        "0이면 비활성.")
    p.add_argument("--dry-run", action="store_true",
                   help="docker 호출 안 하고 인터페이스 검증만")
    p.add_argument("--seed-shuffle", type=int, default=None,
                   help="시나리오 선택 RNG seed (None이면 매번 다름)")
    return p.parse_args(argv)


# --------------------------------------------------------------------- #
# 시나리오 분포
# --------------------------------------------------------------------- #
def parse_dist(spec: str) -> dict[str, float]:
    items: dict[str, float] = {}
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, _, weight = chunk.partition(":")
        items[name.strip()] = float(weight)
    if not items:
        raise ValueError("dist must contain at least one entry")
    total = sum(items.values())
    if total <= 0:
        raise ValueError("dist weights must sum to a positive number")
    return {k: v / total for k, v in items.items()}


def sample_scenario(rng: random.Random, dist: dict[str, float]) -> str:
    r = rng.random()
    cum = 0.0
    for name, p in dist.items():
        cum += p
        if r <= cum:
            return name
    return next(iter(dist))  # numerical edge


# --------------------------------------------------------------------- #
# Per-episode work
# --------------------------------------------------------------------- #
def make_scene(scenario: str, seed: int, scene_yaml: Path,
               grip_jitter: float | None = None) -> None:
    """randomize_scene.py CLI 호출."""
    # edge 시나리오는 grip_jitter를 ±3mm로 키워 sfp1과 동일 trial 사용
    real_scenario = "sfp1" if scenario == "edge" else scenario
    real_jitter = 0.003 if scenario == "edge" else (grip_jitter or 0.002)

    cmd = [sys.executable, str(RANDOMIZE_SCRIPT),
           "--scenario", real_scenario,
           "--seed", str(seed),
           "--out", str(scene_yaml),
           "--grip-jitter", str(real_jitter)]
    subprocess.run(cmd, check=True, cwd=ROOT)


def run_episode_via_compose(ep_dir: Path, scene_yaml: Path, args: argparse.Namespace) -> bool:
    """docker compose up + 종료. dry_run이면 시뮬만."""
    env = os.environ.copy()
    env.update({
        "AIC_POLICY_MODULE":  args.policy_module,
        "AIC_GROUND_TRUTH":   args.ground_truth,
        "AIC_RESULTS_DIR":    str(ep_dir),
        "AIC_TRIAL_CONFIG":   str(scene_yaml),
        "AIC_EP_DIR":         str(ep_dir),
    })
    compose_args = [
        "docker", "compose",
        "-f", "/workspace/aic/docker/docker-compose.yaml",
        "-f", str(ROOT / "docker" / "collect-override.yaml"),
    ]
    up = [*compose_args, "up", "--abort-on-container-exit",
          "--exit-code-from", "eval", "eval", "model"]
    down = [*compose_args, "down", "--remove-orphans"]

    if args.dry_run:
        logger.info("[dry_run] would run: %s", " ".join(up))
        return True

    try:
        proc = subprocess.run(up, env=env, cwd=ROOT,
                              timeout=args.episode_timeout_s,
                              check=False)
        ok = proc.returncode == 0
    except subprocess.TimeoutExpired:
        logger.warning("episode timed out at %.1fs", args.episode_timeout_s)
        ok = False
    finally:
        subprocess.run(down, env=env, cwd=ROOT, check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return ok


def collect_one(args: argparse.Namespace, idx: int, scenario: str, seed: int,
                raw_dir: Path) -> EpisodeResult:
    ep_id = f"ep_{idx:06d}_{scenario}_{seed}"
    ep_dir = raw_dir / ep_id
    ep_dir.mkdir(parents=True, exist_ok=True)
    scene_yaml = ep_dir / "scene.yaml"

    t0 = time.monotonic()
    try:
        make_scene(scenario, seed, scene_yaml)
        ok = run_episode_via_compose(ep_dir, scene_yaml, args)
    except Exception as e:
        logger.exception("episode %s failed", ep_id)
        return EpisodeResult(ep_id=ep_id, scenario=scenario, seed=seed,
                             success=False, duration_s=time.monotonic() - t0,
                             error=str(e))

    # scoring.yaml + bag duration 기반 사후 검증.
    # 이전 v2 회귀의 원인 = "approach 시작 직후 cut된 trajectory가 dataset 대부분"
    # 을 차단. 실패 ep는 raw_dir/_failed/ 로 격리되어 학습에 안 들어감.
    validation_ok = True
    validation_reason = ""
    if not args.dry_run:
        from data.validate_episode import quarantine, validate
        result = validate(
            ep_dir,
            min_trial_score=50.0,
            require_full_insertion=False,
            min_bag_duration=20.0,
        )
        validation_ok = result.valid
        validation_reason = result.reason
        if not result.valid:
            quarantine_root = raw_dir / "_failed"
            try:
                quarantine(ep_dir, quarantine_root)
                logger.warning("%s quarantined: %s", ep_id, result.reason)
            except OSError as e:
                logger.warning("%s quarantine failed: %s", ep_id, e)

    return EpisodeResult(
        ep_id=ep_id, scenario=scenario, seed=seed,
        success=ok and validation_ok,
        duration_s=time.monotonic() - t0,
        error=validation_reason if not validation_ok else None,
    )


# --------------------------------------------------------------------- #
# Post-processing: LeRobot conversion + HF upload
# --------------------------------------------------------------------- #
def _maybe_export_sample(ep_dir: Path, args: argparse.Namespace) -> None:
    """수집된 ep에서 center_camera 비디오 1개를 mp4로 추출 + HF push.
    사용자가 콘솔에서 한 번 보고 OK 사인 → 다음 batch.
    잘못된 데이터 1000ep 만들기 전에 차단."""
    if args.dry_run:
        return
    if shutil.which("ros2") is None or shutil.which("ffmpeg") is None:
        logger.debug("ros2 or ffmpeg not available — skip sample export")
        return

    # rosbag 안의 center_camera 토픽을 mp4로 변환하는 정확한 명령은
    # vast.ai 환경에서 검증 후 보강. 우선 placeholder:
    mp4_path = ep_dir / "sample.mp4"
    logger.info("[sample] would extract video from %s (NOT YET IMPLEMENTED)", ep_dir)
    # TODO(vast.ai): ros2 image_publisher + ffmpeg pipe 로 mp4 생성
    #   pixi run ros2 bag play <ep_dir> --topics /center_camera/image &
    #   pixi run ros2 run image_view image_saver image:=/center_camera/image
    #   ffmpeg -framerate 20 -i frame%04d.png -c:v libx264 -pix_fmt yuv420p mp4_path

    # HF dataset repo에 _samples/ 디렉토리로 push
    if args.push_hf and mp4_path.exists() and shutil.which("hf"):
        try:
            subprocess.run(
                ["hf", "upload", args.push_hf, str(mp4_path),
                 f"_samples/{ep_dir.name}.mp4",
                 "--repo-type", "dataset",
                 "--commit-message", f"sample {ep_dir.name}"],
                check=False, cwd=ROOT,
            )
            logger.info("[sample] pushed → hf://%s/_samples/%s.mp4",
                        args.push_hf, ep_dir.name)
        except Exception as e:
            logger.warning("sample push failed: %s", e)


def maybe_convert_to_lerobot(args: argparse.Namespace, raw_dir: Path) -> None:
    if not args.convert_to_lerobot:
        return
    if args.dry_run:
        logger.info("[dry_run] would convert %s → %s", raw_dir, args.convert_to_lerobot)
        return
    cmd = [sys.executable, str(ROOT / "data" / "bag_to_lerobot.py"),
           "--bag_root", str(raw_dir),
           "--out_root", args.convert_to_lerobot,
           "--repo_id", "nggw519/aic_v0",
           "--fps", "20"]
    logger.info("converting bags → LeRobot: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def maybe_push_hf(args: argparse.Namespace) -> None:
    if not args.push_hf or not args.convert_to_lerobot:
        return
    if args.dry_run:
        logger.info("[dry_run] would upload %s → hf://%s", args.convert_to_lerobot, args.push_hf)
        return
    # huggingface_hub 1.x: CLI 이름이 `huggingface-cli` → `hf` 로 변경.
    if shutil.which("hf") is None:
        logger.warning("hf CLI not found — skipping HF upload (pip install huggingface_hub)")
        return
    cmd = ["hf", "upload", args.push_hf,
           args.convert_to_lerobot, ".",
           "--repo-type", "dataset",
           "--commit-message", "demos batch"]
    logger.info("uploading to HF: %s", " ".join(cmd))
    subprocess.run(cmd, check=False, cwd=ROOT)


# --------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    dist = parse_dist(args.dist)
    rng = random.Random(args.seed_shuffle)
    raw_dir = (ROOT / args.out).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    logger.info("collecting %d episodes → %s (dist=%s)", args.n, raw_dir, dist)

    results: list[EpisodeResult] = []
    abort_min = max(10, args.n // 10)  # 처음 max(10, N/10)개 후부터 success rate 게이트
    for i in range(args.n):
        scenario = sample_scenario(rng, dist)
        seed = args.seed_base + i
        r = collect_one(args, i, scenario, seed, raw_dir)
        results.append(r)
        logger.info("[%d/%d] %s scenario=%s seed=%d ok=%s dur=%.1fs",
                    i + 1, args.n, r.ep_id, r.scenario, r.seed, r.success, r.duration_s)

        # 검증 게이트 1: success rate 너무 낮으면 즉시 중단 (잘못된 데이터 다량 방지)
        if len(results) >= abort_min:
            success_rate = sum(1 for x in results if x.success) / len(results)
            if success_rate < args.min_success_rate:
                logger.error(
                    "ABORT: success_rate=%.2f < %.2f after %d episodes. "
                    "Likely systemic issue (timeout, CheatCode misconfig, etc). "
                    "Inspect %s/_failed/ before retrying.",
                    success_rate, args.min_success_rate, len(results), raw_dir,
                )
                break

        # 검증 게이트 2: N마다 sample mp4 추출 + HF dataset에 업로드
        if args.sample_every > 0 and (i + 1) % args.sample_every == 0 and r.success:
            try:
                _maybe_export_sample(raw_dir / r.ep_id, args)
            except Exception as e:
                logger.warning("sample export failed for %s: %s", r.ep_id, e)

    summary_path = raw_dir / "collection_summary.json"
    summary = {
        "n_total": len(results),
        "n_success": sum(1 for r in results if r.success),
        "n_failure": sum(1 for r in results if not r.success),
        "scenarios": {s: sum(1 for r in results if r.scenario == s) for s in dist},
        "duration_total_s": sum(r.duration_s for r in results),
        "args": vars(args),
        "episodes": [asdict(r) for r in results],
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("summary → %s (success %d/%d)",
                summary_path, summary["n_success"], summary["n_total"])

    maybe_convert_to_lerobot(args, raw_dir)
    maybe_push_hf(args)

    return 0 if summary["n_failure"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
