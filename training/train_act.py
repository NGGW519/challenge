#!/usr/bin/env python3
"""LeRobot ACT를 우리 데이터셋(aic_v0 / aic_v1_*)으로 fine-tune.

- 자동 재개: --resume 시 output_dir의 최신 step_*.pt에서 시작.
- 매 ckpt_every step마다 R2/원격으로 push (rclone).
- vast.ai 인스턴스 다운에 강건.

스모크 테스트 (실데이터 없이 import만 검증):
    python3 training/train_act.py --smoke

실 학습 (인스턴스 안에서):
    python3 training/train_act.py \
        --repo_id      nggw519/aic_v0 \
        --dataset_root datasets/aic_v0 \
        --output_dir   models/act_v1 \
        --steps        200000 \
        --batch_size   32 \
        --lr           1e-4 \
        --remote       hf://nggw519/aic-ckpts/act_v1

remote 형식: `hf://<repo_id>/<path_in_repo>` — HuggingFace Hub model repo로 push.
미지정 시 push 생략 (로컬에만 저장).

LeRobot 0.2x API에 맞춤. 실제 버전 차이 시 ACTConfig / select_action 호출
시그니처를 그에 맞춰 조정해야 한다 (LeRobot CHANGELOG 확인).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("train_act")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id", default="nggw519/aic_v0")
    p.add_argument("--dataset_root", default="datasets/aic_v0")
    p.add_argument("--output_dir", default="models/act_v1")
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--ckpt_every", type=int, default=5_000)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--remote", default=None,
                   help="HF Hub 원격 (예: hf://nggw519/aic-ckpts/act_v1). 미지정 시 push 생략.")
    p.add_argument("--smoke", action="store_true",
                   help="import만 확인하고 종료 — 의존성 사전 점검용.")
    return p.parse_args(argv)


def _parse_hf_remote(remote: str) -> tuple[str, str]:
    """`hf://<repo_id>/<path_in_repo>` → (repo_id, path_in_repo).

    repo_id는 `org/name` 형식이라 항상 슬래시 1개를 포함한다.
    """
    if not remote.startswith("hf://"):
        raise ValueError(f"remote must start with 'hf://': {remote}")
    body = remote[len("hf://"):]
    parts = body.split("/")
    if len(parts) < 2:
        raise ValueError(f"remote repo_id must be 'org/name': {remote}")
    repo_id = "/".join(parts[:2])
    path_in_repo = "/".join(parts[2:]) or "."
    return repo_id, path_in_repo


def push_ckpt(local: Path, remote: str | None, commit_message: str = "") -> None:
    """local 디렉토리(또는 파일)를 HuggingFace Hub repo로 업로드."""
    if not remote:
        return
    try:
        from huggingface_hub import HfApi
    except ImportError:
        logger.warning("huggingface_hub not installed — skip push")
        return
    try:
        repo_id, path_in_repo = _parse_hf_remote(remote)
        api = HfApi()
        if local.is_dir():
            api.upload_folder(
                repo_id=repo_id,
                folder_path=str(local),
                path_in_repo=path_in_repo,
                repo_type="model",
                commit_message=commit_message or f"upload {local.name}",
            )
        else:
            api.upload_file(
                repo_id=repo_id,
                path_or_fileobj=str(local),
                path_in_repo=f"{path_in_repo}/{local.name}",
                repo_type="model",
                commit_message=commit_message or f"upload {local.name}",
            )
        logger.info("pushed %s → %s/%s", local.name, repo_id, path_in_repo)
    except Exception as e:
        logger.error("HF Hub push failed: %s", e)


def _build_config(ds_features: dict, image_keys: list[str], state_dim: int, action_dim: int):
    """LeRobot ACTConfig — 학습 hyperparameter 표 (strategy/12 §2)와 일치."""
    from lerobot.common.policies.act.configuration_act import ACTConfig
    return ACTConfig(
        n_obs_steps=1,
        chunk_size=100,
        n_action_steps=20,
        input_features={
            **{k: {"shape": tuple(ds_features[k]["shape"])} for k in image_keys},
            "observation.state": {"shape": (state_dim,)},
        },
        output_features={"action": {"shape": (action_dim,)}},
        vision_backbone="resnet18",
        replace_final_stride_with_dilation=False,
        pre_norm=False,
        dim_model=512,
        n_heads=8,
        dim_feedforward=3200,
        n_encoder_layers=4,
        n_decoder_layers=7,
        feedforward_activation="relu",
        latent_dim=32,
        kl_weight=10.0,
        use_vae=True,
        temporal_ensemble_coeff=0.01,
        normalization_mapping={"VISUAL": "MEAN_STD", "STATE": "MEAN_STD", "ACTION": "MEAN_STD"},
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.smoke:
        logger.info("smoke mode — importing deps")
        try:
            import lerobot  # noqa: F401
            import torch
            logger.info("torch + lerobot import ok")
            return 0
        except Exception as e:
            logger.error("smoke import failed: %s", e)
            return 1

    import torch
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.policies.act.modeling_act import ACTPolicy
    from torch.utils.data import DataLoader

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device=%s", device)

    ds = LeRobotDataset(args.repo_id, root=args.dataset_root)
    image_keys = [k for k in ds.features if k.startswith("observation.images.")]
    state_dim = int(ds.features["observation.state"]["shape"][-1])
    action_dim = int(ds.features["action"]["shape"][-1])

    cfg = _build_config(ds.features, image_keys, state_dim, action_dim)
    policy = ACTPolicy(cfg, dataset_stats=ds.meta.stats).to(device)
    optim = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)

    start = 0
    if args.resume:
        ckpts = sorted(out.glob("step_*.pt"))
        if ckpts:
            sd = torch.load(ckpts[-1], map_location=device)
            policy.load_state_dict(sd["policy"])
            optim.load_state_dict(sd["optim"])
            start = sd["step"] + 1
            logger.info("resumed from step %d (%s)", start, ckpts[-1].name)

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=True,
    )

    iterator = iter(loader)
    last_log_t = time.monotonic()
    for step in range(start, args.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        batch = {k: (v.to(device, non_blocking=True) if hasattr(v, "to") else v) for k, v in batch.items()}
        out_dict: Any = policy.forward(batch)
        loss = out_dict[0] if isinstance(out_dict, tuple) else out_dict["loss"]
        info = out_dict[1] if isinstance(out_dict, tuple) and len(out_dict) > 1 else {}

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        optim.step()

        if step % args.log_every == 0:
            now = time.monotonic()
            sps = args.log_every / max(now - last_log_t, 1e-6)
            last_log_t = now
            extra = " ".join(f"{k}={v:.4f}" for k, v in (info or {}).items() if isinstance(v, (int, float)))
            logger.info("step=%d loss=%.4f %s | %.2f step/s", step, float(loss.item()), extra, sps)

        if step > 0 and step % args.ckpt_every == 0:
            ckpt = out / f"step_{step:08d}.pt"
            torch.save(
                {"policy": policy.state_dict(), "optim": optim.state_dict(), "step": step,
                 "cfg": cfg.to_dict() if hasattr(cfg, "to_dict") else cfg.__dict__},
                ckpt,
            )
            logger.info("saved ckpt %s", ckpt.name)
            push_ckpt(ckpt, args.remote, commit_message=f"step {step}")

    final = out / "final.pt"
    torch.save(
        {"policy": policy.state_dict(), "step": args.steps,
         "cfg": cfg.to_dict() if hasattr(cfg, "to_dict") else cfg.__dict__},
        final,
    )
    logger.info("training done — saved %s", final)
    push_ckpt(out, args.remote, commit_message=f"final step {args.steps}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
