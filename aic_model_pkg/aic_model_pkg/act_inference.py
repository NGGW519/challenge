"""ACT 추론 wrapper (Stage B IL).

학습된 가중치는 weights/act_v1.pt 형태로 둔다.
LeRobot의 ACTPolicy를 직접 import할 수 있을 때만 활성화.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ACTPlus.py 패턴: checkpoint 해상 우선순위
# 1. AIC_POLICY_LOCAL_DIR — 제출 이미지에 bake된 디렉토리
# 2. AIC_POLICY_HF_REPO   — HuggingFace 모델 repo
# 3. (constructor 인자)    — 로컬 .pt 파일
DEFAULT_HF_REPO = os.environ.get("AIC_POLICY_HF_REPO", "nggw519/aic-ckpts")


def resolve_ckpt_path(arg_path: str | Path | None) -> Path | None:
    """checkpoint 위치를 환경변수 우선순위로 해상.

    반환: 디렉토리 또는 단일 .pt 파일 경로. 미해상 시 None.

    Priority:
      1. AIC_POLICY_LOCAL_DIR (env) — 제출 컨테이너에 bake된 디렉토리
      2. arg_path (constructor 인자) — 직접 지정한 ckpt 파일
      3. AIC_POLICY_HF_REPO (env) — HuggingFace snapshot_download
    """
    local_dir = os.environ.get("AIC_POLICY_LOCAL_DIR")
    if local_dir:
        p = Path(local_dir)
        if p.exists():
            logger.info("ACT ckpt: AIC_POLICY_LOCAL_DIR=%s", p)
            return p
        logger.warning("AIC_POLICY_LOCAL_DIR set but missing: %s", p)

    if arg_path is not None:
        p = Path(arg_path)
        if p.exists():
            logger.info("ACT ckpt: arg=%s", p)
            return p
        logger.warning("constructor arg path missing: %s", p)

    hf_repo = os.environ.get("AIC_POLICY_HF_REPO", DEFAULT_HF_REPO)
    if hf_repo:
        try:
            from huggingface_hub import snapshot_download
            p = Path(snapshot_download(
                repo_id=hf_repo,
                allow_patterns=["*.pt", "*.safetensors", "config.json", "stats.json"],
            ))
            logger.info("ACT ckpt: HF repo %s → %s", hf_repo, p)
            return p
        except Exception as e:
            logger.warning("HF snapshot_download(%s) failed: %s", hf_repo, e)

    return None


class ACTInference:
    """학습된 ACT를 chunk 단위로 추론, 1 step씩 반환."""

    def __init__(self, ckpt_path: str | Path | None = None, device: str = "cuda") -> None:
        self.disabled = True
        self.device = device
        self._policy: Any = None
        self._chunk: np.ndarray | None = None
        self._chunk_idx = 0

        # ACTPlus 패턴: env var (LOCAL_DIR > HF_REPO) > 생성자 인자 순으로 해상.
        resolved = resolve_ckpt_path(ckpt_path)
        if resolved is None:
            logger.warning("ACT ckpt unresolvable — Stage B is a no-op")
            return

        # 디렉토리면 최신 step_*.pt 또는 final.pt 선택
        if resolved.is_dir():
            ckpt = (resolved / "final.pt")
            if not ckpt.exists():
                ckpts = sorted(resolved.glob("step_*.pt"))
                ckpt = ckpts[-1] if ckpts else None
            if ckpt is None or not ckpt.exists():
                logger.warning("no .pt file in %s — Stage B is a no-op", resolved)
                return
            resolved = ckpt

        try:
            self._init_policy(resolved)
            self.disabled = False
            logger.info("ACTInference loaded weights: %s", resolved)
        except Exception as e:  # pragma: no cover
            logger.warning("ACTInference init failed (%s) — Stage B will be a no-op", e)

    def _init_policy(self, ckpt: Path) -> None:
        import torch
        from lerobot.common.policies.act.configuration_act import ACTConfig
        from lerobot.common.policies.act.modeling_act import ACTPolicy

        sd = torch.load(ckpt, map_location=self.device)
        cfg = ACTConfig(**sd["cfg"]) if isinstance(sd.get("cfg"), dict) else sd["cfg"]
        self._policy = ACTPolicy(cfg).to(self.device).eval()
        self._policy.load_state_dict(sd["policy"])
        self._torch = torch  # 보관

    def reset(self) -> None:
        self._chunk = None
        self._chunk_idx = 0

    def select_action(self, obs: dict[str, np.ndarray]) -> np.ndarray | None:
        """obs: {observation.images.left, ..., observation.state}.
        반환: action (shape=(7,)) — Cartesian delta + gripper. 비활성 시 None."""
        if self.disabled or self._policy is None:
            return None
        if self._chunk is None or self._chunk_idx >= len(self._chunk):
            self._chunk = self._infer_chunk(obs)
            self._chunk_idx = 0
        a = self._chunk[self._chunk_idx]
        self._chunk_idx += 1
        return a.copy()

    def _infer_chunk(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        torch = self._torch
        with torch.no_grad():
            batch = {k: torch.from_numpy(v).unsqueeze(0).to(self.device) for k, v in obs.items()}
            out = self._policy.select_action(batch)  # LeRobot 0.2x: returns single step
        # LeRobot은 한 step씩만 반환할 수도 있음 → 안전하게 1-step로 fallback
        a = out.squeeze(0).cpu().numpy()
        if a.ndim == 1:
            return a[None, ...]
        return a
