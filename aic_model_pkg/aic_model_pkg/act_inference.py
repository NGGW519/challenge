"""ACT 추론 wrapper (Stage B IL).

학습된 가중치는 weights/act_v1.pt 형태로 둔다.
LeRobot의 ACTPolicy를 직접 import할 수 있을 때만 활성화.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ACTInference:
    """학습된 ACT를 chunk 단위로 추론, 1 step씩 반환."""

    def __init__(self, ckpt_path: str | Path | None = None, device: str = "cuda") -> None:
        self.disabled = True
        self.device = device
        self._policy: Any = None
        self._chunk: np.ndarray | None = None
        self._chunk_idx = 0
        if ckpt_path and Path(ckpt_path).exists():
            try:
                self._init_policy(Path(ckpt_path))
                self.disabled = False
                logger.info("ACTInference loaded weights: %s", ckpt_path)
            except Exception as e:  # pragma: no cover
                logger.warning("ACTInference init failed (%s) — Stage B will be a no-op", e)
        else:
            logger.warning("ACT ckpt missing (%s) — Stage B is a no-op", ckpt_path)

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
