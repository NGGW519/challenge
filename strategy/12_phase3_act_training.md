# 12. Phase 3 — ACT (Action Chunking Transformer) 학습

> **목표**: Phase 2에서 수집한 LeRobot 데이터셋(`aic_v0`)으로 LeRobot의 ACT 모델을 fine-tune. Stage B (vision-servoed alignment)의 IL 백본을 만든다.
>
> **기간**: 1.5~2일 (vast.ai L40S 48GB 또는 A100 80GB 1대)
>
> **결과물**: `aic_work/models/act_v1/` — pretrained 가중치 + config + 추론 wrapper.

---

## 1. 왜 ACT인가

| 모델 | 장점 | 단점 |
|---|---|---|
| **ACT (Action Chunking Transformer)** | 짧은 horizon에서 SOTA, smooth action chunk → low jerk, 학습 빠름 | exploration 부족 |
| Diffusion Policy | multi-modal action 분포 표현 | 학습/추론 둘 다 더 비쌈 |
| BC-RNN / LSTM | 가벼움 | 시퀀스 의존성 약함 |
| ALOHA Bimanual | 두 팔 작업용 | 우리 1팔 |

**선택**: ACT v1 baseline → Phase 7에서 Diffusion Policy ablation 비교.

---

## 2. ACT 핵심 hyperparameter

LeRobot 기본값을 토대로 우리 task에 맞춤:

| 파라미터 | 값 | 비고 |
|---|---|---|
| backbone | ResNet18 | 가벼움 / 3 cam × 480×640 |
| dim_model | 512 | encoder/decoder 공통 |
| dim_feedforward | 3200 | |
| num_encoder_layers | 4 | |
| num_decoder_layers | 7 | |
| n_heads | 8 | |
| chunk_size | 100 | 5초 분량 (20FPS) |
| n_action_steps | 20 | 1초어치 실행 후 재추론 (20FPS) |
| kl_weight | 10 | VAE latent 정규화 |
| use_vae | True | |
| latent_dim | 32 | |
| input cameras | left + center + right | |
| input state dim | 12 | 6 joint + grip + 6 wrench |
| action dim | 7 | 6 cart_delta + grip |
| optimizer | AdamW | |
| lr | 1e-5 (backbone) / 1e-4 (head) | |
| lr_schedule | cosine + 500 warmup | |
| batch_size | 32 (L40S) / 64 (A100) | |
| training steps | 200_000 | ~1.5~2일 |
| temporal_ensembling | True (m=0.01) | inference smoothing |

> chunk_size=100의 이유: insertion task는 ~10초 → 절반을 한 chunk로 예측하고, 나머지 절반은 새 observation으로 재추론.

---

## 3. 학습 환경 부팅

```bash
# vast.ai L40S 48GB instance
ssh root@<instance>
cd /workspace/aic_work

# pixi env에 lerobot 추가 (이미 toml에 들어있다면 skip)
pixi add lerobot
pixi run python -c "import lerobot; print(lerobot.__version__)"

# dataset 풀 (R2 또는 HF)
hf auth login
hf download nggw519/aic_v0 --repo-type dataset \
    --local-dir aic_work/datasets/aic_v0
```

---

## 4. 학습 스크립트

`aic_work/training/train_act.py`:

```python
"""LeRobot ACT를 우리 데이터셋으로 fine-tune."""
import argparse
import logging
import os
import subprocess
from pathlib import Path

import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.utils.utils import get_safe_torch_device, set_global_seed
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_id", default="nggw519/aic_v0")
    p.add_argument("--dataset_root", default="aic_work/datasets/aic_v0")
    p.add_argument("--output_dir", default="aic_work/models/act_v1")
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--ckpt_every", type=int, default=5_000)
    p.add_argument("--remote", default="r2:aic-ckpts/act_v1/")
    return p.parse_args()

def push_ckpt(local_dir, remote):
    subprocess.run(["rclone", "copy", str(local_dir), remote, "--progress"], check=True)

def main():
    args = parse_args()
    set_global_seed(args.seed)
    device = get_safe_torch_device("cuda")

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cfg = ACTConfig(
        n_obs_steps=1,
        chunk_size=100,
        n_action_steps=20,
        input_features={
            "observation.images.left":   {"shape": (3, 480, 640)},
            "observation.images.center": {"shape": (3, 480, 640)},
            "observation.images.right":  {"shape": (3, 480, 640)},
            "observation.state":         {"shape": (12,)},
        },
        output_features={"action": {"shape": (7,)}},
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
        normalization_mapping={
            "VISUAL": "MEAN_STD",
            "STATE":  "MEAN_STD",
            "ACTION": "MEAN_STD",
        },
    )

    ds = LeRobotDataset(args.repo_id, root=args.dataset_root)
    cfg.input_features["observation.state"]["shape"] = (ds.features["observation.state"]["shape"][-1],)
    cfg.output_features["action"]["shape"] = (ds.features["action"]["shape"][-1],)

    policy = ACTPolicy(cfg, dataset_stats=ds.meta.stats).to(device)
    optim = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-4)

    start_step = 0
    if args.resume:
        latest = sorted(out.glob("step_*.pt"))[-1] if list(out.glob("step_*.pt")) else None
        if latest:
            sd = torch.load(latest, map_location=device)
            policy.load_state_dict(sd["policy"])
            optim.load_state_dict(sd["optim"])
            start_step = sd["step"]
            logger.info("resumed from step %d", start_step)

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
    iterator = iter(loader)

    for step in range(start_step, args.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)

        batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        loss, info = policy.forward(batch)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        optim.step()

        if step % 100 == 0:
            logger.info("step=%d loss=%.4f l1=%.4f kld=%.4f", step, loss.item(), info["l1_loss"], info["kld_loss"])

        if step > 0 and step % args.ckpt_every == 0:
            ckpt = out / f"step_{step:08d}.pt"
            torch.save({"policy": policy.state_dict(), "optim": optim.state_dict(), "step": step, "cfg": cfg.to_dict()}, ckpt)
            push_ckpt(ckpt, args.remote)

    final = out / "final.pt"
    torch.save({"policy": policy.state_dict(), "step": args.steps, "cfg": cfg.to_dict()}, final)
    push_ckpt(out, args.remote)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
```

위 스크립트는 LeRobot 0.2x API를 가정 — 실제 버전 차이 시 호출 시그니처 (`policy.forward`, `LeRobotDataset(...)`)를 맞춰야 한다. 첫 실행 전 LeRobot 공식 examples로 1 step만 돌려 호환성 확인.

---

## 5. 학습 실행

```bash
cd /workspace/aic_work

# wrapper로 자동 재개
bash scripts/train_with_resume.sh
# = while true; do pixi run python training/train_act.py --resume ... && break; sleep 30; done
```

tmux 윈도우 `train`에서 띄워두고:
- 윈도우 `gpu`에서 `nvtop` 또는 `nvidia-smi -l 1` 모니터
- 윈도우 `disk`에서 ckpt 디스크 사용량
- 윈도우 `logs`에서 `tail -F logs/train_act.log`

---

## 6. 학습 진행 모니터링

| 지표 | 목표 |
|---|---|
| total loss | 시작 ~5 → 50k step에서 < 0.5, 200k에서 < 0.2 |
| L1 action loss | 0.05 이하 (action 정규화 후) |
| KL loss | 1~10 수준 유지 (너무 작으면 collapse) |
| GPU util | ≥ 80% (낮으면 num_workers/batch_size 조정) |
| step/sec | L40S: ~3, A100: ~5 |

10k step마다 (또는 ckpt마다) 짧게 5 trial × 1 rollout 평가하여 점수 추세 추적:

```bash
bash scripts/eval_quick.sh aic_work/models/act_v1/step_00050000.pt
```

---

## 7. 추론 wrapper (Stage B IL 정책)

ACT는 chunk_size=100 action을 한 번에 예측하고 n_action_steps=20개 실행 후 재추론.

`aic_work/policies/act_inference.py`:

```python
"""Hybrid policy의 Stage B에서 호출하는 ACT 추론 모듈."""
import torch
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.act.configuration_act import ACTConfig

class ACTInference:
    def __init__(self, ckpt_path, device="cuda"):
        sd = torch.load(ckpt_path, map_location=device)
        cfg = ACTConfig(**sd["cfg"])
        self.policy = ACTPolicy(cfg).to(device).eval()
        self.policy.load_state_dict(sd["policy"])
        self.device = device
        self._chunk = None
        self._chunk_idx = 0

    @torch.no_grad()
    def reset(self):
        self._chunk = None
        self._chunk_idx = 0

    @torch.no_grad()
    def select_action(self, obs):
        """
        obs: dict with keys observation.images.{left,center,right}, observation.state
        returns: action np.ndarray shape (7,)
        """
        if self._chunk is None or self._chunk_idx >= self._chunk.shape[0]:
            batch = {k: v.unsqueeze(0).to(self.device) for k, v in obs.items()}
            self._chunk = self.policy.select_action(batch)[0].cpu().numpy()  # (chunk, 7)
            self._chunk_idx = 0
        a = self._chunk[self._chunk_idx]
        self._chunk_idx += 1
        return a
```

(LeRobot의 `select_action`은 temporal_ensembling을 내부에서 처리. API 차이가 있으면 그에 맞춰 조정.)

---

## 8. 평가 (Phase 5에서 hybrid policy 합치기 전, 단독 ACT)

ACT만으로 평가:
```bash
# 우선 단독 ACT 정책 wrapper 작성: aic_work/policies/ACTOnlyPolicy.py
bash scripts/run_baseline.sh ACTOnly 30  # 30 rollouts × 3 trial = 90 trial
```

기대 점수 (ACT 단독, hybrid 결합 전): 평균 130~180 / 300.
- Tier 1: 1점 (lifecycle OK)
- Tier 2: smoothness 5+, duration 6~10
- Tier 3: 50% 정확 삽입 / 30% 부분 / 20% miss → 평균 ~40점/trial

미달 시 진단:
- 정확도 낮음 → 데이터 늘리기 (Phase 2 추가) or chunk_size 변경
- jerk 큼 → temporal_ensembling 강화 또는 action 후처리 low-pass
- contact penalty 발생 → safety bbox clamp 추가 (Stage A의 일부)

---

## 9. 결정 로그

### 9.1 chunk_size = 100, n_action_steps = 20
- **결정**: chunk_size 100 (5초 분량 @ 20Hz) + n_action_steps 20 (1초 실행 후 재추론)
- **근거**: ACT 원본 (ALOHA RSS'23) 의 검증된 값. Comp-ACT 도 동일. 우리 task 길이 (~10-30s) 에 적합.
- **트리거**: Phase 5 평가에서 stage 전환 시 stale action (마지막 5초 chunk 가 contact 상황 변화에 둔감) 관찰되면 chunk_size 50 으로 ablation.

### 9.2 backbone = ResNet18
- **결정**: ResNet18 (vision backbone). pretrained_backbone_weights = None (제출 컨테이너의 isolated network 가 torchvision 다운로드 차단 — `aic_act_plus.ros.ACTPlus` 가 확인한 함정).
- **근거**: ACT 원본 + Comp-ACT 동일. 우리 L4 24GB 에서 batch_size=32 + 3 cam 가 안전. 학습된 weights 가 model.safetensors 에서 복원되므로 pretrained 안 받아도 OK.
- **트리거**: Phase 6 multi-sim DR 후에도 OOD 점수 ↓ 면 ResNet34 또는 ViT-Tiny ablation.

### 9.3 학습 step = 200k (cosine + warmup 500)
- **결정**: 200,000 step, AdamW lr=1e-4 (head)/1e-5(backbone), cosine schedule.
- **근거**: ALOHA 학습 step 50k~200k. 우리 1100 ep × ~200 frame ≈ 220k sample → 200k step 이면 ~1 epoch 가까이. L40S 에서 ~1.5-2일.
- **트리거**: total_loss > 0.5 @ step 100k 면 lr/schedule 재고. < 0.2 @ step 150k 면 조기 종료 가능.

### 9.4 state vector = 13-D (joint 7 + force 3 + torque 3) — D2 contract
- **결정**: `observation.state` 차원 13. `HybridPolicy.ACT_STATE_DIM`/`ACT_STATE_LAYOUT` 가 학습/추론 contract.
- **근거**: Day 5 진단 (3/300) — state shape mismatch (26 vs 7) 가 marginal-mean collapse 1차 원인. FARM 논문이 F/T modality 빼면 contact-rich subtask -20~40pt 보고. joint 만으로 부족.
- **트리거**: `tests/test_act_obs_contract.py` 가 회귀 즉시 잡음. 변경 시 학습/추론 양쪽 동시 수정 필요.

### 9.5 action_amplify (bag_to_lerobot) — 데이터셋 std 인위 확대
- **결정**: default `--action-amplify 1.0`, 권장 4.0 (ACTPlus RAW_ACTION_SCALE 패턴).
- **근거**: Day 5 진단 — CheatCode descent 속도 ~1cm/s 라 action.std 너무 작아 ACT 가 평균만 학습. 4배 증폭 시 std 가 학습 의미 있는 범위로 이동.
- **트리거**: Phase 3 학습 후 raw action 변화 < 1% 면 amplify ↑. 학습/추론 amplify 가 동일하면 평가 시 영향 없음.

### 9.6 ckpt 백업 = HF Hub 매 5k step
- **결정**: `train_act.py` 의 `--ckpt_every 5000` + `push_ckpt` 가 `nggw519/aic-ckpts` 에 push.
- **근거**: vast.ai 인스턴스 휘발성. 5,000 step = ~25분 = 손실 상한.

### 9.7 ACT v1 학습 후 ablation 우선순위
1. **Comp-ACT (stiffness head)** — Phase 5 평가에서 force penalty 평균 ≥ 6 이면 발동 (project_design_decisions 참고).
2. **chunk_size 50** — 위 9.1 트리거 조건 시.
3. **state 차원 확장** — wrench 외 controller_state.tcp_velocity (3) + joint_velocity (7) 추가 시도. 단 변경 시 contract test + 재학습 필수.

---

## 10. 완료 기준

- [ ] 학습 200k step 완주 (loss < 0.2)
- [ ] `aic_work/models/act_v1/final.pt` + cfg + stats.json 저장
- [ ] R2/HF Hub 업로드
- [ ] 단독 ACT 평균 점수 ≥ 130 (90 rollout)
- [ ] act_inference.py 동작 확인
