"""Frozen LeRobot Diffusion Policy that samples K action chunks per env in one batched call.

We bypass `select_action` (single sample, internal queues) and keep our own
per-env observation history, so that B envs x K candidates go through the
U-Net as one (B*K)-sized batch.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch

HUB_ID = "lerobot/diffusion_pusht"


def migrate_checkpoint(out_dir: str | Path, hub_id: str = HUB_ID) -> Path:
    """The hub checkpoint predates lerobot>=0.4 processors; convert it once."""
    out_dir = Path(out_dir)
    if (out_dir / "policy_preprocessor.json").exists():
        return out_dir
    import subprocess
    import sys

    subprocess.run([sys.executable, "-m", "lerobot.processor.migrate_policy_normalization",
                    "--pretrained-path", hub_id, "--output-dir", str(out_dir)], check=True)
    return out_dir


class ChunkPolicy:
    def __init__(self, ckpt_dir: str | Path, device: str = "cuda", ddim_steps: int | None = None):
        from lerobot.envs.utils import preprocess_observation
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        from lerobot.policies.factory import make_pre_post_processors

        self._prep_obs = preprocess_observation
        self.policy = DiffusionPolicy.from_pretrained(str(ckpt_dir)).to(device).eval()
        self.cfg = self.policy.config
        self.cfg.device = device
        self.pre, self.post = make_pre_post_processors(
            self.cfg, pretrained_path=str(ckpt_dir),
            preprocessor_overrides={"device_processor": {"device": device}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )
        self.device = device
        self.n_obs = self.cfg.n_obs_steps
        self.T_a = self.cfg.n_action_steps
        self.horizon = self.cfg.horizon
        self.act_dim = self.cfg.action_feature.shape[0]
        if ddim_steps is not None:
            self.use_ddim(ddim_steps)
        self.hist: list[deque] = []

    def use_ddim(self, steps: int):
        """Deterministic sampler: the initial noise xi then fully determines the chunk
        (needed for noise-space steering). Note: this changes the base policy slightly."""
        from diffusers import DDIMScheduler

        d = self.policy.diffusion
        old = d.noise_scheduler.config
        d.noise_scheduler = DDIMScheduler(
            num_train_timesteps=old.num_train_timesteps, beta_start=old.beta_start, beta_end=old.beta_end,
            beta_schedule=old.beta_schedule, clip_sample=old.clip_sample,
            clip_sample_range=old.clip_sample_range, prediction_type=old.prediction_type,
        )
        d.num_inference_steps = steps

    # -------------------------------------------------------------- history
    def reset(self, n_envs: int):
        self.hist = [deque(maxlen=self.n_obs) for _ in range(n_envs)]

    @torch.no_grad()
    def observe(self, obs_list: list[dict], idx: list[int] | None = None):
        """Append raw env observations (pixels_agent_pos dicts) for envs `idx`."""
        idx = list(range(len(obs_list))) if idx is None else idx
        raw = {"pixels": np.stack([o["pixels"] for o in obs_list]),
               "agent_pos": np.stack([o["agent_pos"] for o in obs_list])}
        b = self.pre(self._prep_obs(raw))
        img, st = b["observation.image"], b["observation.state"]
        for j, i in enumerate(idx):
            h = self.hist[i]
            item = (img[j], st[j])
            if len(h) == 0:  # first step: pad history by repetition (as LeRobot does)
                for _ in range(self.n_obs - 1):
                    h.append(item)
            h.append(item)

    # ------------------------------------------------------------- sampling
    def noise_shape(self, n: int):
        return (n, self.horizon, self.act_dim)

    @torch.no_grad()
    def sample_chunks(self, idx: list[int], K: int, generator: torch.Generator | None = None,
                      noise: torch.Tensor | None = None):
        """Returns (actions_env [B,K,T_a,2] numpy, noise [B,K,horizon,2] tensor)."""
        B = len(idx)
        imgs = torch.stack([torch.stack([x[0] for x in self.hist[i]]) for i in idx])  # B,n_obs,C,H,W
        sts = torch.stack([torch.stack([x[1] for x in self.hist[i]]) for i in idx])  # B,n_obs,D
        batch = {
            "observation.state": sts.repeat_interleave(K, 0),
            "observation.images": imgs.repeat_interleave(K, 0).unsqueeze(2),  # one camera
        }
        d = self.policy.diffusion
        if noise is None:
            noise = torch.randn(self.noise_shape(B * K), generator=generator, device=self.device)
        else:
            noise = noise.reshape(self.noise_shape(B * K)).to(self.device)
        cond = d._prepare_global_conditioning(batch)
        sample = d.conditional_sample(B * K, global_cond=cond, noise=noise, generator=generator)
        start = self.n_obs - 1
        act_norm = sample[:, start:start + self.T_a]
        act = self.post(act_norm)
        act = act.reshape(B, K, self.T_a, self.act_dim).cpu().numpy()
        return act, noise.reshape(B, K, self.horizon, self.act_dim)
