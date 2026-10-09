"""Shared helpers for the scripts (argument defaults, loading, batching, saving)."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from .envs import PushT
from .policy import ChunkPolicy, migrate_checkpoint
from .rollout import Episode, Start, _context
from .world_model import ContrastiveScorer, RandomScorer, WMUncertaintyScorer, load_wm


def base_parser(desc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--root", default="runs", help="output root (put this on Google Drive in Colab)")
    ap.add_argument("--ckpt", default=None, help="migrated policy dir (default: <root>/dp_pusht_migrated)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ddim", type=int, default=None, help="DDIM steps; default = original 100-step DDPM")
    ap.add_argument("--batch", type=int, default=64, help="envs run in lockstep")
    ap.add_argument("--seed", type=int, default=0)
    return ap


def load_policy(args) -> ChunkPolicy:
    ckpt = migrate_checkpoint(args.ckpt or Path(args.root) / "dp_pusht_migrated")
    return ChunkPolicy(ckpt, device=args.device, ddim_steps=args.ddim)


def load_scorer(name: str, root: str | Path, device: str, beta: float = 1.0):
    """name in {random, u, u_epi, contrast}."""
    root = Path(root)
    if name == "random":
        return RandomScorer()
    ws = load_wm(root / "wm_success.pt", device)
    if name == "u":
        return WMUncertaintyScorer(ws, "aleatoric")
    if name == "u_epi":
        return WMUncertaintyScorer(ws, "epistemic")
    if name == "contrast":
        wf = load_wm(root / "wm_failure.pt", device)
        return ContrastiveScorer(WMUncertaintyScorer(ws), WMUncertaintyScorer(wf), beta=beta)
    raise ValueError(name)


def init_states(n: int, seed: int) -> np.ndarray:
    env = PushT(seed=seed)
    return np.stack([env.sample_init_state() for _ in range(n)])


def batches(n: int, bs: int):
    for i in range(0, n, bs):
        yield list(range(i, min(n, i + bs)))


def sample_at_starts(policy: ChunkPolicy, starts: list[Start], K: int, seed: int):
    """K candidate chunks at captured branch points, plus WM contexts."""
    N = len(starts)
    policy.reset(N)
    for i, s in enumerate(starts):
        policy.hist[i].extend(s.policy_hist)
    g = torch.Generator(device=policy.device).manual_seed(seed)
    noise = torch.randn((N,) + policy.noise_shape(K), generator=g, device=policy.device)
    chunks, _ = policy.sample_chunks(list(range(N)), K, noise=noise)
    fake = [Episode(states=s.states, actions=s.actions) for s in starts]
    return chunks, fake


def context_for(fake_eps, H):
    return _context(fake_eps, list(range(len(fake_eps))), H)


def save_pickle(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
