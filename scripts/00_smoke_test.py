"""Checks every moving part on a handful of episodes. Run first on a new machine/Colab.

python scripts/00_smoke_test.py --ckpt /content/drive/MyDrive/dnh/dp_pusht_migrated
"""
import argparse
import time

import numpy as np
import torch

from dnh.envs import PushT, make_envs
from dnh.policy import ChunkPolicy, migrate_checkpoint
from dnh.rollout import Start, run_batch

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="dp_pusht_migrated")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--n", type=int, default=4)
ap.add_argument("--K", type=int, default=4)
ap.add_argument("--ddim", type=int, default=None, help="DDIM steps (default: original 100-step DDPM)")
ap.add_argument("--max-replans", type=int, default=None)
args = ap.parse_args()

# 1. snapshot / restore is exact
env = PushT()
env.reset(init_state=env.sample_init_state())
for _ in range(10):
    env.step(np.array([256.0, 256.0]))
snap = env.snapshot()
a = [np.array([300.0, 200.0])] * 15
o1 = [env.step(x)[0]["agent_pos"] for x in a][-1]
env.restore(snap)
o2 = [env.step(x)[0]["agent_pos"] for x in a][-1]
print(f"[1] snapshot/restore max abs diff: {np.abs(o1 - o2).max():.2e}")

# 2. perturbations construct and step
for p in ["friction", "cog", "damping", "recolor", "distractor", "action_noise"]:
    e = PushT(p, 3); e.reset(init_state=e.sample_init_state()); e.step(np.array([256.0, 256.0]))
print("[2] all perturbation families step OK")

# 3. policy loads and samples K chunks per env
ckpt = migrate_checkpoint(args.ckpt)
pol = ChunkPolicy(ckpt, device=args.device, ddim_steps=args.ddim)
print(f"[3] policy on {args.device}: n_obs={pol.n_obs} T_a={pol.T_a} horizon={pol.horizon}")

# 4. batched base-policy episodes (K candidates sampled, candidate 0 executed)
envs = make_envs(args.n)
starts = [Start(init_state=e.sample_init_state()) for e in envs]
t0 = time.time()
eps, _ = run_batch(envs, pol, starts, K=args.K, max_replans=args.max_replans)
dt = time.time() - t0
print(f"[4] {args.n} episodes in {dt:.1f}s; success={[e.success for e in eps]} "
      f"max_cov={[round(e.max_coverage, 2) for e in eps]} steps={[e.steps for e in eps]}")

# 5. common random numbers: same seeds + lam=0 reproduce the same episode
eps2, _ = run_batch(make_envs(args.n), pol, starts, K=args.K, max_replans=args.max_replans)
same = all(np.allclose(a.as_arrays()["actions"], b.as_arrays()["actions"]) for a, b in zip(eps, eps2))
print(f"[5] paired re-run identical: {same}")
