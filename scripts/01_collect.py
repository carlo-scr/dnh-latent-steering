"""Roll out the frozen base policy (K=1) to get WM training data and a base success rate.

Nominal episodes -> success-WM. Perturbed failures -> failure-WM (Ho et al. style).
    python scripts/01_collect.py --root $ROOT --n 500
    python scripts/01_collect.py --root $ROOT --n 300 --perturbation friction --severity 2
"""
import time
from pathlib import Path

import numpy as np

from dnh.envs import make_envs
from dnh.experiment import base_parser, batches, init_states, load_policy, save_pickle
from dnh.rollout import Start, run_batch

ap = base_parser(__doc__)
ap.add_argument("--n", type=int, default=500)
ap.add_argument("--perturbation", default="none")
ap.add_argument("--severity", type=int, default=0)
args = ap.parse_args()

pol = load_policy(args)
S0 = init_states(args.n, seed=10_000 + args.seed)  # disjoint from calibration seeds
out, t0 = [], time.time()
for idx in batches(args.n, args.batch):
    envs = make_envs(len(idx), args.perturbation, args.severity, seed=args.seed + idx[0])
    eps, _ = run_batch(envs, pol, [Start(init_state=S0[i]) for i in idx], K=1,
                       seeds=[args.seed * 100_000 + i for i in idx])
    out += [e.as_arrays() | {"init_state": S0[i]} for e, i in zip(eps, idx)]
    sr = np.mean([e["success"] for e in out])
    print(f"{len(out)}/{args.n} episodes  success={sr:.3f}  ({time.time() - t0:.0f}s)", flush=True)

tag = "nominal" if args.perturbation == "none" else f"{args.perturbation}{args.severity}"
path = Path(args.root) / "data" / f"rollouts_{tag}.pkl"
save_pickle(out, path)
print("saved", path)
