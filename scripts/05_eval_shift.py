"""Q3/Q4: success under shift at the certified (lambda_hat, eta_hat) vs base and
uncalibrated best-of-K; stall fraction measures the idle attractor.

    python scripts/05_eval_shift.py --root $ROOT --scorer contrast --n 100 --K 16
"""
import json
from pathlib import Path

import numpy as np

from dnh.envs import PERTURBATIONS, make_envs
from dnh.experiment import base_parser, batches, init_states, load_policy, load_scorer, save_json
from dnh.rollout import Start, run_batch

ap = base_parser(__doc__)
ap.add_argument("--scorer", default="u")
ap.add_argument("--family", default="tilted")
ap.add_argument("--K", type=int, default=16)
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--severities", default="1,2,3")
ap.add_argument("--perturbations", default=",".join(p for p in PERTURBATIONS if p != "none"))
args = ap.parse_args()
root = Path(args.root)
pol = load_policy(args)
scorer = load_scorer(args.scorer, root, args.device)
cal = json.load(open(root / "results" / f"ltt_{args.scorer}_{args.family}.json"))
lam_hat, eta_hat = float(cal["lam_hat"]), float(cal["eta_hat"])
selectors = {"base": (0.0, np.inf), "uncal_bestofK": (np.inf, np.inf), "ours_certified": (lam_hat, eta_hat)}
S0 = init_states(args.n, 50_000 + args.seed)


def stall_fraction(states, thresh=1.0):
    s = np.asarray(states)
    if len(s) < 2:
        return 0.0
    return float((np.linalg.norm(np.diff(s[:, :2], axis=0), axis=1) < thresh).mean())


rows = []
cells = [("none", 0)] + [(p, s) for p in args.perturbations.split(",") for s in map(int, args.severities.split(","))]
for pert, sev in cells:
    for name, (lam, eta) in selectors.items():
        eps_all = []
        for idx in batches(args.n, args.batch):
            envs = make_envs(len(idx), pert, sev, seed=5_000_000 + idx[0])
            eps, _ = run_batch(envs, pol, [Start(init_state=S0[i]) for i in idx], K=args.K, scorer=scorer,
                               lam=lam, eta=eta, family=args.family, seeds=[5_000_000 + i for i in idx])
            eps_all += eps
        row = {"perturbation": pert, "severity": sev, "selector": name,
               "success": np.mean([e.success for e in eps_all]),
               "abstain": np.mean([e.abstained for e in eps_all]),
               "max_coverage": np.mean([e.max_coverage for e in eps_all]),
               "stall": np.mean([stall_fraction(e.states) for e in eps_all])}
        rows.append(row)
        print(row, flush=True)
save_json({"lam_hat": lam_hat, "eta_hat": eta_hat, "rows": rows}, root / "results" / f"shift_{args.scorer}.json")
