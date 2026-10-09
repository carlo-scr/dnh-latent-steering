"""RQ1 / week-1 GO-NO-GO: do imagined WM scores rank candidate chunks by eventual success?

For N branch states (captured mid-episode from base-policy rollouts, nominal and
perturbed), sample K chunks, score each with every scorer, then EXECUTE every
candidate from the restored state and let the base policy finish the episode.
Reports pooled AUROC (score -> failure) and the within-state "steering gain":
P(success | lowest-score candidate) - mean P(success) over states with mixed outcomes.

    python scripts/03_rq1_ranking.py --root $ROOT --n-states 200 --K 16
"""
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from dnh.envs import make_envs
from dnh.experiment import (base_parser, batches, context_for, init_states, load_policy, load_scorer,
                            sample_at_starts, save_json, save_pickle)
from dnh.rollout import Start, run_batch

ap = base_parser(__doc__)
ap.add_argument("--n-states", type=int, default=200)
ap.add_argument("--K", type=int, default=16)
ap.add_argument("--scorers", default="random,u,u_epi,contrast")
ap.add_argument("--perturbation", default="none")
ap.add_argument("--severity", type=int, default=0)
ap.add_argument("--branch-replans", default="2,6,10,16", help="replan indices to branch at")
args = ap.parse_args()
root = Path(args.root)

pol = load_policy(args)
scorers = {}
for name in args.scorers.split(","):
    try:
        scorers[name] = load_scorer(name, root, args.device)
    except FileNotFoundError as e:
        print(f"skip scorer {name}: {e}")
H = 4
branch_at = [int(x) for x in args.branch_replans.split(",")]
S0 = init_states(args.n_states, seed=20_000 + args.seed)

# 1. branch points: run base policy, capture state at a replan index (cycled per state)
starts, t0 = [], time.time()
for r in branch_at:
    idx = [i for i in range(args.n_states) if i % len(branch_at) == branch_at.index(r)]
    for b in batches(len(idx), args.batch):
        ii = [idx[j] for j in b]
        envs = make_envs(len(ii), args.perturbation, args.severity, seed=args.seed + ii[0])
        _, cap = run_batch(envs, pol, [Start(init_state=S0[i]) for i in ii], K=1, capture_at=r, max_replans=r + 1,
                           seeds=list(ii))
        starts += [c for c in cap if c is not None]  # episodes that ended before r are dropped
print(f"{len(starts)} branch states ({time.time() - t0:.0f}s)")

# 2. candidates + scores
chunks, fake = sample_at_starts(pol, starts, args.K, seed=args.seed)
ctx = context_for(fake, H)
scores = {n: s.score(ctx, chunks) for n, s in scorers.items()}  # each [N,K]

# 3. execute every candidate, then base policy to the end (B = states x K envs)
N, K = len(starts), args.K
succ = np.zeros((N, K), bool)
flat = [(i, k) for i in range(N) for k in range(K)]
bs = max(args.batch, K)
for b in batches(len(flat), bs):
    pairs = [flat[j] for j in b]
    envs = make_envs(len(pairs), args.perturbation, args.severity, seed=args.seed + b[0])
    forced = np.stack([chunks[i, k] for i, k in pairs])
    eps, _ = run_batch(envs, pol, [starts[i] for i, _ in pairs], K=1, forced_first=forced,
                       seeds=[1_000_000 + j for j in b])
    for (i, k), e in zip(pairs, eps):
        succ[i, k] = e.success
    print(f"  executed {b[-1] + 1}/{len(flat)} candidates ({time.time() - t0:.0f}s)", flush=True)

# 4. metrics
mixed = (succ.any(1)) & (~succ.all(1))
res = {"n_states": N, "K": K, "base_success": float(succ.mean()), "frac_mixed_states": float(mixed.mean()),
       "oracle_gain": float(succ[mixed].any(1).mean() - succ[mixed].mean()) if mixed.any() else 0.0, "scorers": {}}
for n, s in scores.items():
    r = {}
    if 0 < succ.mean() < 1:
        r["auroc_fail_pooled"] = float(roc_auc_score((~succ).ravel(), s.ravel()))
    if mixed.any():
        best = s[mixed].argmin(1)
        r["p_success_best"] = float(succ[mixed][np.arange(mixed.sum()), best].mean())
        r["p_success_mean"] = float(succ[mixed].mean())
        r["steering_gain"] = r["p_success_best"] - r["p_success_mean"]
        # within-state AUROC averaged over mixed states
        r["auroc_fail_within"] = float(np.mean([roc_auc_score(~succ[i], s[i]) for i in np.flatnonzero(mixed)]))
    res["scorers"][n] = r
    print(n, r)

tag = "nominal" if args.perturbation == "none" else f"{args.perturbation}{args.severity}"
save_json(res, root / "results" / f"rq1_{tag}.json")
save_pickle({"scores": scores, "succ": succ}, root / "results" / f"rq1_{tag}_raw.pkl")
gate = max([v.get("auroc_fail_within", 0) for v in res["scorers"].values() if v] + [0])
print(f"\nGO/NO-GO: best within-state AUROC = {gate:.3f}  ->  {'GO' if gate > 0.6 else 'NO-GO / pivot'}")
