"""Algorithm 1: do-no-harm calibration of (lambda, eta) with Learn-then-Test.

Paired nominal episodes (same init state + same candidate noise seeds) under the base
policy (lambda=0) and every steered configuration, on a calibration split and a
held-out test split. Certifies theta on calibration, reports empirical risk on test.

    python scripts/04_calibrate_ltt.py --root $ROOT --scorer contrast --n-cal 200 --n-test 200 --K 16
"""
import time
from pathlib import Path

import numpy as np

from dnh.envs import make_envs
from dnh.experiment import base_parser, batches, init_states, load_pickle, load_policy, load_scorer, save_json
from dnh.ltt import calibrate, min_n_zero_losses, paired_harm
from dnh.rollout import Episode, Start, _context, run_batch

ap = base_parser(__doc__)
ap.add_argument("--scorer", default="u")
ap.add_argument("--family", default="tilted", choices=["tilted", "bestofk"])
ap.add_argument("--lams", default="0,0.5,1,2,4,inf")
ap.add_argument("--eta-quantiles", default="0.95,0.99", help="eta from nominal WM-training rollouts; inf always added")
ap.add_argument("--K", type=int, default=16)
ap.add_argument("--n-cal", type=int, default=200)
ap.add_argument("--n-test", type=int, default=200)
ap.add_argument("--eps", type=float, default=0.05)
ap.add_argument("--alpha", type=float, default=0.10)
ap.add_argument("--delta", type=float, default=0.10)
args = ap.parse_args()
root = Path(args.root)
pol = load_policy(args)
scorer = load_scorer(args.scorer, root, args.device)
lams = [float(x) for x in args.lams.split(",")]
H = 4
print(f"zero-harm rejection needs n >= {min_n_zero_losses(args.eps, args.delta / 3)} per eta")


def eta_grid():
    """Per-episode max imagined score of the EXECUTED chunks in held-out nominal rollouts
    (WM-training data, disjoint from calibration) -> quantiles. Fixed before calibration."""
    eps = load_pickle(root / "data" / "rollouts_nominal.pkl")
    T = pol.T_a
    maxes = []
    for e in eps[:200]:
        S, A = e["states"], e["actions"]
        per = []
        for t in range(0, len(A) - T + 1, T):
            fake = Episode(states=list(S[:t + 1]), actions=list(A[:t]))
            ctx = _context([fake], [0], H)
            per.append(float(scorer.score(ctx, A[None, None, t:t + T])[0, 0]))
        if per:
            maxes.append(max(per))
    qs = [float(q) for q in args.eta_quantiles.split(",") if q]
    return [float(np.quantile(maxes, q)) for q in qs] + [np.inf]


etas = eta_grid() if args.scorer != "random" else [np.inf]
print("eta grid:", etas)


def run_split(S0, seed_base):
    """returns y[(lam,eta)] success arrays and a[(lam,eta)] abstain arrays."""
    y, a = {}, {}
    configs = [(0.0, np.inf)] + [(l, e) for e in etas for l in lams if l != 0]
    for lam, eta in configs:
        ys, ab = [], []
        for idx in batches(len(S0), args.batch):
            envs = make_envs(len(idx), seed=seed_base + idx[0])
            eps, _ = run_batch(envs, pol, [Start(init_state=S0[i]) for i in idx], K=args.K, scorer=scorer,
                               lam=lam, eta=eta, family=args.family, seeds=[seed_base + i for i in idx])
            ys += [e.success for e in eps]; ab += [e.abstained for e in eps]
        y[(lam, eta)], a[(lam, eta)] = np.array(ys, int), np.array(ab, int)
        print(f"  lam={lam:<5} eta={eta:<8.4g} success={np.mean(ys):.3f} abstain={np.mean(ab):.3f}", flush=True)
    return y, a


t0 = time.time()
print("calibration split"); yc, ac = run_split(init_states(args.n_cal, 30_000 + args.seed), 3_000_000)
print("test split"); yt, at = run_split(init_states(args.n_test, 40_000 + args.seed), 4_000_000)
base_c, base_t = yc[(0.0, np.inf)], yt[(0.0, np.inf)]
harm_c = {k: paired_harm(base_c, v) for k, v in yc.items() if k[0] != 0}
harm_t = {k: paired_harm(base_t, v) for k, v in yt.items() if k[0] != 0}
abst_c = {k: v for k, v in ac.items() if k[0] != 0}
res = calibrate(lams, etas, harm_c, abst_c, args.eps, args.alpha, args.delta)
th = (res["lam"], res["eta"])
out = {"args": vars(args), "etas": etas, "certified": res["certified"], "lam_hat": th[0], "eta_hat": th[1],
       "cal_table": res["table"], "base_success_cal": base_c.mean(), "base_success_test": base_t.mean(),
       "test": [{"lam": k[0], "eta": k[1], "success": yt[k].mean(), "harm": harm_t[k].mean(), "abst": at[k].mean()}
                for k in harm_t]}
if res["certified"]:
    out["test_at_hat"] = {"success": yt[th].mean(), "harm": harm_t[th].mean(), "abst": at[th].mean()}
save_json(out, root / "results" / f"ltt_{args.scorer}_{args.family}.json")
print(f"\ncertified={res['certified']} lam_hat={th[0]} eta_hat={th[1]:.4g}  ({time.time() - t0:.0f}s)")
print("test at hat:", out.get("test_at_hat"))
