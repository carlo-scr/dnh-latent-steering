"""Learn-then-Test (Angelopoulos et al. 2021) for do-no-harm steering calibration.

H_theta: R_harm(theta) > eps  OR  R_abs(theta) > alpha.
Both losses are Bernoulli per episode, so exact binomial-tail p-values are valid;
max(p_harm, p_abs) is a valid p-value for the union null. Fixed-sequence testing
along increasing steering strength controls FWER at delta without assuming the
risk is monotone in lambda. Across an eta grid we Bonferroni-split delta.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binom


def binom_pvalue(n_losses: int, n: int, level: float) -> float:
    """P[Bin(n, level) <= n_losses]: small when the observed risk is well below `level`."""
    return float(binom.cdf(n_losses, n, level))


def paired_harm(y_base: np.ndarray, y_steer: np.ndarray) -> np.ndarray:
    """1[base succeeded and steered failed] per paired episode."""
    return ((np.asarray(y_base) == 1) & (np.asarray(y_steer) == 0)).astype(int)


def fixed_sequence(pvals: list[float], delta: float) -> int:
    """Index of the last hypothesis rejected before the first non-rejection (-1 if none)."""
    last = -1
    for i, p in enumerate(pvals):
        if p > delta:
            break
        last = i
    return last


def calibrate(lams: list[float], etas: list[float], harm: dict, abst: dict,
              eps: float = 0.05, alpha: float = 0.10, delta: float = 0.10) -> dict:
    """harm[(lam, eta)], abst[(lam, eta)]: 0/1 arrays over the n calibration episodes.
    lams must be ordered by increasing steering strength (lams[0] may be 0 = base).
    Returns the certified (lam, eta) with the largest lam (ties -> largest eta, i.e. least abstention).
    """
    d = delta / len(etas)
    best, table = None, []
    for eta in etas:
        seq = [lam for lam in lams if lam != 0]
        ps = []
        for lam in seq:
            h, a = np.asarray(harm[(lam, eta)]), np.asarray(abst[(lam, eta)])
            p = max(binom_pvalue(int(h.sum()), len(h), eps), binom_pvalue(int(a.sum()), len(a), alpha))
            ps.append(p)
            table.append({"lam": lam, "eta": eta, "harm": h.mean(), "abst": a.mean(), "p": p})
        k = fixed_sequence(ps, d)
        if k >= 0:
            cand = (seq[k], eta)
            if best is None or cand[0] > best[0] or (cand[0] == best[0] and cand[1] > best[1]):
                best = cand
    return {"lam": 0.0 if best is None else best[0], "eta": np.inf if best is None else best[1],
            "certified": best is not None, "table": table, "delta_per_eta": d}


def min_n_zero_losses(eps: float, delta: float) -> int:
    """Smallest n such that zero observed losses rejects at level delta."""
    return int(np.ceil(np.log(delta) / np.log(1 - eps)))
