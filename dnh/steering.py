"""Steering operators S_lambda. All satisfy S_0 == base policy (in distribution).

Candidates are i.i.d. policy samples, so "take candidate 0" is distributed exactly
like the base policy; we use that at lambda=0 so paired runs share random numbers.
"""

from __future__ import annotations

import numpy as np


def select(scores: np.ndarray, lam: float, eta: float = np.inf, family: str = "tilted",
           rng: np.random.Generator | None = None) -> np.ndarray:
    """scores [B,K] (lower = better). Returns chosen index per row, or -1 = abstain.

    tilted : sample k ~ softmax(-lam * z_k), z = within-row standardised score
             (lam=0 -> base policy, lam=inf -> argmin, i.e. best-of-K)
    bestofk: lam = number of candidates considered (1 -> base policy)
    Abstention: a candidate is certified iff its raw score <= eta; no certified
    candidate -> -1.
    """
    rng = rng or np.random.default_rng()
    B, K = scores.shape
    certified = scores <= eta
    out = np.full(B, -1, dtype=int)
    for b in range(B):
        s, ok = scores[b], certified[b].copy()
        if family == "bestofk":
            k = int(max(1, min(K, lam)))
            ok[k:] = False
            if ok.any():
                out[b] = int(np.flatnonzero(ok)[np.argmin(s[ok])]) if k > 1 else (0 if ok[0] else -1)
            continue
        if not ok.any():
            continue
        if lam == 0:
            out[b] = int(np.flatnonzero(ok)[0])
            continue
        z = (s - s.mean()) / (s.std() + 1e-8)
        if np.isinf(lam):
            z_ok = np.where(ok, z, np.inf)
            out[b] = int(np.argmin(z_ok))
            continue
        logits = np.where(ok, -lam * z, -np.inf)
        p = np.exp(logits - logits.max())
        out[b] = int(rng.choice(K, p=p / p.sum()))
    return out
