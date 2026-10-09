"""Scorers: map (per-env history, K candidate chunks) -> score [B, K] (lower = better).

`ProbWM` is a STAND-IN for the lab's Cosmos-latent probabilistic WM (Ward et al.):
same recipe (history-conditioned Gaussian head, NLL, trained on nominal data
only, uncertainty = mean predicted std), but on Push-T's low-dimensional state
so the whole pipeline runs before the real checkpoint is available. Swap in
the real model by implementing `Scorer.score` with the same signature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

STATE_DIM, ACT_DIM = 6, 2


@dataclass
class Context:
    """Per-env history at a replanning point (rows = envs)."""
    states: np.ndarray   # [B, H, STATE_DIM]  (last row = current state)
    actions: np.ndarray  # [B, H, ACT_DIM]    (actions that led into each state; zeros at start)


class Scorer:
    def score(self, ctx: Context, chunks: np.ndarray) -> np.ndarray:  # chunks [B,K,T,2]
        raise NotImplementedError


class RandomScorer(Scorer):
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def score(self, ctx, chunks):
        return self.rng.random(chunks.shape[:2])


# --------------------------------------------------------------------------- WM
class _Member(nn.Module):
    def __init__(self, H: int, hidden: int = 256):
        super().__init__()
        d_in = H * (STATE_DIM + ACT_DIM) + ACT_DIM
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 2 * STATE_DIM))

    def forward(self, x):
        mu, log_std = self.net(x).chunk(2, dim=-1)
        return mu, log_std.clamp(-6.0, 2.0).exp()


class ProbWM(nn.Module):
    """Ensemble of history-conditioned Gaussian next-state-delta models."""

    def __init__(self, H: int = 4, n_members: int = 5, hidden: int = 256):
        super().__init__()
        self.H = H
        self.members = nn.ModuleList([_Member(H, hidden) for _ in range(n_members)])
        for name in ("s_mean", "s_std", "a_mean", "a_std", "d_mean", "d_std"):
            self.register_buffer(name, torch.zeros(STATE_DIM if name[0] in "sd" else ACT_DIM))

    # normalisation ---------------------------------------------------------
    def fit_norm(self, S: np.ndarray, A: np.ndarray, D: np.ndarray):
        for (m, s), X in zip([("s_mean", "s_std"), ("a_mean", "a_std"), ("d_mean", "d_std")], [S, A, D]):
            getattr(self, m).copy_(torch.as_tensor(X.mean(0)))
            getattr(self, s).copy_(torch.as_tensor(X.std(0) + 1e-6))

    def _inp(self, hs, ha, a_next):
        hs = (hs - self.s_mean) / self.s_std
        ha = (ha - self.a_mean) / self.a_std
        a_next = (a_next - self.a_mean) / self.a_std
        return torch.cat([hs.flatten(1), ha.flatten(1), a_next], dim=-1)

    def forward(self, hs, ha, a_next):
        """-> mu, std of normalised delta, each [M, N, STATE_DIM]."""
        x = self._inp(hs, ha, a_next)
        outs = [m(x) for m in self.members]
        return torch.stack([o[0] for o in outs]), torch.stack([o[1] for o in outs])

    def nll(self, hs, ha, a_next, delta):
        mu, std = self(hs, ha, a_next)
        d = (delta - self.d_mean) / self.d_std
        return (0.5 * ((d - mu) / std) ** 2 + std.log()).mean()

    @torch.no_grad()
    def imagine(self, hs, ha, chunk):
        """Mean-propagate a chunk. hs [N,H,6], ha [N,H,2], chunk [N,T,2].
        Returns aleatoric std [N,T] and ensemble disagreement [N,T] (normalised units)."""
        alea, epi = [], []
        hs, ha = hs.clone(), ha.clone()
        for t in range(chunk.shape[1]):
            a = chunk[:, t]
            mu, std = self(hs, ha, a)
            alea.append(std.mean(dim=(0, 2)))
            epi.append(mu.std(dim=0).mean(dim=-1))
            delta = mu.mean(0) * self.d_std + self.d_mean
            s_next = hs[:, -1] + delta
            hs = torch.cat([hs[:, 1:], s_next[:, None]], 1)
            ha = torch.cat([ha[:, 1:], a[:, None]], 1)
        return torch.stack(alea, 1), torch.stack(epi, 1)


def windows_from_episodes(episodes: list[dict], H: int):
    """episodes: dicts with 'states' [T+1,6], 'actions' [T,2]. Returns training tensors."""
    HS, HA, AN, D = [], [], [], []
    for ep in episodes:
        S, A = ep["states"], ep["actions"]
        A_in = np.concatenate([np.zeros((1, ACT_DIM), np.float32), A], 0)  # action leading into state t
        for t in range(len(A)):
            lo = max(0, t - H + 1)
            hs, ha = S[lo:t + 1], A_in[lo:t + 1]
            if len(hs) < H:  # left-pad by repetition
                pad = H - len(hs)
                hs = np.concatenate([np.repeat(hs[:1], pad, 0), hs], 0)
                ha = np.concatenate([np.repeat(ha[:1], pad, 0), ha], 0)
            HS.append(hs); HA.append(ha); AN.append(A[t]); D.append(S[t + 1] - S[t])
    f = lambda x: np.asarray(x, np.float32)
    return f(HS), f(HA), f(AN), f(D)


def train_wm(episodes, H=4, n_members=5, epochs=30, bs=1024, lr=1e-3, device="cuda", log=print):
    HS, HA, AN, D = windows_from_episodes(episodes, H)
    wm = ProbWM(H, n_members).to(device)
    wm.fit_norm(HS.reshape(-1, STATE_DIM), AN, D)
    t = lambda x: torch.as_tensor(x, device=device)
    HS, HA, AN, D = t(HS), t(HA), t(AN), t(D)
    opt = torch.optim.Adam(wm.parameters(), lr=lr)
    n = len(HS)
    for ep in range(epochs):
        perm = torch.randperm(n, device=device)
        tot = 0.0
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            loss = wm.nll(HS[j], HA[j], AN[j], D[j])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(j)
        if ep % 5 == 0 or ep == epochs - 1:
            log(f"[wm] epoch {ep:3d}  nll {tot / n:.4f}")
    return wm.eval()


# ---------------------------------------------------------------------- scorers
class WMUncertaintyScorer(Scorer):
    """u(A) = max_t mean predicted std over the imagined chunk (Ward et al. score).
    mode='aleatoric' matches the paper; 'epistemic' uses ensemble disagreement."""

    def __init__(self, wm: ProbWM, mode: str = "aleatoric"):
        self.wm, self.mode = wm, mode
        self.device = next(wm.parameters()).device

    def per_step(self, ctx: Context, chunks: np.ndarray) -> np.ndarray:
        B, K, T, _ = chunks.shape
        H = self.wm.H
        t = lambda x: torch.as_tensor(np.asarray(x, np.float32), device=self.device)
        hs = t(ctx.states[:, -H:]).repeat_interleave(K, 0)
        ha = t(ctx.actions[:, -H:]).repeat_interleave(K, 0)
        alea, epi = self.wm.imagine(hs, ha, t(chunks.reshape(B * K, T, -1)))
        x = alea if self.mode == "aleatoric" else epi
        return x.reshape(B, K, T).cpu().numpy()

    def score(self, ctx, chunks):
        return self.per_step(ctx, chunks).max(-1)


class ContrastiveScorer(Scorer):
    """c(A) = u_s(A) - beta * min(u_f(A), kappa): nominal-like AND unlike known failures."""

    def __init__(self, success: WMUncertaintyScorer, failure: WMUncertaintyScorer, beta=1.0, kappa=np.inf):
        self.s, self.f, self.beta, self.kappa = success, failure, beta, kappa

    def score(self, ctx, chunks):
        return self.s.score(ctx, chunks) - self.beta * np.minimum(self.f.score(ctx, chunks), self.kappa)


def save_wm(wm: ProbWM, path):
    torch.save({"H": wm.H, "n_members": len(wm.members), "state_dict": wm.state_dict()}, path)


def load_wm(path, device="cuda") -> ProbWM:
    ck = torch.load(path, map_location=device)
    wm = ProbWM(ck["H"], ck["n_members"]).to(device)
    wm.load_state_dict(ck["state_dict"])
    return wm.eval()
