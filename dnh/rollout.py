"""Batched lockstep rollouts: B envs, K candidate chunks each, one policy call per replan."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from .envs import PushT
from .policy import ChunkPolicy
from .steering import select
from .world_model import ACT_DIM, Context, Scorer


@dataclass
class Episode:
    success: bool = False
    abstained: bool = False
    max_coverage: float = 0.0
    steps: int = 0
    states: list = field(default_factory=list)   # low-dim states (for WM training)
    actions: list = field(default_factory=list)
    chunk_log: list = field(default_factory=list)  # per replan: (t, scores[K], chosen)

    def as_arrays(self):
        return {"states": np.asarray(self.states, np.float32), "actions": np.asarray(self.actions, np.float32),
                "success": self.success, "abstained": self.abstained, "max_coverage": self.max_coverage}


@dataclass
class Start:
    """Where an env starts: a reset state, or a captured mid-episode branch point."""
    init_state: np.ndarray | None = None
    snapshot: object = None
    t: int = 0
    policy_hist: list | None = None
    states: list | None = None
    actions: list | None = None


def run_batch(envs: list[PushT], policy: ChunkPolicy, starts: list[Start], *, K: int = 1,
              scorer: Scorer | None = None, lam: float = 0.0, eta: float = np.inf, family: str = "tilted",
              seeds: list[int] | None = None, wm_history: int = 4, forced_first: np.ndarray | None = None,
              max_replans: int | None = None, capture_at: int | None = None, log_chunks: bool = False):
    """Run len(envs) episodes in lockstep.

    seeds        : per-env seed for candidate noise (common random numbers across lambda)
    forced_first : [B, T_a, 2] actions executed at the first replan instead of steering
    max_replans  : stop after this many chunks (for short-horizon branches)
    capture_at   : if set, return Start objects captured at that replan index (branch points)
    """
    B = len(envs)
    dev = policy.device
    seeds = seeds if seeds is not None else list(range(B))
    gens = [torch.Generator(device=dev).manual_seed(int(s)) for s in seeds]
    sel_rng = np.random.default_rng(int(seeds[0]) + 7919)
    policy.reset(B)
    eps = [Episode() for _ in range(B)]
    obs = []
    for i, (env, st) in enumerate(zip(envs, starts)):
        if st.snapshot is not None:
            o = env.restore(st.snapshot, st.t)
            if st.policy_hist is not None:
                policy.hist[i].extend(st.policy_hist)
            eps[i].states = list(st.states)
            eps[i].actions = list(st.actions)
        else:
            o, _ = env.reset(init_state=st.init_state)
            eps[i].states = [env.low_dim_state()]
        obs.append(o)
    fresh = [i for i in range(B) if len(policy.hist[i]) == 0]
    if fresh:
        policy.observe([obs[i] for i in fresh], fresh)
    active = np.ones(B, bool)
    captured: list[Start | None] = [None] * B
    replan = 0
    while active.any() and (max_replans is None or replan < max_replans):
        idx = list(np.flatnonzero(active))
        if capture_at is not None and replan == capture_at:
            for i in idx:
                captured[i] = Start(snapshot=envs[i].snapshot(), t=envs[i].t, policy_hist=list(policy.hist[i]),
                                    states=list(eps[i].states), actions=list(eps[i].actions))
        if forced_first is not None and replan == 0:
            chunks = forced_first[idx][:, None]  # [b,1,T,2]
            pick = np.zeros(len(idx), int)
            scores = np.zeros((len(idx), 1))
        else:
            noise = torch.stack([torch.randn(policy.noise_shape(K), generator=gens[i], device=dev) for i in idx])
            chunks, _ = policy.sample_chunks(idx, K, noise=noise)
            need_scores = scorer is not None and (lam != 0 or np.isfinite(eta) or log_chunks)
            if need_scores:
                ctx = _context(eps, idx, wm_history)
                scores = scorer.score(ctx, chunks)
            else:
                scores = np.zeros((len(idx), K))
            pick = select(scores, lam, eta, family, sel_rng)
        chunk_of = {}
        for j, i in enumerate(idx):
            if log_chunks:
                eps[i].chunk_log.append((envs[i].t, scores[j].copy(), int(pick[j])))
            if pick[j] < 0:
                eps[i].abstained, active[i] = True, False
            else:
                chunk_of[i] = chunks[j, pick[j]]
        idx = list(chunk_of)
        for k in range(policy.T_a):
            stepped, new_obs = [], []
            for i in idx:
                if not active[i]:
                    continue
                a = chunk_of[i][k]
                o, r, term, trunc, info = envs[i].step(a)
                ep = eps[i]
                ep.actions.append(np.asarray(a, np.float32))
                ep.states.append(envs[i].low_dim_state())
                ep.max_coverage = max(ep.max_coverage, float(info["coverage"]))
                ep.steps = envs[i].t
                if term:
                    ep.success = True
                if term or trunc:
                    active[i] = False
                else:
                    stepped.append(i); new_obs.append(o)
            if stepped:
                policy.observe(new_obs, stepped)
        replan += 1
    return eps, captured


def _context(eps: list[Episode], idx: list[int], H: int) -> Context:
    S, A = [], []
    for i in idx:
        s = np.asarray(eps[i].states[-H:], np.float32)
        a_in = [np.zeros(ACT_DIM, np.float32)] + list(eps[i].actions)  # action leading into each state
        a = np.asarray(a_in[len(eps[i].states) - len(s):len(eps[i].states)], np.float32)
        if len(s) < H:
            pad = H - len(s)
            s = np.concatenate([np.repeat(s[:1], pad, 0), s]); a = np.concatenate([np.repeat(a[:1], pad, 0), a])
        S.append(s); A.append(a)
    return Context(np.stack(S), np.stack(A))
