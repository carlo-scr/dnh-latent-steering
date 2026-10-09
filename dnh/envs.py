"""Push-T with exact snapshot/restore and parametric perturbations.

`copy.deepcopy` on the gym env fails (pymunk holds cffi handles), so we
snapshot the only dynamic state Push-T has: agent and block kinematics.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless pygame (Colab)

import gym_pusht  # noqa: F401  (registers the env)
import gymnasium as gym
import numpy as np
import pygame

PERTURBATIONS = ("none", "friction", "cog", "damping", "recolor", "distractor", "action_noise")

# severity in {0,1,2,3}; 0 == nominal
_SEVERITY = {
    # gym_pusht sets body.friction (ignored by pymunk), so every shape has friction 0 nominally;
    # scaling it does nothing, so we SET the agent/block contact friction instead.
    "friction": [0.0, 0.3, 0.7, 1.5],
    # Block mass is irrelevant under Push-T's full damping (quasi-static), so the "mass"
    # family is a centre-of-gravity shift (gym_pusht's block_cog), which changes rotation.
    "cog": [None, (0, -10), (0, -20), (0, -30)],
    "damping": [0.0, 0.3, 0.6, 0.9],  # pymunk space.damping = velocity fraction kept per second;
    # nominal Push-T uses 0.0 (fully damped), so higher = block slides/coasts
    "recolor": [None, "IndianRed", "Gold", "MediumSeaGreen"],
    "distractor": [0, 1, 2, 4],  # number of non-colliding distractor blobs
    "action_noise": [0.0, 5.0, 15.0, 30.0],  # std (pixels) of Gaussian noise on target position
}


@dataclass
class Snapshot:
    agent_pos: tuple
    agent_vel: tuple
    block_pos: tuple
    block_angle: float
    block_vel: tuple
    block_angvel: float


class PushT:
    """Thin wrapper around gym_pusht.PushTEnv (pixels_agent_pos observations)."""

    def __init__(self, perturbation: str = "none", severity: int = 0, seed: int = 0):
        assert perturbation in PERTURBATIONS, perturbation
        self.perturbation, self.severity = perturbation, severity
        cog = _SEVERITY["cog"][severity] if perturbation == "cog" else None
        self.env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos", render_mode="rgb_array",
                            block_cog=cog)
        self.u = self.env.unwrapped
        self.rng = np.random.default_rng(seed)
        self.max_steps = 300
        # fixed per env so that resets/restores (paired episodes) see the same scene
        n_d = _SEVERITY["distractor"][severity] if perturbation == "distractor" else 0
        self._distractor_xy = [tuple(self.rng.uniform(60, 450, size=2)) for _ in range(n_d)]

    # ---------------------------------------------------------------- reset
    def reset(self, init_state=None, seed: int | None = None):
        opts = None if init_state is None else {"reset_to_state": np.asarray(init_state, dtype=np.float64)}
        obs, info = self.env.reset(seed=seed, options=opts)
        self._apply_perturbation()
        self.t = 0
        return self.u.get_obs(), info

    def sample_init_state(self) -> np.ndarray:
        """Same distribution as PushTEnv.reset (agent xy, block xy, block angle)."""
        r = self.rng
        return np.array([r.integers(50, 450), r.integers(50, 450), r.integers(100, 400),
                         r.integers(100, 400), r.uniform(-np.pi, np.pi)], dtype=np.float64)

    # ----------------------------------------------------------------- step
    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if self.perturbation == "action_noise" and self.severity > 0:
            action = action + self.rng.normal(0.0, _SEVERITY["action_noise"][self.severity], size=2)
        obs, reward, terminated, truncated, info = self.u.step(np.clip(action, 0, 512))
        self.t += 1
        truncated = truncated or self.t >= self.max_steps
        return obs, reward, terminated, truncated, info

    def low_dim_state(self) -> np.ndarray:
        """[agent_x, agent_y, block_x, block_y, sin th, cos th] (for the stand-in WM)."""
        a, b = self.u.agent.position, self.u.block.position
        th = self.u.block.angle
        return np.array([a[0], a[1], b[0], b[1], np.sin(th), np.cos(th)], dtype=np.float32)

    def coverage(self) -> float:
        return float(self.u._get_coverage())

    # ---------------------------------------------------- snapshot / restore
    def snapshot(self) -> Snapshot:
        a, b = self.u.agent, self.u.block
        return Snapshot(tuple(a.position), tuple(a.velocity), tuple(b.position), float(b.angle),
                        tuple(b.velocity), float(b.angular_velocity))

    def restore(self, s: Snapshot, t: int = 0):
        """Rebuild the space (clears contact caches) and write kinematics back."""
        self.u._setup()
        self._apply_perturbation()
        a, b = self.u.agent, self.u.block
        a.position, a.velocity = s.agent_pos, s.agent_vel
        b.position, b.angle = s.block_pos, s.block_angle
        b.velocity, b.angular_velocity = s.block_vel, s.block_angvel
        self.u.space.reindex_shapes_for_body(a)
        self.u.space.reindex_shapes_for_body(b)
        self.t = t
        return self.u.get_obs()

    # --------------------------------------------------------- perturbations
    def _apply_perturbation(self):
        p, s = self.perturbation, self.severity
        if p == "none" or s == 0:
            return
        u = self.u
        if p == "friction":
            for shape in [*u.agent.shapes, *u._block_shapes]:
                shape.friction = _SEVERITY["friction"][s]
        elif p == "damping":
            u.space.damping = _SEVERITY["damping"][s]
        elif p == "recolor":
            for shape in u._block_shapes:
                shape.color = pygame.Color(_SEVERITY["recolor"][s])
        elif p == "distractor":
            import pymunk

            for xy in self._distractor_xy:
                c = pymunk.Circle(u.space.static_body, 18, offset=xy)
                c.sensor = True  # rendered, never collides
                c.color = pygame.Color("DarkOrange")
                u.space.add(c)
        # cog is set at construction (gym_pusht applies it in _setup); action_noise in step()


def make_envs(n: int, perturbation="none", severity=0, seed=0) -> list[PushT]:
    return [PushT(perturbation, severity, seed=seed + i) for i in range(n)]
