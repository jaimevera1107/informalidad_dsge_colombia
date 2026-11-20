"""Simulation from the estimated state space (identification experiments)."""
from __future__ import annotations

import numpy as np

from .statespace import StateSpace


def simulate(space: StateSpace, n_periods: int, rng: np.random.Generator, burn: int = 200) -> np.ndarray:
    """Draw observables from the linear Gaussian state space (zero-mean deviations)."""
    n = space.A.shape[0]
    q_chol = np.sqrt(np.diag(space.Q))
    h_chol = np.sqrt(np.diag(space.H))
    x = np.zeros(n)
    out = np.empty((n_periods, space.Z.shape[0]))
    for t in range(burn + n_periods):
        x = space.A @ x + space.B @ (q_chol * rng.standard_normal(len(q_chol)))
        if t >= burn:
            out[t - burn] = space.Z @ x + h_chol * rng.standard_normal(len(h_chol))
    return out
