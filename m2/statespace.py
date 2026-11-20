"""State-space form of the linearized M2 and the Kalman filter.

Observables are growth rates, so the state is augmented with its own lag:
    x_t = [s_t, s_{t-1}],   x_{t+1} = A x_t + B eps_{t+1}
    obs_t = Z x_t + e_t,    e_t ~ N(0, diag(me^2))
Quarters with NaN observations are skipped in the update step (missing data).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from .data import OBSERVABLES
from .model import VARIABLES, LinearSolution

_IDX = {name: i for i, name in enumerate(VARIABLES)}


def observation_rows(s_bar: float, names: tuple[str, ...] = OBSERVABLES,
                     earnings_loadings: dict[str, float] | None = None,
                     levels: tuple[str, ...] = ()) -> tuple[np.ndarray, np.ndarray]:
    """Rows mapping (Y_t, Y_{t-1}) into the observables (log deviations).

    The first six are growth rates; the optional 'obs_w' (relative earnings) is a level.
    """
    n_obs, n_var = len(names), len(VARIABLES)
    now, lag = np.zeros((n_obs, n_var)), np.zeros((n_obs, n_var))
    for row, var in enumerate(("ly", "lc", "li")):
        now[row, _IDX[var]] = 1.0
        lag[row, _IDX[var]] = -1.0
        now[row, _IDX["mu"]] = 1.0
    now[3, _IDX["zg"]], lag[3, _IDX["zg"]], now[3, _IDX["mu"]] = 1.0, -1.0, 1.0
    for var, weight in (("lnf", 1.0 - s_bar), ("lni", s_bar)):
        now[4, _IDX[var]] = weight
        lag[4, _IDX[var]] = -weight
    now[5, _IDX["lni"]], now[5, _IDX["lnf"]] = 1.0, -1.0
    if "obs_s" not in levels:
        lag[5, _IDX["lni"]], lag[5, _IDX["lnf"]] = -1.0, 1.0
    if "obs_k" in names:              # log(K_t / Y_t) = lk - gamma - mu - ly, in levels
        row = names.index("obs_k")
        now[row, _IDX["lk"]], now[row, _IDX["mu"]], now[row, _IDX["ly"]] = 1.0, -1.0, -1.0
    if "obs_w" in names:
        for var, coef in (earnings_loadings or {"zr": 1.0}).items():
            now[names.index("obs_w"), _IDX[var]] += coef
    return now, lag


@dataclass(frozen=True)
class StateSpace:
    A: np.ndarray
    B: np.ndarray
    Z: np.ndarray
    Q: np.ndarray
    H: np.ndarray
    n_model_states: int = 0
    noisy: tuple[int, ...] = ()

    @classmethod
    def from_solution(cls, sol: LinearSolution, s_bar: float, shock_sd: np.ndarray,
                      level_noise_sd: np.ndarray, growth_noise_sd: float = 1e-3,
                      names: tuple[str, ...] = OBSERVABLES, earnings_noise_sd: float = 0.0,
                      earnings_loadings: dict[str, float] | None = None,
                      levels: tuple[str, ...] = ()) -> "StateSpace":
        """Build the augmented system.

        level_noise_sd: sd of i.i.d. errors on the LOG LEVEL of each observable
        (survey or statistical noise). In growth rates they appear as e_t - e_{t-1},
        which produces the negative autocorrelation seen in the data.
        A tiny growth-rate error keeps the innovation covariance non-singular.
        """
        ns = sol.transition.shape[0]
        n_obs = len(names)
        level_noise_sd = np.asarray(level_noise_sd, dtype=float).copy()
        iid = {}
        for name in levels:              # level observables get i.i.d. errors in H, not error states
            j = names.index(name)
            if name != "obs_w":
                iid[j] = level_noise_sd[j]
                level_noise_sd[j] = 0.0
        noisy = [j for j, sd in enumerate(level_noise_sd) if sd > 0]
        ne = len(noisy)
        core = ns + ne
        A = np.zeros((2 * core, 2 * core))
        A[:ns, :ns] = sol.transition
        A[core:, :core] = np.eye(core)
        n_shocks = sol.impact.shape[1]
        B = np.zeros((2 * core, n_shocks + ne))
        B[:ns, :n_shocks] = sol.impact
        for k in range(ne):
            B[ns + k, n_shocks + k] = 1.0
        mapping = sol.full_map()
        now, lag = observation_rows(s_bar, names, earnings_loadings, levels)
        Z = np.zeros((n_obs, 2 * core))
        Z[:, :ns] = now @ mapping
        Z[:, core:core + ns] = lag @ mapping
        for k, j in enumerate(noisy):
            Z[j, ns + k] = 1.0
            Z[j, core + ns + k] = -1.0
        Q = np.diag(np.r_[np.asarray(shock_sd) ** 2, np.asarray(level_noise_sd)[noisy] ** 2])
        H = np.eye(n_obs) * growth_noise_sd ** 2
        if "obs_w" in names:
            H[names.index("obs_w"), names.index("obs_w")] = earnings_noise_sd ** 2
        for j, sd in iid.items():
            H[j, j] = sd ** 2
        return cls(A=A, B=B, Z=Z, Q=Q, H=H, n_model_states=ns, noisy=tuple(noisy))

    def initial_covariance(self) -> np.ndarray:
        P0 = solve_discrete_lyapunov(self.A, self.B @ self.Q @ self.B.T)
        if not np.all(np.isfinite(P0)):
            raise np.linalg.LinAlgError("Non-finite initial covariance")
        return 0.5 * (P0 + P0.T)


@dataclass
class FilterOutput:
    loglik: float
    predicted_obs: np.ndarray | None = None
    filtered_states: np.ndarray | None = None
    smoothed_states: np.ndarray | None = None


class KalmanFilter:
    """Standard Kalman filter (Joseph update) with missing observations."""

    def __init__(self, space: StateSpace) -> None:
        self.ss = space

    def run(self, data: np.ndarray, keep: bool = False, smooth: bool = False) -> FilterOutput:
        A, B, Z, Q, H = self.ss.A, self.ss.B, self.ss.Z, self.ss.Q, self.ss.H
        n = A.shape[0]
        T = data.shape[0]
        W = B @ Q @ B.T
        x = np.zeros(n)
        P = self.ss.initial_covariance()
        loglik = 0.0
        pred_obs = np.full(data.shape, np.nan) if keep else None
        xf = np.zeros((T, n)) if (keep or smooth) else None
        xp_all = np.zeros((T, n)) if smooth else None
        Pp_all = np.zeros((T, n, n)) if smooth else None
        Pf_all = np.zeros((T, n, n)) if smooth else None
        eye = np.eye(n)
        for t in range(T):
            if t > 0:
                x = A @ x
                P = A @ P @ A.T + W
            if smooth:
                xp_all[t], Pp_all[t] = x, P
            y_full = data[t]
            if keep:
                pred_obs[t] = Z @ x
            ok = np.isfinite(y_full)
            if ok.any():
                y, Zt, Ht = y_full[ok], Z[ok], H[np.ix_(ok, ok)]
                v = y - Zt @ x
                PZt = P @ Zt.T
                F = Zt @ PZt + Ht
                L = np.linalg.cholesky(F)
                Finv = np.linalg.inv(F)
                Finv_v = Finv @ v
                K = PZt @ Finv
                logdet = 2.0 * np.log(np.diag(L)).sum()
                loglik -= 0.5 * (len(y) * np.log(2 * np.pi) + logdet + v @ Finv_v)
                x = x + K @ v
                IKZ = eye - K @ Zt
                P = IKZ @ P @ IKZ.T + K @ Ht @ K.T
            if xf is not None:
                xf[t] = x
            if smooth:
                Pf_all[t] = P
        out = FilterOutput(loglik=float(loglik), predicted_obs=pred_obs, filtered_states=xf)
        if smooth:
            xs = xf.copy()
            for t in range(T - 2, -1, -1):
                J = Pf_all[t] @ A.T @ np.linalg.pinv(Pp_all[t + 1])
                xs[t] = xf[t] + J @ (xs[t + 1] - xp_all[t + 1])
            out.smoothed_states = xs
        return out
