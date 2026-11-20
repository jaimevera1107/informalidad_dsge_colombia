"""Posterior, posterior mode and random-walk Metropolis-Hastings for M2."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .data import ObservableSet
from .model import Calibration, LaborOnlyInformalModel, ModelError, Structural, TwoSectorModel
from .priors import PriorSet
from .statespace import KalmanFilter, StateSpace



def calibration_from_data(obs: ObservableSet, **overrides: float) -> Calibration:
    values = dict(gamma=obs.mean_growth, s_bar=obs.informality_mean,
                  c_y=obs.ratios["c_y"], i_y=obs.ratios["i_y"], g_y=obs.ratios["g_y"])
    values.update(overrides)
    return Calibration(**values)


class Posterior:
    """Log posterior kernel in the unconstrained parameter space."""

    def __init__(self, obs: ObservableSet, calibration: Calibration,
                 priors: PriorSet | None = None, fixed: dict[str, float] | None = None,
                 model_class: type[TwoSectorModel] = TwoSectorModel) -> None:
        self.model_class = model_class
        self.obs = obs
        self.cal = calibration
        self.priors = priors or PriorSet()
        self.fixed = fixed or {}
        self.data = obs.values
        self.names = obs.names
        self.earnings_offset = 0.0   # log(observed ratio) - log(model productivity ratio); calibrated
        self.evaluations = 0

    def structural(self, params: dict[str, float]) -> Structural:
        keys = Structural.__dataclass_fields__.keys()
        merged = {**self.fixed, **params}
        return Structural(**{k: merged[k] for k in keys if k in merged})

    def state_space(self, params: dict[str, float]) -> tuple[TwoSectorModel, StateSpace]:
        model = self.model_class(self.cal, self.structural(params))
        sol = model.solve()
        merged = {**self.fixed, **params}
        errors = {"obs_i": "me_i", "obs_n": "me_n", "obs_s": "me_s", "obs_k": "me_k"}
        level_noise = np.array([merged.get(errors[n], 0.0) if n in errors else 0.0 for n in self.names])
        space = StateSpace.from_solution(sol, self.cal.s_bar, model.par.volatilities(), level_noise,
                                         names=self.names, earnings_noise_sd=merged.get("me_w", 0.0),
                                         earnings_loadings=model.earnings_loadings(),
                                         levels=self.obs.level_observables)
        return model, space

    def _set_break(self, params: dict[str, float]) -> None:
        self._break = float({**self.fixed, **params}.get("w_break", 0.0))
        if not hasattr(self, "_break_mask"):
            self._break_mask = np.array([str(q) >= "2022Q1" for q in self.obs.quarters])

    def adjusted_data(self, model: TwoSectorModel) -> np.ndarray:
        """Remove the parameter-dependent constant of the level observable obs_w."""
        if "obs_w" not in self.names:
            return self.data
        data = self.data.copy()
        j = self.names.index("obs_w")
        data[:, j] = data[:, j] - (model.earnings_level() + self.earnings_offset)
        shift = getattr(self, "_break", 0.0)
        if shift:
            data[self._break_mask, j] -= shift
        return data

    def loglik(self, params: dict[str, float]) -> float:
        self._set_break(params)
        model, space = self.state_space(params)
        return KalmanFilter(space).run(self.adjusted_data(model)).loglik

    def __call__(self, u: np.ndarray) -> float:
        self.evaluations += 1
        lp = self.priors.logpdf(u)
        if not np.isfinite(lp):
            return -np.inf
        try:
            ll = self.loglik(self.priors.to_params(u))
        except (ModelError, np.linalg.LinAlgError, ValueError, FloatingPointError):
            return -np.inf
        return lp + ll if np.isfinite(ll) else -np.inf


@dataclass
class ModeResult:
    u: np.ndarray
    logpost: float
    hessian_inv: np.ndarray
    attempts: list[float] = field(default_factory=list)


class ModeFinder:
    """Multi-start optimization followed by a numerical Hessian."""

    def __init__(self, posterior: Posterior, seed: int = 11) -> None:
        self.post = posterior
        self.rng = np.random.default_rng(seed)

    def _objective(self, u: np.ndarray) -> float:
        value = self.post(u)
        return -value if np.isfinite(value) else 1e10

    def find(self, n_starts: int = 6) -> ModeResult:
        base = self.post.priors.prior_means()
        starts = [base] + [base + self.rng.normal(0, 0.5, size=base.size) for _ in range(n_starts - 1)]
        best, attempts = None, []
        for start in starts:
            res = minimize(self._objective, start, method="Nelder-Mead",
                           options={"maxiter": 6000, "maxfev": 8000, "xatol": 1e-5, "fatol": 1e-6})
            res = minimize(self._objective, res.x, method="BFGS", options={"maxiter": 500, "gtol": 1e-5})
            attempts.append(-float(res.fun))
            if best is None or res.fun < best.fun:
                best = res
        hess = self.hessian(best.x)
        try:
            cov = np.linalg.inv(hess)
            eigval, eigvec = np.linalg.eigh(0.5 * (cov + cov.T))
            cov = eigvec @ np.diag(np.clip(eigval, 1e-6, 4.0)) @ eigvec.T
        except np.linalg.LinAlgError:
            cov = np.eye(best.x.size) * 0.01
        return ModeResult(u=best.x, logpost=-float(best.fun), hessian_inv=cov, attempts=attempts)

    def hessian(self, u: np.ndarray, step: float = 1e-3) -> np.ndarray:
        d = u.size
        f0 = self._objective(u)
        hess = np.zeros((d, d))
        for i in range(d):
            ei = np.zeros(d)
            ei[i] = step
            hess[i, i] = (self._objective(u + ei) - 2 * f0 + self._objective(u - ei)) / step**2
            for j in range(i):
                ej = np.zeros(d)
                ej[j] = step
                val = (self._objective(u + ei + ej) - self._objective(u + ei - ej)
                       - self._objective(u - ei + ej) + self._objective(u - ei - ej)) / (4 * step**2)
                hess[i, j] = hess[j, i] = val
        return hess


@dataclass
class ChainResult:
    draws: np.ndarray          # (n_keep, d) unconstrained
    logpost: np.ndarray
    acceptance: float
    scale: float


class RandomWalkMH:
    """RWMH with Hessian-based proposal; scale adapted during burn-in only."""

    def __init__(self, posterior: Posterior, proposal_cov: np.ndarray, target: float = 0.28) -> None:
        self.post = posterior
        self.chol = np.linalg.cholesky(proposal_cov)
        self.target = target

    def run(self, start: np.ndarray, n_draws: int, n_burn: int, seed: int,
            scale: float | None = None) -> ChainResult:
        rng = np.random.default_rng(seed)
        d = start.size
        c = scale or 2.38 / np.sqrt(d)
        current, lp = start.copy(), self.post(start)
        if not np.isfinite(lp):
            raise RuntimeError("Chain start has zero posterior density")
        kept, kept_lp = [], []
        accepted_window, accepted_total = 0, 0
        for it in range(n_burn + n_draws):
            proposal = current + c * (self.chol @ rng.standard_normal(d))
            lp_new = self.post(proposal)
            if np.log(rng.random()) < lp_new - lp:
                current, lp = proposal, lp_new
                accepted_window += 1
                if it >= n_burn:
                    accepted_total += 1
            if it < n_burn and (it + 1) % 100 == 0:
                rate = accepted_window / 100
                c *= np.exp(np.clip(rate - self.target, -0.5, 0.5))
                accepted_window = 0
            if it >= n_burn:
                kept.append(current.copy())
                kept_lp.append(lp)
        return ChainResult(draws=np.array(kept), logpost=np.array(kept_lp),
                           acceptance=accepted_total / n_draws, scale=c)
