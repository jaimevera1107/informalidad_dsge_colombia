"""Post-estimation analysis: profile likelihood, variance decomposition."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .estimation import Posterior
from .model import SHOCKS, VARIABLES
from .priors import PriorSet

_LNI, _LNF, _LY = VARIABLES.index("lni"), VARIABLES.index("lnf"), VARIABLES.index("ly")


def profile_theta(post: Posterior, u_mode: np.ndarray, grid: np.ndarray, maxfev: int = 2500) -> pd.DataFrame:
    """Maximized log likelihood (other parameters re-optimized) for fixed theta."""
    names = post.priors.names
    j = names.index("theta")
    others = PriorSet(tuple(p for p in post.priors.priors if p.name != "theta"))
    rows = []
    for value in grid:
        fixed_post = Posterior(post.obs, post.cal, priors=others, fixed={**post.fixed, "theta": float(value)},
                               model_class=post.model_class)
        fixed_post.earnings_offset = post.earnings_offset
        start = np.delete(u_mode, j)
        res = minimize(lambda x: -fixed_post(x) if np.isfinite(fixed_post(x)) else 1e10, start,
                       method="Nelder-Mead", options={"maxfev": maxfev, "xatol": 1e-4, "fatol": 1e-4})
        params = fixed_post.priors.to_params(res.x)
        rows.append({"theta": float(value), "loglik": fixed_post.loglik(params),
                     "logpost_without_theta_prior": -float(res.fun)})
    return pd.DataFrame(rows)


def variance_decomposition(post: Posterior, params: dict[str, float],
                           horizons: tuple[int, ...] = (1, 4, 20, 80)) -> pd.DataFrame:
    """Forecast-error variance shares of each structural shock.

    Variables: detrended output and the logit of the (noise-free) informality
    rate. Because logit(theta_t) = log(rho_tilde) + logit(s_t), the shares for
    the informality rate are also the shares for the informal output share.
    """
    model = post.state_space(params)[0]
    sol = model.solve()
    mapping = sol.full_map()
    targets = {"detrended_output": mapping[_LY], "logit_informality": mapping[_LNI] - mapping[_LNF]}
    T, R = sol.transition, sol.impact
    sd = model.par.volatilities()
    rows = []
    for name, row in targets.items():
        for h in horizons:
            contrib = np.zeros(len(SHOCKS))
            power = np.eye(T.shape[0])
            for _ in range(h):
                contrib += (row @ power @ R * sd) ** 2
                power = T @ power
            rows.append({"variable": name, "horizon": h,
                         **{f"share_{s}": c / contrib.sum() for s, c in zip(SHOCKS, contrib)}})
    return pd.DataFrame(rows)
