"""Informal output share implied by the estimated model.

With a common Cobb-Douglas technology and mobile capital, the informal share
of output equals the informal share of effective labor:
    theta_t = R_t s_t / ((1 - s_t) + R_t s_t),
where s_t is the labor-informality rate cleaned of survey noise by the
Kalman smoother and R_t the relative productivity of informal work.
With a labor-only informal sector the model's own mapping is used instead
(see model.output_share_labor_only). In both cases R_t is the model value of
informal earnings over the gross formal wage, from the smoothed states.
"""
from __future__ import annotations

import numpy as np

from .data import ObservableSet
from .estimation import Posterior
from .model import STATE_INDEX, VARIABLES, output_share_common
from .statespace import KalmanFilter

_LNI, _LNF = VARIABLES.index("lni"), VARIABLES.index("lnf")


def informal_share(rho_tilde: float | np.ndarray, s: np.ndarray) -> np.ndarray:
    """Equal-technology mapping (kept for backward compatibility)."""
    return output_share_common(rho_tilde, s)


class SmoothedInformality:
    """Recovers the noise-free informality rate and the informal output share."""

    def __init__(self, posterior: Posterior, obs: ObservableSet) -> None:
        self.post = posterior
        self.obs = obs

    def path(self, params: dict[str, float]) -> dict[str, np.ndarray]:
        self.post._set_break(params)
        model, space = self.post.state_space(params)
        out = KalmanFilter(space).run(self.post.adjusted_data(model), keep=True, smooth=True)
        mapping = model.solve().full_map()
        ns = len(STATE_INDEX)
        states = out.smoothed_states[:, :ns]
        dev_logit = states @ (mapping[_LNI] - mapping[_LNF])          # model deviation of logit s
        observed_logit = np.log(self.obs.informality_rate / (1 - self.obs.informality_rate))
        if "obs_s" in self.obs.level_observables:
            # level observable: the cleaned rate is the model's smoothed level plus the seasonal pattern
            s_bar = self.post.cal.s_bar
            seasonal = self.obs.seasonal_means["obs_s_level"]
            season = np.array([seasonal[q.quarter] for q in self.obs.quarters])
            cleaned_logit = np.log(s_bar / (1 - s_bar)) + season + dev_logit
        else:
            k = space.noisy.index(5)
            level_noise = out.smoothed_states[:, space.n_model_states + k]   # smoothed survey error
            cleaned_logit = observed_logit - level_noise
        s_clean = 1.0 / (1.0 + np.exp(-cleaned_logit))
        loading = sum(coef * mapping[VARIABLES.index(var)] for var, coef in model.earnings_loadings().items())
        rho = np.exp(model.earnings_level() + states @ loading)        # time-varying earnings ratio (gross)
        return {
            "s_observed": self.obs.informality_rate,
            "s_clean": s_clean,
            "theta_observed": model.output_share(rho, self.obs.informality_rate),
            "theta_clean": model.output_share(rho, s_clean),
            "model_logit_deviation": dev_logit,
            "rho_tilde": rho,
        }
