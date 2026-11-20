"""Prior distributions and the unconstrained parameterization used by the sampler."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.special import betaln, expit, gammaln, logit


@dataclass(frozen=True)
class Prior:
    """Prior for one parameter, specified by mean and standard deviation."""

    name: str
    family: str      # "beta", "invgamma" or "normal"
    mean: float
    sd: float

    def frozen(self) -> stats.rv_continuous:
        m, s = self.mean, self.sd
        if self.family == "beta":
            common = m * (1 - m) / s**2 - 1
            if common <= 0:
                raise ValueError(f"Infeasible beta prior for {self.name}")
            return stats.beta(m * common, (1 - m) * common)
        if self.family == "invgamma":
            shape = 2 + (m / s) ** 2
            return stats.invgamma(shape, scale=m * (shape - 1))
        if self.family == "normal":
            return stats.norm(m, s)
        raise ValueError(f"Unknown prior family {self.family}")

    # unconstrained transform u <-> parameter
    def to_param(self, u: float) -> float:
        if self.family == "normal":
            return float(u)
        return float(expit(u)) if self.family == "beta" else float(np.exp(u))

    def to_unconstrained(self, value: float) -> float:
        if self.family == "normal":
            return float(value)
        return float(logit(value)) if self.family == "beta" else float(np.log(value))

    def log_jacobian(self, u: float) -> float:
        if self.family == "normal":
            return 0.0
        if self.family == "beta":
            p = expit(u)
            return float(np.log(p) + np.log1p(-p))
        return float(u)


PRIORS: tuple[Prior, ...] = (
    Prior("theta", "beta", 0.35, 0.10),
    Prior("rho_mu", "beta", 0.40, 0.20),
    Prior("rho_g", "beta", 0.70, 0.15),
    Prior("rho_v", "beta", 0.70, 0.15),
    Prior("rho_psi", "beta", 0.70, 0.15),
    Prior("rho_om", "beta", 0.85, 0.10),
    Prior("rho_x", "beta", 0.70, 0.15),
    Prior("sig_mu", "invgamma", 0.010, 0.020),
    Prior("sig_g", "invgamma", 0.030, 0.060),
    Prior("sig_v", "invgamma", 0.050, 0.100),
    Prior("sig_psi", "invgamma", 0.020, 0.040),
    Prior("sig_om", "invgamma", 0.020, 0.040),
    Prior("sig_x", "invgamma", 0.020, 0.040),
    Prior("me_i", "invgamma", 0.030, 0.030),
    Prior("me_n", "invgamma", 0.010, 0.010),
    Prior("me_s", "invgamma", 0.010, 0.010),
)
NAMES: tuple[str, ...] = tuple(p.name for p in PRIORS)
ETA_PRIOR = Prior("eta", "beta", 0.70, 0.10)
PRIORS_LABOR_ONLY: tuple[Prior, ...] = PRIORS[:1] + (ETA_PRIOR,) + PRIORS[1:]
EARNINGS_PRIORS: tuple[Prior, ...] = (
    Prior("rho_zr", "beta", 0.80, 0.10),
    Prior("sig_zr", "invgamma", 0.020, 0.040),
    Prior("me_w", "invgamma", 0.030, 0.030),
    # level shift of the earnings ratio from 2022Q1: new definition of informality (marco 2018)
    Prior("w_break", "normal", 0.0, 0.20),
)
PRIORS_WITH_EARNINGS: tuple[Prior, ...] = PRIORS + EARNINGS_PRIORS
# measurement error of the perpetual-inventory capital-output ratio (level)
CAPITAL_PRIOR = Prior("me_k", "invgamma", 0.020, 0.020)


class PriorSet:
    """Joint prior in the unconstrained space (including Jacobians)."""

    def __init__(self, priors: tuple[Prior, ...] = PRIORS) -> None:
        self.priors = priors
        self._frozen = [p.frozen() for p in priors]
        self._shapes = [self._shape(p) for p in priors]

    @staticmethod
    def _shape(prior: Prior) -> tuple[float, float]:
        m, s = prior.mean, prior.sd
        if prior.family == "beta":
            common = m * (1 - m) / s**2 - 1
            return m * common, (1 - m) * common
        if prior.family == "normal":
            return m, s
        shape = 2 + (m / s) ** 2
        return shape, m * (shape - 1)

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.priors)

    def to_params(self, u: np.ndarray) -> dict[str, float]:
        return {p.name: p.to_param(x) for p, x in zip(self.priors, u)}

    def to_unconstrained(self, values: dict[str, float]) -> np.ndarray:
        return np.array([p.to_unconstrained(values[p.name]) for p in self.priors])

    def prior_means(self) -> np.ndarray:
        return self.to_unconstrained({p.name: p.mean for p in self.priors})

    def logpdf(self, u: np.ndarray) -> float:
        total = 0.0
        for prior, (a, b), x in zip(self.priors, self._shapes, u):
            if prior.family == "normal":
                total += -0.5 * ((x - a) / b) ** 2 - np.log(b) - 0.5 * np.log(2 * np.pi)
                continue
            if prior.family == "beta":
                p = 1.0 / (1.0 + np.exp(-x))
                if not 0.0 < p < 1.0:
                    return -np.inf
                # density times Jacobian dp/du = p(1-p)
                total += a * np.log(p) + b * np.log1p(-p) - betaln(a, b)
            else:
                value = np.exp(x)
                # inverse-gamma density times Jacobian dvalue/du = value
                total += a * np.log(b) - gammaln(a) - a * np.log(value) - b / value
        return float(total) if np.isfinite(total) else -np.inf

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        cols = [dist.rvs(size=size, random_state=rng) for dist in self._frozen]
        return np.column_stack(cols)
