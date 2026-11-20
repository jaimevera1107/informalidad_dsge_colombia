"""Regression tests for M2 (run with: python -m pytest tests/test_m2.py -q)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from m2.data import ObservableBuilder
from m2.estimation import Posterior, calibration_from_data
from m2.informal import informal_share
from m2.model import Calibration, ModelError, Structural, TwoSectorModel
from m2.priors import PriorSet
from m2.statespace import KalmanFilter

DATA = Path(__file__).resolve().parents[1] / "datos" / "datos_dsge_estimacion.csv"


@pytest.fixture(scope="module")
def obs():
    return ObservableBuilder().build(DATA)


def test_steady_state_solves_equilibrium():
    model = TwoSectorModel(Calibration(), Structural())
    x = model.ss.vector()
    assert np.abs(model.residuals(x, x)).max() < 1e-12


def test_informal_share_equals_theta_in_steady_state():
    cal = Calibration()
    model = TwoSectorModel(cal, Structural(theta=0.4))
    assert informal_share(model.ss.rho_tilde, cal.s_bar) == pytest.approx(0.4)


def test_blanchard_kahn_and_stable_transition():
    sol = TwoSectorModel(Calibration(), Structural()).solve()
    assert np.max(np.abs(np.linalg.eigvals(sol.transition))) < 1.0


def test_policy_satisfies_linearized_equilibrium():
    model = TwoSectorModel(Calibration(), Structural())
    sol = model.solve()
    d_now, d_next = model.jacobians()
    mapping = sol.full_map()
    # E_t Y_{t+1} = M T s_t  must satisfy  d_now M s + d_next M T s = 0 for every s
    err = d_now @ mapping + d_next @ mapping @ sol.transition
    assert np.abs(err).max() < 1e-6


def test_invalid_theta_raises():
    with pytest.raises(ModelError):
        TwoSectorModel(Calibration(), Structural(theta=1.2))


def test_prior_density_matches_scipy():
    priors = PriorSet()
    u = priors.prior_means() + 0.1
    ref = sum(d.logpdf(p.to_param(x)) + p.log_jacobian(x)
              for p, d, x in zip(priors.priors, priors._frozen, u))
    assert priors.logpdf(u) == pytest.approx(ref, rel=1e-10)


def test_kalman_matches_exact_gaussian_likelihood(obs):
    post = Posterior(obs, calibration_from_data(obs))
    params = post.priors.to_params(post.priors.prior_means())
    _, space = post.state_space(params)
    data = obs.values[:12]
    T, n_obs = data.shape
    n = space.A.shape[0]
    P0 = space.initial_covariance()
    W = space.B @ space.Q @ space.B.T
    # covariance of stacked states x_1..x_T under the stationary distribution
    cov_x = np.zeros((T * n, T * n))
    powers = [np.eye(n)]
    for _ in range(T):
        powers.append(space.A @ powers[-1])
    for t in range(T):
        for s in range(t + 1):
            block = powers[t - s] @ P0
            cov_x[t * n:(t + 1) * n, s * n:(s + 1) * n] = block
            cov_x[s * n:(s + 1) * n, t * n:(t + 1) * n] = block.T
    big_z = np.kron(np.eye(T), space.Z)
    cov_y = big_z @ cov_x @ big_z.T + np.kron(np.eye(T), space.H)
    exact = multivariate_normal(np.zeros(T * n_obs), cov_y, allow_singular=False).logpdf(data.ravel())
    assert W.shape == (n, n)
    assert KalmanFilter(space).run(data).loglik == pytest.approx(exact, rel=1e-8)


def test_observables_exclude_pandemic_quarters(obs):
    quarters = [str(q) for q in obs.quarters]
    missing = np.isnan(obs.values).all(axis=1)
    assert [q for q, m in zip(quarters, missing) if m] == ["2020Q2", "2020Q3", "2020Q4"]


# ---------------------------------------------------------------- labor-only informal sector
from m2.capital_share import AnnualData, CapitalConfig, CapitalShareEstimator  # noqa: E402
from m2.earnings import ValueAddedRatio  # noqa: E402
from m2.model import VARIABLES, LaborOnlyInformalModel, output_share_labor_only  # noqa: E402


def _labor_only(theta: float = 0.3) -> LaborOnlyInformalModel:
    return LaborOnlyInformalModel(Calibration(alpha=0.35, delta=0.015), Structural(theta=theta, eta=0.7))


def test_labor_only_steady_state_and_solution():
    model = _labor_only()
    x = model.ss.vector()
    assert np.abs(model.residuals(x, x)).max() < 1e-12
    assert np.max(np.abs(np.linalg.eigvals(model.solve().transition))) < 1.0


def test_labor_only_share_mapping_recovers_theta():
    model = _labor_only(0.27)
    share = model.output_share(np.exp(model.earnings_level()), model.cal.s_bar)
    assert share == pytest.approx(0.27)
    # equal technologies give a larger share for the same earnings ratio
    assert informal_share(np.exp(model.earnings_level()), model.cal.s_bar) > share


def test_labor_only_earnings_loadings_match_numerical_derivative():
    model = _labor_only()
    cal, par, ss = model.cal, model.par, model.ss

    def log_ratio(v: np.ndarray) -> float:
        lnf, lni, lk, mu, zr = (v[VARIABLES.index(n)] for n in ("lnf", "lni", "lk", "mu", "zr"))
        yi = ss.rho_tilde * np.exp(zr + par.eta * lni)
        yf = np.exp(cal.alpha * (lk - cal.gamma - mu) + (1 - cal.alpha) * lnf)
        return np.log(yi / np.exp(lni)) - np.log((1 - cal.alpha) * yf / np.exp(lnf))

    x0 = ss.vector()
    assert log_ratio(x0) == pytest.approx(model.earnings_level())
    grad = np.array([(log_ratio(x0 + e) - log_ratio(x0 - e)) / 2e-6 for e in np.eye(len(x0)) * 1e-6])
    expected = np.zeros(len(x0))
    for var, coef in model.earnings_loadings().items():
        expected[VARIABLES.index(var)] = coef
    assert np.allclose(grad, expected, atol=1e-6)


def test_value_added_ratio_scales_large_firm_employees():
    import pandas as pd
    row = {"quarter": "2020Q1", "ratio": 0.5, "mean_formal": 100.0}
    for g, (share, mean) in {"own": (0.5, 40.0), "emp_small": (0.2, 30.0), "emp_large": (0.2, 60.0),
                             "domestic": (0.1, 30.0), "emp_nosize": (0.0, np.nan)}.items():
        row[f"share_informal_{g}"], row[f"mean_informal_{g}"] = share, mean
    for g in ("own", "emp_small", "emp_large", "domestic", "emp_nosize"):
        row[f"share_formal_{g}"] = 1.0 if g == "emp_large" else 0.0
        row[f"mean_formal_{g}"] = 100.0 if g == "emp_large" else np.nan
    out = ValueAddedRatio(alpha=0.4).build(pd.DataFrame([row]))
    expected = (0.5 * 40 + 0.2 * 30 + 0.2 * 60 / 0.6 + 0.1 * 30) / 100.0
    assert out.ratio_va.iloc[0] == pytest.approx(expected)


def test_capital_share_recovers_known_alpha():
    import pandas as pd
    rng = np.random.default_rng(3)
    years = np.arange(1990, 2026)
    n = 10.0 * np.exp(0.02 * np.arange(len(years)))
    inv = 20.0 * np.exp(0.04 * np.arange(len(years)) + rng.normal(0, 0.05, len(years)))
    data = AnnualData(pd.DataFrame({"quarter": [f"{y}Q{q}" for y in years for q in range(1, 5)],
                                    "pib": 1.0, "i": np.repeat(inv / 4, 4), "ocupados": np.repeat(n, 4),
                                    "informalidad_ponderada_ocupados": 0.5}),
                      CapitalConfig(depreciation=0.06, start_year=1990))
    k = data.frame.k
    data.frame["y"] = 2.0 * k ** 0.35 * data.frame.n ** 0.65 * np.exp(0.005 * np.arange(len(k))
                                                                      + rng.normal(0, 0.005, len(k)))
    assert CapitalShareEstimator(data).aggregate().alpha == pytest.approx(0.35, abs=0.05)


def test_output_share_labor_only_is_below_common():
    s = np.array([0.5, 0.6])
    assert np.all(output_share_labor_only(0.4, s, 0.33) < informal_share(0.4, s))


def test_informality_in_levels_gives_finite_likelihood():
    from m2.data import DataConfig
    from m2.model import LaborOnlyInformalModel as LO
    obs_lv = ObservableBuilder(DataConfig(start="1993Q1", informality_in_levels=True)).build(DATA)
    assert obs_lv.level_observables == ("obs_s",)
    assert abs(np.nanmean(obs_lv.values[:, 5])) < 0.2
    post = Posterior(obs_lv, calibration_from_data(obs_lv, alpha=0.35, delta=0.0096), model_class=LO)
    params = {**{p.name: p.mean for p in post.priors.priors}, "eta": 0.6}
    assert np.isfinite(post.loglik(params))
