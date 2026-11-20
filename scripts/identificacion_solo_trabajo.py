"""Identification checks for the baseline (labor-only informal sector, informality in levels).

1. Profile likelihood of theta on the actual data, with and without the earnings observable:
   for each fixed theta the other parameters are re-optimized from the baseline posterior median.
2. Recovery in simulated data: data are drawn from the baseline model at its posterior median
   with a known theta (0.20 and 0.40), with the same sample length and missing quarters as the
   actual data, and theta is re-estimated (posterior mode and Laplace standard deviation) with
   and without the earnings observable.

Usage:
    python scripts/identificacion_solo_trabajo.py --base resultados_m2/solo_trabajo ^
        --ingresos datos/geih/ratio_va_0354.csv --salida resultados_m2/identificacion_solo_trabajo --workers 8
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from m2.data import DataConfig, ObservableBuilder, ObservableSet  # noqa: E402
from m2.estimation import ModeFinder, Posterior  # noqa: E402
from m2.informal import SmoothedInformality  # noqa: E402
from m2.model import Calibration, LaborOnlyInformalModel  # noqa: E402
from m2.priors import ETA_PRIOR, PRIORS, PRIORS_WITH_EARNINGS, PriorSet  # noqa: E402
from m2.simulate import simulate  # noqa: E402

THETA_GRID = (0.10, 0.15, 0.20, 0.25, 0.277, 0.30, 0.35, 0.40, 0.45)
PATH_GRID = (0.10, 0.20, 0.30, 0.40)       # theta values at which the share path is recomputed
THETA_TRUE = (0.20, 0.40)
PRIOR_SD_THETA = 0.10


def labor_only_priors(with_earnings: bool) -> tuple:
    base = PRIORS_WITH_EARNINGS if with_earnings else PRIORS
    return base[:1] + (ETA_PRIOR,) + base[1:]


@dataclass
class Setup:
    """Data, calibration and baseline median shared by every check."""

    obs: ObservableSet            # with earnings
    obs_aggregates: ObservableSet  # without earnings
    cal: Calibration
    median: dict[str, float]
    offset: float | np.ndarray

    @classmethod
    def build(cls, data: Path, earnings: Path, column: str, base: Path,
              offset_file: Path | None = None, offset_column: str = "offset_log") -> "Setup":
        summary = json.loads((base / "run_summary.json").read_text())
        cal = Calibration(**summary["calibration"])
        aggregates = ObservableBuilder(DataConfig(start="1993Q1", informality_in_levels=True)).build(data)
        ratio = pd.read_csv(earnings).set_index("quarter")[column]
        obs = aggregates.with_relative_earnings(ratio)
        table = pd.read_csv(base / "posterior_summary.csv").set_index("parameter")
        median = {k: float(v) for k, v in table["post_median"].items()}
        offset = -np.log(1.0 - cal.tau_bar)
        if offset_file is not None:
            table = pd.read_csv(offset_file).set_index("quarter")[offset_column].astype(float)
            offset = table.reindex(obs.quarters.astype(str)).ffill().bfill().to_numpy()
        return cls(obs, aggregates, cal, median, offset)

    def posterior(self, with_earnings: bool, obs: ObservableSet | None = None, fixed: dict | None = None,
                  offset: float | None = None) -> Posterior:
        data = obs if obs is not None else (self.obs if with_earnings else self.obs_aggregates)
        priors = PriorSet(tuple(p for p in labor_only_priors(with_earnings) if not fixed or p.name not in fixed))
        post = Posterior(data, self.cal, priors, fixed=fixed or {}, model_class=LaborOnlyInformalModel)
        post.earnings_offset = self.offset if offset is None else offset
        return post


# ---------------------------------------------------------------- profile likelihood
def _profile_point(args: tuple) -> dict:
    setup, with_earnings, theta, maxfev = args
    post = setup.posterior(with_earnings, fixed={"theta": theta})
    start = post.priors.to_unconstrained({k: setup.median.get(k, p.mean) for k, p in
                                          ((p.name, p) for p in post.priors.priors)})
    objective = lambda u: -post(u) if np.isfinite(post(u)) else 1e10  # noqa: E731
    res = minimize(objective, start, method="Nelder-Mead",
                   options={"maxfev": maxfev, "xatol": 1e-4, "fatol": 1e-4})
    res = minimize(objective, res.x, method="BFGS", options={"maxiter": 200, "gtol": 1e-4})
    params = post.priors.to_params(res.x)
    return {"observables": "with_earnings" if with_earnings else "aggregates_only", "theta": theta,
            "loglik": post.loglik(params), "logpost_without_theta_prior": -float(res.fun)}


def _path_point(args: tuple) -> pd.DataFrame:
    """Re-optimize the other parameters at fixed theta and return the smoothed share path."""
    setup, theta, maxfev = args
    post = setup.posterior(True, fixed={"theta": theta})
    start = post.priors.to_unconstrained({p.name: setup.median.get(p.name, p.mean) for p in post.priors.priors})
    objective = lambda u: -post(u) if np.isfinite(post(u)) else 1e10  # noqa: E731
    res = minimize(objective, start, method="Nelder-Mead", options={"maxfev": maxfev, "xatol": 1e-4, "fatol": 1e-4})
    res = minimize(objective, res.x, method="BFGS", options={"maxiter": 200, "gtol": 1e-4})
    params = {**post.priors.to_params(res.x), "theta": theta}
    path = SmoothedInformality(post, setup.obs).path(params)
    return pd.DataFrame({"quarter": setup.obs.quarters.astype(str), "theta_fixed": theta,
                         "loglik": post.loglik(params), "informal_output_share": path["theta_clean"]})


# ---------------------------------------------------------------- simulation
def _simulate_and_estimate(args: tuple) -> list[dict]:
    setup, theta_true, rep, seed = args
    truth = {**setup.median, "theta": theta_true}
    dgp = setup.posterior(True, offset=0.0)
    model, space = dgp.state_space(truth)
    rng = np.random.default_rng(seed)
    sim = simulate(space, len(setup.obs.quarters), rng)
    sim[:, -1] += model.earnings_level()
    sim[np.isnan(setup.obs.values)] = np.nan                    # same missing pattern as the data
    rows = []
    for label, with_earnings in (("aggregates_only", False), ("with_earnings", True)):
        names = setup.obs.names if with_earnings else setup.obs_aggregates.names
        template = setup.obs if with_earnings else setup.obs_aggregates
        data = replace(template, values=sim[:, :len(names)].copy())
        post = setup.posterior(with_earnings, obs=data, offset=0.0)
        mode = ModeFinder(post, seed=seed).find(n_starts=2)
        j = post.priors.names.index("theta")
        theta_hat = post.priors.to_params(mode.u)["theta"]
        sd = float(np.sqrt(mode.hessian_inv[j, j])) * theta_hat * (1 - theta_hat)   # delta method
        rows.append({"theta_true": theta_true, "replication": rep, "observables": label,
                     "theta_mode": theta_hat, "theta_sd_laplace": sd, "sd_ratio_post_prior": sd / PRIOR_SD_THETA})
        print(rows[-1], flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datos", type=Path, default=ROOT / "datos" / "datos_dsge_estimacion.csv")
    parser.add_argument("--ingresos", type=Path, default=ROOT / "datos" / "geih" / "ratio_va_0354.csv")
    parser.add_argument("--columna-ingresos", default="ratio_va")
    parser.add_argument("--base", type=Path, default=ROOT / "resultados_m2" / "solo_trabajo")
    parser.add_argument("--salida", type=Path, default=ROOT / "resultados_m2" / "identificacion_solo_trabajo")
    parser.add_argument("--replicas", type=int, default=3)
    parser.add_argument("--maxfev", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--solo", choices=("perfil", "trayectorias", "simulacion"), default=None)
    parser.add_argument("--ajuste-ingresos-archivo", type=Path, default=None)
    parser.add_argument("--columna-ajuste", default="offset_log")
    args = parser.parse_args()

    setup = Setup.build(args.datos, args.ingresos, args.columna_ingresos, args.base,
                        args.ajuste_ingresos_archivo, args.columna_ajuste)
    args.salida.mkdir(parents=True, exist_ok=True)
    with Pool(args.workers) as pool:
        if args.solo in (None, "perfil"):
            jobs = [(setup, w, t, args.maxfev) for w in (True, False) for t in THETA_GRID]
            profile = pd.DataFrame(pool.map(_profile_point, jobs))
            profile["loglik_rel"] = profile.groupby("observables")["loglik"].transform(lambda s: s - s.max())
            profile.to_csv(args.salida / "perfil_theta.csv", index=False)
            print(profile.pivot(index="theta", columns="observables", values="loglik_rel").round(2))
        if args.solo in (None, "trayectorias"):
            paths = pd.concat(pool.map(_path_point, [(setup, t, args.maxfev) for t in PATH_GRID]))
            paths.to_csv(args.salida / "trayectorias_theta_fijo.csv", index=False)
            wide = paths.pivot(index="quarter", columns="theta_fixed", values="informal_output_share")
            print((100 * wide.loc[["1993Q2", "2000Q4", "2007Q2", "2015Q2", "2019Q2", "2026Q2"]]).round(1))
        if args.solo in (None, "simulacion"):
            jobs = [(setup, t, r, 100 * k + r) for k, t in enumerate(THETA_TRUE) for r in range(args.replicas)]
            sim = pd.DataFrame([row for rows in pool.map(_simulate_and_estimate, jobs) for row in rows])
            sim.to_csv(args.salida / "simulacion.csv", index=False)
            print(sim.groupby(["theta_true", "observables"])[["theta_mode", "theta_sd_laplace",
                                                              "sd_ratio_post_prior"]].mean().round(3))


if __name__ == "__main__":
    main()
