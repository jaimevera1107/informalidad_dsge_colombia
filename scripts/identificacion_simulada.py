"""Identification experiment: can the model recover the informal output share?

Data are simulated from M2 at known parameters (theta_true differs from the prior
mean). The model is then re-estimated (posterior mode + Laplace standard error)
(a) with the six aggregate observables used today and
(b) adding the level of relative labor earnings informal/formal (GEIH).
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from m2.data import ObservableBuilder  # noqa: E402
from m2.estimation import ModeFinder, Posterior, calibration_from_data  # noqa: E402
from m2.priors import PRIORS, PRIORS_WITH_EARNINGS, PriorSet  # noqa: E402
from m2.simulate import simulate  # noqa: E402

THETA_TRUE = 0.45
TRUTH_EXTRA = {"rho_zr": 0.90, "sig_zr": 0.02, "me_w": 0.03}


def main(n_reps: int = 3, seed: int = 7) -> None:
    obs = ObservableBuilder().build(ROOT / "datos" / "datos_dsge_estimacion.csv")
    cal = calibration_from_data(obs)
    base = pd.read_csv(ROOT / "resultados_m2" / "base" / "posterior_summary.csv").set_index("parameter")
    truth = {k: float(v) for k, v in base["post_median"].items()}
    truth.update(theta=THETA_TRUE, **TRUTH_EXTRA)
    template = obs.with_relative_earnings(pd.Series(dtype=float))
    dgp = Posterior(template, cal, PriorSet(PRIORS_WITH_EARNINGS))
    model, space = dgp.state_space(truth)
    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(n_reps):
        sim = simulate(space, len(obs.quarters), rng)
        sim[:, -1] += np.log(model.ss.rho_tilde)          # earnings ratio in levels
        sim[np.isnan(obs.values[:, 0])] = np.nan            # same missing quarters as the data
        for label, names, priors in (("six_aggregates", obs.names, PRIORS),
                                     ("plus_relative_earnings", template.names, PRIORS_WITH_EARNINGS)):
            data_obs = replace(template, values=sim[:, :len(names)].copy(), names=names)
            post = Posterior(data_obs, cal, PriorSet(priors))
            mode = ModeFinder(post, seed=rep).find(n_starts=2)
            j = post.priors.names.index("theta")
            theta_hat = post.priors.to_params(mode.u)["theta"]
            sd_logit = float(np.sqrt(mode.hessian_inv[j, j]))
            sd_theta = sd_logit * theta_hat * (1 - theta_hat)          # delta method
            rows.append({"replication": rep, "observables": label, "theta_true": THETA_TRUE,
                         "theta_mode": theta_hat, "theta_sd_laplace": sd_theta,
                         "prior_sd": 0.10, "sd_ratio_post_prior": sd_theta / 0.10})
            print(rows[-1], flush=True)
    out = pd.DataFrame(rows)
    target = ROOT / "resultados_m2" / "identificacion_simulada.csv"
    out.to_csv(target, index=False)
    print(out.groupby("observables")[["theta_mode", "theta_sd_laplace", "sd_ratio_post_prior"]].mean())


if __name__ == "__main__":
    main()
