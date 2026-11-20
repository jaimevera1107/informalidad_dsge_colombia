"""End-to-end estimation of M2: mode, MCMC, diagnostics, fit and informal share."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from .data import OBSERVABLES, DataConfig, ObservableBuilder, ObservableSet
from .diagnostics import summarize
from .estimation import ChainResult, ModeFinder, Posterior, RandomWalkMH, calibration_from_data
from .informal import SmoothedInformality
from .model import LaborOnlyInformalModel, TwoSectorModel, output_share_common, output_share_labor_only
from .priors import CAPITAL_PRIOR, ETA_PRIOR, PRIORS, PRIORS_WITH_EARNINGS, Prior, PriorSet

MODELS: dict[str, type[TwoSectorModel]] = {"common": TwoSectorModel, "labor_only": LaborOnlyInformalModel}
from .statespace import KalmanFilter


@dataclass(frozen=True)
class RunConfig:
    data_path: Path
    output: Path
    n_chains: int = 4
    n_draws: int = 15000
    n_burn: int = 5000
    n_starts: int = 3
    n_paths: int = 400
    seed: int = 2026
    workers: int = 2
    theta_prior: tuple[float, float] = (0.35, 0.10)
    data_config: DataConfig = DataConfig()
    earnings_path: Path | None = None   # CSV: quarter, ratio (GEIH informal/formal labor earnings)
    earnings_column: str = "ratio"
    earnings_offset: float | None = None  # log(observed ratio / productivity ratio); None: -log(1 - tau_bar)
    earnings_offset_path: Path | None = None   # CSV quarter,<column>: time-varying log offset (non-wage costs)
    earnings_offset_column: str = "offset_log"
    eta_prior: tuple[float, float] | None = None  # labor_only: (mean, sd) of the beta prior of eta
    model: str = "common"                 # "common" technology or "labor_only" informal sector
    alpha: float | None = None            # capital share (of formal output in labor_only); None: 0.33
    delta: float | None = None            # quarterly depreciation, labor_only only; None: 0.025

    def __post_init__(self) -> None:
        if self.model not in MODELS:
            raise ValueError(f"model must be one of {sorted(MODELS)}")
        if self.alpha is not None and not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if self.delta is not None and not 0.0 < self.delta < 0.1:
            raise ValueError("delta must lie in (0, 0.1)")


def _chain_job(args: tuple) -> ChainResult:
    post, cov, start, n_draws, n_burn, seed = args
    return RandomWalkMH(post, cov).run(start, n_draws, n_burn, seed)


class M2Estimation:
    """Runs the full pipeline and writes every artifact to `config.output`."""

    def __init__(self, config: RunConfig) -> None:
        self.cfg = config
        self.obs: ObservableSet = ObservableBuilder(config.data_config).build(config.data_path)
        base_priors = PRIORS
        if config.earnings_path is not None:
            earnings = pd.read_csv(config.earnings_path)
            self.obs = self.obs.with_relative_earnings(earnings.set_index("quarter")[config.earnings_column])
            base_priors = PRIORS_WITH_EARNINGS
        priors = tuple(Prior("theta", "beta", *config.theta_prior) if p.name == "theta" else p
                       for p in base_priors)
        if "obs_k" in self.obs.names:
            priors = priors + (CAPITAL_PRIOR,)
        if config.model == "labor_only":
            eta = ETA_PRIOR if config.eta_prior is None else Prior("eta", "beta", *config.eta_prior)
            priors = priors[:1] + (eta,) + priors[1:]
        overrides = {k: v for k, v in (("alpha", config.alpha), ("delta", config.delta)) if v is not None}
        self.post = Posterior(self.obs, calibration_from_data(self.obs, **overrides), PriorSet(priors),
                              model_class=MODELS[config.model])
        # GEIH formal income is take-home pay, i.e. net of the formality wedge tau_bar
        self.post.earnings_offset = (-np.log(1.0 - self.post.cal.tau_bar) if config.earnings_offset is None
                                     else config.earnings_offset)
        if config.earnings_offset_path is not None:
            self.post.earnings_offset = self._offset_path(config.earnings_offset_path, config.earnings_offset_column)
        config.output.mkdir(parents=True, exist_ok=True)

    def _offset_path(self, path: Path, column: str) -> np.ndarray:
        """Quarterly log offset aligned with the sample; quarters outside the file take the nearest value."""
        table = pd.read_csv(path)
        if column not in table:
            raise KeyError(f"column {column!r} not in {path}")
        series = table.set_index("quarter")[column].astype(float)
        aligned = series.reindex(self.obs.quarters.astype(str)).ffill().bfill()
        if aligned.isna().any():
            raise ValueError(f"offset file {path} has no usable values")
        return aligned.to_numpy()

    # ------------------------------------------------------------------ sampling
    def sample(self) -> tuple[np.ndarray, list[ChainResult], dict]:
        mode = ModeFinder(self.post, seed=self.cfg.seed).find(self.cfg.n_starts)
        rng = np.random.default_rng(self.cfg.seed)
        chol = np.linalg.cholesky(mode.hessian_inv)
        starts = []
        for _ in range(self.cfg.n_chains):
            for _ in range(200):  # over-dispersed starts around the mode
                cand = mode.u + 1.5 * chol @ rng.standard_normal(mode.u.size)
                if np.isfinite(self.post(cand)):
                    break
            starts.append(cand)
        jobs = [(self.post, mode.hessian_inv, s, self.cfg.n_draws, self.cfg.n_burn, self.cfg.seed + 17 * k)
                for k, s in enumerate(starts)]
        with Pool(self.cfg.workers) as pool:
            chains = pool.map(_chain_job, jobs)
        info = {"mode_logpost": mode.logpost, "mode_attempts": mode.attempts,
                "mode": self.post.priors.to_params(mode.u)}
        return mode.u, chains, info

    # ------------------------------------------------------------------ reporting
    def posterior_table(self, chains: list[ChainResult]) -> pd.DataFrame:
        names = self.post.priors.names
        draws = np.stack([np.array([[p.to_param(x) for p, x in zip(self.post.priors.priors, row)]
                                    for row in c.draws]) for c in chains])
        rows = []
        for j, prior in enumerate(self.post.priors.priors):
            values = draws[:, :, j]
            diag = summarize(values)
            dist = prior.frozen()
            flat = values.ravel()
            rows.append({
                "parameter": names[j], "prior": prior.family, "prior_mean": prior.mean, "prior_sd": prior.sd,
                "post_mean": flat.mean(), "post_median": np.median(flat),
                "p05": np.quantile(flat, 0.05), "p95": np.quantile(flat, 0.95),
                "post_sd_over_prior_sd": flat.std() / dist.std(),
                **diag,
            })
        return pd.DataFrame(rows)

    def fit_table(self, params: dict[str, float]) -> pd.DataFrame:
        model, space = self.post.state_space(params)
        data = self.post.adjusted_data(model)
        out = KalmanFilter(space).run(data, keep=True)
        ok = np.all(np.isfinite(data[:, :len(OBSERVABLES)]), axis=1)
        rows = []
        for j, name in enumerate(self.obs.names[:len(OBSERVABLES)]):
            x = data[ok, j]
            err = x - out.predicted_obs[ok, j]
            series = data[:, j]
            lagged = np.column_stack((series[:-1], series[1:]))
            lagged = lagged[np.all(np.isfinite(lagged), axis=1)]
            b = np.polyfit(lagged[:, 0], lagged[:, 1], 1)
            ar_err = lagged[:, 1] - np.polyval(b, lagged[:, 0])
            rows.append({"observable": name, "sd_data": x.std(ddof=1),
                         "rmse_model_over_sd": np.sqrt(np.mean(err**2)) / x.std(ddof=1),
                         "rmse_ar1_over_sd": ar_err.std() / lagged[:, 1].std(),
                         "autocorr_data": pd.Series(series).autocorr()})
        return pd.DataFrame(rows)

    def informal_paths(self, chains: list[ChainResult]) -> pd.DataFrame:
        rng = np.random.default_rng(self.cfg.seed)
        pooled = np.vstack([c.draws for c in chains])
        idx = rng.choice(len(pooled), size=min(self.cfg.n_paths, len(pooled)), replace=False)
        smoother = SmoothedInformality(self.post, self.obs)
        theta_clean, s_clean, rho = [], [], []
        for i in idx:
            path = smoother.path(self.post.priors.to_params(pooled[i]))
            theta_clean.append(path["theta_clean"])
            s_clean.append(path["s_clean"])
            rho.append(float(np.median(path["rho_tilde"])))
        theta_clean, s_clean = np.array(theta_clean), np.array(s_clean)
        frame = pd.DataFrame({"quarter": self.obs.quarters.astype(str),
                              "informality_rate_observed": self.obs.informality_rate,
                              "informality_rate_clean_median": np.median(s_clean, axis=0)})
        for q, label in ((0.05, "p05"), (0.5, "median"), (0.95, "p95")):
            frame[f"informal_output_share_{label}"] = np.quantile(theta_clean, q, axis=0)
        frame.attrs["rho_tilde_draws"] = np.array(rho)
        return frame

    def anchor_map(self, s: np.ndarray) -> pd.DataFrame:
        """Informal output share for a grid of relative productivities (no estimation)."""
        grid = np.round(np.arange(0.20, 0.81, 0.05), 2)
        return pd.DataFrame({"quarter": self.obs.quarters.astype(str),
                             **{f"rho_{r:.2f}": self.share(r, s) for r in grid}})

    def share(self, ratio: float | np.ndarray, s: np.ndarray) -> np.ndarray:
        """Informal output share for a gross earnings ratio under the configured model."""
        if self.cfg.model == "labor_only":
            return output_share_labor_only(ratio, s, self.post.cal.alpha)
        return output_share_common(ratio, s)

    def run(self) -> dict:
        t0 = time.time()
        _, chains, info = self.sample()
        table = self.posterior_table(chains)
        table.to_csv(self.cfg.output / "posterior_summary.csv", index=False)
        pooled = np.vstack([c.draws for c in chains])
        names = self.post.priors.names
        draws_frame = pd.DataFrame([self.post.priors.to_params(u) for u in pooled[::5]])
        draws_frame.to_csv(self.cfg.output / "posterior_draws_thin5.csv", index=False)
        median = {n: float(table.set_index("parameter").loc[n, "post_median"]) for n in names}
        fit = self.fit_table(median)
        fit.to_csv(self.cfg.output / "fit_one_step.csv", index=False)
        paths = self.informal_paths(chains)
        paths.to_csv(self.cfg.output / "informal_output_share.csv", index=False)
        self.anchor_map(paths["informality_rate_clean_median"].to_numpy()).to_csv(
            self.cfg.output / "informal_share_by_relative_productivity.csv", index=False)
        summary = {
            "sample": [str(self.obs.quarters[0]), str(self.obs.quarters[-1])],
            "excluded_quarters": list(self.cfg.data_config.missing_quarters),
            "calibration": self.post.cal.__dict__,
            "chains": self.cfg.n_chains, "draws_per_chain": self.cfg.n_draws, "burn_in": self.cfg.n_burn,
            "acceptance": [c.acceptance for c in chains],
            "rhat_max": float(table["rhat"].max()),
            "ess_bulk_min": float(table["ess_bulk"].min()),
            "ess_tail_min": float(table["ess_tail"].min()),
            "converged": bool(table["rhat"].max() < 1.01 and table["ess_bulk"].min() > 400),
            "theta_prior": list(self.cfg.theta_prior),
            "model": self.cfg.model,
            "earnings": None if self.cfg.earnings_path is None else
            {"file": str(self.cfg.earnings_path), "column": self.cfg.earnings_column,
             "log_offset": (float(self.post.earnings_offset) if np.ndim(self.post.earnings_offset) == 0
                            else {"file": str(self.cfg.earnings_offset_path), "column": self.cfg.earnings_offset_column,
                                  "min": float(np.min(self.post.earnings_offset)),
                                  "max": float(np.max(self.post.earnings_offset))})},
            "rho_tilde_median": float(np.median(paths.attrs["rho_tilde_draws"])),
            "minutes": (time.time() - t0) / 60,
            **info,
        }
        (self.cfg.output / "run_summary.json").write_text(json.dumps(summary, indent=2, default=float))
        return summary
